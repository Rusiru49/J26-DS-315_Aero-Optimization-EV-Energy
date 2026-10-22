import { Canvas, useThree } from "@react-three/fiber";
import { ContactShadows, Environment, Lightformer, OrbitControls, useGLTF } from "@react-three/drei";
import { Suspense, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import type { Config } from "../api";
import { buildCar } from "../lib/carGeometry";

type Mode = "baseline" | "optimized" | "both";
type Source = "file" | "shape";

const BASE_COLOR = "#4a86c5";
const OPT_COLOR = "#19a58e";

const existsCache = new Map<string, boolean>();

/** Is there a real .glb file in /public/models? (Vite answers 200 with a web page for missing files, so we check the file header.) */
function useGlbExists(file: string | null): "checking" | "yes" | "no" {
  const cached = file ? existsCache.get(file) : false;
  const [state, setState] = useState<"checking" | "yes" | "no">(
    cached === undefined ? "checking" : cached ? "yes" : "no",
  );
  useEffect(() => {
    let alive = true;
    if (!file) {
      setState("no");
      return;
    }
    const known = existsCache.get(file);
    if (known !== undefined) {
      setState(known ? "yes" : "no");
      return;
    }
    setState("checking");
    fetch(`/models/${file}`)
      .then((r) => r.arrayBuffer())
      .then((buf) => {
        const magic = new TextDecoder().decode(new Uint8Array(buf, 0, Math.min(4, buf.byteLength)));
        existsCache.set(file, magic === "glTF");
        if (alive) setState(magic === "glTF" ? "yes" : "no");
      })
      .catch(() => {
        existsCache.set(file, false);
        if (alive) setState("no");
      });
    return () => {
      alive = false;
    };
  }, [file]);
  return state;
}

/** Moves the camera so everything inside `target` fills the view. Runs again when `trigger` changes. */
function FitCamera({ target, trigger }: { target: React.RefObject<THREE.Group | null>; trigger: string }) {
  const { camera, size, controls, invalidate } = useThree();
  useLayoutEffect(() => {
    const fit = () => {
      const group = target.current;
      if (!group) return;
      const box = new THREE.Box3().setFromObject(group);
      if (box.isEmpty() || size.width === 0 || size.height === 0) return;
      const center = box.getCenter(new THREE.Vector3());
      const dims = box.getSize(new THREE.Vector3());
      const cam = camera as THREE.PerspectiveCamera;
      const vfov = (cam.fov * Math.PI) / 180;
      const hfov = 2 * Math.atan(Math.tan(vfov / 2) * (size.width / size.height));
      const dist = Math.max(dims.x / 2 / Math.tan(hfov / 2), dims.y / 2 / Math.tan(vfov / 2)) * 1.5 + dims.z / 2;
      const dir = new THREE.Vector3(0.18, 0.3, 1).normalize();
      cam.position.copy(center).addScaledVector(dir, dist);
      cam.near = 0.1;
      cam.far = dist * 10 + 100;
      cam.updateProjectionMatrix();
      cam.lookAt(center);
      const orbit = controls as unknown as { target: THREE.Vector3; update: () => void } | null;
      if (orbit) {
        orbit.target.copy(center);
        orbit.update();
      }
      invalidate();
    };
    fit();
    // fit again a moment later: the panel size and the model files can settle after the first fit
    const t1 = setTimeout(fit, 300);
    const t2 = setTimeout(fit, 1200);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, [trigger, size.width, size.height, camera, controls, target, invalidate]);
  return null;
}

function GlbModel({ url, onReady }: { url: string; onReady: () => void }) {
  const { scene } = useGLTF(url);
  const clone = useMemo(() => scene.clone(true), [scene]);
  useEffect(onReady, [onReady]);
  return <primitive object={clone} />;
}

function ShapeModel({ config, color, onReady }: { config: Config; color: string; onReady: () => void }) {
  const car = useMemo(() => buildCar(config.shape_parameters, color), [config, color]);
  useEffect(onReady, [onReady]);
  return <primitive object={car} />;
}

function CarModel({
  config,
  color,
  position,
  onSource,
  onReady,
}: {
  config: Config;
  color: string;
  position: [number, number, number];
  onSource: (id: string, s: Source) => void;
  onReady: (id: string) => void;
}) {
  const ready = useCallback(() => onReady(config.config_id), [onReady, config.config_id]);
  const glb = useGlbExists(config.geometry_file);
  useEffect(() => {
    if (glb === "yes") onSource(config.config_id, "file");
    if (glb === "no") onSource(config.config_id, "shape");
  }, [glb, config.config_id, onSource]);
  if (glb === "checking") return null;
  return (
    <group position={position}>
      {glb === "yes" ? (
        <Suspense fallback={null}>
          <GlbModel url={`/models/${config.geometry_file}`} onReady={ready} />
        </Suspense>
      ) : (
        <ShapeModel config={config} color={color} onReady={ready} />
      )}
    </group>
  );
}

export default function Viewer3D({ baseline, optimized }: { baseline: Config; optimized: Config }) {
  const [mode, setMode] = useState<Mode>("both");
  const [sources, setSources] = useState<Record<string, Source>>({});
  const onSource = useMemo(
    () => (id: string, s: Source) => setSources((prev) => (prev[id] === s ? prev : { ...prev, [id]: s })),
    [],
  );

  // The camera fits itself to what is in the scene. We refit when a car has really appeared.
  const [ready, setReady] = useState<string[]>([]);
  const onReady = useCallback(
    (id: string) => setReady((prev) => (prev.includes(id) ? prev : [...prev, id].sort())),
    [],
  );

  const carsRef = useRef<THREE.Group>(null);
  const showBase = mode !== "optimized";
  const showOpt = mode !== "baseline";
  const gap = mode === "both" ? 2.9 : 0; // the cars stand next to each other along the X axis

  const usingFiles = Object.values(sources).length > 0 && Object.values(sources).every((s) => s === "file");
  const usingShape = Object.values(sources).some((s) => s === "shape");

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="mb-2 flex flex-none flex-wrap items-center justify-between gap-2">
        <div className="inline-flex rounded-xl border border-line bg-white p-0.5 text-xs">
          {(
            [
              ["baseline", "Baseline"],
              ["optimized", "Optimized"],
              ["both", "Side by side"],
            ] as [Mode, string][]
          ).map(([m, label]) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`rounded-lg px-2.5 py-1 font-medium transition ${
                mode === m ? "bg-brand text-white shadow-sm" : "text-muted hover:text-ink"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-3 text-xs text-muted">
          <span className="flex items-center gap-1">
            <i className="inline-block h-2.5 w-2.5 rounded-full bg-base" /> Baseline
          </span>
          <span className="flex items-center gap-1">
            <i className="inline-block h-2.5 w-2.5 rounded-full bg-opt" /> Optimized
          </span>
        </div>
      </div>

      <div className="min-h-[180px] flex-1 overflow-hidden rounded-xl border border-line bg-gradient-to-b from-[#eef6fb] to-[#dfeef5]">
        <Canvas frameloop="demand" dpr={[1, 2]} camera={{ position: [3, 3.2, 11], fov: 34 }}>
          <ambientLight intensity={0.35} />
          <hemisphereLight args={["#ffffff", "#bcd3e2", 0.35]} />
          <directionalLight position={[5, 8, 4]} intensity={1.4} />
          <directionalLight position={[-6, 4, -5]} intensity={0.5} />
          <Environment resolution={256} frames={1}>
            <Lightformer form="rect" intensity={2.4} position={[0, 6, 3]} scale={[14, 5, 1]} />
            <Lightformer form="rect" intensity={1.6} position={[-7, 2, 0]} rotation-y={Math.PI / 2} scale={[10, 3, 1]} />
            <Lightformer form="rect" intensity={1.6} position={[7, 2, 0]} rotation-y={-Math.PI / 2} scale={[10, 3, 1]} />
            <Lightformer form="rect" intensity={1.2} position={[0, 2, -7]} rotation-y={Math.PI} scale={[14, 3, 1]} />
          </Environment>
          <group ref={carsRef}>
            {showBase && (
              <CarModel config={baseline} color={BASE_COLOR} position={[-gap, 0, 0]} onSource={onSource} onReady={onReady} />
            )}
            {showOpt && (
              <CarModel config={optimized} color={OPT_COLOR} position={[gap, 0, 0]} onSource={onSource} onReady={onReady} />
            )}
          </group>
          <FitCamera target={carsRef} trigger={`${mode}|${ready.join(",")}`} />
          <ContactShadows position={[0, 0, 0]} opacity={0.35} scale={14} blur={2.4} far={3} />
          <OrbitControls makeDefault enablePan={false} minPolarAngle={0.2} maxPolarAngle={Math.PI / 2.05} />
        </Canvas>
      </div>

      <p className="mt-1 flex-none text-[11px] leading-tight text-muted">
        Drag to rotate, scroll to zoom.{" "}
        {usingFiles && <>Showing the model files {baseline.geometry_file} and {optimized.geometry_file}.</>}
        {usingShape && <>Model file not found in public/models, so the body is drawn from the shape parameters.</>}
      </p>
    </div>
  );
}