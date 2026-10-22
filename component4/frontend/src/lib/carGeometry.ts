import * as THREE from "three";

/**
 * Builds a sedan from C3's five shape parameters (sample model).
 * Used (1) by the 3D viewer when no .glb file is found and
 *      (2) by scripts/make_sample_glb.mjs to write the sample .glb files.
 * The body is made from smooth cross-sections (a "loft"), so it has no hard box edges.
 * Units: metres. Front of the car is +X, up is +Y, width is Z. Wheels touch y = 0.
 */

const clamp01 = (t: number) => Math.min(1, Math.max(0, t));
/** 0 at a, 1 at b, smooth in between (works when b < a too). */
const smooth = (a: number, b: number, x: number) => {
  const t = clamp01((x - a) / (b - a));
  return t * t * (3 - 2 * t);
};
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const spow = (v: number, e: number) => Math.sign(v) * Math.pow(Math.abs(v), e);

interface Station {
  x: number;
  yb: number; // bottom of the cross-section
  yt: number; // top of the cross-section
  w: number; // half width
}

interface LoftOptions {
  rings: number; // points around one cross-section
  n: number; // roundness: 2 = ellipse, higher = boxier
  arc?: [number, number]; // only part of the ring (open surface, e.g. the roof panel)
  shape?: (x: number, y: number, z: number, st: Station) => [number, number, number];
  color?: (x: number, y: number, z: number, theta: number) => THREE.Color;
}

/** Joins cross-sections along X into one smooth surface. */
function loft(stations: Station[], o: LoftOptions): THREE.BufferGeometry {
  const open = !!o.arc;
  const [t0, t1] = o.arc ?? [0, Math.PI * 2];
  const M = o.rings;
  const per = open ? M + 1 : M;
  const pos: number[] = [];
  const col: number[] = [];
  for (const st of stations) {
    for (let j = 0; j < per; j++) {
      const th = t0 + ((t1 - t0) * j) / M;
      const zc = spow(Math.cos(th), 2 / o.n);
      const yc = spow(Math.sin(th), 2 / o.n);
      let y = (st.yb + st.yt) / 2 + ((st.yt - st.yb) / 2) * yc;
      let z = st.w * zc;
      let x = st.x;
      if (o.shape) [x, y, z] = o.shape(x, y, z, st);
      pos.push(x, y, z);
      if (o.color) {
        const c = o.color(x, y, z, th);
        col.push(c.r, c.g, c.b);
      }
    }
  }
  const idx: number[] = [];
  const S = stations.length;
  const jn = (j: number) => (open ? j + 1 : (j + 1) % M);
  const cnt = open ? M : M;
  for (let i = 0; i < S - 1; i++) {
    for (let j = 0; j < cnt; j++) {
      const a = i * per + j;
      const b = (i + 1) * per + j;
      const c = i * per + jn(j);
      const d = (i + 1) * per + jn(j);
      idx.push(a, b, c, b, d, c);
    }
  }
  if (!open) {
    // close both ends with a fan
    for (const [si, flip] of [[0, false], [S - 1, true]] as [number, boolean][]) {
      const st = stations[si];
      const centre = pos.length / 3;
      pos.push(st.x, (st.yb + st.yt) / 2, 0);
      if (o.color) {
        const c = o.color(st.x, (st.yb + st.yt) / 2, 0, -Math.PI / 2);
        col.push(c.r, c.g, c.b);
      }
      for (let j = 0; j < M; j++) {
        const a = si * per + j;
        const b = si * per + ((j + 1) % M);
        if (flip) idx.push(centre, b, a);
        else idx.push(centre, a, b);
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  if (o.color) g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

function xs(a: number, b: number, fine: number, coarse: number, edge: number): number[] {
  const out: number[] = [];
  let x = a;
  while (x < b - 1e-6) {
    out.push(x);
    x += x < a + edge || x > b - edge ? fine : coarse;
  }
  out.push(b);
  return out;
}

export function buildCar(p: Record<string, number>, color = "#4a86c5"): THREE.Group {
  const rad = (d: number) => (d * Math.PI) / 180;
  const gc = (p.ground_clearance_mm ?? 150) / 1000;
  const wsAngle = rad(p.windshield_angle_deg ?? 28);
  const hoodAngle = rad(p.hood_angle_deg ?? 10);
  const tailDeg = p.boat_tail_angle_deg ?? 15;
  const tailAngle = rad(tailDeg);
  const spoilerAngle = rad(p.spoiler_angle_deg ?? 10);

  // fixed sizes of the sample sedan
  const xf = 2.4; // front
  const xr = -2.4; // rear
  const W0 = 0.94; // half width
  const H = 1.48; // roof height
  const wheelR = 0.345;
  const wheelX = 1.45;
  const wheelZ = 0.8;
  const arcR = 0.38;

  // ---- key heights and positions that depend on the shape parameters
  const yb0 = gc + 0.08;
  const xCowl = 0.95;
  const xNose = xf - 0.2;
  const yHoodFront = gc + 0.62;
  const yCowl = yHoodFront + Math.tan(hoodAngle) * (xNose - xCowl);
  const yBelt = yCowl + 0.02;
  const yRoofEdge = H - 0.045;
  const xRoofFront = xCowl - (yRoofEdge - yBelt) / Math.tan(wsAngle);
  const xRoofRear = xRoofFront - 1.05;
  const backAngle = rad(12 + 1.3 * tailDeg);
  const xGlassEnd = xRoofRear - (yRoofEdge - yBelt) / Math.tan(backAngle);
  const yBeltCabin = yBelt - 0.06;

  const paint = new THREE.Color(color);
  const dark = new THREE.Color("#171a1e");
  const lightWhite = new THREE.Color("#eef4fa");
  const lightRed = new THREE.Color("#b3131b");

  // ---- body (hood, doors, boot) -------------------------------------------------
  const yBottom = (x: number) => {
    let y = yb0;
    for (const xw of [wheelX, -wheelX]) {
      const dx = Math.abs(x - xw);
      if (dx < arcR) y = Math.max(y, wheelR + Math.sqrt(arcR * arcR - dx * dx));
    }
    return y + 0.06 * smooth(xf - 0.45, xf, x) + 0.1 * smooth(xr + 0.5, xr, x);
  };
  const inArch = (x: number) => [wheelX, -wheelX].some((xw) => Math.abs(x - xw) < arcR + 0.02);

  const bodyTop = (x: number) => {
    if (x >= xNose) {
      const t = (x - xNose) / (xf - xNose);
      const s = Math.sqrt(Math.max(0, 1 - t * t));
      const yTip = yBottom(x) + 0.17;
      return yTip + (yHoodFront - yTip) * s;
    }
    if (x >= xCowl) return yCowl - Math.tan(hoodAngle) * (x - xCowl);
    if (x >= xGlassEnd) return yBelt;
    const yDeck = yBelt - Math.tan(tailAngle) * (xGlassEnd - x) * 0.7;
    const xRound = xr + 0.28;
    if (x <= xRound) {
      const t = (xRound - x) / (xRound - xr);
      const s = Math.sqrt(Math.max(0, 1 - t * t));
      const yTip = yBottom(x) + 0.2;
      const yAt = yBelt - Math.tan(tailAngle) * (xGlassEnd - xRound) * 0.7;
      return yTip + (yAt - yTip) * s;
    }
    return yDeck;
  };

  const bodyWidth = (x: number) => {
    let w = W0 * (1 - 0.16 * smooth(1.3, xf, x));
    w *= 1 - 0.9 * Math.tan(tailAngle) * smooth(-1.2, xr, x);
    if (x >= xNose) w *= 0.62 + 0.38 * Math.sqrt(Math.max(0, 1 - Math.pow((x - xNose) / (xf - xNose), 2)));
    if (x <= xr + 0.28) w *= 0.7 + 0.3 * Math.sqrt(Math.max(0, 1 - Math.pow((xr + 0.28 - x) / 0.28, 2)));
    return w;
  };

  const bodyStations: Station[] = xs(xr, xf, 0.012, 0.03, 0.5).map((x) => ({
    x,
    yb: yBottom(x),
    yt: Math.max(bodyTop(x), yBottom(x) + 0.1),
    w: bodyWidth(x),
  }));

  const bodyColor = (x: number, y: number, z: number, th: number) => {
    const sn = Math.sin(th);
    let d = smooth(-0.5, -0.88, sn);
    if (inArch(x) && sn < 0.2) d = 1;
    if (x > xf - 0.5 && Math.abs(z) < 0.55 && y < yBottom(x) + 0.28) d = Math.max(d, 0.95); // air intake
    if (x < xr + 0.35 && y < yBottom(x) + 0.16) d = Math.max(d, 0.9); // rear diffuser
    const c = paint.clone().lerp(dark, d);
    // headlights (front corners) and tail lights (rear corners)
    if (x > xf - 0.32 && Math.abs(z) > 0.42 && Math.abs(z) < 0.8 && sn > 0.1 && sn < 0.92) {
      c.lerp(lightWhite, 0.92);
    }
    if (x < xr + 0.22 && Math.abs(z) > 0.32 && sn > 0.05 && sn < 0.85) {
      c.lerp(lightRed, 0.9);
    }
    return c;
  };

  const paintMat = new THREE.MeshPhysicalMaterial({
    color: "#ffffff",
    vertexColors: true,
    metalness: 0.55,
    roughness: 0.3,
    clearcoat: 1,
    clearcoatRoughness: 0.06,
  });
  const solidPaint = new THREE.MeshPhysicalMaterial({
    color: paint,
    metalness: 0.55,
    roughness: 0.3,
    clearcoat: 1,
    clearcoatRoughness: 0.06,
  });
  const glassMat = new THREE.MeshPhysicalMaterial({
    color: "#16293a",
    metalness: 0.8,
    roughness: 0.08,
    clearcoat: 1,
    clearcoatRoughness: 0.02,
  });
  const blackMat = new THREE.MeshStandardMaterial({ color: "#14161a", roughness: 0.85, metalness: 0.1 });
  const rimMat = new THREE.MeshStandardMaterial({ color: "#d8dee3", roughness: 0.3, metalness: 0.65 });
  const rimDarkMat = new THREE.MeshStandardMaterial({ color: "#454c53", roughness: 0.5, metalness: 0.4 });

  const car = new THREE.Group();

  const body = new THREE.Mesh(loft(bodyStations, { rings: 64, n: 2.6, color: bodyColor }), paintMat);
  body.name = "body";
  car.add(body);

  // ---- cabin (glass) -----------------------------------------------------------
  const xm = (xRoofFront + xRoofRear) / 2;
  const hl = (xRoofFront - xRoofRear) / 2;
  const cabinTop = (x: number) => {
    if (x >= xRoofFront) return yBelt + (xCowl - x) * Math.tan(wsAngle); // windscreen
    if (x >= xRoofRear) return H - 0.045 * Math.pow((x - xm) / hl, 2); // roof, slightly curved
    return yRoofEdge - (xRoofRear - x) * Math.tan(backAngle); // rear window
  };
  const cabinWidth = (x: number) =>
    0.8 -
    0.07 * smooth(xCowl, xRoofFront, x) -
    (0.06 + 0.25 * Math.tan(tailAngle)) * smooth(xRoofRear, xGlassEnd, x);
  const tumblehome = (x: number, y: number, z: number, st: Station): [number, number, number] => [
    x,
    y,
    z * (1 - 0.2 * clamp01((y - st.yb) / (H - st.yb))),
  ];
  const cabinStation = (x: number, grow = 0): Station => ({
    x,
    yb: yBeltCabin,
    yt: Math.max(cabinTop(x), yBelt) + grow,
    w: cabinWidth(x) + grow,
  });
  const cabinXs = xs(xGlassEnd, xCowl, 0.02, 0.04, 0.3);

  const cabin = new THREE.Mesh(
    loft(cabinXs.map((x) => cabinStation(x)), { rings: 40, n: 2.4, shape: tumblehome }),
    glassMat,
  );
  cabin.name = "glass";
  car.add(cabin);

  // roof panel and the two pillars (painted, drawn just outside the glass)
  const roof = new THREE.Mesh(
    loft(xs(xRoofRear, xRoofFront, 0.04, 0.04, 0).map((x) => cabinStation(x, 0.006)), {
      rings: 14,
      n: 2.4,
      arc: [Math.PI / 2 - 0.62, Math.PI / 2 + 0.62],
      shape: tumblehome,
    }),
    solidPaint,
  );
  roof.name = "roof";
  car.add(roof);
  for (const xb of [xm - 0.1, xRoofRear - 0.12]) {
    const pillar = new THREE.Mesh(
      loft([xb - 0.05, xb, xb + 0.05].map((x) => cabinStation(x, 0.006)), { rings: 40, n: 2.4, shape: tumblehome }),
      solidPaint,
    );
    pillar.name = "pillar";
    car.add(pillar);
  }

  // mirrors
  for (const side of [-1, 1]) {
    const mirror = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.09, 0.17), solidPaint);
    mirror.position.set(xCowl - 0.08, yBelt + 0.05, side * (cabinWidth(xCowl) + 0.01));
    car.add(mirror);
  }

  // ---- rear spoiler: pitch comes from spoiler_angle_deg ----------------------------
  const xSp = xr + 0.5;
  const spoiler = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.022, 2 * bodyWidth(xSp) * 0.8), solidPaint);
  spoiler.position.set(xSp, bodyTop(xSp) + 0.03, 0);
  spoiler.rotation.z = -spoilerAngle; // rear edge goes up
  spoiler.name = "spoiler";
  car.add(spoiler);

  // ---- wheels --------------------------------------------------------------------------
  const tireProfile = [
    [0.21, -0.12], [0.29, -0.125], [0.33, -0.1], [0.343, -0.05], [0.345, 0],
    [0.343, 0.05], [0.33, 0.1], [0.29, 0.125], [0.21, 0.12],
  ].map(([r, a]) => new THREE.Vector2(r, a));
  const tireGeo = new THREE.LatheGeometry(tireProfile, 40);
  const rimGeo = new THREE.CylinderGeometry(0.215, 0.215, 0.22, 32);
  const faceGeo = new THREE.CylinderGeometry(0.2, 0.2, 0.012, 32);
  const spokeGeo = new THREE.BoxGeometry(0.2, 0.045, 0.014);
  const capGeo = new THREE.CylinderGeometry(0.045, 0.045, 0.016, 20);
  for (const xw of [wheelX, -wheelX]) {
    for (const side of [-1, 1]) {
      const wheel = new THREE.Group();
      wheel.name = "wheel";
      const tire = new THREE.Mesh(tireGeo, blackMat);
      tire.rotation.x = Math.PI / 2;
      wheel.add(tire);
      const rim = new THREE.Mesh(rimGeo, rimMat);
      rim.rotation.x = Math.PI / 2;
      wheel.add(rim);
      const face = new THREE.Mesh(faceGeo, rimDarkMat);
      face.rotation.x = Math.PI / 2;
      face.position.z = side * 0.114;
      wheel.add(face);
      for (let k = 0; k < 5; k++) {
        const holder = new THREE.Group();
        holder.rotation.z = (k * Math.PI * 2) / 5 + 0.3;
        const sp = new THREE.Mesh(spokeGeo, rimMat);
        sp.position.set(0.1, 0, side * 0.12);
        holder.add(sp);
        wheel.add(holder);
      }
      const cap = new THREE.Mesh(capGeo, rimMat);
      cap.rotation.x = Math.PI / 2;
      cap.position.z = side * 0.122;
      wheel.add(cap);
      wheel.position.set(xw, wheelR, side * wheelZ);
      car.add(wheel);
    }
  }

  return car;
}