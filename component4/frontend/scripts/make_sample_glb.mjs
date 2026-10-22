// Writes sample .glb files (one per design in the data file) into public/models/.
// Run from the frontend folder:
//   node --experimental-strip-types scripts/make_sample_glb.mjs
// When C3 / Member 2 gives you the real .glb files, just copy them into public/models/
// with the names listed as "geometry_file" in the data. Nothing else changes.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { GLTFExporter } from "three/addons/exporters/GLTFExporter.js";
import { buildCar } from "../src/lib/carGeometry.ts";

// GLTFExporter needs FileReader (it only exists in browsers), so give it a tiny stand-in.
globalThis.FileReader = class {
  readAsArrayBuffer(blob) {
    blob.arrayBuffer().then((buf) => {
      this.result = buf;
      this.onloadend?.();
    });
  }
};

const here = path.dirname(fileURLToPath(import.meta.url));
const dataFile = path.resolve(here, "../../data/c3_package_dummy.json");
const outDir = path.resolve(here, "../public/models");
fs.mkdirSync(outDir, { recursive: true });

const pkg = JSON.parse(fs.readFileSync(dataFile, "utf-8"));
const exporter = new GLTFExporter();

for (const ev of pkg.evs) {
  for (const cfg of ev.configurations) {
    const color = cfg.type === "baseline" ? "#4a86c5" : "#19a58e";
    const car = buildCar(cfg.shape_parameters, color);
    const glb = await exporter.parseAsync(car, { binary: true });
    const file = path.join(outDir, cfg.geometry_file);
    fs.writeFileSync(file, Buffer.from(glb));
    console.log("wrote", path.relative(process.cwd(), file), `(${Math.round(glb.byteLength / 1024)} KB)`);
  }
}