# Component 2: සම්පූර්ණ චිත්‍රය සහ "ඇයි මේක මෙහෙම කළේ" (Overview)

**S.I.C. Senevirathne (IT23213708), Component 2, Project J26-DS-315 (EV Aero-Optimizer)**
මේ ලිපි තුනෙන් මේක පළමුවැන්නයි: සම්පූර්ණ චිත්‍රය. වෙනම ලිපි: `02_CAD_Model_Workflow.md` (CAD කොටස), `03_Simulation_Workflow.md` (Simulation කොටස).

> **ලකුණු:** `[confirm]` කියන තැන් ඔයා ඔයාගේ ඇත්ත අගයෙන්/මෙවලමෙන් තහවුරු කරන්න ඕන දේවල්. මම ඒවා අනුමාන කරලා නැහැ.

---

## 1. Project එක මොකක්ද

**EV Aero-Optimizer**: Sri Lanka තත්ත්ව යටතේ electric vehicle එකක aerodynamic drag අඩු කරලා energy/range වැඩි කරන්න ML (surrogate) පාදක පද්ධතියක් හදන එක.

ඒක කොටස් 4කට බෙදිලා තියෙනවා:

| Component | කවුද | මොකක්ද කරන්නේ |
|---|---|---|
| **1** | Fernando T.R.T (IT23278080) | Sri Lankan Driving Cycle (SLDC) සහ design space (shape parameters 5, LHS sampling) |
| **2 (ඔයා)** | S.I.C. Senevirathne | CAD designs + CFD simulations → **ground-truth Cd dataset** |
| **3** | L.L. Chandanayaka | CTGAN + XGBoost/ANN surrogate model (Cd predict කරන ML) |
| **4** | Wijerathne | Dashboard, energy/range, 3D visualisation (Three.js) |

**ඔයාගේ role එක = "Data Engine".** Component 3 ගේ ML ඉගෙන ගන්නේ ඔයා හදන Cd දත්ත වලින්. ඒවා වැරදි නම් surrogate, optimisation, dashboard සියල්ලම වැරදියි. ඒකයි ඔයාගේ වැඩේ වැදගත්.

## 2. ඇයි CFD data ඕන (ප්‍රශ්නය)

- Drag force = ½ · ρ · Cd · A · v². Energy use එකට Cd (හැඩයෙන් එන) සහ v (ධාවන වේගය) අවශ්‍යයි.
- Real wind-tunnel data හෝ ANSYS CFD data **Sri Lanka තත්ත්ව** (tropical air, local speeds) සඳහා publicly නැහැ.
- Manual CAD → CFD → CAD කරලා designs සිය ගණනක් කරන්න බැහැ. ඒක slow, reproducible නැහැ.
- සාමාන්‍ය datasets වල Cd විතරයි. Uncertainty, convergence, mesh quality නැහැ.

**ඒ නිසා ඔයාගේ විසඳුම:** CAD designs ස්වයංක්‍රීයව හදලා, OpenFOAM වලින් ඒ සියල්ල එකම ක්‍රමයට simulate කරලා, Cd + surface pressure (Cp) + quality flags + uncertainty සහිත dataset එකක් දෙනවා.

## 3. සම්පූර්ණ workflow (එක බැල්මකින්)

```
[Component 1] Design space: parameters 5 + LHS sampling
        │
        ▼
┌─────────── CAD කොටස (ලිපිය 02) ───────────┐
│ Reference → defeature → loft shell (D000)  │
│ 5 parameters → design table (LHS)          │
│ designs හදලා STEP/STL export + watertight  │
└───────────────────────────────────────────┘
        │  (car surface STL)
        ▼
┌─────────── Simulation කොටස (ලිපිය 03) ───────────┐
│ Wind tunnel domain (blockMesh)                    │
│ snappyHexMesh (D000 mesh) + mesh-independence     │
│ FFD morph (design එකක් = mesh එක morph)           │
│ simpleFoam k-ω SST, 100 සහ 40 km/h, tropical air  │
│ Cd (forceCoeffs) + Cp field (foamToVTK → NPZ)     │
│ QC: checkMesh, residuals, Cd stability, re-mesh   │
└───────────────────────────────────────────────────┘
        │
        ▼
Dataset: run_table.csv + per-face NPZ ──► Component 3 (surrogate)
        │
        ▼
Data-science analysis + Dashboard ──► Component 4 / panel
```

## 4. තීරණ සහ "ඇයි" (Why we used each thing)

| තීරණය | ඇයි |
|---|---|
| **OpenFOAM (ANSYS නෙවෙයි)** | නොමිලේ, open-source, scripting (headless batch) පහසුයි. ANSYS Fluent licence එකක් ඕන, GUI-based. ඒත් **ANSYS එක්ක validate කරලා නැහැ**, ඒක honest disclosure එකක්. |
| **simpleFoam + k-ω SST** | Steady-state, incompressible RANS. වාහන aerodynamics වලට විවිධ-flow separation හොඳින් හසුකරන, නිතර පාවිච්චි වන turbulence model එකක්. කාලය අඩුයි (transient LES ට වඩා). |
| **Tropical air (≈30 °C) + 100 සහ 40 km/h** | Sri Lanka highway සහ urban වේග, local වායුගෝලය. European/standard air නෙවෙයි (ρ = 1.165 kg/m³ `[confirm]` case එකේ). |
| **D000 එකම base mesh + FFD morph** | සෑම design එකටම අලුතින් mesh කළොත් faces/order වෙනස්. Component 3 ට pressure patterns සසඳන්න **එකම faces, එකම පිළිවෙල** ඕන. Morph එකෙන් ඒක ලැබෙනවා. බිලි: morph වලදී mesh quality අඩු වෙනවා. |
| **Coarse mesh (~0.57 M cells)** | 8 GB MacBook Air M2 එකේ designs ගණනාවක් run කරන්න ඕන. සියුම් mesh එකක් නොහැකි. බිලි: absolute Cd නිරවද්‍ය නැහැ, designs අතර වෙනස්කම් විතරයි විශ්වාසයි. |
| **Mesh-independence study (D000)** | Mesh size එක Cd වෙනස් කරනවාද කියලා පෙන්වන්න. නැත්නම් "ඇයි මේ mesh එක?" කියන ප්‍රශ්නයට පිළිතුරක් නැහැ. |
| **Cd = අන්තිම iterations 100 හි සාමාන්‍යය** | Steady solver එකේ පවා Cd oscillate වෙනවා. අවසාන iteration එක විතරක් ගත්තොත් noise. Standard error (SE = std(last100)/√n_eff) ද දෙනවා. |
| **Cp surface field (NPZ)** | Cd විතරක් නෙවෙයි, ඒක හේතු වෙන pressure distribution එක Component 3 ට (PCA/ANN) සහ Component 4 ට (3D colour) ඕන. |
| **Quality flags + uncertainty** | සමහර runs විශ්වාස කරන්න බැහැ (mesh checks fail, Cd unstable, morph error). ML එකට weights දෙන්න ඕන. |
| **Python batch scripts** | Manual click නැතුව, reproducible, අන්සේ බලපෑම අඩු. |
| **Data-science analysis (8ක්)** | Panel feedback: "data science scope එකක් එකතු කරන්න". Coverage, surrogate, Sobol, uncertainty, PCA, active learning, what-if. |
| **Dashboard (HTML → Next.js)** | Simulation process එක සහ ප්‍රතිඵල real data වලින් පෙන්වන්න (panel + Component 4 ට). |

## 5. Component 3 සහ 4 සමඟ සම්බන්ධය

- **Component 3 ට** (ඔහුගේ request file එකට අනුව): හැම design × speed එකකට NPZ (`centre, area, normal, p [Pa], tau [Pa]`, float32), `run_table.csv` (Cd, Cd_mean, Cd_std, Cd_p, Cd_f, Cl, A_frontal, q_ref, quality columns), `design_table.csv` (`is_corner`, `run_order`), parameter නම් (`windshield_angle_deg, spoiler_angle_deg, boat_tail_angle_deg, hood_angle_deg, ground_clearance_mm`), ඒකක (m, m²). **ඔයාගේ දැන් තියෙන files ඒ format එකට ගැලපෙනවාද කියලා තහවුරු කරන්න** `[confirm]`.
- **Component 4 ට**: 3D car (Cp), flow results, dashboard (Three.js ට GLB අවශ්‍ය නම් පස්සේ).
- **Component 1 ට**: parameter ranges (windshield/hood) තහවුරු කිරීම; sensitivity analysis feedback.

## 6. අවංක තත්ත්වය (Panel එකේ කියන්න ඕන දේවල්)

- OpenFOAM, ANSYS නෙවෙයි; coarse mesh (~0.57 M cells).
- Design space එකේ 70 designs (+ D000), simulate කළේ **30ක්**. ඉතිරි 40 active learning වලින් තෝරනවා.
- `checkMesh`: සියලු runs වල concave-cell check එක fail වෙනවා; සමහර morphed designs වල non-orthogonality 70° ට වැඩියි (D000: 64.7°). Absolute Cd වලට පරිස්සම් වෙන්න, relative differences විතරයි.
- Morph එකෙන් Cd අඩුවෙන් predict කරන ප්‍රවණතාවක් (≈ −11.8 % දක්වා); re-mesh validation සීමිතයි.
- "Cd stability flag" අපේම rule එකක් (>1 % drift/oscillation), solver verdict එකක් නෙවෙයි.
- Proposal එකේ 200 designs / ANSYS / Cl, gradient data කියලා තිබුණා; ඇත්තටම කළ දේ මෙහෙමයි. Presentation එකේ ඇත්ත අගයන් පාවිච්චි කරන්න.

## 7. Panel එකට සරල වාක්‍ය 3ක්

1. "Component 2 turns the design space from Component 1 into physics-based drag data under Sri Lankan conditions."
2. "Every design is simulated the same way (one baseline mesh, FFD-morphed), so the data are comparable and ready for Component 3."
3. "We show uncertainty and limitations openly: OpenFOAM, a coarse mesh, and a subset of designs."
