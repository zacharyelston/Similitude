# Similitude: a shared platform for machines that work in a medium

*Proposal, 13 November 2026. Decided on 13 November 2026: the name Similitude, the Apache-2.0 license and a
public repo, `zacharyelston/Similitude`. The project began as Froude, a fluids-only optimizer; the old plan is in
this file's history.*

## Summary

Three public projects are building the same platform, each in its own way:

- **[Osprey](https://github.com/zacharyelston/Osprey):** flying and floating craft.
- **[eOx](https://github.com/zacharyelston/EOx):** an electric tractor cut from sheet steel.
- **[AtlanteanConstructor](https://github.com/zacharyelston/AtlanteanConstructor):** earthworks dug by a
  tethered eOx.

Between them they already have:

- design files and engineering checks;
- an evolutionary design search;
- MuJoCo models built from the design;
- models of water and of ground;
- a CAD-to-manufacturing path;
- Blender views and kept runs.

They also copy each other's code, and the copies are drifting apart.

Similitude pulls those pieces into one platform: a Python core, with Rust for field solvers and hot loops. Each
project keeps its own designs, its special models and its runs, and depends on Similitude for the rest.

The guiding idea is a machine working in a medium: air, water, ground or a field (electromagnetic, thermal,
potential flow, seepage). Each medium gets a ladder of models. Cheap ones run in the design search, and
expensive ones are run now and then to check the cheap ones and correct them.

## The three projects today

| | Osprey | eOx | AtlanteanConstructor |
|---|---|---|---|
| What it does | Sizes, floats and simulates crewed and drone craft | Designs and checks an electric tractor kit cut from sheet steel | Simulates digging ring moats and berms |
| Language | Python, with a Rust sim loop through PyO3 | Python | Python and NumPy |
| Main tools | OpenMDAO, OpenAeroStruct, MuJoCo, Blender | build123d, MuJoCo, ezdxf, Blender | NumPy, Blender |
| Strongest reusable piece | sizing with gradients; hydrostatics; MuJoCo with our own water forces | the trial engine (genes, objectives, constraints); checks; tyres on soil; sheet-metal rules | the terrain engine (bank and loose soil, exact mass balance, slumping); cost baselines |

## What they repeat

| Piece | Osprey | eOx | AtlanteanConstructor |
|---|---|---|---|
| Design as data, with overrides | TOML, `**over` arguments | TOML, dotted `[overrides]` | JSON, dotted overrides on the command line |
| Engineering checks | `calc` exits 1 on FAIL | `checks.py` PASS/WARN/FAIL | `config.py` validation |
| Exploring designs | sweep lists written by hand | the GA trial engine | a scenario × vehicle matrix |
| Dynamics built from the design | MuJoCo (water) | MuJoCo (terrain, tyres) | step-by-step tether physics |
| The medium | water, air | ground and tyres | terrain and soil |
| Geometry | numpy meshes | build123d, STEP, DXF | heightmaps |
| Viewing | Blender, matplotlib | Blender, SVG in a set palette | Blender, hillshades |
| Kept runs | `runs/` with the commit | trial history | `runs/` |

## Where they're drifting

- **Tyres:** AtlanteanConstructor carries a copy of eOx's tyre model with the note "keep the two in step".
  Nothing enforces that.
- **Part data:** AtlanteanConstructor hard-codes eOx's part data from one pinned eOx commit.
- **Repeated machinery:** three parameter loaders, three ways to sweep designs, three Blender renderers, three
  figure styles, and two MuJoCo builders.

## The platform

### 1. The core (Python)

- **Design schema:**
  - designs in TOML, with dotted overrides;
  - uncertain values as ranges, such as `{ref = 9.0, range = [7, 12]}`;
  - units carried with the values.
- **Checks:** PASS/WARN/FAIL, as in eOx. A run exits non-zero on a FAIL, so CI can gate on it.
- **Trials:**
  - Start from eOx's trial engine: genes, objectives, constraints, and every candidate getting the full checks.
    Its rule carries over too: a loophole a trial finds becomes a new check.
  - Add pymoo for trade-off fronts across several goals, BoTorch for expensive runs, and OpenTURNS to
    propagate the uncertain ranges.
- **Runs and provenance:**
  - kept runs record the commit they came from, and are never edited;
  - each run's pass/fail test is written down before the run.
- **Figures:** one validated palette for every figure.
- **Remote runs:** heavy jobs run on `gpu1` and `gpu2` from a `git archive` snapshot of a commit.

### 2. Geometry and manufacturing

- build123d is the source of truth, as eOx already decided: watertight solids, STEP and DXF.
- eOx's sheet-metal and SendCutSend rules apply to any part that is ordered cut and bent.
- Exports:
  - surface meshes for hydrostatics and Capytaine (Osprey's clipper tests carry over);
  - volume meshes from Gmsh;
  - heightfields for terrain;
  - simplicial complexes for the field solvers;
  - Blender scenes for viewing.

### 3. Media, each with a ladder of models

| Medium | Cheap (in the search) | Mid | Expensive (checks) |
|---|---|---|---|
| Air | AeroSandbox buildup, NeuralFoil | AVL, vortex-lattice, DUST or FLOWUnsteady for props and wings together | SU2 or OpenFOAM |
| Water | Osprey's hydrostatics and MuJoCo water | Capytaine: added mass, damping, wave forces | DualSPHysics or OpenFOAM `interFoam` for slamming |
| Ground | eOx's Brixius tyres; AtlanteanConstructor's terrain engine and handbook production rates | terramechanics (Bekker–Wong) | discrete-element soil (Project Chrono) |
| Fields (EM, heat, potential flow, seepage) | analytic and lumped circuits | discrete exterior calculus with Whitney (edge and face) elements | the same, refined, checked against published cases |
| Structure | beams, weight per area | OpenAeroStruct, CalculiX shells | TACS through MPhys |

### 4. Field solvers (Rust)

Field problems (electromagnetics, heat, potential flow and seepage under a berm) share one mathematical
structure, discrete exterior calculus:

- incidence matrices with d∘d = 0 exactly;
- Hodge stars;
- Whitney forms, so curl-curl problems have no spurious modes;
- CG and eigensolvers.

That core belongs in Rust, with Python bindings through PyO3, the route Osprey already uses for its sim loop.

An engineering layer goes on top:

- units;
- materials (ε, μ and conductivity, or permeability per cell);
- boundary conditions (conductor walls, absorbing layers, ports, fixed heads);
- sources;
- validation against textbook cases, such as conductor-cavity resonances, with convergence under refinement.

### 5. Dynamics

- One MuJoCo builder takes a design and its media.
- Force plug-ins come from each medium:
  - water (buoyancy, added mass, waves);
  - air (lift, drag, prop wash);
  - ground (tyre grip and rolling resistance, soil cutting);
  - tethers.

### 6. Parts library, with versions

- Shared parts are data with a version, such as eOx's 3 HP unit, drive axle and power box, and Osprey's props
  and motors.
- AtlanteanConstructor would depend on a released version of eOx's parts rather than copy them.

### 7. Power and cost

- eOx's 48 V bus and battery calcs.
- Osprey's battery sizing.
- Bills of materials.
- AtlanteanConstructor's economics and its conventional-machine baselines.

### 8. Agents and serving

- Each layer is served as MCP tools, with WebSocket streaming for live views.
- Agents can propose and run designs. Kept runs need human sign-off.

## Surrogates and AI

- **Surrogates:** train fast models on the expensive solvers' results, as NeuralFoil does for XFoil. Run the
  real solver where the surrogate is least sure, then retrain.
- **Multi-fidelity correction:** fit the gap between the cheap and expensive models, rather than replacing the
  cheap ones.
- **Physics-informed neural nets:** don't use them as solver replacements.
- **Every surrogate reports its own uncertainty,** and a kept result is checked against the real solver.

## Moving the projects over

Each step removes a real duplication or fills a real gap:

1. **The core, from eOx and Osprey:** the schema, checks, trials, runs and figures. eOx's trial engine becomes
   the platform's, and Osprey's sweeps become trials.
2. **Ground:** the tyre model and terrain engine move into the platform, and the parts library ships eOx's
   modules. That ends AtlanteanConstructor's copy and its pinned part data.
3. **Water and air, from Osprey:** hydrostatics and MuJoCo water, then Capytaine for added mass (the Trident
   first). Then AeroSandbox and AVL for whole-vehicle stability.
4. **Fields:** the Rust core with PyO3 bindings, then the engineering layer, validated against textbook cases.
   First uses: motor and cable fields for eOx's 48 V bus, and seepage under AtlanteanConstructor's berms.
5. **Expensive solvers and surrogates** on `gpu1` and `gpu2`, with agents through MCP.

## Tools and licenses

These are the licenses as I remember them. Confirm each one before adopting the tool.

| Tool | What it's for | License (to confirm) |
|---|---|---|
| OpenMDAO, OpenAeroStruct | Coupling, optimization with gradients, wing structure | Apache-2.0 |
| MuJoCo | Rigid-body dynamics | Apache-2.0 |
| build123d / CadQuery, Gmsh | Parametric CAD, meshing | Apache-2.0; GPL with exceptions |
| AeroSandbox, NeuralFoil | Whole-aircraft aero, a fast airfoil model | MIT |
| XFoil, AVL | Airfoils; stability derivatives | GPL |
| OpenVSP, VSPAERO | Aircraft geometry and panel aero | NASA Open Source Agreement |
| CCBlade | Blade-element momentum for props | Apache-2.0 |
| DUST / FLOWUnsteady | Prop–wing interaction | check |
| Capytaine | Added mass, radiation damping, wave forces | GPL-3.0 |
| DualSPHysics | Slamming, breaking waves | LGPL |
| OpenFOAM | General CFD, two-phase water | GPL-3.0 |
| SU2 | CFD with adjoint shape optimization | LGPL-2.1 |
| Project Chrono | Discrete-element soil, multibody | BSD-3 |
| CalculiX | Finite-element structure | GPL |
| TACS, MPhys | Composite shells coupled into OpenMDAO | Apache-2.0 (check) |
| pymoo, BoTorch, Optuna | Multi-objective and Bayesian search | Apache-2.0, MIT, MIT |
| OpenTURNS, chaospy | Uncertainty propagation | LGPL, MIT |
| PhysicsNeMo, neuraloperator | Neural surrogates for whole fields | Apache-2.0, MIT |
| Blender | Viewing and rendering | GPL |

## Structure and licensing

- **Repos:** one platform repo, holding a Python package and a Rust workspace, with optional extras for each
  medium. The projects depend on it by version.
- **License:** Similitude is Apache-2.0. It is compatible with eOx's MIT code. eOx's hardware stays under
  CERN-OHL-W in eOx.
- **GPL and LGPL tools** are called as separate programs, through files or subprocesses, and never linked. That
  keeps Similitude Apache-licensed.
- **Avoid tools that are free only for non-commercial use,** such as FluidX3D, and commercial optimizers such as
  SNOPT.
- **Osprey and AtlanteanConstructor need LICENSE files.**

## Decisions needed

- **One platform repo or several:** I recommend one, with optional extras.
- **The Python–Rust boundary:** I recommend Rust for fields and hot loops, and Python for everything else.
- **What agents may do:** propose and run designs, but keep runs only with sign-off.

## Risks

- **Scope.** Move one piece at a time, and only when a real project needs it.
- **Breaking working projects.** Each move keeps the project's existing tests and kept runs, and gives the same
  numbers.
- **Licensing mistakes with GPL code.** The rule of separate programs only needs CI to check it.
- **Surrogates and agents trusted too far.** Every model reports its uncertainty, and kept runs need sign-off.
