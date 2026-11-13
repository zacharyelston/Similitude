# Froude: an open-source, AI-assisted engineering optimizer

*Proposal, 13 November 2026. Decided on 13 November 2026: the name Froude, the Apache-2.0 license, and a public repo, `zacharyelston/Froude`.*

## Summary

Froude would be a separate open-source project: a toolkit that sizes, simulates and optimizes physical designs
across air, water and structure. Each discipline gets a ladder of models. Cheap models run inside the optimizer,
and expensive ones are run now and then to check the cheap ones and correct them. AI fits in two places: as fast
stand-ins for the slow solvers, and as agents that drive the whole loop through MCP.

Osprey would be its first user. The Trident, the Puffin and the Beet would move onto it one at a time, and later
projects would start on it from day one.

## Why a new project

Osprey showed what works, and where each new craft starts again from scratch.

**What works and should carry over:**

- Design inputs in TOML, with code that never hard-codes a design number.
- Kept runs that record the commit they came from, and are never edited.
- Tests that hold each model to an answer known in advance, such as the roll period from GM and inertia.
- MuJoCo carrying the rigid body while our own code supplies the water's forces.
- OpenMDAO for sizing with gradients.

**What doesn't scale:**

- Each craft builds its own meshes in numpy: the Beet, the Puffin and the Trident each have their own.
- The physics, the geometry and the design live in one file (`trident.py` does all three).
- Every model has one fidelity. A shortcut such as "the static margin counts only the wing" stays in place,
  because nothing better can be swapped in.
- The REF stand-ins are single numbers, so a result is a yes or a no, never a confidence.
- The gaps found during the Trident work keep coming back: no added mass, no prop–wing interaction, no
  structural check.

## Goals and non-goals

**Goals:**

- **One design description** that every tool reads: parameters plus parametric geometry.
- **A fidelity ladder in each discipline,** with a common interface, so the optimizer can swap one model for
  another without other changes.
- **Optimization across several goals at once, with uncertainty.** The output is a front of trade-offs with
  confidence levels, not a single point.
- **Fast stand-ins for the slow solvers,** trained where they are least sure.
- **Agents that can run the whole loop,** with every step tested and every run traceable to its commit.
- **Reusable beyond aircraft and boats:** buoys, hulls, drones, and anything else with flow and structure.

**Non-goals:**

- Writing our own CFD or FEM solvers. Froude wraps established ones.
- A graphical design tool. CAD stays in Onshape or build123d, and viewing stays in Blender.
- Replacing physical testing. Froude says where a tank or wind-tunnel test is worth its cost.

## Architecture

### 1. The design description

The design is TOML plus parametric geometry code, with no physics in it. A design names its components (body,
lifting surface, rotor, leg, hull), their parameters, their materials and where the parameters are uncertain.
REF stand-ins become ranges, such as `wing_kgm2 = {ref = 9.0, range = [7.0, 12.0]}`.

### 2. The geometry layer

One parametric model produces everything the other tools need:

- watertight solids, and STEP for Onshape and manufacturing;
- surface meshes for hydrostatics, Capytaine and MuJoCo;
- volume meshes for CFD and FEM;
- OpenVSP and AVL inputs for aircraft;
- Blender scenes for viewing.

It would be built on build123d (or CadQuery) and Gmsh. Osprey's numpy meshes become test cases: the new layer
must give the same volumes and centroids.

### 3. Disciplines, each with a fidelity ladder

Every discipline has the same interface: it takes a design and a fidelity level, and returns quantities with
their uncertainty.

| Discipline | Cheap (in the loop) | Mid | High (spot checks) |
|---|---|---|---|
| Airfoils | NeuralFoil | XFoil | SU2 or OpenFOAM (2D) |
| Whole-vehicle aero and stability | AeroSandbox's buildup | AVL, VSPAERO, AeroSandbox's vortex lattice | SU2 or OpenFOAM (RANS) |
| Props and rotors | Momentum theory, CCBlade | DUST or FLOWUnsteady (prop–wing interaction) | OpenFOAM with an actuator disc or overset grids |
| Hydrostatics and righting | Osprey's own mesh clipper | — | — |
| Hydrodynamics (motion in waves) | Osprey's MuJoCo water | Capytaine (added mass, damping, wave forces) | DualSPHysics or OpenFOAM `interFoam` (slamming, breaking waves) |
| Structure | Beams and weight per area | OpenAeroStruct, CalculiX shells | TACS through MPhys (composites, with gradients) |
| Flight and motion | MuJoCo with our aero and water forces | JSBSim; PX4 software-in-the-loop for control | — |

### 4. Coupling and optimization

- **OpenMDAO stays the backbone,** for coupling disciplines and for optimization with gradients. Use IPOPT
  through pyOptSparse, not SNOPT, which is commercial.
- **pymoo** for multi-objective fronts. For the Trident, weight against stall against how well it rights itself
  is a three-way trade.
- **BoTorch or Optuna** for searching with expensive, non-differentiable runs such as CFD or MuJoCo.
- **OpenTURNS or chaospy** to propagate the REF ranges, so a result reads like "rights itself with 95 %
  confidence".

### 5. Surrogates and AI

- **Surrogates:** train fast models on the expensive solvers' results, as NeuralFoil does for XFoil. Run the
  real solver where the surrogate is least sure, then retrain (active learning). NVIDIA PhysicsNeMo and
  `neuraloperator` provide neural operators and graph networks for whole fields, such as pressures over a hull.
- **Multi-fidelity correction:** fit the gap between the cheap and the expensive models, rather than replacing
  the cheap ones.
- **Physics-informed neural nets:** skip as solver replacements. They rarely beat a real solver.
- **Agents:** each tool (sizing, simulation, geometry, Blender, CFD jobs) is wrapped as an MCP server, so an
  agent can propose a design, run it, read the results and iterate. The guardrails are the same as in Osprey:
  every model is tested, and every run records its commit.

### 6. Runs, provenance and validation

- **Runs:** Osprey's `runs/` convention carries over, with the inputs, the source commit, the outputs and the
  command line. Optionally add MLflow or DVC once the runs grow large.
- **Validation suite:** checks against published data, such as NACA wind-tunnel data for airfoils, analytic
  added mass for spheres and cylinders, and towing-tank benchmarks for hulls. A model doesn't ship until it
  passes.
- **Compute:** heavy runs go to gpu1 and gpu2 as a `git archive` snapshot of a commit, as Osprey does now.

## Tools to integrate

These are the open-source pieces, what each is for, and their licenses as I remember them. Confirm each license
before adopting the tool.

| Tool | What it's for | License (to confirm) |
|---|---|---|
| OpenMDAO, OpenAeroStruct | Coupling, optimization with gradients, wing structure | Apache-2.0 |
| MuJoCo | Rigid-body dynamics | Apache-2.0 |
| AeroSandbox, NeuralFoil | Whole-aircraft aero, a fast airfoil model | MIT |
| XFoil, AVL | Airfoils; stability derivatives and the neutral point | GPL |
| OpenVSP, VSPAERO | Aircraft geometry and panel aero | NASA Open Source Agreement |
| CCBlade | Blade-element momentum for props | Apache-2.0 |
| DUST / FLOWUnsteady | Prop–wing interaction, transition | check |
| Capytaine | Added mass, radiation damping, wave forces | GPL-3.0 |
| DualSPHysics | Slamming, breaking waves (particles, GPU) | LGPL |
| OpenFOAM | General CFD, two-phase water | GPL-3.0 |
| SU2 | CFD with adjoint shape optimization | LGPL-2.1 |
| CalculiX | Finite-element structure | GPL |
| TACS, MPhys | Composite shells coupled into OpenMDAO | Apache-2.0 (check) |
| build123d / CadQuery, Gmsh | Parametric CAD, meshing | Apache-2.0; GPL with exceptions |
| pymoo, BoTorch, Optuna | Multi-objective and Bayesian search | Apache-2.0, MIT, MIT |
| OpenTURNS, chaospy | Uncertainty propagation | LGPL, MIT |
| PhysicsNeMo, neuraloperator | Neural surrogates for whole fields | Apache-2.0, MIT |
| Blender | Viewing and rendering | GPL |
| JSBSim, PX4 | Flight dynamics, control software-in-the-loop | LGPL, BSD-3 |

## Licensing

- **Froude is licensed under Apache-2.0.** It matches OpenMDAO and MuJoCo, and includes a patent grant.
- **Call GPL tools as separate programs** (OpenFOAM, Capytaine, XFoil, AVL, CalculiX, Blender), through files or
  subprocesses, and never link them in. That keeps Froude itself Apache-licensed.
- **Avoid tools that are free only for non-commercial use,** such as FluidX3D, and commercial optimizers such as
  SNOPT.
- **Osprey has no LICENSE file.** It is public but not legally open source yet. Fix that whatever happens to
  Froude.

## How Osprey moves over

Osprey keeps its own designs, runs and docs, and adds Froude as a dependency. Each craft moves over when it next
needs new physics:

1. **The Trident first.** Its geometry moves into the geometry layer, and Capytaine gives it added mass and
   damping. Test it against today's kept runs: the same volumes and righting curves, and a longer roll period.
2. **The Beet and the Puffin next,** and the Recon craft last, since it is already deep in OpenMDAO.
3. **Osprey's MuJoCo water and mesh clipper** become Froude's cheapest hydrodynamics level, with their tests.

## Roadmap

1. **Set up:** the repo and CI (done); the design schema; set up runs and provenance;
   build the plugin interface for disciplines.
2. **Geometry and water:** the geometry layer (build123d and Gmsh), Capytaine, and irregular seas. The Trident
   afloat with added mass, checked against analytic cases.
3. **Air:** AeroSandbox and AVL for the whole vehicle's neutral point, and CCBlade for props. This replaces the
   Trident's wing-only static margin and its 60 % tail stand-in.
4. **Optimization and uncertainty:** pymoo fronts and OpenTURNS ranges. The Trident's front of weight, stall and
   righting, with confidence levels.
5. **Mid and high fidelity:** DUST or FLOWUnsteady for transition; OpenFOAM or DualSPHysics for slamming; TACS for
   the hull and wings. These run on gpu1 and gpu2, and train the surrogates.
6. **Agents:** MCP servers for each layer, and an agent loop with human sign-off on kept runs.

## Decisions needed

- **Python only, or Julia too:** FLOWUnsteady is Julia. DUST is a compiled program that could be called on its
  own.
- **How far the agents go:** whether an agent may keep a run, or only propose one.

## The name

I checked PyPI and GitHub on 13 November 2026. I haven't searched for trademarks.

- **Froude (chosen).** William Froude invented towing-tank testing with scale models: test small and cheap,
  correct, then scale up. That is this project's whole approach. The Froude number governs wave-making
  resistance, and it also scales wave-tank and drop tests. The name is free on PyPI, and about 30 small GitHub
  repos use it, none of them notable.
- **Kittiwake.** It continues Osprey's seabird names: a gull that both skims the water and flies. It is free on
  PyPI, with about 50 small GitHub repos.
- **Cormorant.** A bird that flies and swims. It is free on PyPI, but a well-known open-source font family has the
  name.

Names I ruled out because they were taken: Gannet, Fulmar, Skua, Petrel (an established geoscience software
product), Tern, Shearwater, Strake and Slipstream.

## Risks

- **Scope:** this is many tools. Keep each phase tied to a real gap in a real craft, starting with the Trident.
- **Licensing mistakes** with GPL code. The rule of separate programs only needs CI to check it.
- **Surrogates trusted too far.** Every surrogate reports its own uncertainty, and a kept result is checked
  against the real solver.
- **Compute:** high-fidelity CFD and particle runs need gpu1 and gpu2. Queue them, and don't put them in the
  loop.
