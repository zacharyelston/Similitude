# Froude

An open-source, AI-assisted toolkit for sizing, simulating and optimizing physical designs across air, water and
structure.

It is named for William Froude, who invented towing-tank testing with scale models: test small and cheap,
correct, then scale up. Froude works the same way:

- **A ladder of models in each discipline.** Cheap models run inside the optimizer, and expensive ones are run
  now and then to check the cheap ones and correct them.
- **AI in two roles:** fast stand-ins trained on the slow solvers, and agents that drive the loop through MCP.
- **Established solvers, wrapped, not rewritten:** OpenMDAO, MuJoCo, Capytaine, AeroSandbox, AVL, OpenFOAM, SU2,
  TACS and others.

**Status:** just started. So far there is the plan and the repo; no models yet. See the
[proposal](docs/proposal.md) for the architecture, the tools and their licenses, and the roadmap.

Its first user is [Osprey](https://github.com/zacharyelston/Osprey), whose crafts (the Trident, the Puffin and the
Beet) move onto Froude as each needs new physics.

## Developing

Froude uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

## License

Apache-2.0; see [LICENSE](LICENSE). GPL tools (OpenFOAM, Capytaine, XFoil, AVL, CalculiX, Blender) are called as
separate programs, never linked in.
