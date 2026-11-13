# Similitude

An open-source, AI-assisted platform for sizing, simulating and optimizing machines that work in a medium: air,
water, ground or a field.

The name comes from engineering similitude: the rules that let a cheap scale model stand in for the real thing.
William Froude's towing tanks are the classic case. Similitude works the same way:

- **A ladder of models in each medium.** Cheap models run inside the design search, and expensive ones are run
  now and then to check the cheap ones and correct them.
- **One design description for every tool,** with uncertain values carried as ranges.
- **AI in two roles:** fast stand-ins trained on the slow solvers, and agents that drive the loop through MCP.
- **Established solvers, wrapped, not rewritten:** OpenMDAO, MuJoCo, build123d, Capytaine, AeroSandbox, AVL,
  OpenFOAM, SU2 and others.

**Status:** just started. So far there is the plan and the repo; no models yet. See the
[proposal](docs/proposal.md) for the architecture, the tools and their licenses, and the roadmap.

It is being pulled out of three projects that each built parts of it:

- [Osprey](https://github.com/zacharyelston/Osprey): flying and floating craft;
- [eOx](https://github.com/zacharyelston/EOx): an electric tractor cut from sheet steel;
- [AtlanteanConstructor](https://github.com/zacharyelston/AtlanteanConstructor): earthworks dug by a tethered
  eOx.

The project began as Froude, a fluids-only optimizer; "froude" may live on as its fluids package.

## Developing

Similitude uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

## License

Apache-2.0; see [LICENSE](LICENSE). GPL tools (OpenFOAM, Capytaine, XFoil, AVL, CalculiX, Blender) are called as
separate programs, never linked in.
