# Working in Similitude

- **Branches and PRs:** after the initial commit, make every change on a branch (`feat/…`, `fix/…`, `chore/…`)
  and merge it through a PR into `main`. Don't stack PRs; target `main`. Never commit credentials, keys or
  personal data.
- **Public repo:** never commit internal hostnames, logins, company domains or other projects' names. The compute
  hosts are `gpu1` and `gpu2`.
- **Licensing:** Similitude is Apache-2.0.
  - Don't import or link GPL or LGPL code. Call those tools (OpenFOAM, Capytaine, XFoil, AVL, CalculiX, Blender) as
    separate programs, through files or subprocesses.
  - Don't add tools that are free only for non-commercial use, or commercial optimizers.
  - Record each tool's license in `docs/proposal.md` when you adopt it.
- **Designs are data:** design inputs are TOML. Code reads them and never hard-codes a design number. An uncertain
  value is a range, not a single number.
- **Disciplines:** each discipline sits behind one interface that takes a design and a fidelity level, and
  returns quantities with their uncertainty. Cheap levels run in the optimizer's loop; expensive ones are for
  checks and for training surrogates.
- **Checks:** every model has a test against an answer known in advance (analytic, or published data) before it
  is used. Run `uv run pytest` before any commit.
- **Runs:** `out/` is scratch and gitignored. Runs worth keeping go to `runs/<date>_<name>/` with the source
  commit, and are listed in `runs/README.md`. Never edit a kept run.
- **Remote runs:** heavy runs (CFD, particle methods) run on `gpu1`/`gpu2` from a `git archive` snapshot of a
  commit, never a working tree.
