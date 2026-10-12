# Agent rules for this repo

You are implementing one step of the RECAP value-function pipeline. Before writing code:

1. Read `README.md`, `docs/background.md`, `docs/conventions.md`, and **your step doc** in `docs/steps/`.
2. Follow the file/batch contracts in `docs/conventions.md` exactly. Import shared constants from
   `recap_vf.constants`, paths from `recap_vf.paths`, the value support from `recap_vf.bins` — never redefine them.
3. Develop and test **on this laptop** with `uv run …` (unit tests on CPU, smoke tests on ≤ 10 episodes). Full-scale
   runs happen later on the shared RTX 5090 workstation after `git pull` — make every script work at both scales via
   flags/config. Mind the laptop's free disk (see "Machines" in `docs/conventions.md`). Do not commit or push.
   ⚠ The laptop shell has ROS Humble's Python 3.10 packages on `PYTHONPATH`; they leak into our 3.12 venv and break
   pytest. **Always run `env -u PYTHONPATH uv run …`** on the laptop.
4. Only touch files your step owns (listed in its doc). If you need to change a shared file or a contract, stop and
   write the request in your report instead.
5. Finish by writing `reports/step_XX_<name>.md` using the template in `docs/conventions.md`, including every
   command needed to reproduce your results and the ticked definition-of-done checklist.
6. Report honestly: failed checks, skipped items and deviations go in the report, not under the rug.
