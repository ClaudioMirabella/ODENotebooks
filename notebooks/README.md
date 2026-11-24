# MATH 331 notebooks

Open `00_Index.ipynb` first; students new to Python start with `08_First_Steps_in_Python.ipynb`. Each notebook: problem statement, existence and uniqueness, analytical
solution (verified with SymPy), reproduction of reference examples (with `assert`), numerical
solutions, sliders (ipywidgets), interpretation, exercises.

* `odelib.py`: tested helper functions (analytic solutions, Euler/Heun/RK4, slope fields, phase lines,
  characteristic roots, free response of the mass-spring-damper, Wronskians; `solution_curve` for curves on a slope field; Torricelli's law; the integrating factor with numerical integrals; relaxation to an equilibrium and quadratic drag; forced vibrations; equations of order n and boundary value problems; linear systems x' = A x (`companion_matrix`, `two_by_two_case`, `eigen_solution`, notebook 17); the phase plane and nonlinear systems (`classify_linear_2d`, `jacobian`, `classify_equilibrium_2d`, `trajectories`, `direction_field`, `phase_portrait`, notebook 18); the Laplace transform (`laplace_numeric`, the integral that defines it, and `laplace_table`, the table of the course text as data, notebook 19); steps, impulses and convolution (`step`, the unit step u(t - a); `convolution`, (f * g)(t) by a numerical integral; `impulse_response`, notebook 20); series solutions (`series_coefficients`, the coefficients of a power series from a recurrence; `series_value`, its partial sum; `indicial_roots`, the roots of the indicial equation and their case, notebook 21); `check`, `check_formula`, `check_solution`, `check_vector` and `check_choice` (words such as "saddle") for the
  exercises "by hand, then check"). The list of every public name is at the top of `odelib.py`.
* `tests/`: `python -m pytest` (from this folder, about nine minutes) checks `odelib` against the
  values printed in the books (`test_odelib.py`), against SymPy's own solutions and an
  independent numerical integration (`test_cross_validation.py`), and runs every numbered
  notebook untouched, with right answers and with wrong answers (`test_notebooks_run.py`).
* `plans/`: the plan of each new notebook (08 onwards) and the script that builds it
  (`python plans/make_08.py`, `make_09.py`, ...; `nbbuild.py` is shared; notebook 01 is built this
  way too since its conversion, see `plans/01-07_Conversion_plan.md`): edit the script, not the
  `.ipynb` file.
* Sliders need a running kernel (Jupyter, JupyterLab or VS Code). Every slider plot also has a static
  version, so the notebooks can be read on GitHub.
* Notebooks 05-07 (second-order linear equations) should be read in order.
* To rebuild all the notebooks: run every `plans/make_NN.py` (each takes under a minute).
* **For students who have never used a notebook** (review of 2026-11-17):
  * the student guide is one text, in `odelib.py` (`GUIDE_START`, `GUIDE_TYPING`, `GUIDE_ERRORS`,
    `GUIDE_HOW`, `GUIDE_WORDS`). It is printed by `ol.guide()`, shown by notebook 00
    (`plans/make_00.py`) and by the home page of the site: change it in `odelib.py` only;
  * `odelib` adds a line that starts with `Hint (MATH 331):` next to the red message of the usual
    beginner's errors (`hint_for`). It only adds a line after the cell has run: the error itself
    is untouched. Do not replace this with IPython's `set_custom_exc`: it makes the kernel
    swallow the errors (a test guards against it);
  * `nbbuild.py` adds, in every notebook, a note under the heading of the exercises and a line
    `# ---- your answers go above this line ...` in every answer cell; `code(..., keep_output=False)`
    does not store an output that would give an answer away.
