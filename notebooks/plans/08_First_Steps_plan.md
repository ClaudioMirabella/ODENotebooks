# Notebook 08 - First steps in Python: plan

MATH 331 - Differential Equations - Spring 2027 - phase C of the plan of the student site

**Session 2** (lecture plan: "Toolkit II ... setup of Python and Jupyter"; practical exercise:
"First Jupyter notebook: plot functions and solve a linear system"). About **30 minutes in class**
(sections 1-5, on the course site, nothing to install), the rest as homework.

**Purpose.** The students' first contact with Python, built on what they already know: the
questions of the diagnostic exam. They compute by hand in the appendix of Lecture 1 and in the
videos R01-R11; here the computer **checks** the same answers. Two ideas are planted for the rest
of the course: *Python checks and visualises what you do by hand*, and *a computer computes
approximately* (the first lesson in numerical methods).

No prior programming is assumed. Nothing in this notebook is needed to follow the lectures by
hand; everything in it is reused by notebooks 01-07 and 09-22.

---

## 1. Contents

| # | Section | What the student does | Values shown (all checked) |
|---|---|---|---|
| 0 | Title, session, what you will learn | reads | - |
| 1 | **A notebook in two minutes** | runs a cell (Shift+Enter), changes a number, runs again; cells run in order; a harmless, deliberate error is caught and explained: "read the last line" | - |
| 2 | **Python as a calculator** | arithmetic, powers `**` (not `^`), variables, `print`; functions from numpy (`np.exp`, `np.sqrt`, `np.pi`) | `2**3**2 = 512`; `7/2 = 3.5`, `7//2 = 3` |
| 3 | **Plot a function** | `np.linspace`, a plot of f(x) = x e^(-x) (Question 11) with labels; reads the maximum from the arrays | maximum at x = 1.0, value 0.3679 = 1/e |
| 4 | **Solve a linear system** (the plan's practical exercise) | Question 8 with `np.linalg.solve`; check with `A @ x`; then the 3 x 3 system of Lebl Example 2.3.3 | (x, y) = (2, 1); (C1, C2, C3) = (-1/4, 1, 1/4) |
| 5 | **The diagnostic exam, checked by the computer** | SymPy redoes Q1, Q2, Q6, Q9 (derivative, integral, roots, Taylor series); complex numbers for Q5 | x(3x+2)e^(3x); e^(x^2)/2; 2 +- 3i; x - x^3/6 + x^5/120; (3+4i)(1-2i) = 11 - 2i |
| 6 | **Computers compute approximately** | `0.1 + 0.2`; `np.exp(1j*np.pi)`; a derivative by finite differences with smaller and smaller h | 0.30000000000000004; -1 + 1.2e-16 i; slopes 2.1, 2.0001, 1.99999999, 2.0002, 0 for h = 1e-1 ... 1e-16 |
| 7 | **A first differential equation** (preview) | the coffee of Lecture 1: `solve_ivp` against the formula, and a slider for the room temperature and the cooling rate | k = 0.0308 per minute; 60 degrees after 18.15 minutes |
| 8 | **Exercises** | the four steps below | - |
| 9 | **Answers** (collapsed) | opens after trying | - |

Section 6 matters for the course: the last row (the slope becomes 2.0002 and then 0 when h is too
small) is the students' first sight of rounding error, which returns in notebook 04 (Euler's
method) and in notebook 22.

## 2. Exercises (the four steps of the site plan)

1. **Predict** (no code): what do `2 + 3 * 4`, `2**3**2` and `10 / 4` give? What is the type of
   the result of `np.exp(1j*np.pi)`: exactly -1, or not? Then run and compare.
2. **By hand, then check**: solve 4x - y = 2, x + 2y = 5 by hand (practice sheet of R08), then
   with `np.linalg.solve`; differentiate x^3 e^(-2x) by hand (practice sheet of R01), then with
   SymPy. The cell says whether the two agree.
3. **Fill in**: complete `g(x)` so that it returns x e^(-2x); the next cell finds its maximum on a
   grid and `assert`s that it is at x = 0.5 with value 1/(2e) = 0.1839 (pause card of R11).
4. **Explore**: plot x^n e^(-x) for n = 1, 2, 3, 4 on one figure. Where are the maxima? State a
   conjecture and prove it by hand. (They are at x = n.)

Answers (final values only) are in a collapsed block at the end; full solutions are released after
the session, as proposed in the site plan (decision 3, still open: this notebook follows the
proposal).

## 3. Conventions for this and all new notebooks

* **Same setup cell as notebooks 01-07** (it starts with `import sys`): the site build adds the
  slider guard and the reminder cell to it automatically.
* Reference values are checked in the notebook with `assert`, as today; the notebook must run
  from top to bottom without errors, in ordinary Jupyter and on the site.
* **One new helper in `odelib.py`**, with tests: `check(label, value, expected, tol)` prints
  "label: correct" or "label: not yet (your value ..., try again)", without stopping the notebook,
  for the "by hand, then check" exercises. `assert` stays for the fill-in exercises.
* Every plot gets one sentence underneath saying what it shows (accessibility, site plan risk 9);
  curves differ by line style as well as colour.
* Every slider is preceded by a question to answer before moving it (risk 10).
* Plain ASCII; no numbers typed twice (values are computed, or defined once and reused).

## 4. Checks

* `tests/test_odelib.py`: tests of `check` (correct, wrong, tolerance, arrays and complex values).
* The notebook is executed from the command line (`nbclient`, as `tools/run_notebooks.py` does) in
  a new test `tests/test_notebooks_run.py`, which runs every numbered notebook and fails on any
  error: this also protects notebooks 01-07.
* The site is rebuilt and notebook 08 is run in the browser (all cells, one slider), as for 01-07.
* The index notebook (`00_Index.ipynb`) and the site's home page list notebook 08 first, as the
  place to start.

## 5. Files

```
ode/ode/notebooks/
    08_First_Steps_in_Python.ipynb     <- to write (generated by a small script, so that it is reviewable)
    plans/08_First_Steps_plan.md       <- this document
    plans/make_08.py                   <- builds the notebook from readable cell sources
    odelib.py                          <- + check()
    tests/test_odelib.py               <- + tests of check()
    tests/test_notebooks_run.py        <- new: every notebook runs without errors
    00_Index.ipynb                     <- + one line for notebook 08
```

**After 08** (same workflow, one plan each): 09 slope fields and what a solution is (Session 3),
10 separable equations (Session 4), 11 linear first-order equations and uniqueness (Session 5),
12 exact equations and substitutions (Session 6), 13 first-order models (Sessions 7-8).
