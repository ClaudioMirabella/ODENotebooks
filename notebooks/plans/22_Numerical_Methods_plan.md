# Notebook 22 - Numerical methods, part 2: plan

MATH 331 - Differential Equations - Spring 2027 - phase E of the notebooks (19-22), the last notebook of the
phase

**Status (2026-11-25): the eight decisions of Section 6 approved by the instructor "as proposed"; being built.**

**Session 29, Unit 14** (lecture plan: "Euler, improved Euler (Heun), and fourth-order Runge-Kutta methods;
local and global error; order of accuracy; stiffness; scipy solve_ivp"; theoretical exercise "Derive the order
of accuracy of Euler's method"; practical exercise "Convergence study on a log-log plot"). About **15 minutes
in class**, the rest as homework. Notebook 04 ends its "Going further" with "notebook 22 (numerical methods,
part 2)", and notebook 21 ends with "Where to go next: notebook 22": this notebook keeps both promises.

The instructor's standing requests: caution with the existing code, step by step; the mathematics
double-checked; the new concepts understandable for inexperienced students; the finished notebook put on the
website (exported when finished) so that the instructor can review it there.

**What the students already have (notebook 04, Sessions 8 and 29; the deck of Session 8).** Euler's method by
hand and by `ol.euler`; Heun and RK4 (`ol.heun`, `ol.rk4`); the error table of Lebl's Table 1.7.1 (halve h,
the error halves); one log-log plot of the three methods on y' = x + y with `ol.observed_order`; a step that is
too large (y' = -5y); `solve_ivp` with `rtol`; and, under "Going further", Euler's error bound with a proof,
the regions of absolute stability, and stiffness shown by step counts (RK45 against Radau on
y' = -1000(y - cos t)). **This notebook does not repeat those**: it explains them (why the orders are 1, 2, 4),
uses them on a system, and adds what 04 only names (an implicit method; how a solver chooses its step).

**References** (found in the text of the book on 2026-11-25; the numbers of this plan computed with odelib,
SciPy and SymPy the same day).
* Lebl, **Section 1.7, Numerical methods: Euler's method**: the method, Table 1.7.1 (the error halves with h),
  the sentence that Runge-Kutta is "a fourth order method", and the Exercises of 1.E.7 (1.7.1 to 1.7.9, the
  Runge-Kutta formulas printed before 1.7.5, the improved Euler before 1.7.9).
* Lebl, **Section 3.1.2**: software solves first-order systems; "It is not very hard to adapt computer code for
  the Euler or Runge-Kutta method ... We simply treat the dependent variable not as a number but as a vector."
* A. Meseguer, *Fundamentals of Numerical Mathematics for Physicists and Engineers* (Wiley, 2020), Section 8.2,
  already cited by notebook 04 for the error bound and the stability regions (backward Euler and its
  stability: the same chapter).
* SciPy's documentation of `solve_ivp` (methods RK45, BDF, Radau; `rtol`, `atol`; `nfev`), for what the solver
  does, stated in words only.

**Purpose. One idea: the order p of a method is a promise about how the error shrinks with h - each step
makes an error of size h^(p+1), and about T/h steps add up to h^p - and it is a promise that holds only while
the step is stable; when stability, not accuracy, sets the step, an implicit method is needed, and a solver
like `solve_ivp` chooses its steps by estimating its own error.**

---

## 1. Contents

1. **Before you start**: what notebook 04 did (one paragraph, with links); the conventions box: t the
   independent variable (time-like; Lebl writes x in Section 1.7, notebook 04 too: said once), y the unknown,
   h the step, N the number of steps, T the end time, y_n the computed value at t_n = n h; **local error** (the
   error of one step started on the exact solution) and **global error** (the error at T after N steps); the
   **order** p; the **cost** (evaluations of f).
2. **Local and global error: why Euler's method has order 1** (the theoretical exercise).
   * The notebook's example: y' = y cos t, y(0) = 1, exact solution y = e^(sin t). One Euler step by hand with
     h = 0.2: y_1 = 1.2 against e^(sin 0.2) = 1.21978: the local error 0.01978.
   * Taylor's formula y(t + h) = y(t) + h y'(t) + (h^2/2) y''(s) for some s between t and t + h; Euler keeps the
     first two terms, so the local error is (h^2/2) y''(s): **order h^2 per step**. Here y''(0) = 1, so about
     h^2/2 = 0.02. Checked: local errors 1.98e-2, 4.99e-3, 1.25e-3, 3.13e-4 for h = 0.2, 0.1, 0.05, 0.025
     (divided by 4 each time).
   * N = T/h steps, each adding an error of size h^2 (and each carrying the earlier errors along): about
     (T/h) h^2 = T h, **order h**. The precise statement with its growth factor is notebook 04's Appendix A
     (cited, not repeated). Checked at T = 4: global errors 7.81e-2, 3.28e-2, 1.51e-2, 7.25e-3, 3.55e-3 for
     N = 10, 20, 40, 80, 160; observed orders 1.25, 1.12, 1.06, 1.03 (towards 1).
   * The same for Heun (local h^3, global h^2) and RK4 (local h^5, global h^4), measured: local orders 3.0 and
     5.0, global 2.0 and 3.97. **A log-log plot** with the local and the global errors of the three methods:
     six straight lines, slopes 2, 3, 5 and 1, 2, 4. `ol.local_error` and `ol.global_error` (Section 3).
3. **A convergence study on a system** (the practical exercise).
   * x'' + 4x = 0, x(0) = 1, x'(0) = 0 (a spring, as notebook 07; exact x = cos 2t), written as the system
     x' = v, v' = -4x (Lebl, Section 3.1.2: the same code, the unknown a vector). The error of x at T = 5 (not a
     whole number of periods: at T = 2 pi, a full period, RK4's error at the end looks like order 5, a
     coincidence of the oscillator, said in one line).
   * Errors for N = 25, 50, 100, 200, 400, 800: Euler 5.5 ... 5.4e-2 (orders 1.82 ... 1.05), Heun 0.107 ...
     1.40e-4 (towards 2), RK4 5.1e-4 ... 1.1e-9 (towards 4): the log-log plot, the observed orders printed.
   * **Cost against accuracy**: the same errors against the number of evaluations of f (Euler 1 per step, Heun
     2, RK4 4). To reach 1e-4 Euler needs far more work than RK4 (the plot says how much): why RK4 is the
     usual choice.
   * **Euler's wrong energy**: the points (x, v/2) of Euler's solution spiral outwards (the amplitude grows by
     sqrt(1 + 4h^2) each step: 6.57 after one period with N = 40), RK4's stay on the circle (0.99974). Figure.
   * **Slider 1** (question first: "with how many steps does Euler's error of x at T = 5 fall below 0.01? and
     RK4's?"): the number of steps; the three methods against cos 2t, the errors in the title.
4. **Stiffness and an implicit method.**
   * y' = -100 (y - sin t), y(0) = 1: the solution follows sin t after a fast start (formula given: a decaying
     term e^(-100 t) plus a slowly varying part). Euler multiplies the gap by 1 - 100h each step (notebook 04's
     Appendix B for the test equation): stable only for h < 0.02, although the solution after the start is
     smooth.
   * **Backward Euler**: y_(n+1) = y_n + h f(t_(n+1), y_(n+1)), the slope at the END of the step; y_(n+1) is on
     both sides, so each step solves an equation (here linear: y_(n+1) = (y_n + 100 h sin t_(n+1)) / (1 + 100h)).
     On the test equation it multiplies by 1/(1 + 100h), less than 1 for every h > 0: stable for every step.
     Order 1, like Euler.
   * `ol.backward_euler` (Newton's method on each step, for equations that are not linear).
   * **Slider 2** (question first: "Euler with h = 0.03: will it follow sin t?"): the step h; Euler and backward
     Euler against the exact solution. The answer: no, Euler's values grow (factor -2), backward Euler follows.
   * `solve_ivp` on the same problem on [0, 10]: RK45 367 steps and 2234 evaluations of f, BDF (an implicit
     method) 114 steps and 285 evaluations (rtol 1e-4): the professional version of the same idea.
5. **How `solve_ivp` chooses its step.**
   * **Two methods, one step**: Euler and Heun from the same point; their difference estimates Euler's local
     error (Heun is far more accurate). `ol.embedded_step` returns Heun's value and that estimate. Checked
     against the true local error on y' = y cos t.
   * A small adaptive loop in the notebook (about 12 lines, every line commented): keep a step if the estimate
     is below the tolerance, otherwise halve it; grow it after a good step. Its steps on y' = y cos t, plotted
     against t: small where the solution bends, large where it is straight.
   * `solve_ivp` does the same with a pair of orders 4 and 5 (RK45): the tolerance against the true error and the
     number of steps at T = 4 (rtol 1e-3: 7 points, error 1.4e-4; 1e-6: 16, 1.5e-7; 1e-9: 54, 2.5e-10).
6. **Where to go next**: the course's synthesis (Session 30); Project 3 and 4 work with `solve_ivp`; what was
   left out (multistep methods, error control of systems, symplectic methods).
7. **Exercises**, **answers**.

Target length: about 2800 words before the exercises.

---

## 2. Exercises (the four steps)

1. **Predict** (no code). (a) A method's global errors are 3.2e-3, 4.1e-4, 5.1e-5 for h, h/2, h/4: its order?
   (8: order 3.) (b) Euler's global error is 0.02 with N = 50 steps; about how many steps for 0.005? (200.)
   (c) For y' = -100 (y - sin t), is Euler stable with h = 0.015? with h = 0.025? (yes: factor -0.5; no: -1.5.)
   (d) Does backward Euler's order beat Euler's? (no: both order 1; it wins on stability, not accuracy.)
2. **By hand, then check**. (a) y' = -2t y, y(0) = 1 (exact e^(-t^2)): one Euler step with h = 0.1, its local
   error (1 against 0.99005: 0.00995), and the Taylor estimate (h^2/2)|y''(0)| = 0.01. (b) y' = -3y, y(0) = 1,
   h = 0.5: one Euler step (-0.5) and one backward Euler step (1/2.5 = 0.4) against e^(-1.5) = 0.223; which keeps
   the sign? (c) With `ol.global_error`: the observed order of Heun on y' = -2t y on [0, 2].
3. **Fill in** (code). The midpoint method: k_1 = f(t, y), y_(n+1) = y_n + h f(t + h/2, y + (h/2) k_1). Write
   its step, measure its global order with `ol.global_error` (2) and its local order with `ol.local_error` (3).
4. **Explore** (code). (i) RK4 with very small h: the error stops falling near 1e-13 (rounding, notebook 08):
   find the h where the line bends. (ii) Euler's energy on the oscillator: the amplitude after one period
   against N, and the factor (1 + 4h^2)^(N/2). (iii) The tolerance of `solve_ivp` against the true error on the
   oscillator.
5. **Explain** (two or three sentences). Why does a smaller step not cure stiffness cheaply, and what does an
   implicit method do instead?

Answers checked with the existing `ol.check`, `ol.check_vector`, `ol.check_choice`.

---

## 3. New in `odelib.py` (with tests)

A new section, "Numerical methods, part 2 (notebook 22)", after the section of notebook 21. NumPy only at the
top. Type hints; docstrings citing Lebl 1.7, the notebook, and what each function does not promise. The
existing `euler`, `heun`, `rk4`, `observed_order` and their helpers are reused **unchanged** (they integrate
forward only and raise FloatingPointError on overflow: documented, not changed).

* `local_error(method, f, t0, y0, h, exact) -> float`: the error of ONE step of `method` (ol.euler, ol.heun,
  ol.rk4, or a student's function with the same call) started on the exact value y0 = exact(t0): |y_1 -
  exact(t0 + h)| (the norm for a system: the largest component, said in the docstring).
* `global_error(method, f, t_span, y0, exact, steps) -> tuple[np.ndarray, np.ndarray]`: for each N in `steps`,
  h = (t1 - t0)/N and the error at t1; returns (h values, errors), ready for a log-log plot and
  `observed_order` (whose `ratio` argument is the ratio of consecutive N).
* `backward_euler(f, t_span, y0, h, jacobian=None)`: the implicit method; each step solves y = y_n + h f(t_(n+1), y)
  by Newton's method from the Euler guess, with the Jacobian given or by central differences (the existing
  `JACOBIAN_STEP`), a named tolerance and iteration limit, and a ValueError naming t when Newton does not
  converge. Same grid rule and same return shape as `euler`.
* `embedded_step(f, t, y, h) -> tuple`: Heun's new value and the estimate |Heun - Euler| of Euler's local error.
* Named constants (Newton tolerance and iteration limit); no other public name. Validation with ValueError and
  one sentence, as the existing functions.

**Tests**: against the book (Lebl's Table 1.7.1 numbers through `global_error`; Exercise 1.7.7's errors 16, 8,
4 where Euler's answer is printed); against independent methods (exact solutions; `solve_ivp` at tight
tolerances; the closed form of backward Euler on linear equations; SymPy's Taylor series for the local error
constants h^2/2 y''); the orders measured for the three methods and for backward Euler; stability of backward
Euler on y' = lambda y for large |lambda| h; invalid input with `match`; seeded cases (SEED = 331).

---

## 4. Conventions, checks, and care with the existing code

As for notebooks 19 to 21: `assert` for every reference value against an independent method; one sentence
under every plot, written after looking at the figure; a question before every slider and its honest answer
after; cells where a student changes a number do not stop on a bare AssertionError; named constants; plain
ASCII; generated by `plans/make_22.py` with `nbbuild.py`; the setup cell of notebook 21 with `import sys`;
`SETUP_NAMES` unchanged; every loop of the notebook within the browser budget (largest: 800 RK4 steps on a
system, a few hundred backward Euler steps).

* **Letters**: t the independent variable, y (or x, v for the spring) the unknowns, h the step, N the number of
  steps, T the end time, p the order; "local error" and "global error" defined once; lambda only for the test
  equation, as notebook 04. Rows for STYLE_SHEET Section 4 with the deck of Session 29.
* **Hard points for beginners**, named now so that the reviews test them: reading a log-log plot (a slope is an
  order); "error of one step" against "error at the end"; why h^2 per step becomes h after T/h steps; why an
  equation must be solved at each step of an implicit method; stability against accuracy; a tolerance is not
  the error.
* **Care with the existing code** (as for notebooks 20 and 21): baseline first; copies of `odelib.py` and the
  three test files before the work; strictly additive, proved mechanically (no existing line edited or deleted,
  apart from the list of names in the module docstring); the name-clash scan of notebook 21; every pre-existing
  function compared old against new; the four registries extended, never reordered; all gates again; the
  public folder touched only by `site/export_public.py --update`.

---

## 5. Order of work

1. Baseline: library tests (two environments), notebook tests, `--collect-only`, browser run.
2. `odelib.py`: the four functions and their constants; tests first; ruff; the whole suite unchanged.
3. `plans/make_22.py`: Sections 1 to 3 with slider 1; every figure looked at.
4. Sections 4 to 6 with slider 2.
5. Exercises and answers; `EXERCISES`, `EXTREMES`, `NAME_22`.
6. Index row, `READING_ORDER` (Week 15, Session 29), READMEs; site build and its tests; `check_site.py`.
7. Independent reviews: mathematics, weakest student, code (with the additivity check); corrections; fresh
   re-check; final pass; all gates again.
8. Trial export into a scratch folder.
9. The real export into the public folder (standing request); the instructor commits and pushes.

---

## 6. Decisions for the instructor

1. **Part 2, not a repeat**: notebook 04 keeps Euler, Heun, RK4 and the first log-log plot; notebook 22
   explains the orders (local against global error), applies them to a system, and adds backward Euler and step
   control. About 15 minutes in class.
2. **Four library functions**, as the master plan listed: `local_error`, `global_error`, `backward_euler`,
   `embedded_step`; the existing methods reused unchanged.
3. **The theoretical exercise** (the order of Euler's method) by Taylor's formula, made visible: local errors of
   slope 2 and global errors of slope 1 on one log-log plot, with the same for Heun and RK4.
4. **The practical exercise** on a system, x'' + 4x = 0 at T = 5, with the cost plot (error against evaluations
   of f) and Euler's growing energy.
5. **Stiffness on y' = -100 (y - sin t)** with backward Euler (stable for every h) and `solve_ivp`'s BDF; a
   relative of notebook 04's y' = -1000 (y - cos t), on purpose, now with the method that cures it.
6. **Step control** shown with an Euler-Heun pair and a small adaptive loop written in the notebook;
   RK45's own pair described in words only.
7. **The letter t** as the independent variable (Lebl and notebook 04 write x in Section 1.7; said once).
8. **Export to the public folder when the notebook is finished**, after the reviews and all gates; you commit
   and push.

---

## 7. Files

```
ode/ode/notebooks/
    22_Numerical_Methods.ipynb                 <- to build
    plans/22_Numerical_Methods_plan.md         <- this document
    plans/make_22.py                           <- builds the notebook
    odelib.py                                  <- + local_error(), global_error(), backward_euler(), embedded_step()
    tests/test_odelib.py, test_cross_validation.py, test_notebooks_run.py
    00_Index.ipynb, plans/make_00.py, README.md
ode/ode/site/build_site.py, site/README.md, ode/ode/README.md
```

---

## 8. As built (2026-11-25)

**The notebook.** `22_Numerical_Methods.ipynb`, 56 cells (28 code), built by `plans/make_22.py`; 560 984
bytes (the limit of `export_public.py` is 2 000 000), plain ASCII, no local path stored; ruff on the
notebook itself and on `make_22.py`: 0 findings. Before the exercises: 2710 whitespace tokens, **2316**
with each formula as one word (`nb21_build/words_per_cell.py`, the measure used for notebook 21, which gives
3112 for notebook 21 as it stands); the target was about 2800, so the notebook is shorter than planned (see
the departures). The setup cell is that of notebook 21 (with `import sys`); the cell after it holds the three
methods with their costs, orders, markers and colors, the example y' = y cos t (`cosine_rate`,
`cosine_exact`) and one helper of the checks (`close_to`). `SETUP_NAMES` unchanged.

**Sections.** 1 Before you start (what notebook 04 did, with its link and its "Going further" A, B, C; the
conventions: t, y, h, N, T, y_n, local error, global error, order p, cost, lambda only for the test
equation); 2 Local and global error (one Euler step by hand, 1.2 against 1.21978; Taylor's formula and
y''(0) = 1; the table of local errors 1.98e-2 ... 3.13e-4 divided by 3.97, 3.99, 4.00 and against h^2/2,
checked against the formula without the library; N h^2 = T h; the table of global errors 7.81e-2 ...
3.55e-3 with observed orders 1.25, 1.12, 1.06, 1.03, checked against Euler written out in the cell; how to
read a log-log plot; the figure of the six lines, slopes 2.00, 1.03, 3.02, 2.00, 5.02, 3.97, each asserted
within 0.1); 3 A convergence study on a system (x'' + 4x = 0 as x' = v, v' = -4x, Lebl's sentence of
Section 3.1.2, the error of x at T = 5 with `component=0`, the table for N = 25 ... 800 and the last
observed orders 1.05, 1.98, 3.98, checked by the matrix powers (I + hA)^N and their Heun and RK4
analogues; the remark on whole periods; cost against accuracy, a figure and the smallest N for an error
1e-4 found by bisection on the matrix powers: Euler 419 566 evaluations, Heun 1894, RK4 168, about 2500
times; Euler's wrong energy with the factor sqrt(1 + 4h^2) derived, the figure of the spiral, Euler's radius
6.5698 = (1 + 4h^2)^20 asserted, RK4's 0.99974; slider 1 with its question, static figure, answer and
checking cell); 4 Stiffness and an implicit method (the solution formula, the factor 1 - 100h, Lebl's
sentence "Such problems are sometimes called stiff"; a table of Euler and backward Euler at t = 1.5 for
h = 0.1, 0.05, 0.025, 0.0125; backward Euler, explicit against implicit, the closed form on this equation,
the factor 1/(1 + 100h), order 1, Newton's method in one sentence; `ol.backward_euler` against the
closed-form recurrence and the halving of its errors asserted; slider 2 with its question, static
figure, answer and checking cell; `solve_ivp` RK45 against BDF on [0, 10]); 5 How `solve_ivp` chooses its
step (two methods, one step; `ol.embedded_step` against the true local error; the adaptive loop of 14
commented lines; its figure with Taylor's prediction sqrt(2 tolerance / |y''|); RK45's pair in words, after
SciPy's documentation; the tolerance against the true error at T = 4: 7, 16, 54 points, errors 1.4e-4,
1.5e-7, 2.5e-10); 6 Where to go next (Session 30; Projects 3 and 4; what was left out); 7 Exercises (the
five of plan Section 2); 8 Answers.

**Figures: 6 stored** (the log-log plot of local and global errors; error against work; the energy spiral;
the adaptive steps, two panels; the static frames of the two sliders). Every figure was exported to PNG
(`nb22_build/figs`) and looked at, and both slider functions were drawn at N = 5, 12, 90, 4224, 4225, 5000
and h = 0.005, 0.015, 0.02, 0.025, 0.03, 0.1 (`nb22_build/sliders`). Changed after the first look: the
tick labels of the log-log plot (the minor labels "3 x 10^-2 4 x 10^-2 ..." overlapped; now the five steps
written plainly); the legend of the spiral (it touched the spiral; now outside); the errors in the title of
slider 1 (".1e" printed Euler's 0.009999 at N = 4225 as "1.0e-02", which looked like not below 0.01; now
".4g"); the default of slider 2 (h = 0.01 gave Euler's factor exactly 0, a jump onto the curve in one
step, which shows nothing; now 0.015, factor -0.5, with its alternating start), and the label of the
values off the picture ("2e+00" became "2"); the caption of the stiff frame ("the slow rise and fall": on
[0, 1.5] there is no fall); the caption of the adaptive figure (the points are dense over the hump, t from
1 to 2.2, and sparse where y'' = 0; the saw-teeth of the rule named); the caption of the log-log plot
(RK4's global error at h = 0.025 is 7.3e-10, "below 1e-9", not "about"). **Sliders: 2** (`plot_spring(n)`,
N from 5 to 5000 in steps of 1, default 50; `plot_stiff(h)`, h from 0.005 to 0.1 in steps of 0.005,
default 0.015), each with its question first, a static figure, its answer, and a checking cell after the
answer that asserts the answer at every position of the slider (slider 1: the first N with the error of x
below 0.01 is 4225 for Euler, 90 for Heun, 12 for RK4, and every larger N on the slider stays below,
computed by the matrix powers and checked against `ol.euler` at 4224 and 4225; slider 2: at all 20
positions backward Euler ends within 0.01 of the solution, Euler within 0.01 when |1 - 100h| < 0.99 and
farther than 1 otherwise).

**Added to `odelib.py`** (a new section "Numerical methods, part 2 (notebook 22; Lebl, Section 1.7)"
after the section of notebook 21; 313 lines, all insertions: 5 in the module docstring, 308 of code;
`SETUP_NAMES` unchanged; numpy-only at import, checked with SciPy, SymPy and matplotlib blocked):
* `local_error(method, f, t0, y0, h, exact, *, component=None) -> float`
* `global_error(method, f, t_span, y0, exact, steps, *, component=None) -> tuple[np.ndarray, np.ndarray]`
  (the h values and the errors)
* `backward_euler(f, t_span, y0, h, jacobian=None)`: same grid rule (`_grid`) and same return shape as
  `euler`; Newton's method from Euler's guess at each step, with the given Jacobian or central differences
  with the existing `JACOBIAN_STEP`; a ValueError naming t when Newton does not converge or the matrix
  I - hJ is singular.
* `embedded_step(f, t, y, h) -> tuple[float | np.ndarray, float]`: Heun's value and |Heun - Euler|.
* constants `NEWTON_TOLERANCE` (1e-10, relative to max(1, |y|)) and `NEWTON_MAX_ITERATIONS` (50); no other
  public name. Private helpers: `_state`, `_slope_at`, `_exact_at`, `_component`, `_start_on_exact`,
  `_last_value`, `_newton_matrix`, `_backward_euler_step`. The existing `_grid`, `_real_array`,
  `_one_real_number`, `_require_positive`, `MAX_STEPS`, `JACOBIAN_STEP` and `CHECK_TOLERANCE` are used, not
  changed; `euler`, `heun`, `rk4`, `observed_order` are reused unchanged (forward only, FloatingPointError on
  overflow: documented in the new docstrings, not changed). No existing function changed; no change to an
  existing function is proposed.

**Tests added: 118** (`--collect-only` 1231 -> 1349).
* `test_odelib.py`, 75: Lebl's Table 1.7.1 (errors and ratios to five decimals) through `global_error`;
  Exercise 1.7.7 (errors 16, 8, 4, observed orders exactly 1); the numbers of the notebook (local and global
  errors of y' = y cos t, the spring at T = 5 for the three methods with `component=0` and the largest
  component never smaller, Euler's radius 6.57 = the formula); local and global orders of the three methods
  (2, 3, 5 and 1, 2, 4 within 0.05); a method written by a student (the midpoint method: orders 2 and 3);
  Euler's factor on the stiff equation (stable exactly when |1 - 100h| < 1, at h = 0.015, 0.025, 0.03);
  backward Euler against its closed form on the stiff equation at five steps (with and without a given
  Jacobian), its order 1, its stability for lambda h = -10, -100, -1e4, -1e6 (positive, decreasing,
  (1/(1 - lambda h))^n; Euler's values growing), its shapes against `euler`, the equation of each step
  satisfied on y' = -y^3; `embedded_step` (values, shapes, the estimate against Euler's true local error at
  h = 0.1, 0.01, 0.001); 20 seeded linear equations (SEED = 331); 41 tests of invalid input, each with
  `match`; warning filters and NumPy settings unchanged, `SETUP_NAMES` and the two constants.
* `test_cross_validation.py`, 14 (behind `importorskip("sympy")`): the local error of each method against
  SymPy's Taylor series of the exact solution minus one symbolic step (nothing below h^(p+1); the series to
  h^7 at h = 0.05 and 0.02 to 1e-6; the leading constant at a small h); Euler's constant y''(0)/2 from
  SymPy; the global orders of the three methods and of backward Euler on the pendulum against `solve_ivp`
  at tight tolerances (its dense output as the exact solution); backward Euler step by step against
  `scipy.optimize.brentq` on y' = -y^3 + sin t; 10 seeded 2 x 2 linear systems against the matrix power
  ((I - hA)^-1)^N, with and without the Jacobian; the embedded estimate against (h^2/2)|y''| from SymPy at
  four times.
* `test_notebooks_run.py`, 29: the notebook as it is, with right answers, with wrong answers, the answer
  markers; 20 particular answers (other accepted forms; each taught mistake with its sentence; a NameError
  on an answer line stops only that cell; a student's step that raises an error or returns text is reported
  in one line); the sections run again after the exercises; the invited TOLERANCE = 1e-4 (391 steps kept,
  105 refused); 2 cells of the text that stop with a sentence; 12 slider values with RuntimeWarning as an
  error.

**Counts of tests, before and after.**

| Gate | Before (2026-11-25) | After |
|---|---|---|
| Library, env `ode` | 909 passed (127 s) | 998 passed (117 s) |
| Library as in CI (`site/.venv`, no SymPy) | 695 passed, 11 skipped (81 s) | 770 passed, 11 skipped (65 s) |
| Notebooks (`test_notebooks_run.py`, whole file) | 322 passed (8 min 15 s) | 351 passed (7 min 54 s) |
| `--collect-only` | 1231 | 1349 (README.md: 1349) |

**The other gates (2026-11-25).** Lint: `ruff check --no-cache` clean on `make_22.py` and on the notebook,
and no finding on the lines added to `odelib.py`, the three test files, `make_00.py` and `build_site.py`
(`nb22_build/ruff_added.py`: the findings on the lines that difflib finds inserted). Additive-only proof
(`nb22_build/prove_additive.py`, difflib on the copies made before the work): for each of the four files the
opcodes are "equal" and "insert" only (313, 328, 135 and 123 lines added), every old line still there in
order, line endings as they were (LF for `odelib.py`, CRLF for the tests). Name clashes
(`nb22_build/name_clash.py`): no new top-level name of the test files rebinds an old one; it found one while
building (a generator variable `i` in a constant of `test_odelib.py`, harmless in Python 3 but reported;
the constant was written out), and the only "clashes" left are the two dictionaries `EXERCISES` and
`EXTREMES`, extended on purpose. Old against new (`nb22_build/compare_old_new.py`, the script of notebook
21 with 17 more calls: the three functions of notebook 21, `euler`, `heun`, `rk4` on the spring and on the
stiff equation, an overflow, `observed_order`): 51 pre-existing functions, 86 calls (values, exceptions with
their messages, printed text; the nine rows of the Laplace table used on numbers): 0 different; no old
public name missing, no constant changed; new public names: the four functions and two constants. Site
build: 27 notebooks, 137 MB (its own run of the library tests: 770 passed); `site/test_build_site.py`: 5 passed. Browser (`check_site.py`, headless Edge): 27 of 27 notebooks without errors, 15 to 27 s each, about 8 min in all
(baseline before the work: 26 of 26). Notebook 22 in the browser: 28 cells, 8 plots, 2 sliders, 20 s; by `nb19_build/cell_times.py` 19.2 s in all, the setup cell 15.1 s, every other cell at most 0.8 s (the checking cell of slider 1, three times 4996 matrix powers). Redrawing slider 1 at its end, N = 5000 (35 000 evaluations of f), takes 0.14 s in CPython. Trial export with
`export_public.py` into a new, empty folder of the scratch area (`nb22_build/export_trial`): 93 files, 16.2 MB, all checks passed (only files of the allow-list, no large file, no mention of a file that was left out, plain ASCII). Compared with the public folder (read only, `diff -rq`): new are the notebook, its plan and `make_22.py`; changed are `odelib.py`, the three test files, `00_Index.ipynb`, `make_00.py`, `build_site.py` and the two READMEs that are exported (`notebooks/README.md`, `site/README.md`); nothing else.
Nothing was committed or pushed; the public folder was not touched.

**Not run.** The library tests at the browser pins (numpy 2.4.6, SciPy 1.18.0) in an environment of their
own, as for notebooks 19 to 21; the browser run executes the new functions at those pins inside the
notebook. The independent reviews (step 7) and the real export (step 9) are left to the coordinator.

**Departures from the plan, and why.**
1. `local_error` and `global_error` take a keyword `component=None`. The plan measures "the error of x at
   T = 5" on the spring, while the norm of a system is its largest component (the error of v is about twice
   that of x): with `component=0` the plan's numbers (Euler 5.5 ... 5.4e-2, Heun 0.107 ... 1.40e-4, RK4
   5.1e-4 ... 1.1e-9) are reproduced. No new public name.
2. Both functions refuse a y0 that is not exact(t0), within the existing `CHECK_TOLERANCE` (1e-6): a start
   off the solution would measure something else with no error. A method's own errors (its grid sentences,
   a FloatingPointError, an error in a student's function) are not caught; only a result that is not (t, y)
   is reported.
3. Euler's energy: the plan's "6.57 after one period with N = 40" is the radius at t = 2 pi, which is two
   periods of cos 2t (the period is pi). The notebook says "after two periods (t = 2 pi)"; at one period
   (t = pi, N = 40) it would be 1.63. RK4's 0.99974 is also at t = 2 pi.
4. `solve_ivp` on y' = -100 (y - sin t), [0, 10], rtol 1e-4 (SciPy 1.18.1, default atol): RK45 365 steps
   (366 points) and 2222 evaluations, BDF 117 steps (118 points) and 283 evaluations; the plan had 367 and
   2234, 114 and 285. The text says "about three times as many steps and about eight times as many
   evaluations", and the cell asserts more than 2 and 5 times, so that SciPy 1.18.0 in the browser cannot
   contradict it.
5. Exercise 4(i): the plan's "the error stops falling near 1e-13" is not what y' = y cos t gives at T = 4:
   RK4's error falls to 1.4e-15 at N = 5120 (h = 7.8e-4), then wanders (4.4e-15, 2.4e-15). The answer says
   so. On the spring at T = 5 the floor is similar (5e-15).
6. Exercise 4(iii) (the spring, atol 1e-12): the errors 1.5e-3, 1.2e-6, 1.0e-9 are a little above the
   tolerances; the text of Section 5 cites it as the example that a tolerance is not a bound on the error
   (at T = 4 on y' = y cos t they were below).
7. Slider 1 runs from N = 5 to 5000 in steps of 1, so that Euler's answer, 4225, is on the slider (the
   plan did not fix the range); RK4's answer is 12, Heun's 90. Slider 2: h from 0.005 to 0.1, default
   0.015 (see the figures above).
8. The adaptive loop: tolerance 1e-3, growth 1.2, first step 0.1 (with growth 1.5 there were 73 refused
   steps for 125 kept; with 1.2, 33 for 124); its error at T = 4 is 8.5e-5. 14 lines with their comments
   (the plan: about 12).
9. Length: 2316 words on the measure above, against the target of about 2800. The plan's contents are all
   there; the hard points each have their sentence or paragraph (the log-log plot; one step against the end;
   N h^2 = T h; explicit against implicit, with the equation of Exercise 2(b) as the example; stability
   against accuracy; a tolerance is not the error). The reviews may ask for more words where a beginner
   stops.
10. Exercises, details the plan left open: 1(c) and 1(d) are True or False (`ol.check`); 2(b) asks for the
    method that keeps the sign as one of four words (`ol.check_choice`); 2(c) is the observed order from
    N = 80 to N = 160 (2.04; 2 is also accepted, tolerance 0.05 relative); 3 is `midpoint_step(f, t, y, h)`,
    run by a given `midpoint(f, t_span, y0, h)` with the call of `ol.euler`, and its check prints the two
    orders measured (1.99 and 2.98). Hints for the taught mistakes: 1(a) the factor 8; 1(b) 100 (order 2);
    2(a) the sign of y''(0); 2(b) the sign of backward Euler's equation; 3 Euler's step.
11. Registries: `NAME_22` is the first entry of `EXERCISES` and of `EXTREMES` (inserted, nothing
    reordered); the week is "Week 15 - Numerical methods, part 2 (Session 29)"; the row of `make_00.py`
    shows backward Euler's formula, Session 29, Lebl 1.7 and 3.1. Rebuilding `00_Index.ipynb` also changed
    its stored widget ids and the order of its widget state, as every rebuild does. READMEs:
    `ode/ode/README.md` (1231 -> 1349 tests, equal to `--collect-only`), `notebooks/README.md` (the four
    functions in the list of `odelib.py`), `site/README.md` (27 notebooks, the row of notebook 22 in the
    table of tested notebooks, the last browser run).
12. Two more tests of the library than the plan names: the orders of a method written by a student, and
    the matrix-power check of backward Euler on systems.

**Lebl's text.** Checked: Section 1.7 (the method, Table 1.7.1 with its errors and ratios, "a fourth order
method", "Such problems are sometimes called stiff"); Exercise 1.7.7 (printed errors 16, 8, 4, factors 0.5);
Exercise 1.7.9 (already in notebook 04); Section 3.1.2 ("We simply treat the dependent variable not as a
number but as a vector."). Not in Lebl and said so: backward Euler and Newton's method (Meseguer, Section
8.2), RK45's pair (SciPy's documentation).

**Points for the reviewers.**
* The sentence "Euler multiplies the gap between its values and that slow curve by 1 - 100h" is true up to
  a term of size h^2 at each step (the local error of the slow curve); the notebook says "as for the test
  equation".
* The length is below the target; Sections 4 and 5 are each about one screen of text per idea.
* The widget state stores the two slider pictures a second time, as in notebooks 17 to 21.
* The rows for the reserved-letter table of the style sheet (Section 4) are not written: they go with the
  deck of Session 29.

### Corrections after the reviews (2026-11-25)

Three reviews (`reviews_nb/math_review_nb22.md`: 0 ERROR, 4 IMPORTANT, 11 MINOR; `reviews_nb/clarity_review_nb22.md`:
1 BLOCKING, 10 IMPORTANT, 17 MINOR; `reviews_nb/code_review_nb22.md`: 0 ERROR, 0 IMPORTANT, 10 MINOR, existing
code unharmed) and the coordinator's decisions, applied in one pass. Copies of `odelib.py`, the three test
files and `make_22.py` were taken before the pass (`nb22_build/before_fix`). Only lines added for notebook 22
were changed: the library section, the test blocks and the registry entries were edited in their scratch
sources and inserted again into the copies taken before notebook 22, so the additivity proof still runs
against those first copies. What Section 8 says above about the cells, the length, the tests and the
counts is superseded by this part.

| Finding | What was done |
|---|---|
| Math 1 (IMPORTANT), errors "grow or shrink a little" | The reviewer's sentence: the carried errors can grow; notebook 04's bound (K h / 2L)(e^(LT) - 1) with its two hypotheses; "leaving that factor aside" before N h^2 = T h. |
| Math 2 (IMPORTANT), whole periods | The reviewer's reason, no "coincidence": at whole periods x is at its top, only the amplitude error shows, Heun and RK4 then fall like h^3 and h^5. |
| Math 3 (IMPORTANT), "never fails" | "a method that stays stable at every one of these steps"; the docstring of `backward_euler` has the bullet for lambda > 0 (h lambda = 1 cannot be solved, h lambda > 1 changes sign), and a test pins both. |
| Math 4 (IMPORTANT), 4(iii) given away | "(Exercise 4(iii) asks whether it holds for the spring)". |
| Math 5 | Heun's line ends just above 1e-4 at 1600 evaluations; the numbers say it crosses at about 1900. |
| Math 6 | "for small h" in the estimate; the kept value is Heun's, on the safe side only for small steps (merged with clarity 9). |
| Math 7 | (a) "and the small change of the curve during one step is added" (with clarity 7); (b) the test equation y' = lambda y, 1/(1 - h lambda), lambda = -100, "for every lambda < 0" (with clarity 8). |
| Math 8 | "the kind of problem of notebook 04's "Going further", C" (with clarity 22). |
| Math 9 | Symplectic methods keep the energy "close to its true value for all time". |
| Math 10 | Exercise 1(c) is now y' = -50 (y - cos t) with h = 0.03 and 0.05 (factors -0.5 and -1.5; `my_stable_003`, `my_stable_005`); also settles clarity 27. |
| Math 11, clarity 2 | `;` after the last `ol.check` of Exercise 1; Exercise 2 now ends with an if/else: no stray True/False is stored or shown. |
| Math 12 | xi for Taylor's point; C for the constant of the stiff solution. |
| Math 13, clarity 13 | One parenthesis after the slopes: one first step can be worse than the whole run, whose errors partly cancel. |
| Math 14 | Docstring of `global_error`: the ratios go off below about 1e-13, and the errors stop near 1e-15. |
| Math 15 | (a) docstring of `embedded_step`: on the safe side for small h only; (b) the lambda > 0 bullet of `backward_euler`. |
| Clarity 1 (BLOCKING), Exercise 2(c) | A fill-in cell before the answer cell: `rate_2c` to complete, `exact_2c` given, the order printed when `rate_2c` is filled; the statement points to it; the answer needs the computed number (tolerance 0.005 relative: 2.04 passes, a typed 2 gets "2 is the order Heun should have; compute the observed one"); an array of one number (what `observed_order` returns) is accepted. |
| Clarity 3 | `true` renamed `true_error`. |
| Clarity 4 | "The observed order" defined once, at the end of the step-to-many paragraph, with log2 8 = 3. |
| Clarity 5 | The log-log paragraph replaced (the two log rules, the slope read one factor 10 to the left). |
| Clarity 6 | The three lines of algebra for sqrt(1 + 4h^2) and the power 20; the energy of notebook 07; "(square both and add)" cut. |
| Clarity 7 | Euler's gap derived (y_(n+1) - sin t_n = (1 - 100h)(y_n - sin t_n)), notebook 04 Section 6 cited, "stable" defined, -1 < 1 - 100h < 1 solved; "dies within t = 0.05" shortened. |
| Clarity 8 | (a) f substituted in backward Euler; (b) the test equation line; (c) a caption for the backward Euler cell ("five times Euler's limit": h = 0.1 against 0.02; the reviewer wrote "ten"). The Newton sentence cut to one clause (clarity 21). |
| Clarity 9 | Heun's formula restated, why the difference estimates Euler's error, "embedded" explained, why Heun's value is kept. |
| Clarity 10 | The tolerance tied to the LOCAL error of each step and the GLOBAL error at the end; `atol` defined; "our loop 124 steps, RK45 6". |
| Clarity 11, pacing | In class: Section 2 and Section 3 up to the table of the spring; at home: from "Cost against accuracy" on (marked "(At home.)", as is the energy paragraph), slider 1, Sections 4 and 5. |
| Clarity 12 | "(because y' = f(t, y))", the line y'' = (y cos t)' = ..., "read it as ...". |
| Clarity 14 | Replaced by math 2 (the coordinator's decision). |
| Clarity 15 | `x, v = u` commented; `SPRING_START = [1.0, 0.0]  # (x(0), v(0))` replaces the repeated list. |
| Clarity 16 | Why evaluations of f; the search cell's comment says to read what it prints. |
| Clarity 17 | The exact circle drawn first, grey and wide; "the grey circle" in the caption. |
| Clarity 18 | Type a value in the slider's box; the handle 800 x 0.054 for the guess. |
| Clarity 19 | With clarity 7. |
| Clarity 20 | Why h = 0.05 is worse than h = 0.1 (4^30 against 9^15). |
| Clarity 21 | Newton in one clause. |
| Clarity 22 | With math 8. |
| Clarity 23 | The step of sqrt(2 tolerance / |y''|) written; the comment of ROUNDING explained. |
| Clarity 24 | The caption sends the eye to the lower panel; why the dashed curve is infinite where y'' = 0. |
| Clarity 25 | "(points = steps + 1, the start included)". |
| Clarity 26 | "(a whole number)" in 1(a); a decimal within 0.5 of the order gets "an order is a whole number: round it". |
| Clarity 27 | Settled by math 10. |
| Clarity 28 | The template of Exercise 3 has a line `k1 = None  # replace None with the slope at the start, f(t, y)`; the statement says that `y` is y_n. Both one-line and two-line answers are tested. |
| Code 1 | `h = _require_positive("h", _one_real_number("h", h))` in `local_error` and `embedded_step`; 6 invalid-input rows (None, a list, text, True). |
| Code 2 | The Jacobian is called outside the try; a test: an error inside `jacobian` is its own message. |
| Code 3 | Overflow in backward Euler: "the values became too large for the computer ..."; one row. |
| Code 4 | `-> tuple[np.ndarray, np.ndarray]` on `backward_euler`. |
| Code 5 | The spring test asserts that the largest component is the maximum of the two components (a sum would fail). |
| Code 6 | `MAX_TRIES = 50_000` in the adaptive loop, with the sentence "stopped at t = ... TOLERANCE ... is too small"; a test with TOLERANCE = 1e-10. |
| Code 7 | The index of the first good N, not N - N_min. |
| Code 8 | The neutral case |1 - 100h| = 1 made explicit: the gap kept, `INITIAL_GAP` = C (difference 3e-7 at h = 0.02). |
| Code 9 | `BACKWARD_MAX_STEPS = 100_000`, checked in `backward_euler` with the sentence of the step limit, in its docstring and in the docstring's list of constants; one row. A new public constant (the plan's "no other public name" relaxed for this, as the reviewer proposed). |
| Code 10 | Named numbers: in the notebook `AGREEMENT`, `LOOSE_AGREEMENT`, `ORDER_WINDOW`, `POWERS_AGREEMENT`, `HALVING`, `HALVING_DIGITS`, `RK45_STEPS_RATIO`, `RK45_WORK_RATIO`, `ESTIMATE_AGREEMENT`, `ROUND_WINDOW` and the figure sizes; in the tests `SEEDED_DIGITS`, `STIFF_LONG`, `STABILITY_SPAN`, `NONLINEAR_SPAN`, `NEWTON_MARGIN`, `SYSTEM_STARTS`, and `GLOBAL_STEPS` reused. |

**Not applied, or applied otherwise.** Clarity 14 gave way to math 2, and clarity 27 to math 10 (the
coordinator's decisions). Clarity 8(c) says "five times", not "ten times", Euler's limit. Ruff on the notebook
reports one finding, F841 on `k1 = None` in the template of Exercise 3 (an unused name until the student fills
it in): left, because the template is what clarity 28 asks for.

**The notebook after the corrections.** 58 cells (29 code), 6 stored figures and 2 sliders; 571 871 bytes,
plain ASCII, no local path. Before the exercises: 2948 words on the measure of Section 8 (2316 before this
pass): the derivations and definitions the reviews asked for cost about 700 words; about 70 were cut (the
Newton paragraph, "(square both and add)", the first-term sentence). Above the target of about 2800, as
notebook 21 ended above its own. All six figures were rendered again from the final notebook and looked
at (`nb22_build/figs_fix`), and the twelve slider frames (`nb22_build/sliders`): changed, the energy figure
(the exact circle now grey and wide under RK4's points); no other figure changed its look; every number
printed is the same as before the pass.

**Counts.**

| Gate | As built | After the corrections |
|---|---|---|
| Library, env `ode` | 998 passed | 1008 passed (113 s) |
| Library as in CI (`site/.venv`) | 770 passed, 11 skipped | 780 passed, 11 skipped (64 s) |
| Notebooks (`test_notebooks_run.py`, whole file) | 351 passed | 356 passed (8 min 20 s) |
| `--collect-only` | 1349 | 1364 (README.md: 1364) |

Tests added by notebook 22 in all: 133 (`test_odelib.py` 85, `test_cross_validation.py` 14,
`test_notebooks_run.py` 34).

**Gates after the corrections (2026-11-25).** Lint: `make_22.py` clean; ruff on the notebook: the one F841
named above; no finding on the added lines of `odelib.py` and the three test files. Additive-only proof
against the copies made before notebook 22: opcodes "equal" and "insert" only (330, 367, 136 and 132 lines
added), every old line in order, line endings as they were. Name clashes: none, apart from the two
dictionaries extended on purpose. Old against new: 51 functions, 86 calls, 0 different; no old public name
missing; no constant changed; new public names: the four functions and three constants. Site build: 27
notebooks, 137 MB; `site/test_build_site.py`: 5 passed. Browser (`check_site.py`, headless Edge): 26 of 27
at the first run; notebook 12, unchanged by this work, stopped before its first cell ("the page did not
respond", a timeout of the browser's click on Run All), and passed when run again alone
(`check_site.py 12`: 18 cells, 18 s): so 27 of 27, 16 to 30 s each, about 9 min. Notebook 22: 29 cells,
8 plots, 2 sliders, 22 s; by `cell_times.py` 20.6 s in all, the setup cell 16.5 s, every other cell at most
0.8 s. Trial export into a new, empty folder of the scratch area (`nb22_build/export_trial_3`): 93 files, 16.2 MB, all checks passed (allow-list, no large file, no mention of a left-out file, plain ASCII); repeated into a second new folder (`export_trial_4`) after this part was written, with the same result.
Nothing was committed or pushed; the public folder was not touched.

### Final pass after the re-check (2026-11-25)

The re-check (`reviews_nb/recheck_nb22.md`: "ready to publish"; existing code unharmed; 0 ERROR, 0 IMPORTANT,
5 MINOR) was applied with the re-checker's own fixes; the optional length cuts of its Section 5 were not
applied (the coordinator's decision). Copies before the pass: `nb22_build/before_final`. Only `make_22.py`,
the notebook and the notebook 22 entries of `test_notebooks_run.py` changed; `odelib.py`,
`test_odelib.py` and `test_cross_validation.py` are byte for byte as after the corrections.

| Finding | What was done |
|---|---|
| N1, the log-log caption names RK4 only | "the local errors of Heun (h = 0.4) and RK4 (h = 0.4 and 0.2) are even above their global errors". |
| N2, sin t against "that slow curve" | "the gap between y and sin t (almost the slow curve) ... (and the small change of sin t during one step is added)". |
| N3, Exercise 2(c) | A list or tuple of one number is read as that number (`[2.04]` is now correct); a value outside the accepted window but within 0.05 of the order gets "close: copy the number printed by the cell above, to two decimals" (2.03), not the advice to type an exact expression. Two tests added. |
| N4, the two tolerances | "for its tolerance 1e-3; RK45, with rtol = 1e-3 (relative to |y|, so a little looser here), needs 6". |
| N5, unnamed numbers | `STIFF_START` (five uses), `QUESTION_H_2`, `STEPS_1C` (labels made from the steps), `SPAN_2C`, `START_2C`, `STEPS_2C`, `RATE_2C` in the checker of 2(c) (the student's cell keeps its literals), `TAYLOR_TERMS`. |

**The notebook after the final pass.** 58 cells (29 code), 6 stored figures and 2 sliders, plain ASCII, no
local path; every printed number and all six figures are identical to those after the corrections (the
PNG files compared byte for byte); ruff on `make_22.py` clean, on the notebook the one F841 of the template
of Exercise 3 (kept, as said above). Length unchanged (2948 words, plus the few words of N1, N2, N4).

**Counts.**

| Gate | After the corrections | After the final pass |
|---|---|---|
| Library, env `ode` | 1008 passed | 1008 passed (93 s) |
| Library as in CI (`site/.venv`) | 780 passed, 11 skipped | 780 passed, 11 skipped (52 s) |
| Notebooks (`test_notebooks_run.py`, whole file) | 356 passed | 358 passed (7 min 38 s) |
| `--collect-only` | 1364 | 1366 (README.md: 1366) |

**Gates after the final pass (2026-11-25).** Additive-only proof against the copies made before notebook 22:
opcodes "equal" and "insert" only (330, 367, 136 and 134 lines added), every old line in order, line endings
as they were. Name clashes: none, apart from `EXERCISES` and `EXTREMES`. Old against new: 51 functions, 86
calls, 0 different; no old public name missing; no constant changed. No ruff finding on the added lines of
`odelib.py` and the three test files. Site build: 27 notebooks, 137 MB; `site/test_build_site.py`: 5 passed.
Browser (`check_site.py`, headless Edge): 27 of 27 notebooks without errors at the first run, 12 to 21 s
each, about 7 min in all (no notebook needed a second run this time). Notebook 22: 29 cells, 8 plots,
2 sliders, 15 s; by `cell_times.py` 15.8 s in all, the setup cell 12.3 s, every other cell at most 0.8 s.
`site/README.md` has the times of this run. Trial export into a new, empty folder of the scratch area
(`nb22_build/export_trial_5`): 93 files, 16.3 MB, all checks passed (allow-list, no large file, no mention of a left-out file, plain ASCII); repeated into a second new folder (`export_trial_6`) after this part was written, with the same result. Nothing was committed or pushed; the public folder was not touched.
