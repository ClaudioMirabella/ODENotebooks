# Notebook 09 - Slope fields and what a solution is: plan

MATH 331 - Differential Equations - Spring 2027 - phase C of the plan of the student site

**Session 3, Unit 1** (lecture plan: "Definitions and terminology; order and degree; linear vs.
nonlinear; verifying solutions; IVPs and BVPs; direction fields". Theory exercise: "Classify ODEs
by order, degree, and linearity; verify candidate solutions". Practical exercise: "Plot slope
fields in Python and sketch solution curves"). About **15 minutes in class** (sections 4-5, the
application block), the rest as homework.

**References.** Lebl, *Notes on Diffy Qs*: Section 0.2 (what a solution is; Exercises 0.2.1-0.2.3,
0.2.6, 0.2.13-0.2.15), Section 0.3 (classification), Section 1.2 (slope fields; Figures
1.2.1-1.2.6; Examples 1.2.2 and 1.2.3; Exercises 1.2.1 and 1.2.6). Prerequisites: videos R01
(derivatives), R08 (linear systems), R11 (graphs); notebook 08.

**Purpose.** Three ideas, each done by hand first and then with the computer:
*a solution is a function, and checking one needs only derivatives*; *an equation is a recipe for
slopes, and a solution is a curve that follows them*; *an initial condition picks one curve*.
The notebook does **not** solve any equation by a method: that starts in Session 4.

---

## 1. Contents

| # | Section | What the student does | Values shown (all checked) |
|---|---|---|---|
| 0 | Title, session, references, what you will learn | reads | - |
| 1 | **Theory in brief** | one screen: ODE, order, linear or nonlinear (the test: y and its derivatives appear to the first power, not multiplied together, coefficients depend on x only), solution, general solution, initial value problem, boundary value problem, slope field | - |
| 2 | **Classify** | a table of six equations of the course (cooling, logistic, pendulum, RLC, y' = xy, mass-spring) with order and linearity; the pendulum is the nonlinear one, and its small-angle version is linear | orders 1, 1, 2, 2, 1, 2 |
| 3 | **Verify a solution** | by hand, then SymPy: put the candidate into the equation and simplify the residual (left side minus right side) to zero. Lebl 0.2.1 and 0.2.2 (e^(4t) is a solution, e^t is not), 0.2.3 (sin t), 0.2.13 (y = x^2). A small function `residual` is written in the notebook | residuals 0; -27 e^t; 0; 0 |
| 4 | **From a family to one solution** | Lebl 0.2.15: x = C1 e^t + C2 solves x'' - x' = 0; the conditions x(0) = 10, x'(0) = 100 are a linear system (as in notebook 08). Then initial against boundary conditions for y'' + y = 0: one solution, infinitely many, or none | C1 = 100, C2 = -90 (Lebl's answer); y(0) = 0 with y(pi/2) = 1: sin x; with y(pi) = 0: any b sin x; with y(pi) = 1: none |
| 5 | **The slope field** (the plan's practical exercise) | Lebl Figure 1.2.1: for y' = xy the slope at (2, 1.5) is 3. A slider moves the point and shows its segment. Then the whole field (Figure 1.2.2) with `ol.slope_field`, and by hand the **isoclines** of y' = x - y (the slope is 0 on the line y = x, 1 on y = x - 1) | slope 3; the line y = x - 1 is itself a solution |
| 6 | **Solutions follow the slopes** | Figure 1.2.3: y' = xy with y(0) = 0.2, 0, -0.2, drawn numerically on the field and compared with y = y0 e^(x^2/2); Figure 1.2.4: y' = -y, all solutions tend to 0. Slider for y(0), with a question first: which initial value gives the solution that stays bounded? | y(2) = 0.2 e^2 = 1.4778; only y(0) = 0 stays bounded |
| 7 | **What a picture cannot tell you** (look ahead to Session 5) | Lebl Example 1.2.2, Figure 1.2.6: y' = 2 sqrt(abs(y)), y(0) = 0 has two solutions, y = 0 and y = x^2 (x >= 0); both follow the same slopes. One paragraph; the theorem is in notebook 11 | both residuals 0 |
| 8 | **Exercises** | the four steps below | - |
| 9 | **Answers** (collapsed) | opens after trying | - |

## 2. Exercises (the four steps)

1. **Predict** (no code) - *Lebl, Exercise 1.2.6.* Three slope fields A, B, C are drawn in an order
   chosen by the notebook; match them to y' = 1 - x, y' = x - 2y, y' = x(1 - y). Clues to find by
   hand: where is the slope zero? Does it depend on y? The cell checks the three choices.
2. **By hand, then check** - (a) order and linearity of four equations; (b) *Lebl, Exercise 0.2.6*:
   verify that x = C1 e^(-t) + C2 e^(2t) solves x'' - x' - 2x = 0 and find C1, C2 for x(0) = 10,
   x'(0) = 0. Answers: C1 = 20/3, C2 = 10/3.
3. **Fill in** - *Lebl, Exercise 1.2.1.* Complete `f(x, y)` for y' = e^(x - y); `assert` lines check
   three slopes (f(0, 0) = 1, f(1, 1) = 1, f(1, 0) = e), and the next cell draws the field. Then
   guess a solution from the picture (y = x) and verify it with `residual`.
4. **Explore** - *Lebl, Example 1.2.3.* Draw the field of y' = y^2 and the solutions with
   y(0) = 1 and y(0) = 1/2. How far to the right do they go? Compare with y = 1/(1/A - x). What
   does this say about "a solution exists"? (They stop at x = 1 and x = 2: Session 5.)

Answers (final values only) in a collapsed block; full solutions after the session.

## 3. New in `odelib.py` (with tests)

`solution_curve(f, point, t_range, y_range, h)`: the solution of y' = f(t, y) through a point,
followed **in both directions** with RK4 until it leaves the window (so a solution that blows up,
such as y' = y^2, simply stops at the edge of the picture, without warnings). Returns `t, y`.
Used for every "curves on a slope field" picture of notebooks 09-13. Today's `rk4` only goes
forward and raises an error on blow-up, which is right for it but not for drawing.

Tests: against y0 e^(x^2/2) for y' = xy through (0, 0.2), in both directions; y' = -y; y' = y^2
through (0, 1) stays inside the window and matches 1/(1 - x) there; the point must be inside the
window, h positive (ValueError otherwise).

## 4. Conventions and checks

As for notebook 08 (its plan, sections 3 and 4): same setup cell, `assert` for reference values,
`ol.check` for answers typed by the student, one sentence under every plot, line styles as well as
colours, a question before every slider, plain ASCII, generated by `plans/make_09.py`.

* Sliders: (1) the point (x, y) and its slope segment; (2) the initial value y(0) on the field of
  y' = xy. Both have a static picture first.
* `tests/test_notebooks_run.py` runs the notebook as it is, and a second time with the correct
  answers typed into the exercises (it must print "correct" for 1, 2a, 2b and 3).
* The site is rebuilt and the notebook is run in the browser (all cells, one slider).
* Index notebook: one more row. Notebook 01 (exponential growth) already draws one slope field;
  it gets a one-line pointer to notebook 09 for the idea.

## 5. Files

```
ode/ode/notebooks/
    09_Slope_Fields.ipynb              <- to build
    plans/09_Slope_Fields_plan.md      <- this document
    plans/make_09.py                   <- builds the notebook
    odelib.py                          <- + solution_curve()
    tests/test_odelib.py               <- + its tests
    tests/test_notebooks_run.py        <- + the exercises of notebook 09
    00_Index.ipynb, 01_Exponential_Growth.ipynb   <- one line each
```
