# Notebook 11 - Linear first-order equations; existence and uniqueness: plan

MATH 331 - Differential Equations - Spring 2027 - phase C of the plan of the student site

**Session 5, Unit 2** (lecture plan: "Linear first-order equations; integrating factor; existence
and uniqueness theorems". Theory exercise: "State the hypotheses of the theorem and find the
interval of existence". Practical exercise: "Analyze the non-unique case y' = y^(1/3)". Quiz 1 is
in this session). About **15 minutes in class** (section 6, the non-unique case), the rest as
homework and as preparation for the quiz.

**References.** Lebl, *Notes on Diffy Qs*: **Section 1.4** (integrating factor; Example 1.4.1;
formula (1.4.5); the Note on existence for linear equations; Exercises 1.4.3, 1.4.7, 1.4.11,
1.4.13) and **Section 1.2** (Theorem 1.2.1, Picard; Examples 1.2.1-1.2.3; Exercise 1.2.5).
Prerequisites: videos R01 (product rule), R02 and R03 (integration); notebooks 09 and 10.

**Purpose.** The second method of the course, and the first theorem. *Method:* the integrating
factor turns the left side into the derivative of a product. *Theorem:* when is there a solution,
when is it the only one, and how far does it go; with the three open questions left by notebooks
09 and 10 answered (two solutions through one point; solutions that stop; the empty tank).

---

## 1. Contents

| # | Section | What the student does | Values shown (all checked) |
|---|---|---|---|
| 0 | Title, session, references, what you will learn | reads | - |
| 1 | **Theory in brief** | the standard form y' + p(x) y = f(x); the method in four steps (standard form; r = exp of the integral of p; (r y)' = r f; integrate and divide), with the advice of Lebl: remember the process, not the formula; Picard's theorem (Lebl 1.2.1) and the stronger statement for linear equations | - |
| 2 | **The integrating factor, step by step** | Lebl Example 1.4.1, y' + 2xy = e^(x - x^2), y(0) = -1: by hand, then each step with SymPy (the factor, the product rule check, the integral, the constant); the solution on its slope field | r = e^(x^2); y = e^(x - x^2) - 2 e^(-x^2); residual 0 |
| 3 | **The shape of the answer** | y' = x - y (left open in notebook 09): y = x - 1 + C e^(-x). Every solution is one particular solution plus C times a solution of y' + y = 0; the second part dies out (transient), so all solutions approach y = x - 1. Picture on the slope field | residual 0 |
| 4 | **When the integral has no formula** | Lebl Exercise 1.4.3, y' + y = e^(x^2 - x), y(0) = 10: the integral of e^(t^2) has no closed form; the solution is written with a definite integral (formula 1.4.5) and evaluated by the computer, then compared with `solve_ivp` | y(1) = 4.2169 both ways |
| 5 | **Existence and uniqueness** (the plan's theory exercise) | the two hypotheses of Picard's theorem checked on four problems met so far: y' = xy (both hold); y' = 1/x at x0 = 0 (f not continuous: no solution, Lebl 1.2.1); y' = 2 sqrt(abs(y)) at y0 = 0 (the y-derivative of f not continuous: two solutions, Lebl 1.2.2); y' = y^2 (both hold, yet the solution stops at x = 1/A: the theorem is local, Lebl 1.2.3). Then the linear case: the solution exists on the whole interval where p and f are continuous. Example x y' + y = 2x, y(1) = 2: the theorem promises x > 0, and y = x + 1/x lives exactly there; with y(1) = 1 the solution y = x lives everywhere (a guarantee, not a limit) | residuals 0 |
| 6 | **The non-unique case y' = y^(1/3)** (the plan's practical exercise) | y(0) = 0: y = 0 and y = (2x/3)^(3/2) are both solutions, and so is every curve that stays at 0 until x = c and then takes off: infinitely many. Which hypothesis fails: the y-derivative of f is y^(-2/3)/3, unbounded near y = 0 (picture). What the computer does: `solve_ivp` from y(0) = 0 returns y = 0; from y(0) = 10^(-9) it follows the other branch. **A numerical answer does not tell you that the solution is not unique.** Slider for the take-off point c, with a question first | y(3) = 2 sqrt 2 = 2.8284 on the upper branch; the solver gives 0 and 2.8284 |
| 7 | **Exercises** | the four steps below | - |
| 8 | **Answers** (collapsed) | opens after trying | - |

Section 6 closes two threads: it is the tank of notebook 10 run backwards (h = 0 reached in
finite time), and the example y' = 2 sqrt(abs(y)) of notebook 09.

## 2. Exercises (the four steps)

1. **Predict** (no code) - for four initial value problems, does Picard's theorem guarantee one
   and only one solution near the initial point? y' = y sqrt(abs(x)), y(0) = 0 (*Lebl, Exercise
   1.2.5*: yes); y' = y^(1/3), y(1) = 1 (yes: away from y = 0); y' = y^(1/3), y(1) = 0 (no);
   y' = x/y, y(0) = 0 (no: f is not continuous there). The answers are typed as True or False.
2. **By hand, then check** - (a) *Lebl, Exercise 1.4.11*: y' + 2 sin(2x) y = 2 sin(2x),
   y(pi/2) = 3; type the solution (Lebl: y = 2 e^(cos(2x) + 1) + 1). (b) *Lebl, Exercise 1.4.13*:
   P' = (2 - 0.1 t) P, P(0) = 1000; type P(5) (Lebl: 1000 e^8.75, about 6.31 million; it is linear
   and separable: solve it both ways). (c) the interval of existence promised by the theorem for
   y' + y/(x - 3) = 1/(x + 1), y(0) = 1, without solving: type its two ends (-1 and 3).
3. **Fill in** - solve y' + 2y = 4, y(0) = 0 by hand and complete `y_exact(x)`; the next cell
   compares it with `solve_ivp` (`assert`) and plots both (answer: 2 (1 - e^(-2x))).
4. **Explore** - *Lebl, Exercise 1.4.7.* A room whose temperature oscillates between day and night,
   x' = -k (x - A0 cos(w t)). A function that plots the numerical solutions for several initial
   temperatures is given. Do the initial conditions matter in the long term? What happens to the
   amplitude and to the delay when k is small (a thick wall)? (General solution, checked:
   A0 k (k cos(wt) + w sin(wt))/(k^2 + w^2) + C e^(-kt).)

Answers (final values only) in a collapsed block; full solutions after the session.

## 3. New in `odelib.py` (with tests)

`linear_first_order(p, f, x0, y0, x)`: the solution of y' + p(x) y = f(x), y(x0) = y0 from the
definite-integral formula (Lebl 1.4.5), with the integrals computed numerically. Used in section
4, and in notebook 13 for the mixing tanks. The integration routine of SciPy is imported inside
the function, as matplotlib is today, so `odelib` still loads with NumPy alone.

Tests: Lebl Example 1.4.1 (closed form), Exercise 1.4.13 (1000 e^8.75), **Example 1.4.2** (the
salt tank: 11.86 kg when the tank is full, at t = 20), a constant-coefficient case, x to the left
of x0, y(x0) = y0, invalid arguments.

## 4. Conventions and checks

As for notebooks 08-10: same setup cell, `assert` for the reference values, `ol.check` for the
answers typed by the student, one sentence under every plot, line styles as well as colours, a
question before the slider, plain ASCII, generated by `plans/make_11.py` with `nbbuild.py`.

* `tests/test_notebooks_run.py`: the notebook as it is; with the correct answers (it must print
  "correct" for 1, 2a, 2b, 2c and 3); with wrong answers.
* The site is rebuilt and the notebook is run in the browser (all cells, the slider).
* Index notebook: one more row. Notebook 10 ("where to go next") points to notebook 11.

Not in this notebook, on purpose: mixing tanks and Newton's cooling as models (notebook 13,
Sessions 7-8); Picard's iteration (it can become an optional appendix if you want the idea of the
proof shown).

## 5. Files

```
ode/ode/notebooks/
    11_Linear_First_Order.ipynb                <- to build
    plans/11_Linear_First_Order_plan.md        <- this document
    plans/make_11.py                           <- builds the notebook
    odelib.py                                  <- + linear_first_order()
    tests/test_odelib.py                       <- + its tests
    tests/test_notebooks_run.py                <- + the exercises of notebook 11
    00_Index.ipynb, plans/make_10.py           <- one line each
```
