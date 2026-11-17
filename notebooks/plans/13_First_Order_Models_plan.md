# Notebook 13 - First-order models: plan

MATH 331 - Differential Equations - Spring 2027 - phase C of the plan of the student site
(the last notebook of phase C: after it, Units 1-4 and Midterm 1 are covered)

**Sessions 7 and 8, Unit 4** (lecture plan, Session 7: "Growth and decay; Newton's law of
cooling; mixing tanks; logistic equation; autonomous equations and the phase line"; theory
exercise "Classify the stability of equilibria"; practical exercise "Build a mixing-tank model
from a verbal description". Session 8: "Falling bodies with drag; RC and RL circuits; preview of
Euler's method"; theory exercise "Derive the terminal velocity"; practical exercise "Compare
Euler's method with the exact solution"; Quiz 2). About **15 minutes in class in each session**
(Session 7: sections 2-4; Session 8: sections 5-7), the rest as homework and review for the
midterm.

**References.** Lebl, *Notes on Diffy Qs*: Example 1.3.3 and Exercise 1.3.12 (cooling); Example
1.4.2 and Exercises 1.4.6, 1.4.8, 1.4.9, 1.4.12 (mixing); Section 1.6 (autonomous equations,
phase line); Section 1.7 (Euler's method). Adkins and Davidson, *Ordinary Differential Equations*
(Springer, 2012): Section 1.4, Example 11 (mixing); Section 3.7 (the laws of the resistor, the
capacitor and the inductor). Falling bodies and first-order circuits are not in Lebl: the notebook
derives them from Newton's second law and Kirchhoff's voltage law, and the numbers are mine.
Already covered elsewhere, and only pointed to: growth and decay (notebooks 01, 02), the logistic
equation and its phase line (notebook 03), Euler's method in depth (notebook 04).

**Purpose.** The slide of Lecture 1, "same mathematics, different letters", made concrete: the
cooling coffee, the salt in a tank, the falling body with linear drag, the RC circuit and the RL
circuit are **one equation**,

    x' = (x_inf - x) / tau,     x(t) = x_inf + (x0 - x_inf) e^(-t/tau),

with one stable equilibrium x_inf and one time constant tau. The student learns to *build* the
model from words (a balance of rates), to *read* tau and x_inf from it without solving, and to
see what changes when the model is not of this type (variable volume; quadratic drag).

---

## 1. Contents

| # | Section | What the student does | Values shown (all checked) |
|---|---|---|---|
| 0 | Title, sessions, references, what you will learn | reads | - |
| 1 | **Theory in brief** | modelling as a balance (rate of change = rate in - rate out); the equation x' = (x_inf - x)/tau and its solution; equilibrium and stability on the phase line (Lebl 1.6); the two numbers to remember: 63 % of the gap is closed after one tau, 99 % after five | 1 - 1/e = 0.632; 1 - e^(-5) = 0.993 |
| 2 | **One equation, five stories** | a table: for each model, the law, the equation, x_inf and tau. Cooling: tau = 1/k, x_inf = room temperature. Mixing at constant volume V with flow r: tau = V/r, x_inf = V times the incoming concentration. Linear drag: tau = m/k, v_inf = mg/k. RC: tau = RC, q_inf = CE. RL: tau = L/R, i_inf = E/R. One picture: the five solutions, each divided by its own x_inf and plotted against t/tau, fall on the **same curve** | the five curves coincide (assert) |
| 3 | **Equilibrium and stability** (Session 7, theory exercise) | the phase line of x' = (x_inf - x)/tau with `ol.phase_line`: one equilibrium, stable, because f'(x_inf) = -1/tau < 0; compared in two lines with the logistic equation (two equilibria, notebook 03). Slider: x0, x_inf and tau, with the markers at t = tau and t = 5 tau | - |
| 4 | **Building a mixing model from words** (Session 7, practical exercise) | Adkins-Davidson Example 11 (constant volume): 10 L, 100 g/L at the start, 400 g/L in at 3 L/min, out at 3 L/min: y' = 1200 - 3y/10, y = 4000 - 3000 e^(-3t/10); tau and y_inf read from the equation. Then Lebl Example 1.4.2, where the **volume changes** (60 + 2t litres): the equation is still linear but no longer of the simple type; solved with `ol.linear_first_order` and compared with Lebl's formula | tau = 10/3 min, y_inf = 4000 g; 11.86 kg of salt when the tank is full, at t = 20 |
| 5 | **Falling with drag** (Session 8, theory exercise) | linear drag, m v' = mg - k v: terminal velocity v_inf = mg/k from v' = 0, without solving. Quadratic drag, m v' = mg - c v^2 (a skydiver): terminal velocity v_T = sqrt(mg/c), again from v' = 0; the solution v = v_T tanh(g t / v_T) (separable, partial fractions: video R04), compared with `solve_ivp`; both models on one plot, with the same terminal velocity | m = 80 kg, c = 0.25 kg/m: v_T = 56.0 m/s (202 km/h); 95 % of it after 10.5 s |
| 6 | **RC and RL circuits** | Kirchhoff's voltage law with the three component laws (Adkins-Davidson 3.7): R q' + q/C = E and L i' + R i = E. Charging a capacitor and switching on a coil: same curve, tau = RC and tau = L/R. Question before the plot: doubling R makes the RC circuit slower or faster? And the RL circuit? | R = 1 kOhm, C = 1 mF, E = 5 V: tau = 1 s, q_inf = 5 mC. L = 0.5 H: tau = 0.5 ms, i_inf = 5 mA |
| 7 | **Euler's method against the exact solution** (Session 8, practical exercise) | Euler's method in three lines (follow the slope for a step h; Lebl 1.7), with `ol.euler`, on the falling body with linear drag. Table of the error at a fixed time for h halved four times: the error halves (order 1, `ol.observed_order`). Slider for h, with a question first: what happens when h is larger than tau? Larger than 2 tau? (The numerical solution oscillates, then explodes: a numerical answer can be wrong in a way the exact solution never is.) Pointer to notebook 04 | errors 0.94, 0.50, 0.25, 0.13 for h = 2, 1, 0.5, 0.25 s: observed order 0.92, 0.97, 0.99; for h = 12.5 s (2.2 tau) the iterates are 0, 123, -23, 150, -55 |
| 8 | **Exercises** | the four steps below | - |
| 9 | **Answers** (collapsed) | opens after trying | - |

## 2. Exercises (the four steps)

1. **Predict** (no code) - a quantity relaxes with time constant tau. What fraction of the initial
   gap is left after tau, after 3 tau? In an RC circuit the resistance is doubled: what happens to
   tau and to the final charge? In an RL circuit? (The answers are typed as numbers and checked.)
2. **By hand, then check** - (a) *Lebl, Exercise 1.4.8*: 5 g of salt in 20 L, brine at 2 g/L in at
   3 L/min, out at 3 L/min; after how long are there 20 g? (3.73 min). (b) *Lebl, Exercise 1.4.9*:
   the unknown concentration of the incoming brine (1.735 g/L). (c) *Lebl, Exercise 1.3.12*: a
   coffee at 100, 70 and 50 degrees at t = 0, 10, 20 min: the room temperature (10 degrees).
   (d) the terminal velocity of an 80 kg skydiver whose parachute has c = 30 kg/m (5.1 m/s).
3. **Fill in** - *Lebl, Exercise 1.4.12*, the practical exercise of Session 7. A tank with 10 L of
   clean water is pumped out at 3 L/min; water with a toxic substance flows in at 2 L/min with
   concentration 20 t g/L. Complete `rate_of_change(t, x)` from this description (rate in times
   concentration in, minus rate out times concentration out); the next cell solves the equation
   with `solve_ivp` and checks the amount when the tank is half empty against Lebl's answer, 250 g.
4. **Explore** - *Lebl, Exercise 1.4.6.* Two lakes on a stream (100 and 200 thousand litres, 500
   L/h through both); a truck with 500 kg of a toxic substance crashes into the first. Model both
   lakes, solve numerically, and find when the first lake is below 0.001 kg/L (after about 322 h)
   and when the second is worst (after about 277 h). Which notebook already solved the same
   equations? (Notebook 02, the decay chain.)

Answers (final values only) in a collapsed block; full solutions after the session.

## 3. New in `odelib.py` (with tests)

* `relaxation(t, x0, x_inf, tau)`: the solution of x' = (x_inf - x)/tau, the curve of sections 2-7.
* `quadratic_drag_velocity(t, terminal_velocity, g)`: v_T tanh(g t / v_T), for a body released
  from rest.

Tests: the equation is satisfied (finite differences); x(0) = x0; 63.2 % at t = tau; agreement
with `rk4`; Adkins-Davidson Example 11 (4000 - 3000 e^(-3t/10)); Lebl Example 1.3.3 through
`relaxation` (60 degrees at 9.21 min); the skydiver's 56.0 m/s as the limit and 95 % at 10.5 s;
invalid arguments. `linear_first_order` (notebook 11) is reused for the tank of variable volume.

## 4. Conventions and checks

As for notebooks 08-12: same setup cell, `assert` for the reference values, `ol.check` for the
answers typed by the student, one sentence under every plot, line styles as well as colours, a
question before every slider, plain ASCII, generated by `plans/make_13.py` with `nbbuild.py`.

* Two sliders (sections 3 and 7), each with its static picture first.
* `tests/test_notebooks_run.py`: the notebook as it is; with the correct answers (it must print
  "correct" for 1, 2a-2d and 3); with wrong answers.
* The site is rebuilt and the notebook is run in the browser (all cells, one slider).
* Index notebook: one more row, and its "Next notebooks" paragraph is brought up to date (it
  still lists Sessions 3-8 as future work). Notebook 12 ("where to go next") points to 13.

After this notebook, phase C is complete. Before phase D (notebooks 14-18, Units 5-9) it would be
worth a pass over notebooks 08-13 together (consistency of notation, total length, what a student
sees in order), and phase B, the tests on real devices, is still open.

## 5. Files

```
ode/ode/notebooks/
    13_First_Order_Models.ipynb                <- to build
    plans/13_First_Order_Models_plan.md        <- this document
    plans/make_13.py                           <- builds the notebook
    odelib.py                                  <- + relaxation(), quadratic_drag_velocity()
    tests/test_odelib.py                       <- + their tests
    tests/test_notebooks_run.py                <- + the exercises of notebook 13
    00_Index.ipynb, plans/make_12.py           <- index row and paragraph; one line
```
