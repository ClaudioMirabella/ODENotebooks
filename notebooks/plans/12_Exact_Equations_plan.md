# Notebook 12 - Exact equations and substitutions: plan

MATH 331 - Differential Equations - Spring 2027 - phase C of the plan of the student site

**Session 6, Unit 3** (lecture plan: "Exact equations; integrating factors; Bernoulli and
homogeneous substitutions". Theory exercise: "Prove the test for exactness". Practical exercise:
"Find integrating factors of the form mu(x) or mu(y)"). About **15 minutes in class** (section 4,
the integrating factors), the rest as homework.

**References.** Lebl, *Notes on Diffy Qs*: **Section 1.8** (exact equations: Figure 1.8.1,
Examples 1.8.2, 1.8.3, 1.8.5-1.8.7, Theorem 1.8.1; Exercises 1.8.7, 1.8.8, 1.8.9) and
**Section 1.5** (substitution: Example 1.5.1, Bernoulli; Example 1.5.2, homogeneous; Exercise
1.5.7). Prerequisites: partial derivatives (one paragraph in the notebook), videos R02 and R03;
notebooks 10 and 11.

**Purpose.** Three more tools for first-order equations, each reduced to something already known:
an *exact* equation is a conserved quantity F(x, y) = C in disguise (level curves, as in notebook
10); an *integrating factor* makes an equation exact (the factor of notebook 11 is a special
case); a *substitution* turns a Bernoulli equation into a linear one and a homogeneous equation
into a separable one. The computer does the partial derivatives and draws the level curves; the
student decides which tool applies.

---

## 1. Contents

| # | Section | What the student does | Values shown (all checked) |
|---|---|---|---|
| 0 | Title, session, references, what you will learn | reads | - |
| 1 | **Theory in brief** | partial derivatives in three lines; M dx + N dy = 0 is exact when it is dF = 0; the test M_y = N_x and why it is necessary (mixed derivatives are equal: the plan's theory exercise, stated and proved in four lines); the procedure for F; integrating factors mu(x) and mu(y); Bernoulli (v = y^(1-n)); homogeneous (v = y/x); a table "when you see ... try ..." | - |
| 2 | **A conserved quantity** | Lebl Figure 1.8.1: F = x^2 + y^2, the circles F = C, and the equation 2x + 2y y' = 0 they satisfy. Level curves drawn on the slope field of y' = -x/y: the segments are tangent to the circles | - |
| 3 | **The test, and the potential, step by step** | Lebl Example 1.8.3, (2x + y) + (x - 1) y' = 0, y(0) = 1: test, integrate M in x, differentiate in y, compare with N, find C; by hand and with SymPy. Lebl Example 1.8.2 (M = 2x + y, N = xy) fails the test | M_y = N_x = 1; F = x^2 + xy - y; C = -1; y = (x^2 + 1)/(1 - x). For 1.8.2: M_y - N_x = 1 - y |
| 4 | **Integrating factors mu(x) and mu(y)** (the plan's practical exercise) | Lebl Example 1.8.6: (M_y - N_x)/N = 1/(x + 1) depends on x only, mu = x + 1, and the equation becomes Example 1.8.5 with F = x^3/3 + x y^2 + y^2. Lebl Example 1.8.7: (M_y - N_x)/M = 1/y depends on y only, mu = 1/y, F = xy + ln y: an implicit solution, drawn as level curves with a numerical curve on one of them. Remark: the factor e^(integral of p) of notebook 11 is the case mu(x) | mu = x + 1; mu = 1/y; on the curve through (0, 1), y(1) = 0.5671 from the implicit relation and from `solve_ivp` |
| 5 | **Bernoulli equations** | Lebl Example 1.5.1, x y' + y (x + 1) + x y^5 = 0, y(1) = 1: v = y^(-4) gives the linear equation v' - 4 (x + 1)/x v = 4, v(1) = 1, solved with `ol.linear_first_order` (the integral has no closed form), then y = v^(-1/4); compared with `solve_ivp` on the original equation and with Lebl's formula | y(2) = 0.1654 three ways |
| 6 | **Homogeneous equations** | Lebl Example 1.5.2, x^2 y' = y^2 + xy, y(1) = 1: v = y/x gives x v' = v^2, so y = x/(1 - ln x), valid for 0 < x < e. Picture: the slope depends only on y/x, so it is constant along each ray through the origin (the isoclines are rays) | y = x/(1 - ln x); validity 0 < x < e |
| 7 | **Exercises** | the four steps below | - |
| 8 | **Answers** (collapsed) | opens after trying | - |

## 2. Exercises (the four steps)

1. **Predict** (no code) - which of these are exact? (2x + y) dx + (x - 4y) dy = 0 (*Lebl 1.8.7b*:
   yes); (2y + y^2/x) dx + (2y + x) dy = 0 (*Lebl 1.8.8d*: no); y dx - x dy = 0 (no);
   e^x dx + e^y dy = 0 (*Lebl 1.8.7c*: yes). Typed as a list of True and False.
2. **By hand, then check** - (a) *Lebl, Exercise 1.8.7(d)*: find a potential of
   (3x^2 + 3y) dx + (3y^2 + 3x) dy = 0 (Lebl: x^3 + 3xy + y^3); any constant may be added, and the
   check allows it. (b) *Lebl, Exercise 1.8.8(d)*: find an integrating factor of
   (2y + y^2/x) dx + (2y + x) dy = 0 (Lebl: x); the cell checks that the student's factor really
   makes the equation exact, so any correct factor is accepted. (c) *Lebl, Exercise 1.5.7*:
   x y' + y + y^2 = 0, y(1) = 2 (Lebl: y = 2/(3x - 2)).
3. **Fill in** - complete `is_exact(M, N)`, which returns True when M_y = N_x (one line with
   `sp.diff` and `sp.simplify`); `assert` lines try it on the four equations of Exercise 1 and on
   the examples of the notebook.
4. **Explore** - *Lebl, Exercise 1.8.9.* Every separable equation y' = f(x) g(y) can be written as
   an exact equation. Do it for y' = xy, find the potential, draw its level curves and compare with
   the solutions of notebook 10 (Section 2). Then: is every linear equation exact after multiplying
   by its integrating factor? Check it on Lebl's Example 1.4.1.

Answers (final values only) in a collapsed block; full solutions after the session.

## 3. `odelib.py`

**No new function.** The notebook reuses `slope_field`, `solution_curve`, `linear_first_order`
(section 5) and `check`. The tools of this session are symbolic (partial derivatives, a test, a
potential), and `odelib` deliberately depends on NumPy only; the two helpers the notebook needs
(`residual`, and `exactness(M, N)` returning M_y - N_x) are three lines each and are written in
the notebook, where the student can read them.

## 4. Conventions and checks

As for notebooks 08-11: same setup cell, `assert` for the reference values, `ol.check` for the
answers typed by the student, one sentence under every plot, line styles as well as colours, plain
ASCII, generated by `plans/make_12.py` with `nbbuild.py`.

* **Sliders:** one, in section 6: the direction of a ray through the origin, showing that the
  slope along it is constant (question first: what is the slope along the ray y = x?). The other
  sections are step-by-step computations, where a slider would add nothing.
* `tests/test_notebooks_run.py`: the notebook as it is; with the correct answers (it must print
  "correct" for 1, 2a, 2b, 2c and 3), including a potential shifted by a constant and an
  integrating factor multiplied by a constant; with wrong answers.
* The site is rebuilt and the notebook is run in the browser (all cells, the slider).
* Index notebook: one more row. Notebook 11 ("where to go next") points to notebook 12.

## 5. Files

```
ode/ode/notebooks/
    12_Exact_Equations.ipynb                   <- to build
    plans/12_Exact_Equations_plan.md           <- this document
    plans/make_12.py                           <- builds the notebook
    tests/test_notebooks_run.py                <- + the exercises of notebook 12
    00_Index.ipynb, plans/make_11.py           <- one line each
```
