# Notebook 10 - Separable equations: plan

MATH 331 - Differential Equations - Spring 2027 - phase C of the plan of the student site

**Session 4, Unit 2** (lecture plan: "Separable equations; implicit solutions". Theory exercise:
"Determine the domain of validity of a solution". Practical exercise: "Torricelli's law: time to
drain a tank"). About **15 minutes in class** (section 6, the tank), the rest as homework.

**References.** Lebl, *Notes on Diffy Qs*, **Section 1.3**: Examples 1.3.1 (y' = xy), 1.3.2
(interval of validity), 1.3.3 (Bob's coffee), 1.3.4 (singular solution); Subsection 1.3.1 and
Figure 1.3.1 (implicit solutions); Exercises 1.3.3, 1.3.14, 1.3.16, 1.3.18. Adkins and Davidson,
*Ordinary Differential Equations* (Springer, 2012), Section 1.1, Exercise 3 (Torricelli's law).
Prerequisites: videos R02 (substitution), R04 (partial fractions), R01 (chain rule); notebook 09.

**Purpose.** The first **method** of the course, done by hand and checked by the computer:
*separate, integrate, solve for y, check*. Three things a formula does not say by itself, each
with a picture: a solution lost when dividing (singular solution), a solution that cannot be
solved for y (implicit solution), and a solution that lives only on an interval (validity).
Then the method at work on a tank.

---

## 1. Contents

| # | Section | What the student does | Values shown (all checked) |
|---|---|---|---|
| 0 | Title, session, references, what you will learn | reads | - |
| 1 | **Theory in brief** | y' = f(x) g(y); the four steps: separate, integrate both sides, solve for y if possible, check; the constant solutions g(y) = 0 are found first, because dividing by g(y) loses them; why "integrating in two variables" is legitimate (substitution, video R02) | - |
| 2 | **The method, step by step** | Lebl Example 1.3.1, y' = xy, by hand; then the same steps with SymPy (the two integrals, then `sp.solve`), residual 0; the curves on the slope field of notebook 09 now have a formula | y = D e^(x^2/2); y(2) = 0.2 e^2 = 1.4778 for y(0) = 0.2 |
| 3 | **A solution lost on the way** | Lebl Example 1.3.4, y' = -x y^2/3: the family y = 6/(x^2 + 2C) and the singular solution y = 0, which no value of C gives; plot of the family with y = 0 | residuals 0 |
| 4 | **Implicit solutions** | Lebl Subsection 1.3.1, y' = xy/(y^2 + 1): the integration gives y^2 + 2 ln(abs(y)) = x^2 + C, which cannot be solved for y. Figure 1.3.1 as a contour plot; the numerical curve through (1, 1) lies on the level C = 0; a value of y is computed from the implicit relation with a root finder; checking an implicit solution by differentiating | y(2) = 1.7106 from the implicit relation and from the numerical solution |
| 5 | **Where does the solution live?** (the plan's theory exercise) | Lebl Example 1.3.2: x^2 y' = 1 - x^2 + y^2 - x^2 y^2, y(1) = 0 gives y = tan(-1/x - x + 2). The formula makes sense for many x, the *solution* only on the interval around x = 1 where the tangent has no asymptote. Plot with the asymptotes; the same question for y' = y^2 (notebook 09, Exercise 4) | interval 0.306 < x < 3.264 |
| 6 | **Torricelli's law: draining a tank** (the plan's practical exercise) | model: h' = -(a/A) sqrt(2 g h) for a cylindrical tank of cross-section A with a hole of area a; by hand h(t) = (sqrt(h0) - (a/A) sqrt(g/2) t)^2 and the draining time T = (A/a) sqrt(2 h0/g); compared with `solve_ivp`; sliders for the hole radius and the initial height, with questions first (hole twice as wide: time divided by 4; water twice as high: time times sqrt 2) | tank radius 0.5 m, hole radius 1 cm, h0 = 1 m: T = 1128.8 s = 18.8 min; half the height is lost in the first 29 % of the time |
| 7 | **Look back** | after T the tank stays empty: h = 0 is also a solution, the same equation as y' = 2 sqrt(abs(y)) of notebook 09 run backwards. From an empty tank you cannot tell when it emptied (uniqueness fails at h = 0: Session 5) | - |
| 8 | **Exercises** | the four steps below | - |
| 9 | **Answers** (collapsed) | opens after trying | - |

A note for section 6: the numerical solver needs the square root of a number that rounding can
make slightly negative near the end; the notebook shows the one-line remedy (never take the root
of a negative height) and says why, linking back to "computers compute approximately" (notebook
08).

## 2. Exercises (the four steps)

1. **Predict** (no code) - a tank drains in 20 minutes. Without computing: is it half empty after
   10 minutes, before, or after? What if the hole is twice as wide? Then check with the slider of
   section 6.
2. **By hand, then check** - (a) *Lebl, Exercise 1.3.14*: x' = 3x t^2 - 3t^2, x(0) = 2; type the
   solution as an expression in `t` (answer in Lebl: x = e^(t^3) + 1). (b) *Lebl, Exercise 1.3.16*:
   x y' = y^2, y(1) = 1; type the solution and the right end of its interval of validity (answers:
   y = 1/(1 - ln x), valid for 0 < x < e). (c) *Lebl, Exercise 1.3.3*: x' = (x - 1)^2, x(0) = 0;
   type the left end of the interval of validity (answer: x = t/(t + 1), valid for t > -1).
3. **Fill in** - complete `drain_time(h0, tank_radius, hole_radius)` from the formula of section 6;
   `assert` lines check it against the value of the notebook and against the two scaling laws.
4. **Explore** - *Lebl, Example 1.3.3 and Exercise 1.3.18.* Bob's coffee: 89 degrees at t = 0, 85
   degrees after one minute, room at 22 degrees; drinkable at 60 degrees after 9.21 minutes. If
   each temperature is known within 0.5 degrees, between which times is the coffee drinkable?
   (Lebl: about 7.45 to 12.15 minutes.) What does this say about quoting "9.21"?

Answers (final values only) in a collapsed block; full solutions after the session.

## 3. New in `odelib.py` (with tests)

* `torricelli_height(t, h0, area_ratio, g)`: the exact height, equal to 0 from the draining time
  on (so it can be plotted beyond T).
* `torricelli_drain_time(h0, area_ratio, g)`.

`area_ratio` is hole area over tank cross-section; `g` has the standard value as default, named
once. Tests: h(0) = h0, h(T) = 0 and stays 0, the equation is satisfied (finite differences),
agreement with `rk4` on the clipped equation, the two scaling laws, the value 1128.8 s, invalid
arguments (ValueError). Notebook 13 (first-order models) reuses them.

## 4. Conventions and checks

As for notebooks 08 and 09: same setup cell, `assert` for the reference values, `ol.check` for the
answers typed by the student (expressions are compared through their values at sample points, as
in notebook 08), one sentence under every plot, line styles as well as colours, a question before
every slider, plain ASCII, generated by `plans/make_10.py` with the shared `nbbuild.py`.

* `tests/test_notebooks_run.py`: the notebook as it is; with the correct answers (it must print
  "correct" for 2a, 2b, 2c and 3); with wrong answers.
* The site is rebuilt and the notebook is run in the browser (all cells, one slider).
* Index notebook: one more row. Notebook 09 ("where to go next") points to notebook 10.

## 5. Files

```
ode/ode/notebooks/
    10_Separable_Equations.ipynb               <- to build
    plans/10_Separable_Equations_plan.md       <- this document
    plans/make_10.py                           <- builds the notebook
    odelib.py                                  <- + torricelli_height(), torricelli_drain_time()
    tests/test_odelib.py                       <- + their tests
    tests/test_notebooks_run.py                <- + the exercises of notebook 10
    00_Index.ipynb, plans/make_09.py           <- one line each
```
