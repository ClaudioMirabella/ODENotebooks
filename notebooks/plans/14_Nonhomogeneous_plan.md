# Notebook 14 - Non-homogeneous equations: plan

MATH 331 - Differential Equations - Spring 2027 - phase D of the plan of the student site
(the first notebook of phase D: notebooks 14-18, Units 5-9, Weeks 6-10)

**Status (2026-11-17): built and reviewed.** The independent review found no wrong worked example or answer, and these points, all corrected: Adkins-Davidson 3.4 Example 1 has e^(-x) on the right (a minus sign was lost in the text copy of the book); the guess for a polynomial times a sine or cosine is not a product of guesses (two independent polynomials are needed); `check_solution` measured the residual against the size of the answer, so a very large solution of the homogeneous equation passed (now measured against the right-hand side); an answer written with an absolute value was refused; `root_multiplicity` took a number very close to a double root for a root. The notebook has about 2300 words before the exercises, above the 2100 planned.

**Sessions 13 and 14, Unit 6** (lecture plan, Session 13: "Method of undetermined coefficients";
theory exercise "Explain when the trial solution must be modified"; practical exercise "Solve
equations with a variety of forcing terms". Session 14: "Variation of parameters; free and damped
vibrations"; theory exercise "Derive the variation-of-parameters formula"; the practical exercise
of Session 14, "Classify a spring-mass system by its damping", is notebook 07). About **15 minutes
in class in each session** (Session 13: sections 2-5; Session 14: sections 6-7), the rest as
homework.

**References** (every item below was found in the text of the book and recomputed with SymPy on
2026-11-17; exercise numbers of Lebl are those of the lists at the end of the chapter).
* Lebl, *Notes on Diffy Qs*, **Section 2.5, Nonhomogeneous equations**: Theorem 2.5.1
  (y = yc + yp); the opening example y'' + 5y' + 6y = 2x + 1 with y(0) = 0, y'(0) = 1/3; the
  note on the common mistake with initial conditions; y'' + 2y' + 2y = cos 2x; the "hiccup"
  y'' - 9y = e^(3x) and y'' - 6y' + 9y = e^(3x); sums of terms; variation of parameters on
  y'' + y = tan x. Exercises 2.5.1, 2.5.2, 2.5.3, 2.5.7, 2.5.8, 2.5.9, 2.5.11, 2.5.13, 2.5.16.
* Adkins and Davidson, *Ordinary Differential Equations* (Springer, 2012): **Section 3.4**,
  Examples 1, 5, 7, 9 and Theorem 8 (superposition); **Section 5.6**, Theorem 1 (the formula with
  the Wronskian), Remark 2 ("too complicated to memorize": remember the two conditions), Example 4
  (t y'' + 2y' + t y = 1). Their way of finding the trial function uses Laplace transforms, which
  the course meets only in Unit 10: the notebook takes their examples, not their procedure.
* Notebooks 05 (Wronskian), 06 (characteristic roots, the slider of two roots coming together),
  07 (vibrations), 11 (integrating factor: variation of parameters is its second-order
  relative). Videos R08 (2 x 2 linear systems), R07 (determinants).

**Purpose.** Three ideas, in this order.

1. **Structure.** Every solution is yc + yp: the part that remembers the initial data plus one
   particular solution, found "in any old way" (Lebl). Two different particular solutions differ
   by a solution of the homogeneous equation: this is why the computer can accept *any* correct
   answer, and why the notebook checks a particular solution by substituting it, never by
   comparing it with a formula.
2. **Undetermined coefficients** is a guess with a rule: guess a function of the same family as
   the right-hand side; if the guess already solves the homogeneous equation, multiply by x, as
   many times as the "number" of the right-hand side is a root of the characteristic equation.
   The student sees *why* the factor x appears: it is the limit of ordinary solutions, exactly as
   x e^(rx) appeared in notebook 06.
3. **Variation of parameters** is a formula, not a guess: two conditions, a 2 x 2 system whose
   determinant is the Wronskian (notebook 05), two integrals. It works for any continuous
   right-hand side, also when the integrals have no closed form (then the computer does them).

---

## 1. Contents

| # | Section | What the student does | Values shown (all checked) |
|---|---|---|---|
| 0 | Title, sessions, references, what you will learn | reads | - |
| 1 | **Theory in brief** | y = yc + yp; table of trial functions (polynomial of degree n; e^(ax); cos bx and sin bx together; products); the rule for the factor x^s; the two conditions of variation of parameters and the formulas u1' = -y2 f / W, u2' = y1 f / W | - |
| 2 | **One particular solution is enough** (Lebl's opening example) | by hand: yp = Ax + B in y'' + 5y' + 6y = 2x + 1, then yc, then the initial conditions on yc + yp. The same four lines in SymPy, as a recipe that is reused in the whole notebook: guess with symbols, substitute, `sp.solve` on the coefficients. The **common mistake** (constants fitted on yc alone) is run on purpose: its result fails the initial data. Plot: the solution, yp (dashed) and yc (dash-dot) | A = 1/3, B = -1/9; y = e^(-2x)/3 - 2e^(-3x)/9 + (3x - 1)/9; the wrong way gives y(0) = -1/9, y'(0) = 2/3 instead of 0, 1/3 |
| 3 | **A variety of forcing terms** (Session 13, practical exercise) | three right-hand sides by hand, each then checked by substitution: an exponential (Adkins-Davidson 3.4 Example 1, y'' - y' - 6y = e^(-x)), a cosine (Lebl, y'' + 2y' + 2y = cos 2x: why the sine is needed in the guess), a sum (Adkins-Davidson Example 9 with Theorem 8: one term at a time). Question before the plot: "the right-hand side is cos 2x; will the solution contain sin 2x?" | -e^(-x)/4; (-cos 2x + 2 sin 2x)/10; 2 - 4x e^(2x) |
| 4 | **When the guess must change** (Session 13, theory exercise) | Lebl's hiccup: A e^(3x) in y'' - 9y = e^(3x) gives 0 = e^(3x). The cure, x e^(3x)/6, and the double root y'' - 6y' + 9y = e^(3x) with x^2 e^(3x)/2. Then *why*: for y'' - 9y = e^(cx) with c not 3 the solution (e^(cx) - e^(3x))/(c^2 - 9) is an ordinary one, and as c tends to 3 it tends to x e^(3x)/6 (Lebl, Exercise 2.5.8: "handle every c"). **Slider 1**: c from 2 to 4; the title shows the size of the plain guess 1/(c^2 - 9), which explodes, while the curve passes smoothly through c = 3. The rule: s = number of times a + ib is a root (0, 1 or 2) | limit checked with `sp.limit`; at c = 3.1, 1/(c^2 - 9) = 1.64; at c = 3.01, 16.6 |
| 5 | **What stays and what fades** | y'' + 2y' + 2y = cos 2x from five different initial data: all the solutions join the same curve, yp. Because the roots are -1 +- i, yc dies out: yc is the *transient*, yp the *steady* part (the words of notebook 15). **Slider 2**: y(0) and y'(0). Question first: after which time are all the solutions within 1 % of each other? | yc is multiplied by e^(-x): below 1 % after x = ln 100 = 4.6 |
| 6 | **Variation of parameters** (Session 14, theory exercise) | the derivation in five numbered lines for y'' + p y' + q y = f: yp = u1 y1 + u2 y2; first condition u1' y1 + u2' y2 = 0; substitute; second condition u1' y1' + u2' y2' = f; solve the 2 x 2 system by Cramer's rule, where the determinant is the Wronskian. Then Lebl's example y'' + y = tan x by hand (W = 1; u1' = -sin x tan x, u2' = sin x), checked by substitution and against the numerical solution. Remark: for a first-order equation the same idea is the integrating factor of notebook 11 | yp = (1/2) cos x ln((1 - sin x)/(1 + sin x)) on (-pi/2, pi/2) |
| 7 | **Which method, and what if the integrals cannot be done?** | a table of two lines (undetermined coefficients: constant coefficients and right-hand sides made of polynomials, exponentials, sines and cosines, quick; variation of parameters: any continuous f and any equation whose yc is known, but two integrals). Lebl, Exercise 2.5.7, y'' - 2y' + y = sin(x^2): no formula, but `ol.forced_response` computes the integral; compared with `solve_ivp`. One variable-coefficient example in three lines: Adkins-Davidson 5.6 Example 4, yp = 1/t | agreement with `solve_ivp` to 1e-8 |
| 8 | **Exercises** | the four steps below | - |
| 9 | **Answers** (collapsed), where to go next (notebook 15; 07 for Session 14's second half) | opens after trying | - |

Length: at most 2100 words before the exercises (the range of notebooks 08-13); eight or nine
figures; two sliders, each with its static picture and its question first.

## 2. Exercises (the four steps)

1. **Predict** (no code) - For six equations, by which power of x must the usual guess be
   multiplied: 0, 1 or 2? (a) y'' - y' - 6y = e^(2x) [Lebl 2.5.1]; (b) y'' - 4y' + 4y = e^(2x)
   [Lebl 2.5.2]; (c) y'' + 9y = cos 3x + sin 3x [Lebl 2.5.3]; (d) y'' - 5y' + 6y = 4e^(2x)
   [Adkins-Davidson 3.4, Example 5]; (e) y'' + 2y' + 2y = e^(-x) cos x; (f) y'' + 2y' + 2y = cos x
   (the trap: the roots are -1 +- i, not +- i). Typed as a list, checked: [0, 2, 1, 1, 1, 0]. And
   one True/False: can undetermined coefficients solve y'' + y = tan x? (False.)
2. **By hand, then check** - (a) Lebl 2.5.1: a particular solution (-e^(2x)/4). (b) Lebl 2.5.2
   (x^2 e^(2x)/2). (c) Lebl 2.5.11: y'' - y' + y = 2 sin 3x ((-16 sin 3x + 6 cos 3x)/73, the
   answer printed in the book). (d) Lebl 2.5.13, an initial value problem: y'' + 2y' + y = x^2,
   y(0) = 1, y'(0) = 2 (x^2 - 4x + 6 + e^(-x)(x - 5), the answer of the book). For (a)-(c) **any**
   particular solution is accepted (the check substitutes it in the equation); (d) is unique and is
   also checked on its initial data.
3. **Fill in** - Complete `u_derivatives(y1, y2, f)`, which returns the pair (u1', u2') of
   variation of parameters (the Wronskian function of notebook 05, Exercise 3, is given). The next
   cell integrates them with SymPy, builds yp and checks it by substitution on three equations:
   y'' + y = tan x (Lebl), y'' - y = e^x (Lebl 2.5.9), t y'' + 2y' + t y = 1 (Adkins-Davidson 5.6,
   Example 4, where yp = 1/t). A wrong sign, or the two formulas exchanged, is reported with the
   name of the equation that fails, not with a traceback.
4. **Explore** - (i) Lebl 2.5.9, part (c): undetermined coefficients give x e^x / 2, variation of
   parameters gives x e^x / 2 - e^x / 4. Are they both right? What is going on? (ii) With slider
   1, find the particular solution of y'' - y = e^(cx) for every real c (Lebl 2.5.8). (iii) Lebl
   2.5.16: guessing also works outside constant coefficients: find a polynomial that solves
   y' + x y = x^3 + 2x^2 + 5x + 2 (x^2 + 2x + 3). (iv) Change the right-hand side of section 7 to
   one of your own and plot the solution with `ol.forced_response`.

Answers (final values only) in a collapsed block; full solutions after the session.

## 3. New in `odelib.py` (with tests)

* `check_solution(label, expression, residual, symbol, interval, conditions=(), tol=...)`:
  the check for answers that are **not unique**. `residual` is a function that takes a SymPy
  expression and returns "left-hand side minus right-hand side"; the answer is right when the
  residual is zero at `FORMULA_SAMPLES` points of the interval (and, if given, when each pair
  (point, value, order of derivative) in `conditions` is met). Same manners as `check` and
  `check_formula`: one line, True or False, never shows an answer, never raises for a wrong one,
  `None` means "no answer yet", a formula in another letter gets a hint. The zero function is
  refused for a non-homogeneous equation by construction (its residual is -f).
* `forced_response(t, a, b, c, f, y0=0.0, v0=0.0)`: the solution of a y'' + b y' + c y = f(t)
  with y(0) = y0, y'(0) = v0, for any continuous f: the homogeneous solution with those data
  (`constant_coefficient_solution`) plus the integral from 0 to t of g(t - s) f(s) / a ds, where
  g is the homogeneous solution with g(0) = 0, g'(0) = 1. This is variation of parameters written
  once and for all; integrals by `scipy.integrate.quad`, with the same care as
  `linear_first_order` (a failed integration is a ValueError, not a wrong number). It will be
  used again in notebook 15 (forced vibrations) and in Unit 11 (convolution).
* `root_multiplicity(a, b, c, alpha, beta=0.0, rtol=1e-9)`: 0, 1 or 2, the number of times
  alpha + i beta is a root of a r^2 + b r + c = 0 (the power of x in the rule of section 4).

Tests (`tests/test_odelib.py`, `tests/test_check_formula.py`, `tests/test_cross_validation.py`):
* `forced_response` against the closed forms of this plan (Lebl's opening example with its
  initial data, the cosine, the resonant e^(3x), the double root, Adkins-Davidson Examples 1, 5,
  7, 9) and against `solve_ivp` on random coefficients in the three cases of roots and on
  f = tan t and f = sin(t^2); f = 0 gives back `constant_coefficient_solution`; linear in f;
  invalid arguments (a = 0, non-finite values, f that is not callable).
* `root_multiplicity` on the six equations of Exercise 1 and at a nearly repeated root.
* `check_solution`: accepts yp and yp + any solution of the homogeneous equation; rejects a wrong
  coefficient, a lost sign, the zero function, a string; `None`; a wrong letter; a formula that
  cannot be evaluated; an initial condition that is not met.

## 4. Conventions and checks

As for notebooks 08-13 and the converted 01-07: same setup cell, `assert` for every reference
value, one sentence under every plot (written after looking at the figure and recomputing every
number in it), line styles as well as colours, a question before every slider and its answer
after, named constants, plain ASCII, generated by `plans/make_14.py` with `nbbuild.py`.

* Letters: y(x) for the equations (Lebl); Adkins-Davidson's examples are rewritten in x and say
  so; t only in their Example 4 of Section 5.6, quoted as it is. Coefficients of the guess A, B
  (Lebl). The amplitude letter A of notebooks 06 and 07 does not appear here, so there is no
  clash. p(x) is the coefficient of y' as in notebook 05 (not the p = c/(2m) of notebook 07).
* What must not be said: "the particular solution" as if it were unique (always "a particular
  solution"); "multiply by x" without "as many times as needed, and no more" (Lebl).
* `tests/test_notebooks_run.py`: the notebook as it is; with the correct answers (it must print
  "correct" for 1, 2a-2d and 3); with a *different but correct* particular solution in 2a-2c
  (yp plus a homogeneous solution: must be accepted); with wrong answers (wrong power of x, a
  lost sign, the two formulas of Exercise 3 exchanged); `EXTREMES` for both sliders (c = 3
  exactly, c at both ends, c = 3 +- one step; extreme initial data).
* The site is rebuilt and the notebook is run in the browser with `check_site.py`. `quad` inside
  `forced_response` is the only new cost: the picture of section 7 uses at most 200 points, and
  the time of the cell in the browser is measured before the notebook is accepted (limit: 5 s).
* Index notebook: one more row (14, U6, Sessions 13-14, Lebl 2.5); home page: a line for
  "Week 7" with 14 before 07 and 07E (Session 13 comes before Session 14); notebook 06 and
  notebook 07 ("where to go next") point to 14; notebook 05 gets one sentence (the Wronskian is
  used again in variation of parameters).
* After the build: an independent review as for 05-07E (mathematics, captions, citations,
  exercises, notation, clarity), and its corrections, before the notebook is called done.

## 5. Order of work

1. `odelib.py`: the three functions with their tests (the tests first, from the closed forms
   above); all tests pass.
2. `plans/make_14.py`: sections 1-3, generate, look at the figures.
3. Sections 4-5 with the two sliders; `EXTREMES`.
4. Sections 6-7; time the numerical cell.
5. Exercises and answers; `EXERCISES` in the tests, including the "different but correct" case.
6. Index, home page, links from 05, 06, 07; rebuild; `check_site.py`.
7. Independent review; corrections; final run of all tests.

## 6. Decisions for the instructor

1. **One notebook for both sessions** (proposed), or two (14a undetermined coefficients, 14b
   variation of parameters). One keeps the comparison of the two methods in one place and
   follows the plan of phase D; the 15 minutes of each session use different sections.
2. **An optional notebook 14E later** (proposed: yes, after notebook 15, not now): the same
   formula read as "response to a short push" (Green's function), which prepares the
   convolution of Unit 11; Adkins-Davidson 5.6 Example 3 (a Cauchy-Euler equation); Lebl
   Exercise 2.5.14.
3. **The rule for the factor x** is stated as "the number a + ib of the right-hand side, counted
   as a root of the characteristic equation". Lebl says "multiply by x until there is no
   duplication": the same rule, less precise. Proposed: Lebl's sentence first, then the precise
   form.

## 7. Files

```
ode/ode/notebooks/
    14_Nonhomogeneous_Equations.ipynb          <- to build
    plans/14_Nonhomogeneous_plan.md            <- this document
    plans/make_14.py                           <- builds the notebook
    odelib.py                                  <- + check_solution(), forced_response(), root_multiplicity()
    tests/test_odelib.py                       <- + tests of the two numerical functions
    tests/test_check_formula.py                <- + tests of check_solution
    tests/test_cross_validation.py             <- + forced_response against solve_ivp
    tests/test_notebooks_run.py                <- + the exercises and sliders of notebook 14
    00_Index.ipynb, plans/make_05.py, make_06.py, make_07.py   <- index row; one line each
ode/ode/site/build_site.py                     <- READING_ORDER: 14 in Week 7
```
