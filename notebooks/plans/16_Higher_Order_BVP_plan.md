# Notebook 16 - Higher-order equations and boundary value problems: plan

MATH 331 - Differential Equations - Spring 2027 - phase D of the plan of the student site
(the third notebook of phase D)

**Status (2026-11-17): built and reviewed.** 1975 words before the exercises. The independent review found no wrong formula, table entry or answer. Found while building, by the new tests: a single number typed in place of a list in Exercise 1 stopped the notebook; `check_solution` blamed the formula when a condition given at `sp.pi` was not met. Corrected after the review: `classify_linear_system` gave wrong answers when the equations had very different sizes (now each is divided by its largest coefficient) and its test for a solvable singular system depended on the size of the right-hand side; the check of Exercise 3 refused 0.013021 w0 L^4 / EI, the value printed in Section 6; `beam_deflection` returned values outside the beam; the sentence on conditions at one point now says "initial conditions"; the Fredholm alternative is stated for boundary values and Lebl's theorem cited for what it says; wording in six places.

**Session 16, Unit 7** (lecture plan: "Theory of nth-order equations; reduction of order;
higher-order constant-coefficient equations; two-point BVPs (beam deflection)"; theory exercise
"Show that a BVP can have no solution or infinitely many solutions"; practical exercise "Compute
the deflection of a beam"; Quiz 4). About **15 minutes in class** (sections 5 and 6), the rest as
homework.

**References** (every item was found in the text of the book and recomputed on 2026-11-17).
* Lebl, *Notes on Diffy Qs*, **Section 2.3, Higher order linear ODEs**: Theorem 2.3.1
  (superposition), Theorem 2.3.2 (existence and uniqueness with n initial conditions), Example
  2.3.3 with its initial conditions (C = -1/4, 1, 1/4) and the advice on finding roots (try -1,
  0, 1; a change of sign; the product of the roots), Example 2.3.4 (r (r - 1)^3), Example 2.3.5
  ((r^2 - 2r + 2)^2). Exercises 2.3.2, 2.3.5, 2.3.12, 2.3.13, 2.3.14. Reduction of order:
  Exercises 2.1.7 and 2.1.8 (Chebyshev's equation of order 1).
* Lebl, **Section 4.1, Boundary value problems**: Examples 4.1.1 (infinitely many solutions)
  and 4.1.2 (only the zero solution), Example 4.1.3 (the eigenvalues k^2), Theorem 4.1.2
  (Fredholm alternative). Exercises 4.1.7 (answer in the book: lambda = 4 k^2 pi^2), 4.1.8
  (answer in the book: x = -sin t), 4.1.9.
* J. Kiusalaas, *Numerical Methods in Engineering with Python 3* (Cambridge University Press,
  2013), **Section 8.2, Example 8.4**: the simply supported beam with the load w0 x / L,
  d^4 v / dx^4 = w0 x / (L EI), v = v'' = 0 at both ends; printed results: slopes
  19.444e-3 and -22.222e-3 (times w0 L^3 / EI) and mid-span displacement 6.5104e-3 (times
  w0 L^4 / EI); the remark in Section 8.3 on which pairs of conditions make sense at one end;
  Example 8.8 (beam with both ends fixed). Neither Lebl nor Adkins and Davidson treats the static
  beam (Lebl has the vibrating beam, Section 5.2, with the same end conditions), so this book of
  the repository is the reference for it. The classical values for a uniform load
  (5/384, 1/8, 1/384) are standard results of beam theory, derived in the notebook, not quoted.
* Notebooks 05 (reduction of order is done there, Section 6), 06 (Section 6 already shows
  Example 2.3.3 in three lines), 08 (linear systems), 14, 15. Videos R06 (roots of polynomials),
  R08 (linear systems).

**Purpose.** Two ideas that sound alike and are not.

1. **Order n: n constants, and n conditions at one point always fix them.** Higher order with
   constant coefficients is notebook 06 with more roots; the only new thing is roots repeated
   more than twice, and how to find the roots at all.
2. **The same number of conditions at two different points may fix nothing.** A boundary value
   problem can have one solution, none, or infinitely many, and the reason is elementary: the
   conditions are a linear system for the constants, and a linear system can be singular. The
   beam is the case where everything goes well, and where the four conditions have a physical
   meaning.

---

## 1. Contents

| # | Section | What the student does | Values shown (all checked) |
|---|---|---|---|
| 0 | Title, session, references, what you will learn | reads | - |
| 1 | **Theory in brief** | order n: general solution with n constants; n initial conditions at one point give exactly one solution (Lebl, Theorem 2.3.2); constant coefficients: a root r of multiplicity k gives e^(rx), x e^(rx), ..., x^(k-1) e^(rx) (and the same with cos, sin for a complex pair); boundary conditions: a linear system M C = b for the constants, one solution exactly when det M is not 0; the beam: EI y'''' = w(x), two conditions at each end | - |
| 2 | **n constants, n initial conditions** | Lebl, Example 2.3.3 with y(0) = 1, y'(0) = 2, y''(0) = 3: the three conditions as a 3 x 3 system (video R08), solved by `np.linalg.solve`; its matrix has the Wronskian at 0 as determinant, which is not zero (notebook 05), so there is always exactly one answer. The "trick" exercise of Lebl (2.3.14): ugly coefficients and zero initial data: the solution is y = 0, by uniqueness, with no computation | C = (-1/4, 1, 1/4); determinant 16 |
| 3 | **Roots of any multiplicity** | finding the roots (Lebl's advice, then the computer): plot of r^3 - 3r^2 - r + 3 with its three crossings; `sp.roots` gives roots with multiplicities. Example 2.3.4 (0 once, 1 three times) and Example 2.3.5 (1 +- i twice each), each general solution written by the rule and checked by substitution. Question before the code: how many constants, and which functions, for y^(5) - y^(4) = 0 (Exercise 2.3.12)? | (C1 + C2 x + C3 x^2) e^x + C4; e^x ((C1 + C2 x) cos x + (C3 + C4 x) sin x); C1 e^x + C2 x^3 + C3 x^2 + C4 x + C5 |
| 4 | **One solution gives another** (reduction of order) | six lines, pointing to notebook 05, Section 6: try y2 = y1 v; the equation for w = v' is of first order. One example in SymPy: Lebl, Exercise 2.1.8 (Chebyshev, y1 = x). This section is short on purpose: the method is already in notebook 05 | y2 = sqrt(1 - x^2) (up to a constant factor), on -1 < x < 1 |
| 5 | **Conditions at two points** (Session 16, theory exercise) | x'' + lambda x = 0 on [0, pi] with x(0) = 0 and a condition at pi. General solution A cos + B sin; the two conditions as a 2 x 2 system; its determinant is sin(sqrt(lambda) pi). Three cases by hand: Lebl's Example 4.1.2 (lambda = 2, x(pi) = 0: only x = 0); Example 4.1.1 (lambda = 1, x(pi) = 0: every B sin t); and lambda = 1 with x(pi) = 1: **no solution**, because sin(pi) = 0. Picture of the three. The rule in plain words (Lebl's Theorem 4.1.2): either the problem with zero data has only the zero solution, and then every problem has exactly one solution; or it has others, and then a problem has none or infinitely many. **Slider 1**: lambda from 0.2 to 10 with x(0) = 0, x(pi) = 1: the solution sin(sqrt(lambda) t) / sin(sqrt(lambda) pi) grows without bound as lambda approaches 1, 4, 9 (Example 4.1.3: the eigenvalues k^2) and changes sign across them. One sentence: it is the same picture as the response curve of notebook 15, with the eigenvalues in the place of the natural frequency | determinant sin(sqrt(2) pi) = -0.964; at lambda = 0.9 the largest value is 6.2, at 0.99: 63 |
| 6 | **The deflection of a beam** (Session 16, practical exercise) | the model in five lines: y(x) deflection, y' slope, EI y'' bending moment, EI y''' shear force, EI y'''' = w(x) load per unit length; so the general solution is four integrations: a particular solution plus a cubic (the roots are 0, four times: Section 3). Table of supports: simply supported y = y'' = 0; fixed y = y' = 0; free y'' = y''' = 0 (and Kiusalaas's remark: deflection and shear, or slope and moment, cannot both be prescribed at the same end). Kiusalaas, Example 8.4 by hand in dimensionless form, y'''' = xi, y = y'' = 0 at 0 and 1: y = xi^5/120 - xi^3/36 + 7 xi/360; the 4 x 4 system is not singular. Picture with the beam drawn downwards. Uniform load on three kinds of supports, on one plot. **Slider 2**: kind of supports (three positions) and how uneven the load is | slopes 7/360 = 0.019444 and -1/45 = -0.022222, mid-span 5/768 = 0.0065104 (the three numbers printed in the book); largest deflection 0.006522 at xi = 0.519; uniform load: 5/384 (supported), 1/8 (cantilever, at the free end), 1/384 (both ends fixed): 48 : 5 : 1 |
| 7 | **Exercises** | the four steps below | - |
| 8 | **Answers** (collapsed), where to go next (notebook 17: an equation of order n as a system of n equations of order 1; Unit 12 for the eigenvalue problems behind Fourier series) | opens after trying | - |

Length: at most 2100 words before the exercises (kept for notebook 15: 1947). If the draft is
longer, Section 4 is reduced to two lines and its example moves to the exercises. Seven figures;
two sliders, each with its static picture and its question first.

## 2. Exercises (the four steps)

1. **Predict** (no code) - (a) *Lebl, Exercise 2.3.14*: 1001 y''' + 3.2 y'' + pi y' - sqrt(4) y = 0
   with y(0) = y'(0) = y''(0) = 0: what is y(1)? (0.) (b) *Lebl, Exercise 2.3.12*: how many
   constants in the general solution of y^(5) - y^(4) = 0 (5), and how many of its basic solutions
   are polynomials (4)? (c) How many solutions (0, 1, or `np.inf`) have the three problems
   x'' + x = 0, x(0) = 0 with x(pi) = 0; with x(pi) = 1; and x'' + 2x = 0, x(0) = 0, x(pi) = 1?
   (`[np.inf, 0, 1]`.)
2. **By hand, then check** - (a) *Lebl, Exercise 2.3.2*: the four roots, with repetitions, of the
   characteristic equation of y'''' - 5y''' + 6y'' = 0 (0, 0, 2, 3), typed in increasing order.
   (b) *Lebl, Exercise 2.3.13*: a third-order equation whose characteristic roots are 3 and
   +- 2i: its coefficients (1, -3, 4, -12, the answer of the book). (c) *Lebl, Exercise 4.1.8*:
   x'' + x = 0, x(0) = 0, x'(pi) = 1 (x = -sin t, the answer of the book), checked with
   `ol.check_solution` and its two boundary conditions. (d) *Lebl, Exercise 2.1.8*: the solution
   of Chebyshev's equation (1 - x^2) y'' - x y' + y = 0 with y(0) = 1, y'(0) = 0 (sqrt(1 - x^2);
   asking for these data, not for "a second solution", makes the answer unique, so that y = x or
   y = 0 cannot pass).
3. **Fill in** - Complete `midspan_deflection(w0, L, EI)`: the deflection at the middle of a
   simply supported beam with a uniform load (5 w0 L^4 / (384 EI); found in Section 6 in
   dimensionless form, so the student has to put the units back). The next cell compares it with
   a numerical solution of the boundary value problem (`scipy.integrate.solve_bvp`, which does
   not use the formula) for three beams, then applies it to a steel beam with numbers chosen for
   this exercise (L = 6 m, w0 = 10 kN/m, E = 210 GPa, I = 8.356e-5 m^4: 9.6 mm) and asks whether
   it is within L/300.
4. **Explore** - (i) *Lebl, Exercise 4.1.7*: x'' + lambda x = 0, x(0) = 1, x(1) = 1: for which
   lambda is there more than one solution? (lambda = 4 k^2 pi^2.) (ii) The same uniform load on a
   beam fixed at both ends, simply supported, and fixed at one end only: compare the largest
   deflections with slider 2 (1 : 5 : 48). (iii) *Lebl, Exercise 2.3.5*: a fourth-order equation
   that has the solution 2 e^(4x) x cos x (roots 4 +- i twice: (r^2 - 8r + 17)^2). (iv) *Lebl,
   Exercise 4.1.9*: why has a first-order equation with two end conditions no eigenvalues?

Answers (final values only) in a collapsed block; full solutions after the session.

## 3. New in `odelib.py` (with tests)

* `higher_order_solution(t, coefficients, initial)`: the solution of
  a_n y^(n) + ... + a_1 y' + a_0 y = 0 with y(0), y'(0), ..., y^(n-1)(0) given, for any n >= 1,
  computed from the equivalent first-order system with the matrix exponential
  (`scipy.linalg.expm`), so that repeated and complex roots need no special case. The same
  function is the starting point of notebook 17 (an nth-order equation as a system).
* `classify_linear_system(matrix, right_side, rtol)`: "one solution", "no solution" or
  "infinitely many solutions" for M C = b, from the ranks of M and of [M | b] (singular values
  compared with the largest one), and a solution when there is one. It is the theory exercise
  as a tool: the cells of Section 5 build M and b from the boundary conditions and ask it.
* `beam_deflection(x, load, left, right, length=1.0, rigidity=1.0)`: the deflection of a beam
  with EI y'''' = w(x), where `load` is the list of polynomial coefficients of w and `left`,
  `right` are "supported", "fixed" or "free". Exact (a polynomial, found by solving the 4 x 4
  system with NumPy). A combination without a unique solution (free-free, supported-free: the
  beam is not held) raises a ValueError that says so, through `classify_linear_system`.

Tests:
* `tests/test_odelib.py`: Lebl's Example 2.3.3 with its initial data, Examples 2.3.4 and 2.3.5
  against their general solutions with constants fitted, Exercise 2.3.14 (zero); order 2 against
  `constant_coefficient_solution` and order 1 against `exponential_solution`; Kiusalaas's three
  printed numbers; 5/384, 1/8, 1/384 and the symmetry of the symmetric cases; the unsupported
  beams; `classify_linear_system` on Lebl's Examples 4.1.1 and 4.1.2 and on the no-solution
  case, on a nearly singular matrix, on a rectangular one (ValueError); invalid arguments.
* `tests/test_cross_validation.py`: `higher_order_solution` against `solve_ivp` for random
  equations of order 1 to 5 and against SymPy's `dsolve` for order 3; `beam_deflection` against
  `scipy.integrate.solve_bvp` for random polynomial loads on the three kinds of supports.

## 4. Conventions and checks

As for notebooks 14 and 15: same setup cell, `assert` for every reference value, one sentence
under every plot (written after looking at the figure and recomputing every number), line
styles as well as colours, a question before every slider and its answer after, named
constants, plain ASCII, generated by `plans/make_16.py` with `nbbuild.py`.

* Letters: y(x) in Sections 1-4 and 6 (Lebl, Section 2.3); x(t) and lambda in Section 5, as in
  Lebl's Section 4.1, with one sentence at the change; xi for the dimensionless position along
  the beam, as in Kiusalaas; constants C1, C2, ...; the matrix of the conditions M (not A, which
  is a constant of the general solution in Section 5).
* The deflection of a beam is positive downwards and the pictures are drawn that way (the
  vertical axis is reversed), with a sentence that says so.
* Lessons of the reviews of 14 and 15, applied from the start: no name of a global variable that
  an exercise defines as a function; tolerances of `ol.check` chosen by looking at the nearest
  plausible wrong answer; every number in a title is what the title says it is; quotations only
  when literal; tables of rules tested on a case that would break them (here: a root of
  multiplicity 3, a complex pair repeated twice, the supports that do not hold the beam).
* `tests/test_notebooks_run.py`: the notebook as it is; with the correct answers; with wrong
  answers (number of solutions exchanged, a root counted once, x = sin t, y = x, the formula of
  the cantilever); `EXTREMES` for both sliders (lambda exactly 1, 4 and 9, where the title must
  say "no solution" and nothing is divided by zero; lambda at both ends; the three kinds of
  supports with the load at both extremes).
* **The test run is split in two** before this notebook is built: the library tests
  (`test_odelib.py`, `test_check_formula.py`, `test_cross_validation.py`, a few seconds) and the
  notebook tests (about five minutes and growing). After notebook 15 the single run takes almost
  six minutes; the limit of one command here is ten.
* The site is rebuilt and the notebook is run in the browser with `check_site.py`
  (`scipy.linalg.expm` and `solve_bvp` are in the SciPy of the site; the time of the cells is
  measured).
* Index notebook: one more row (16, U7, Session 16, Lebl 2.3 and 4.1); home page: 16 in the
  line of Week 8; notebook 15 ("where to go next") points to 16 by number; notebook 06, Section
  6, gets one line pointing here for repeated roots of higher order.
* After the build: an independent review as for 14 and 15, and its corrections.

## 5. Order of work

1. Split the test run; `odelib.py`: the three functions, tests first; all tests pass.
2. `plans/make_16.py`: sections 1-4; look at the figures.
3. Section 5 with slider 1.
4. Section 6 with slider 2; word count; decide on Section 4.
5. Exercises and answers; `EXERCISES` and `EXTREMES` in the tests.
6. Index, home page, links from 06 and 15; rebuild; `check_site.py`.
7. Independent review; corrections; final run of all tests.

## 6. Decisions for the instructor

1. **Reduction of order** is in the lecture plan of this session but is already done in
   notebook 05 (Session 11), where it follows from Abel's theorem. Proposed: a short section
   here that points back, with Chebyshev's equation as the example and as Exercise 2(d).
2. **The reference for the beam.** Proposed: Kiusalaas, Example 8.4, because its three printed
   numbers can be reproduced exactly; the book is in the repository but is not one of the two
   course texts, so the notebook cites it as further reading. If you prefer a different
   reference (a statics or strength-of-materials text used by the students), tell me which.
3. **Units in the beam.** Proposed: dimensionless in the text (as Kiusalaas), real units only in
   Exercise 3 (a steel beam, with numbers chosen for the exercise and said to be so).
4. **Optional notebook 16E** (proposed: later, with 14E and 15E): the eigenvalue problems of
   Lebl's Section 4.1 in full (the three kinds of conditions, orthogonality), the buckling of a
   column as an eigenvalue problem, and the shooting method.

## 7. Files

```
ode/ode/notebooks/
    16_Higher_Order_BVP.ipynb                  <- to build
    plans/16_Higher_Order_BVP_plan.md          <- this document
    plans/make_16.py                           <- builds the notebook
    odelib.py                                  <- + higher_order_solution(), classify_linear_system(), beam_deflection()
    tests/test_odelib.py                       <- + their tests against the books
    tests/test_cross_validation.py             <- + against solve_ivp, dsolve and solve_bvp
    tests/test_notebooks_run.py                <- + the exercises and sliders of notebook 16
    00_Index.ipynb, plans/make_06.py, make_15.py   <- index row; one line each
ode/ode/site/build_site.py                     <- READING_ORDER: 16 in Week 8
```
