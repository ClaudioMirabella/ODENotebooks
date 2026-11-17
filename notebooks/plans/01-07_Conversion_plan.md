# Notebooks 01-07: conversion to the template of 08-13 - plan

MATH 331 - Differential Equations - Spring 2027 - stage 1, item 5 of the review of 2026-11-17

**Why.** Notebooks 01-07 were written first, as complete treatments for a reader who already knows
the subject. Their mathematics is right (independent review, section 7 of the review document),
but a first-year student meets them differently from 08-13: long (up to 3200 words), with proofs
in the text and in the exercises, no checks of the student's own answers, no sentence under the
plots, curves told apart by colour, sliders without a question, and their own notation.

**Aim.** Every numbered notebook reads the same way: theory in brief, by hand and then SymPy,
pictures with a sentence, sliders with a question, exercises in four steps with checks and
collapsed answers. Nothing that is in 01-07 today is thrown away: what is beyond a first reading
moves to optional "going further" notebooks.

---

## 1. The idea: a core notebook and, where needed, a "going further" notebook

| Today | Core notebook (same number, same file name) | What moves out | Where it goes |
|---|---|---|---|
| 01 Exponential growth (1460 words) | the model, the solution, Lebl Example 0.2.1, slope field, doubling time, limits of the model | the Picard argument (one line and a pointer to notebook 11); the error study of Euler's method (it is in 04 and 13) | nothing left over: 01 is only slimmed |
| 02 Radioactive decay (1160) | the decay law, half-life and mean life, radiocarbon dating (Adkins and Davidson, Example 8) | Section 4: the decay chain (integrating factor, Bateman's solution, the equal-rates limit, transient equilibrium) | **02E Decay chains** |
| 03 Logistic equation (1560) | the model, equilibria and the phase line, the exact solution for a positive start (partial fractions), harvesting with Lebl's numbers | the four cases with finite-time blow-up; Euler's false oscillations and the logistic map; the bifurcation in detail | **03E The logistic equation, further** |
| 04 Euler's method (1220) | the method, Lebl's Table 1.7.1 by hand and by computer, the error halves, a first look at Heun and RK4, Table 1.7.2, `solve_ivp` in practice | the error bound with its proof; the region of absolute stability; stiffness | **notebook 22** (Numerical methods, part 2, Session 29, phase E): it is exactly that content. Until then these sections stay at the end of 04 under "Going further" |
| 05 Wronskian and Abel (3230) | the setting, superposition, "do two solutions give all?", the Wronskian as a 2 x 2 determinant computed by hand, Abel's theorem (statement and one check), reduction of order with Lebl's example | the phase-plane area, Vandermonde, the counterexample t^3 and abs(t)^3, Airy and Bessel, discontinuous p, Sturm's theorem, the n-th order case, the proofs | **05E The Wronskian, further** |
| 06 Characteristic equation (2050) | the key identity, the three cases with Lebl's Examples 2.2.1-2.2.4, the slider from roots to solutions, higher order (Example 2.3.3) | the factorisation of the operator; the consistency table with Abel; the map of the (b, c) plane | the first two to **05E**, the third to **07E** (it is the root locus seen another way) |
| 07 Mass-spring-damper (2120) | the model, the three regimes, amplitude and phase (Lebl 2.4.1), damped motion (2.4.2), the slider for the damping, the RLC circuit as the same equation (link to notebook 13) | the root locus; the energy balance; phase-plane areas and Liouville; the RLC circuit solved two ways | **07E Vibrations, further** |

So: seven core notebooks and four optional ones (02E, 03E, 05E, 07E). The optional ones keep the
present text almost as it is, with the corrections of the review, a sentence under every plot,
line styles, a question before every slider and plain ASCII; their exercises stay ordinary
problems (several are proofs), without the four steps.

Target for a core notebook, from 08-13 as measured: 1500-2100 words before the exercises (I had
written 1000-1400 here: that was too low), 15-22 code cells, 4-7 plots, one or two sliders, four
exercises. Notebook 01, converted, has about 1200 words before the exercises.

## 2. What every core notebook gets

1. **Theory in brief**, one screen, citing Lebl by section, with the correct credits (the
   Wronskian is from Adkins and Davidson: Lebl does not use it).
2. **By hand first, then SymPy**: each worked example of the book shown in steps, then checked
   (`residual`, `assert` on the book's value).
3. **A sentence under every plot**; curves told apart by line style or marker as well as colour
   (notebook 07 uses red and green today); the legend names every curve.
4. **A question before every slider**, and its answer after it.
5. **Exercises in four steps** with `ol.check` and `ol.check_formula`, final answers collapsed.
   The proofs among today's exercises move to the "going further" notebooks.
6. **Notation of the conventions** in the index (section 4 below).
7. **Built from a script** (`plans/make_NN.py`, with `nbbuild.py`), like 08-13: plain ASCII,
   fixed cell numbers, executed, reviewable as text.
8. **Tests**: untouched, with right answers, with wrong answers, and every slider function at
   both ends of every slider.
9. **Independent mathematical review** before it is called done, and the automatic browser check.

## 3. The exercises (the four steps), notebook by notebook

Sources are today's exercises and the book items already cited; values are those verified in the
review.

| Nb | Predict | By hand, then check | Fill in | Explore |
|---|---|---|---|---|
| 01 | a population grows 3 % a year: about how long to double? (the rule of 70; state whether 3 % means k = 0.03) | Lebl Exercise 0.2.10: the city, 144 thousand; the time of the maximum of P' = (2 - 0.1 t) P (t = 20) | `growth_rate(t1, P1, t2, P2)` from two measurements, checked on Lebl Example 0.2.1 (k = ln 2 / 10) | how much does P(60) change if k is 1 % off? |
| 02 | what fraction is left after three half-lives? after one mean life? | Adkins and Davidson Example 8 redone for one third (their exercise) and for 10 % | `age(fraction, half_life)`, checked on 2378 years | the uncertainty of the age for an uncertainty of 0.005 in the fraction, at 75 % and at 10 % |
| 03 | from the phase line: where does x go from x(0) = 1, 6, -1, with M = 5? | Lebl's harvesting example (M = 8, k = 0.1): the equilibria for h = 1 (1.55 and 6.45) and the critical harvest (1.6) | the right-hand side with harvesting; the cell finds the equilibria and classifies them | proportional harvesting (Lebl Exercise 1.6.4) |
| 04 | Euler with h = 1 on y' = y, y(0) = 1: y(3)? above or below e^3? | the first row of Lebl's Table 1.7.1 by hand (y' = y^2/3, y(0) = 1, h = 1: y(2) = 1.92593, error 1.07407), then the error ratio | one Euler step; the cell rebuilds the table and compares with the book | Lebl Exercise 1.7.4: the two thresholds of h for y' = -5y |
| 05 | are e^x and e^(2x) independent? x and 2x? (no computation) | W for x^2 and 1/x (Lebl 2.1: -3); the constants for y(1) = 2, y'(1) = 1 | `wronskian(y1, y2)` with SymPy, checked on three pairs | Abel's formula for x y'' + 2y' + x y = 0 and the pair cos x / x, sin x / x |
| 06 | for six equations: do the solutions oscillate? decay? (from b and c only) | Adkins and Davidson 3.3, Exercises 14 and 16 (with the right sign); Lebl 2.2 examples | `solve_constant(a, b, c, y0, v0)` for distinct roots, checked against `ol.constant_coefficient_solution` | for which c does y'' + 2y' + c y = 0 stop oscillating? |
| 07 | m = 2, k = 8: the frequency? what changes if the mass is four times larger? | Lebl Example 2.4.1 (amplitude 0.707, phase 0.785); Adkins and Davidson 3.6 Example 2 (R = 0.902, phase 0.983) | `regime(m, c, k)`, checked on four cases | the time for the amplitude to fall to 1 % (18.4 s in today's Exercise 3) |

The exact data of each exercise are fixed, and checked against the book, when the notebook is
built; a short note in the report of each notebook will list any change from this table.

## 4. Notation (to be stated once in the index, and used everywhere)

The review found the same letter with two meanings inside 01-07. Proposals:

| Clash today | Proposal |
|---|---|
| h: Euler's step, and the harvest rate, both in notebook 03 | keep Lebl's h for both, but never in the same notebook: the Euler part of 03 moves to 03E, where the harvest does not appear |
| lambda: decay constant (02), test equation y' = lambda y (04) | decay constant only; the test equation is written y' = -a y in 04 (and lambda returns in notebook 22, with a note) |
| p: coefficient of y' (05, 06), and c/(2m) (07) | both are Lebl's; in 07 the coefficient of x' is always written c/m, and p means c/(2m) only |
| phase: delta (06), gamma (07) | gamma, as in Lebl |
| amplitude: R (06, 07), which is also the resistance in 07 | **A**; Lebl writes C, which is also the capacitance. Your choice (decision 2) |
| independent variable: x or t for the same equation | as in the conventions: y(x) for an equation studied for itself, t for a model; an item quoted from Adkins and Davidson is rewritten in the notebook's letter, and says so |
| initial data: (b0, b1), (y0, v0), (x0, v0) | (y0, v0), or (x0, v0) for a position |
| exercise numbers of Lebl | end-of-chapter lists, "in the text" said when it is so (as in 08-13) |

## 5. Other changes that come with the conversion

* **Site.** The allow-list accepts the "going further" notebooks (`NNE_*.ipynb`); the home page
  lists the notebooks **by week of the course** instead of by number (the plan of the site always
  said so), with the optional ones marked. Today a student sees 08 first and then 00-13 in
  numerical order, which is not the order of the lectures (08, 09, 10, 11, 12, then 01, 02, 03,
  13, 04, then 05, 06, 07).
* **Numbers and file names of 01-07 do not change**: the videos, the slides of Lecture 1 and the
  READMEs refer to them. Two references move to the new files: the decay chain cited by the
  Lecture 1 appendix and by video R11 becomes "notebook 02E".
* **`odelib.py`**: the last small points of the review (non-finite arguments refused everywhere;
  a repeated root detected with a relative tolerance when b = 0; 12 non-ASCII characters in the
  docstrings; the semi-stable dot of the phase line drawn as a half-filled circle).
* **Episodes 1-3** (videos) quote notebooks 03 and 04 in two captions: checked after the
  conversion, changed only if a section moved.

## 6. Order of work

One notebook at a time, in the order the students meet them. After each: tests, browser check,
a short report; you look at it when you like, the work does not wait unless a decision is needed.

| Step | Notebooks | Needed by |
|---|---|---|
| 1 | **01** (the pattern: smallest change; it also fixes how an old notebook is ported to a script) | Week 2 |
| 2 | **02** and **02E** | Weeks 2-4 |
| 3 | **03** and **03E** | Week 4 |
| 4 | **04** | Week 4 |
| 5 | independent review of 01-04, 02E, 03E; site by week; rebuild and browser check | before Midterm 1 |
| 6 | **05** and **05E** | Weeks 5-6 |
| 7 | **06** | Week 6 |
| 8 | **07** and **07E** | Week 7 |
| 9 | independent review of 05-07, 05E, 07E; final rebuild and browser check | before Week 6 |

Size: steps 1-4 are about as much work as notebooks 08-10 were; steps 6-8 as 11-13, plus the
moves. The optional notebooks are mostly moves, not new writing.

## 7. Decisions for you

1. **The split** of section 1: seven core notebooks and four optional ones, with the advanced
   part of 04 waiting for notebook 22. (Alternative: no optional notebooks, the extra sections
   stay at the end of each core notebook under "Going further"; simpler, but the core notebooks
   stay long and the browser loads everything.)
2. **The letter for the amplitude**: A (proposed), or Lebl's C, or today's R.
3. **The home page by week**: yes (proposed) or keep the numerical order.
4. **The proofs** in today's exercises: moved to the optional notebooks (proposed), or kept in
   the core as a fifth, optional exercise.

## 8. Files

```
ode/ode/notebooks/
    01_... to 07_...ipynb                       rebuilt from scripts (same names)
    02E_Decay_Chains.ipynb, 03E_Logistic_Further.ipynb,
    05E_Wronskian_Further.ipynb, 07E_Vibrations_Further.ipynb        new, optional
    plans/make_01.py ... make_07.py, make_02E.py ...                 new
    plans/01-07_Conversion_plan.md                                   this document
    odelib.py, tests/                                                small changes, new tests
    00_Index.ipynb                                                   table, conventions, reading order
ode/ode/site/build_site.py, landing.html, check_site.py              allow-list, home page by week
```

## 9. Progress

| Step | State (2026-11-17) |
|---|---|
| 1. Notebook 01 | **Done.** Built from `make_01.py`; about 1200 words before the exercises; 2 static plots, 1 slider group; four exercises with checks. The two proof exercises of the old version are not in it: the limit of Euler's method goes to notebook 04, and "two solutions cannot cross" is Exercise 1 of notebook 11. |
| 2. Notebooks 02 and 02E | **Done.** 02: decay law, half-life, mean lifetime (the time constant of notebook 13), radiocarbon dating. 02E: the decay chain, and the two lakes of Lebl's Exercise 1.4.6 as the same equations. The half-life is now written T_1/2 and the mean lifetime tau. |
| 3. Notebooks 03 and 03E | **Done.** 03: equilibria and phase line, the exact solution for a positive start, harvesting with Lebl's numbers. 03E: negative starts and blow-up, the Bernoulli route, Euler's false oscillations (the step is written Delta t there, because h is the harvest), the bifurcation diagram. |
| 4. Notebook 04 | **Done.** Euler's method by hand and by computer on Lebl's example, both tables of Section 1.7, Heun and RK4 with Lebl's Exercise 1.7.9, the step that is too large (y' = -5y), the three methods on the models of 01-03, `solve_ivp`. The error bound with its proof, the stability regions and stiffness are at the end under "Going further", until notebook 22. The test equation is y' = -5y in the core; lambda appears only in "Going further". |
| 5. Review of 01-04 and the optional ones; home page by week | Home page by week: **done** (`READING_ORDER` in `site/build_site.py`). References to 02E: **done** (Lecture 1 deck, appendix and handout rebuilt; video R11 script, plan and draft). Independent review: see section 10. |
| 6-9. Notebooks 05, 05E, 06, 07, 07E; second review | to do. **Session numbers to correct** while converting: in the lecture plan Session 9 is the review and Session 10 the first midterm, so 05 belongs to Session 11 (not 9-11), 06 to Sessions 11-12, 07 to Session 14 (free and damped vibrations), not 13. |

Tests after step 4: 201 pass (each converted notebook untouched, with right answers, with wrong
answers, and every slider function at the ends of its sliders). Browser check: all 15 notebooks with code run on the built site without errors, 12-14 s each.

## 10. Independent review of the converted notebooks (01-04, 02E, 03E)

A reviewer re-derived every formula and number of the six sources, ran every cell with its own
methods, and compared every citation with the books. **No serious error; all formulas and almost
all numbers right.** Corrected after the review:

* **03**: the caption said all solutions below 5 are S-shaped; only those that start below 2.5 are.
* **04**: the caption gave Euler's and RK4's smallest errors too roughly; the plot now goes on to
  the rounding plateau, which the text had asserted without showing. The slider of Section 7 now
  reaches 300 steps (the answer, about 280, was beyond its old maximum of 200).
* **Citations**: three items are now marked "adapted from" (01 Exercise 2b, 04 Exercise 2c: the
  equation is Lebl's, the question is ours); the header of 04 named the wrong exercises; in 02 the
  5568-year half-life was credited to Adkins and Davidson, who do not mention it; 02 now warns
  that their book writes tau for the half-life.
* **Checks made stricter or fairer**: 0.019 is no longer accepted for 0.018 (01); a half-life
  computed with lambda rounded to 0.022 is accepted (02); the check cell of 04 no longer prints a
  number that it would accept as the answer; the three fill-in exercises each got one more test,
  so that a wrong function (absolute values in 01, a hard-coded half-life in 02, a slope taken at
  the wrong point in 04) no longer passes.
* **Smaller**: titles at the exact thresholds (03E, 04); the slider of 01 opens on the same k as
  the static picture; a name used twice in 02 (`age`); wording in 01, 02E, 03 and 03E.

Left as they are (noted, not changed): a few letters with two meanings across notebooks (k_1, k_2
for slopes in 04; lambda positive in 02 and negative in the optional part of 04); the stiff example
of 04 written in t.

## 11. Progress, second half

| Step | State (2026-11-17) |
|---|---|
| 6. Notebooks 05 and 05E | **Done.** 05 (Session 11): superposition on Lebl's example, why two solutions are enough (initial conditions as a 2 x 2 system), the Wronskian by hand, Abel's theorem with its three-line proof and a numerical check with a slider, reduction of order. 05E: the Wronskian as an area, Vandermonde, the counterexample, Airy and Bessel functions, three solutions are dependent, Legendre's second solution, Sturm's theorem, systems and order n. Credits corrected: the Wronskian is from Adkins and Davidson; Lebl tests independence by constant multiples. |
| 7. Notebook 06 | **Done.** 06 (Sessions 11-12): the three cases, each by hand first with Lebl's examples (the opening example, 2.2.2, 2.2.4) and Adkins and Davidson's Example 7; where x e^(rx) comes from (slider: two roots coming together); Euler's formula and the envelope; the roots in the complex plane and the shape of the solution (slider); higher order (Lebl 2.3.3). The factorisation of the operator and the agreement with Abel's theorem are now Sections 7 and 8 of 05E. The map of the (b, c) plane waits for 07E. The amplitude is written A and the phase gamma. |
| 8. Notebooks 07 and 07E | **Done.** 07 (Session 14): the model (Newton, Hooke, damper; vertical spring; units; the pendulum), free motion without damping with Lebl's Example 2.4.1 (amplitude and phase), the kind of motion decided before solving (Adkins and Davidson, Section 3.6, Examples 2 and 3), the three kinds side by side with Lebl's Example 2.4.2 checked, sliders for m, c, k and the initial data, the circuit as the same equation (Table 3.4), four exercises. 07E: how the roots move and the decay rate, the energy balance, the cloud of states and Abel's theorem, charge, current and heat in the circuit, the map of the (b, c) plane, six problems. The velocity is now computed exactly (it solves the same equation), not by finite differences. |
| 9. Second review | **Done.** An independent reviewer recomputed the examples, answers and captions of 05, 05E, 06, 07 and 07E and checked the citations in Lebl and in Adkins and Davidson: no wrong formula or number. Corrected after it: the proof that three solutions are dependent (05E, now by uniqueness, not by W = 0); Adkins-Davidson 4.2 Theorem 6 is for constant coefficients; the missing answer 5 of 05E; the two meanings of the letter p (Abel's p(x) is c/m = 2p); the quadrant of the phase in 06 and 07; the 1 % question of 07 needed a longer time axis (plot_response has t_end); tolerance of amplitude and phase; small wording points. Not checked: the equation numbers cited from Arfken, Weber and Harris (the book is not in the repository as text). |

Tests after step 7: 209 pass. After steps 8 and 9: 213 pass.
