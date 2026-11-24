# Notebook 21 - Series solutions: plan

MATH 331 - Differential Equations - Spring 2027 - phase E of the notebooks (19-22), the third notebook of
the phase

**Status (2026-11-25): built, reviewed, corrected, re-checked; exported to the public folder on 2026-11-25 at the instructor's request (the instructor commits and pushes)** (the eight decisions of Section 6 approved by the instructor as proposed; Section 8: as built, the corrections after the reviews, and the final pass after the re-check).

**Sessions 27 and 28, Units 12 and 13** (lecture plan, Session 27: "Power series refresher; series solutions
at ordinary points; recurrence relations (Airy, Hermite)"; theory exercise "Bound the radius of convergence
from the singularities"; practical exercise "Compute the first coefficients numerically". Session 28:
"Regular singular points; Euler equations; Frobenius method; Bessel functions"; theory exercise "Discuss the
indicial equation and its three root cases"; practical exercise "Compare a series solution with the scipy
Bessel functions"). About **15 minutes in class in each session** (Session 27: Sections 2 to 4; Session 28:
Sections 5 to 7), the rest as homework. Notebook 20 ends with "Where to go next: notebook 21": this notebook
keeps that promise.

The instructor's requests (2026-11-24, standing since notebook 20): caution with the existing code, step by
step; the mathematics double-checked; the new concepts understandable for inexperienced students; and, new,
**the finished notebook put on the website** so that the instructor can review it there (Section 5, step 9).

**References** (found in the text of the book on 2026-11-24; the numbers of this plan recomputed with SymPy
and SciPy the same day).
* Lebl, **Section 7.1, Power series**: the definition; Theorem 7.1.1 (radius of convergence) and Theorem 7.1.2
  (ratio and root tests); Examples 7.1.1 to 7.1.5 (the exponential, the geometric series, a series centered at
  1, manipulating series, a rational function whose radius is the distance to its complex roots).
* Lebl, **Section 7.2, Series solutions of linear second order ODEs**: ordinary and singular points;
  Example 7.2.1 (y'' - y = 0, the recurrence a_(k+2) = a_k/((k + 2)(k + 1)), cosh and sinh); Example 7.2.2
  (Airy's equation y'' - x y = 0, a_2 = 0, a_(k+3) = a_k/((k + 3)(k + 2))); Example 7.2.3 (Hermite's
  equation y'' - 2x y' + 2n y = 0, a polynomial solution when n is a whole number).
* Lebl, **Section 7.3, Singular points and the method of Frobenius**: Examples 7.3.1 (2x y' - y = 0, y = C
  x^(1/2)) and 7.3.2; regular singular points and Example 7.3.3; Theorem 7.3.1 (method of Frobenius) with
  its cases i to vi; Section 7.3.3, Bessel's equation of order p, the indicial equation (r - p)(r + p) = 0,
  J_p and Y_n; Example 7.3.4.
* Lebl, **Exercises 7.E** for 7.1, 7.2, 7.3 (answers printed for some; each one used is checked).
* Notebooks 05E (Airy and Bessel functions met through Abel's theorem, with `scipy.special`), 06 (the
  characteristic equation: Euler equations reuse its three cases), 04 and 14 (numerical checks).

**Not in Lebl, stated as a fact and sourced** (Decision 7): the radius of convergence of a series solution at
an ordinary point is at least the distance to the nearest singular point, complex ones included (Boyce and
DiPrima, Theorem 5.3.1; Lebl shows the same idea for rational functions in Example 7.1.5).

**Purpose.** One idea per session.
1. **Session 27: when no formula is known, the solution can still be written as a power series whose
   coefficients follow from a recurrence; the computer gives as many as wanted; the series is good inside a
   radius that the equation itself announces.**
2. **Session 28: at a regular singular point a power series alone is not enough; a power x^r in front fixes
   it, and r comes from a quadratic, the indicial equation, whose three kinds of roots are the three cases
   of the characteristic equation again.**

---

## 1. Contents

1. **Before you start**: what is needed (Taylor series from calculus, the sigma notation, the characteristic
   equation of notebook 06); the conventions box (Section 4).
2. **Power series, refresher** (Session 27).
   * Partial sums: e^x and 1/(1 - x), a table of partial sums at x = 0.5 and x = 1.5.
   * The radius of convergence by the ratio test, numerically: the ratios |a_k / a_(k+1)| printed for growing
     k approach the radius.
   * **Slider 1** (question first: "the series of 1/(1 + x^2) uses only real numbers; where on the real line
     does it stop working, and why there?"): the number of terms N; the partial sums against 1/(1 + x^2) on
     [-2, 2]. The answer: at |x| = 1, the distance to the complex roots +-i (Lebl's Example 7.1.5).
3. **Series at an ordinary point** (Session 27).
   * The method in four lines, on Lebl's Example 7.2.1 (y'' - y = 0): try the series, differentiate, shift
     the index, equate coefficients; the recurrence; cosh and sinh recognized.
   * `ol.series_coefficients(recurrence, initial, n)`: the student writes the recurrence as a function of k
     and the earlier coefficients; the first coefficients by hand checked against the computer's.
   * Airy's equation (Lebl's Example 7.2.2): the two series y_1, y_2; their combination with SciPy's
     constants Ai(0) and Ai'(0) gives `scipy.special.airy` (Ai at x = 1, 2, -3 agrees to 1e-12 with 40
     terms). Figure: the two series solutions and Ai, Bi.
   * Hermite's equation with n = 3 (Lebl's Example 7.2.3): the odd series stops, x - (2/3) x^3.
4. **How far a series solution reaches: the theory exercise** (Session 27).
   * The fact, with its source, and how to use it: find the zeros of p(x), complex ones too; the distance
     from x_0 to the nearest one is a lower bound for the radius.
   * The notebook's own example: (1 + x^2) y'' + 2x y' = 0, whose solution through y(0) = 0, y'(0) = 1 is
     arctan x; the singular points +-i; the bound 1; the series x - x^3/3 + x^5/5 - ... diverges for
     |x| > 1 although arctan x is fine on the whole line. Figure with partial sums.
   * The ratio test on computed coefficients agrees.
5. **Singular points and the method of Frobenius** (Session 28).
   * Why a power series can fail: Lebl's Example 7.3.1 (2x y' - y = 0: only y = 0 as a power series, y =
     x^(1/2) with a power in front).
   * Regular singular points: the two limits, on Lebl's Example 7.3.3 and its "slight change".
   * The Euler equation x^2 y'' + a x y' + b y = 0 as the simplest case: y = x^r gives r(r - 1) + a r + b = 0,
     the same three cases as notebook 06.
   * The indicial equation r(r - 1) + p_0 r + q_0 = 0 with p_0, q_0 the two limits; `ol.indicial_roots(p0,
     q0)` gives the roots and the case: (i) two real roots not differing by an integer; (ii) a double root;
     (iii) two real roots differing by a whole number; (iv) complex roots. What each case means for the second
     solution (Lebl's list, Theorem 7.3.1), in words and a table.
   * Lebl's Example 7.3.2, the first coefficients by `ol.series_coefficients` with r inside the recurrence.
6. **Bessel functions** (Session 28).
   * Bessel's equation of order p; the indicial roots +-p; the series of J_0 by its recurrence, against
     `scipy.special.jv(0, x)`.
   * **Slider 2** (question first: "how many terms are needed for J_0 to be right up to x = 10?"): the number
     of terms N; the partial sum against `jv(0, x)` on [0, 20], with the error printed. The answer: about 20
     terms to 1e-8 at x = 10 (10 terms give -6.2 there instead of -0.246); an infinite radius does not mean
     few terms.
   * Lebl's Example 7.3.4 (an equation reduced to Bessel's) in one cell; Y_n named, with the warning that it
     is infinite at 0.
7. **Where to go next**: notebook 22 (numerical methods, part 2); what was left out (the logarithmic second
   solutions written in full, series at infinity).
8. **Exercises**, **answers**.

Target length: about 3000 words before the exercises (notebook 20 ended at about 3430 after its reviews).

---

## 2. Exercises (the four steps)

Each exercise is marked with its session.

1. **Predict** (no code). (a) The radius of convergence of the series of 1/(4 + x^2) at 0. (b) Is x = 0 an
   ordinary point of (1 - x) y'' + y = 0? Where is the nearest singular point? (c) After Session 28: the
   indicial roots of an Euler equation given by its coefficients. (d) After Session 28: which case of
   Frobenius for Bessel's equation of order 1/2, 1, and 0?
2. **Compute**. (a) The first five coefficients of the series solution of y'' + x y' + y = 0 with y(0) = 1,
   y'(0) = 0, by hand then with `ol.series_coefficients`; recognize e^(-x^2/2). (b) The radius bound for an
   equation with p(x) = x^2 - 2x + 5. (c) After Session 28: the indicial equation of 2x^2 y'' + 3x y' -
   (1 + x) y = 0 (roots 1/2 and -1, the case of a difference that is not a whole number). (d) After Session
   28: J_1 by its series against `jv(1, x)`.
3. **Explore** (code). (i) Hermite with n = 4 (the even series stops). (ii) How the error of the arctan series
   at x = 0.9 and x = 1.1 behaves as N grows. (iii) After Session 28: the case of a double root on an Euler
   equation, and the second solution x^r ln x checked by substitution. (iv) After Session 28: the zeros of J_0
   from the series against `scipy.special.jn_zeros`.
4. **Explain** (two or three sentences). Why can a series solution stop working at |x| = 1 when the equation's
   coefficients are perfectly smooth on the real line?

Answers are checked with the existing `ol.check`, `ol.check_formula`, `ol.check_vector` and `ol.check_choice`.

---

## 3. New in `odelib.py` (with tests)

A new section, "Series solutions (notebook 21)", after the section of notebook 20. NumPy only at the top.
Type hints, docstrings citing Lebl, the notebook, and what each function does not promise.

* `series_coefficients(recurrence, initial, n) -> np.ndarray`: the first n coefficients a_0, ..., a_(n-1),
  where `initial` gives the first ones and `recurrence(k, a)` returns a_k from the list a of the earlier
  ones (the student writes it as in the notebook). ValueError, with one sentence, when n is not a whole
  number at least len(initial), when the recurrence returns something that is not one finite real number, or
  raises an error (the sentence names k).
* `series_value(coefficients, x, x0=0.0) -> np.ndarray`: the partial sum at x (vectorised), by Horner's rule.
  ValueError for coefficients that are not finite real numbers.
* `indicial_roots(p0, q0) -> tuple`: the two roots of r(r - 1) + p0 r + q0 = 0 and the case, one of four named
  constants (the three cases of Lebl's Theorem 7.3.1 plus complex roots), with a tolerance named once for
  "whole-number difference" and "double root". ValueError for inputs that are not finite real numbers.
* Named constants for the cases and the tolerance; no other public name.

**Tests**: against the book (Examples 7.2.1, 7.2.2, 7.2.3, 7.3.2; Bessel's recurrence), against independent
methods (`scipy.special.airy`, `jv`, SymPy's `series` and `dsolve` with `hint="2nd_power_series_ordinary"`
where it applies, exact cosh, sinh, e^(-x^2/2)), invalid input with `match`, seeded cases (SEED = 331), and the
four cases of `indicial_roots` at and near their borders.

---

## 4. Conventions, checks, and care with the existing code

As for notebooks 19 and 20: `assert` for every reference value against an independent method; one sentence
under every plot, written after looking at the figure; a question before every slider and its honest answer
after, with nothing above the slider that gives it away; cells where a student is invited to change a number
do not stop on a bare AssertionError; named constants; plain ASCII; generated by `plans/make_21.py` with
`nbbuild.py`; the setup cell of notebook 20 with `import sys`; `SETUP_NAMES` unchanged.

* **Letters**: x the independent variable (as Lebl in Chapter 7; the time t of Units 10 and 11 is not used);
  a_k the coefficients, k the index; x_0 the center; rho the radius of convergence; r the exponent of
  Frobenius (not a root of the characteristic equation, although it plays the same part: said once); p_0,
  q_0 the two limits. Rows for STYLE_SHEET Section 4 with the deck of Session 27.
* **Hard points for beginners**, named now so that the reviews test them: the index shift in a sum; reading a
  recurrence; "converges" against "agrees with the function"; a radius set by complex points; why x^r in
  front; the indicial equation as "the lowest power must vanish"; the three cases.
* **Care with the existing code** (as for notebook 20): baseline first; copies of `odelib.py` and the three test
  files before the work; strictly additive, proved mechanically (no existing line edited or deleted, apart
  from the list of names in the module docstring); every pre-existing function compared old against new; the
  four registries extended, never reordered; all gates again after the work; the public folder touched only
  by `site/export_public.py --update`.

---

## 5. Order of work

1. Baseline: library tests (two environments), notebook tests, `--collect-only`, browser run.
2. `odelib.py`: the three functions and their constants; tests first; ruff; the whole suite unchanged.
3. `plans/make_21.py`: Sections 1 to 4 with slider 1 (Session 27); every figure looked at.
4. Sections 5 to 7 with slider 2 (Session 28).
5. Exercises and answers; `EXERCISES`, `EXTREMES`, `NAME_21`.
6. Index row, `READING_ORDER` (Week 14, Sessions 27-28), READMEs; site build and its tests; `check_site.py`.
7. Independent reviews: mathematics, weakest student, code (with the additivity check); corrections; fresh
   re-check; final pass; all gates again.
8. Trial export into a scratch folder.
9. **The real export into the public folder** (the instructor's request of 2026-11-24): the instructor then
   commits and pushes, and the GitHub workflow publishes the site, where the instructor reviews the notebook.
   Until the push, the site can also be seen locally with `site/serve_site.py`.

---

## 6. Decisions for the instructor

1. **One notebook for two sessions** (as 17 to 20): Sections 2 to 4 for Session 27, 5 to 7 for Session 28.
2. **Three library functions**: `series_coefficients` (the student writes the recurrence), `series_value`,
   `indicial_roots`, as the master plan listed.
3. **The theory exercise of Session 27 on the notebook's own example**, (1 + x^2) y'' + 2x y' = 0 (arctan x),
   where the bound from the singular points +-i is exactly the radius.
4. **Logarithmic second solutions** (cases ii and iii of Frobenius) shown in words and checked once on an Euler
   equation (x^r ln x), not computed in general.
5. **Bessel functions through J_0 and J_1 only**, with Y_n named; the gamma function not introduced.
6. **The letter x** as the independent variable throughout, as Lebl in Chapter 7.
7. **The radius theorem stated with its source** (Boyce and DiPrima, Theorem 5.3.1), since Lebl states it only
   for rational functions.
8. **Export to the public folder when the notebook is finished**, as you asked, after the reviews and all gates;
   you commit and push.

---

## 7. Files

```
ode/ode/notebooks/
    21_Series_Solutions.ipynb                  <- to build
    plans/21_Series_Solutions_plan.md          <- this document
    plans/make_21.py                           <- builds the notebook
    odelib.py                                  <- + series_coefficients(), series_value(), indicial_roots()
    tests/test_odelib.py, test_cross_validation.py, test_notebooks_run.py
    00_Index.ipynb, plans/make_00.py, README.md
ode/ode/site/build_site.py, site/README.md, ode/ode/README.md
```

---

## 8. As built (2026-11-24)

**The notebook.** `21_Series_Solutions.ipynb`, 50 cells (25 code), built by `plans/make_21.py`; 431 075
bytes (the limit of `export_public.py` is 2 000 000), plain ASCII, no local path stored. Before the
exercises: 4362 whitespace tokens, 3307 with each formula as one word, **2963** on the measure of plan 17
(the target was about 3000; notebook 20 has 3428). The setup cell is that of notebook 20 (with `import sys`);
the cell after it holds SymPy's `x`, `airy`, `jv`, `yv` from `scipy.special`, and two helpers of the checks
(`larger_first`, `close_to`).

**Sections.** 1 Before you start (what is needed; conventions: x, a_k and k, x_0, "a term" is a coefficient
that is not 0, rho, P y'' + Q y' + R y with capital letters because r is the exponent); 2 Power series (the
table of partial sums of e^x and 1/(1 - x) at 0.5 and 1.5; "converges" against "agrees with the function";
the radius and the ratio test with a printed table; slider 1); 3 Series at an ordinary point (the four steps
on Example 7.2.1 with the shift of the index in a line and a table of its own, the recurrence read by hand
with a table, the same rule written for a_k with its own line, `ol.series_coefficients` against the table
by hand; Airy with the shift of x y written out, a_2 = 0, three initial values, the coefficients by hand,
Ai and Bi from the two series against `scipy.special.airy`, figure; Hermite with n = 3, checked by SymPy);
4 How far a series solution reaches (the fact with its source, three steps to use it, the arctan example
with every line of its recurrence, the bound 1, figure of the complex plane and of the partial sums, the
ratio test); 5 Singular points and Frobenius (Example 7.3.1, why x^r with a table of slopes of sqrt x,
regular singular points with Example 7.3.3 and its slight change computed by `sp.limit`, the Euler
equation, the indicial equation as "the lowest power must vanish", `ol.indicial_roots` on four equations
(one per case) and a cell for the student's own p0, q0, the table of the four cases, Example 7.3.2 with
a table of where each power comes from and r inside the rule); 6 Bessel functions (the indicial roots +-p,
the recurrence of J_0 with its shift, the first coefficients by hand, against `jv`; slider 2; Example 7.3.4
checked by SymPy, Y_0 with values near 0 and a figure); 7 Where to go next; 8 Exercises; 9 Answers.

**Figures: 5 stored** (Airy: the two series, and Ai, Bi against the series; the complex plane and the
partial sums of arctan; J_0 and Y_0; the static frames of the two sliders). Every figure was exported to
PNG (`nb21_build/figs`) and looked at, and both slider functions were drawn at N = 1, 2, 10, 30 and
N = 1, 5, 15, 19, 20, 30 (`nb21_build/sliders`). Changed after the first look: the legend of the complex
plane (it hid -i; now labels), the label of the interval (it lay on the circle), "1 terms", the legends of
both sliders, the caption of slider 1 (-0.7 to 0.7, not -0.8 to 0.8), the caption of the arctan figure
("the more terms, the more steeply", not "the sooner"), the caption under the coefficients of cosh
(5e-13 at x = 0.5, not 1e-14). **Sliders: 2** (`plot_bump(n)`, `plot_bessel(n)`), each with its question
first, a static figure, its answer, and a checking cell after the answer that asserts the answer at every
position of the slider (slider 1: the error is exactly (-x^2)^N / (1 + x^2); slider 2: 20 terms are within
1e-8 on [0, 10] and 19 are not; 10 terms give -6.2176 at x = 10).

**Added to `odelib.py`** (a new section "Series solutions (notebook 21; Lebl, Sections 7.1 to 7.3)" after
the section of notebook 20; 240 lines, all insertions: 6 in the module docstring, 234 of code; `SETUP_NAMES`
unchanged; numpy-only at import, checked with scipy, sympy and matplotlib blocked):
* `series_coefficients(recurrence, initial, n) -> np.ndarray`
* `series_value(coefficients, x, x0=SERIES_CENTER) -> np.ndarray` (Horner's rule; the shape of x)
* `indicial_roots(p0, q0) -> tuple[float | complex, float | complex, str]`: (r1, r2, case), r1 > r2
* constants `SERIES_CENTER` (0.0), `INDICIAL_NOT_INTEGER`, `INDICIAL_DOUBLE`, `INDICIAL_INTEGER`,
  `INDICIAL_COMPLEX` (the four cases, as readable text), `INDICIAL_TOLERANCE` (1e-12); `MAX_STEPS` is reused
  as the largest n. Private helpers: `_UnknownCoefficient`, `_EarlierCoefficients` (the list `a` that the
  recurrence sees: it refuses an index outside 0, ..., k - 1, so that a[-1] is never read from the end of
  the list, which would give a wrong a_2 for Airy with two initial values and no error), `_next_coefficient`.
  The existing `_real_array` and `_one_real_number` are called, not changed. No existing function changed;
  no change to an existing function is proposed.

**Tests added: 166** (`--collect-only` 1041 -> 1207).
* `test_odelib.py`, 98: Examples 7.2.1 (1/k!, cosh, sinh), 7.2.2 (Lebl's coefficients, a_(3n+2) = 0, Ai and
  Bi against `scipy.special.airy` at 1, 2, -3 to 1e-12), 7.2.3 (n = 3 and n = 4 as printed; n = 1 to 7 against
  NumPy's Hermite polynomials), 7.3.2 (r = 1/2 from `indicial_roots`, 1/k!, x^(1/2) e^x); Bessel's
  recurrence for p = 0, 1, 1/3, 2.5 against `jv`; the numbers of the notebook (J_0: -6.2176 with 10 terms at
  10, 20 terms within 1e-8 on [0, 10], 19 not); the documented limits (arctan at 1.1: 0.17 with 10 terms,
  13 with 40, the errors at 0.9 decreasing; J_0 at x = 20: 1.3e-5 with 30 terms, between 1e-10 and 1e-8
  with 40, 60, 80 terms, largest term 7.6e6); Horner against `np.polyval` on 40 seeded polynomials, shapes;
  the list `a` (what the recurrence is given, n = len(initial), NumPy numbers); 72 tests of invalid input,
  each with `match` (among them the negative index, a[k], a float index, ZeroDivisionError, inf, NaN, text,
  a list, a complex number, a boolean, None, a missing argument, `k ^ 2`, a NameError: each sentence names
  k); the four cases on ten equations of the book and of the notebook; 40 seeded pairs of roots of each kind;
  the borders (decimals 0.7, 0.0225; gaps of 1e-7, 1e-6 double, 1e-5 not; whole differences 1, 2, 7 and
  each a little off); large and small roots; warning filters and NumPy settings unchanged; `SETUP_NAMES`.
* `test_cross_validation.py`, 21 (behind `importorskip("sympy")`): partial sums put in six equations leave
  only the powers above the cut; four equations against `dsolve(..., hint="2nd_power_series_ordinary")`;
  eight exact series (cosh, sinh, e^(-x^2/2), arctan, J_0, J_1 / x, e^x of Example 7.3.2, Ai / Ai(0));
  Hermite n = 3 and 4; 30 seeded rational equations against `sp.solve` with the case decided exactly; the
  limits of Examples 7.3.2, 7.3.3 (and its slight change), Exercise 2(c), Bessel of order 1/3; x ln x for
  Lebl's Exercise 7.3.10; Example 7.3.4 by `simplify`.
* `test_notebooks_run.py`, 47: the notebook as it is, with right answers, with wrong answers, the answer
  markers; 30 particular answers (other accepted forms; each taught mistake with its sentence; the library's
  sentences reaching the student in Exercise 2(d)); the sections run again after the exercises; 8 runs with
  the numbers that the comments invite a student to change; 2 cells of the text that stop with a sentence;
  N_COEFFICIENTS = 1 stops with the library's sentence; 9 slider values with RuntimeWarning as an error.

**Counts of tests, before and after.**

| Gate | Before (2026-11-24) | After |
|---|---|---|
| Library, env `ode` | 779 passed (117 s) | 898 passed (116 s) |
| Library as in CI (`site/.venv`, no SymPy) | 586 passed, 11 skipped | 684 passed, 11 skipped |
| Notebooks (`test_notebooks_run.py`, whole file) | 262 passed (5 min 31 s) | 309 passed (6 min 31 s) |
| `--collect-only` | 1041 | 1207 |

**The other gates (2026-11-24).** Lint: `ruff check --no-cache` clean on `make_21.py`, and no finding on the
lines added to `odelib.py`, the three test files, `make_00.py` and `build_site.py`. Additive-only proof
(`nb21_build/prove_additive.py`, difflib on the copies made before the work): for each of the four files
the opcodes are "equal" and "insert" only (240, 367, 163 and 160 lines added), every old line still there in
order, line endings as they were. A second check (`nb21_build/name_clash.py`): no new top-level name of
the test files rebinds an old one (it found one while building, `X` in `test_cross_validation.py`, which
broke 20 old phase-plane tests; renamed `XV` before anything else was done); the only "clashes" reported
are the two dictionaries `EXERCISES` and `EXTREMES`, extended on purpose. Old against new
(`nb21_build/compare_old_new.py`): 48 pre-existing functions, 69 calls (values, exceptions with their
messages, printed text; the nine rows of the Laplace table used on numbers): 0 different; no old public
name missing, no constant changed. Site build: 26 notebooks, 137 MB; `site/test_build_site.py`: 5 passed.
Browser (`check_site.py`, headless Edge): 26 of 26 notebooks without errors, 13 to 22 s each, about 7 min in
all (baseline before the work: 25 of 25). Notebook 21 in the browser: 25 cells, 7 plots, 2 sliders, 15 s
(14.4 s by `nb19_build/cell_times.py`): the setup cell 12.5 s, every other cell at most 0.4 s (the arctan
cell with its figure), the SymPy cells (limits, `besselj` simplified) under 0.2 s. Trial export with
`export_public.py` into a new, empty folder of the scratch area (`nb21_build/export_trial`): 90 files, 15.5 MB, all checks passed (only files of the allow-list, no large file, no mention of a file that was left out, plain ASCII), done again into a second new folder after this section was written (`nb21_build/export_trial_2`), with the same result.
Nothing was committed or pushed; the public folder was not touched.

**Not run.** The library tests at the browser pins (numpy 2.4.6, SciPy 1.18.0) in an environment of their
own, as for notebooks 19 and 20; the browser run executes the new functions at those pins inside the
notebook. The independent reviews (step 7), the real export (step 9) are left to the coordinator.

**Departures from the plan, and why.**
1. Slider 1 is not attributed to "Lebl's Example 7.1.5". In the text of the book Section 7.1 has two
   examples numbered 7.1.5; the rational one is x/(1 + 2x + x^2), whose denominator has the real root -1,
   and nothing in Chapter 7 speaks of complex roots and the radius. The notebook quotes Lebl's Section 7.1.5
   on the geometric series ("a series for a function only defines the function on an interval") and gives
   the complex-zeros rule to Boyce and DiPrima (Decision 7), stated as a lower bound with the zeros of P
   (true with or without a common factor).
2. A sixth constant, `SERIES_CENTER = 0.0`, the default x0 of `series_value` (the rule "no hard-coded
   numbers", as `STEP_START` in notebook 20); `MAX_STEPS` reused as the largest n; three private helpers.
3. `INDICIAL_TOLERANCE` is applied to the squared difference (r1 - r2)^2 = (p0 - 1)^2 - 4 q0, relative to
   max(1, (p0 - 1)^2 + 4|q0|), and is 1e-12. On the difference itself (a first version, 1e-6 times
   max(1, |r1|, |r2|)) every pair of roots larger than about 5e5 was called "a whole number" with no error.
   As built: roots closer than about 1e-6 are a double root, a difference within about 5e-13 of a whole
   number is that number (for p0, q0 of size 1); the docstring says so.
4. `series_coefficients` refuses n = 20.0 (as `laplace_table` and `trajectories` refuse a float) and a
   recurrence that returns a boolean; the returned "list" a is the read-only view of departure 2.
5. Letters: capital P, Q, R for the coefficients of the equation (Lebl's p, q, r), because r is the
   exponent; "a term" means a coefficient that is not 0, and both sliders count such terms (2N
   coefficients), so that "10 terms of J_0" is the plan's -6.218 at x = 10.
6. The brief's sentence "the J_0 series at x = 20 with 30 terms is off by about 1e-5" is right, but that
   error is truncation, not cancellation: with 35 terms or more the error stops at about 1.2e-9, the
   rounding of terms as large as 7.6e6. The docstring of `series_value` and the answer of slider 2 say both.
7. Exercises, details the plan left open: 1(b) is Lebl's Exercise 7.2.8(a); 1(c) is x^2 y'' - 2x y' + 2y = 0
   (roots 2 and 1); 1(d) is answered with the four constants `ol.INDICIAL_...` (`ol.check_choice`; text
   with any capitals and spaces is accepted); 2(b) is (x^2 - 2x + 5) y'' + y = 0; 2(d) is a function to
   complete, `bessel_1_rule(k, a)`, checked as x times the partial sum (a_0 = 1/2) against `jv(1, x)`;
   3(iii) is Lebl's Exercise 7.3.10 (printed answer A x + B x ln x); 3(iv) uses `scipy.optimize.brentq`.
   Hints for the taught mistakes: 1(a) the constant 4; 2(a) three wrong recurrences; 2(b) the real or the
   imaginary part alone; 2(c) not divided by 2, and the sign of q0; 2(d) the rule of J_0.
8. Figures: a J_0 and Y_0 figure was added (Section 6: "Y_n named, with the warning that it is infinite at
   0"); there is no figure in Section 5.
9. The J_0 cell above slider 2 uses 15 terms and checks up to x = 4, so that nothing above the slider shows
   that 20 terms are enough at x = 10.
10. SymPy's power-series `dsolve` ignores `ics`: the cross-validation solves for C1 and C2 itself, and
    compares the powers up to x^5 (SymPy stops the odd series early).
11. Registries: `NAME_21` is the first entry of `EXERCISES` and of `EXTREMES` (inserted, nothing
    reordered); the week is "Week 14 - Series solutions (Sessions 27-28)"; the row of `make_00.py` shows
    the two forms of the series. Rebuilding `00_Index.ipynb` also changed its stored widget ids and the
    order of its widget state, as every rebuild does.

**Lebl's text.** Recomputed and right: Examples 7.1.1 to 7.1.4, the geometric series of Section 7.1.5 and
its sentence; Example 7.2.1 and (7.2.2); Example 7.2.2 (a_2 = 0, the recurrence, y_1 and y_2 to x^7);
Example 7.2.3 (the recurrence, n = 4: 1 - 4x^2 + (4/3) x^4); Example 7.3.1; Example 7.3.2 (4r(r - 1) + 1,
the recurrence, 1/k!, x^(1/2) e^x, "That was lucky!"); Example 7.3.3 and the slight change; Theorem 7.3.1,
items iii to vi; Bessel's indicial equation (r - p)(r + p); J_n for n = 0 and 1 against SciPy; Y_n to
minus infinity at 0; Example 7.3.4; the printed answers of Exercises 7.3.8 and 7.3.10. Not in Lebl and
said so: the radius rule (Boyce and DiPrima, Theorem 5.3.1); "the larger root always gives one solution"
is stated without a source (Boyce and DiPrima, Theorem 5.6.1, for a reviewer to confirm).

**Points for the reviewers.**
* The wording of the radius rule and its citation (Boyce and DiPrima state it with the radii of Q/P and
  R/P; the notebook uses the zeros of P, which gives a bound that is never larger).
* `indicial_roots` near the borders of its cases: the tolerance is a choice (departure 3).
* The length is near the target, but Sections 3 and 5 are dense: the four steps, Airy, the cases table and
  Example 7.3.2 are each one long Markdown cell.
* The widget state stores the two slider pictures a second time, as in notebooks 17 to 20.
* The rows for the reserved-letter table of the style sheet (Section 4) are not written: they go with the
  decks of Sessions 27 and 28.

### Corrections after the reviews (2026-11-24)

Three reviews (`reviews_nb/math_review_nb21.md`: the mathematics of the notebook correct, 1 ERROR in
`indicial_roots`, 5 IMPRECISE, 8 COSMETIC; `reviews_nb/clarity_review_nb21.md`: 0 BLOCKING, 14 IMPORTANT,
20 MINOR, five of the seven hard points "partly"; `reviews_nb/code_review_nb21.md`: existing code
unharmed, 1 BUG, 2 RISK, 6 STYLE) and the coordinator's decisions A to G, applied in one pass, library
first. One line per finding. What Section 8 says above about the cells, the figures, the length, the
constants, the tests and the counts is superseded by this part.

**Library** (new code of notebook 21 only; nothing older changed)
* Mathematics 1 (ERROR) and code RISK 3 (decision A): adopted as proposed. A new constant
  `INDICIAL_AMBIGUOUS = 0.25`; `indicial_roots` refuses with one sentence ("the roots, about ..., are too
  large for the computer to tell a double root from two roots 1 apart") when
  `INDICIAL_TOLERANCE max(1, (p0 - 1)^2 + 4|q0|)` reaches it, that is for roots of size about 1.8e5 and
  more; the docstring has the reviewer's sentence with the example 1000 and 1000.002 (returned as the
  double root 1000.001) and the threshold. Tests on both sides (double roots of size 1.5e5 computed, 2e5
  refused), the reviewer's three rows refused, the documented example pinned; two earlier tests of very
  large roots (1e8, 1e100) changed to roots near 1e4 and to a refusal.
* Code STYLE 4: `-0.0` removed (`q0 / larger + 0.0`), with a test on (2, 0).
* Code BUG 1 and mathematics 13 (decision A): `__iter__` and slices with ends 0 or more, as the
  reviewer's tested fix; a negative end of a slice is refused with its own sentence. Tests: Fibonacci with
  `sum(a[k - 2:k])`, `sum(a) / k`, `np.array(a)`, `max(v for v in a)`, a slice past k cut at k, `a[-2:]`
  and `a[k - 3:k]` at k = 2 refused. The docstring says that `a` can be looped over and sliced.
* Code STYLE 5 and clarity 17: the sentence for a negative index now says "(use a[k - 2], not a[-2], for
  the coefficient 2 place(s) back); if your rule reaches back 4 places, give at least 4 initial values",
  true whether the index was typed or computed.
* Code STYLE 6: the "two arguments" advice only when Python's message is about positional arguments; any
  other TypeError is reported as "the recurrence stopped with an error at k = ... (TypeError: ...)".
  Clarity 16: a ZeroDivisionError adds "(if you wrote a power with ^, write ** instead)".
* Code RISK 2: `x - x0` is computed inside the `errstate` block; a test with warnings as errors.
* Mathematics 7: "Theorem 7.3.1, items iii to vi" became "Section 7.3.2, items iii to vi of the list after
  Theorem 7.3.1" in the comment and both docstrings; the Euler equation of the docstring has the
  coefficients p0, q0 (decision D).
* Code STYLE 9: the ranges and tolerances of the new tests are named (19 constants in `test_odelib.py`,
  3 in `test_cross_validation.py`).
* One line outside the new section: `from collections.abc import Iterator` inserted after the existing
  import line (for the type hint of `__iter__`); an insertion, as the additivity proof shows.

**The five hard points that were "partly"** (decision B)
* 1, the shift (clarity 1, 2, 3): "The rule of the shift" at the end of step 3 (write k - m in place of
  k, add m to the start), cited by name for $xy$ in Airy (m = 1), for $x^2y$ in $J_0$ (m = 2), and in the
  two rows of the table of Example 7.3.2 (m = 1); where $2a_2$ comes from in Airy; "this is not a shift of
  a sum" for the $k - 2$ of the computer's form.
* 3, converges against agrees (clarity 6, mathematics 10): the third question "how many terms give the
  accuracy we want" is named in Section 2; slider 2 asks "right to 8 decimals for every x up to 10 (an
  error below 1e-8)", and its answer starts "by eye, about 15 terms; to 8 decimals, 20"; the rounding
  remark is marked "(Optional.)".
* 4, complex points (clarity 4 and 5, mathematics 2): the ratio test is taught once, in the form "next
  term over this term", in an "(At home.)" cell of Section 2 (Lebl, Theorem 7.1.2, "in another form"),
  and cited in Sections 4 and 6; the printed ratios are $|a_{k+1}/a_k|$, as the header says. "Why complex
  points matter" (the disk of the complex plane) is in Section 4 before the exercise that needs it; the
  answer of slider 1 points to it.
* 6, the lowest power (clarity 7, mathematics 4): the line $x^2y'' + x(xQ/P)y' + (x^2R/P)y = 0$ is
  written; the justification is the reviewer's (the two functions are power series $p_0 + p_1x + \cdots$);
  the Euler equation is $x^2y'' + p_0xy' + q_0y = 0$, with "(not Euler's method of notebook 04, nor Euler's
  formula of notebook 06)".
* 7, the cases (clarity 8, 9, 10; mathematics 3): the table has the columns name in odelib, roots, Lebl,
  example, a second solution ("1, 2, ... apart" excludes 0); the code cell before it prints the four
  names with their texts; $x^{\alpha + i\beta} = x^\alpha(\cos(\beta\ln x) + i\sin(\beta\ln x))$ by
  Euler's formula; "the larger root" only for real roots, with the complex case said (Boyce and DiPrima,
  Theorem 5.6.1).

**Mathematics texts** (decision C): 2 the reviewer's sentence for "The fact", and no claim that Lebl's
7.1.5 shows the complex idea; 3 (above); 4 (above); 5 "Not in Lebl: series about the point at infinity";
6 "that have no common factor"; 7 (above); 8 the slider title, the checking table and the answer print
"below 1e-15" (helper `error_text`), and the comment of the checking cell gives 7e-19; 9 "as N grows";
10 (above); 11 "a second solution ... (row iv of the table)"; 12 "(Lebl, (7.2.2), solved for $a_{k+2}$)".

**Letters** (decision D, clarity 11, 12): $P$, $Q$, $R$ for the equation, $p_0$, $q_0$ for the limits,
$\nu$ ("nu") for the order of Bessel's equation, said once in Section 1 and again in Section 6 ("Lebl's
p"; $p_0 = 1$ for every order); the Euler equation with $p_0$, $q_0$; `R_732` renamed `EXPONENT_732`.

**Code a student touches** (decision E)
* Clarity 8, 21: Exercise 1(d) lists the four names and an example (`ol.INDICIAL_COMPLEX`, not an
  answer); its own check prints "no answer yet (replace None with one of the four names ...)", "write
  ol.INDICIAL_INTEGER, with ol. in front" (the four names without `ol.` are defined in the helper cell as
  markers, so the cell does not stop on a NameError), and "write one of the four names ... without
  quotes" for other text or a number; nothing in it stops 1(a) to 1(c). The comment of 1(b): "a number,
  such as 3 (not a list)".
* Clarity 13: Hermite moved to Exercise 3(i) (with the worked n = 3 in its statement); its template
  takes `HERMITE_N`, computes both series and prints each residual with "(0 only when the series stops)".
* Clarity 14 and 19: the formula of 2(a) has an answer cell of its own, with the comment "with SymPy's
  functions: sp.exp (not np.exp)"; the statement says "compare with e^u" and "type it with sp.exp"; with
  `np.exp` only that cell stops, and the hint of the library ("use sp.exp") is printed under it (a test
  runs it with allow_errors and finds the hint and the later checks); the partial sum gets "this is the
  partial sum; the whole series is e^u for some u".
* Clarity 15 and code STYLE 8: the 2(d) checker calls the rule only through `ol.series_coefficients`;
  "no answer yet" is read from its sentence for None; the J_0 hint compares the computed a_2; `a[k]` now
  gets the library's sentence; the generic sentence adds "check the signs, a[k - 2], and that powers are
  written **".
* Clarity 18: 1(c) and 2(c) print "the indicial equation starts with r(r - 1), not r^2" for the roots of
  $r^2 + p_0r + q_0$ (in any order, complex too). Clarity 20: 2(b) hints 1, 2 and 5.
* Clarity 22: the rule of Section 4 has a cell of its own whose figure (the complex plane) follows
  `P_COEFFICIENTS` and `X0` (circle, zeros, labels as "i", "-i", "1+2i", interval from the numbers); the
  comments invite `[1, 0, 4]`, `[1, -2, 5]` and `X0 = 0.5` (tests run them). The arctan figure is a cell
  of its own.
* Code STYLE 7: the unnamed numbers of the code cells are named (`ROWS_SHOWN_1`, `ROWS_SHOWN_2`,
  `RATIO_STEP`, `SHOWN_ARCTAN`, `RATIO_K`, `COEFFICIENTS_732`, `SHOWN_732`, `SHOWN_J0`, `SHOWN_DIGITS`,
  `J0_ENOUGH`, `J0_TOO_FEW`, `TEXT_TERMS`, `TEXT_VALUE`, `TEXT_DIGITS`, `FIVE`); the header of the slider-1
  table is built from `X_ERRORS_1`; the first zero of $J_0$ is `jn_zeros(0, 1)[0]`; the $J_0$ asserts have
  a sentence (`TEXT_J0`, tested).
* Clarity 23: the coefficients of cosh print with `:.10g` (no false zeros); clarity 26: "Python writes j
  for i" in Section 1; clarity 32: the templates say "select only its lines (not its title line)" and
  template (ii) prints with `:.2g`.

**Pacing** (decision F, clarity 33, 34): "(At home.)" cells for the ratio test, Airy (its derivation, and
"Against SciPy", marked optional) and the arctan derivation; Example 7.3.4 "(Optional; at home.)"; the
In class / At home lines of Sections 2 and 5 rewritten. Split: cell 13 into three (steps 1-3 with the
rule of the shift; step 4 with reading the recurrence; the computer), cell 15 into the caption, Airy and
"Against SciPy", cell 19 into the fact (with "Why complex points matter" and the example) and the series
of the example, cell 22 into Example 7.3.1 with "why $x^r$" and regular singular points, cell 24 into the
Euler and indicial equations and, after the code cell that prints the four names, the table of cases.
No step of a hard point left the text.

**Remaining MINOR items** (decision G): clarity 24 (cosh and sinh written), 25 (elementary functions in
words; Frobenius in one clause), 27 (the factorial step written; "just below"), 28 (why $a_0 \ne 0$; why
the two limits), 29 (the substitution of Example 7.3.4 written, "not time"), 30 (the legend of the arctan
figure outside the axes), 31 (with mathematics 8), 10, 12, 16, 17, 21, 23, 26, 32 (above).

**Departures from a proposed fix, and what was not done**
* Clarity 4: Lebl's Theorem 7.1.2 is cited "in another form", not dropped.
* Clarity 5: the paragraph says that a zero of $P$ "can be" a point where the solution blows up (it need
  not be; the rule is a lower bound), not "such as a zero of P".
* Clarity 8: the names without `ol.` are caught by four marker objects defined in the helper cell (a
  NameError on the answer line itself cannot be caught inside the cell); the full text of a case typed in
  quotes is still accepted.
* Clarity 9: the formula for $x^{\alpha + i\beta}$ is placed under the table, as proposed; row vi of the
  table names the imaginary part, the sentence under it the real part.
* Clarity 14: `np.exp` still stops its (own) answer cell, because the error happens on the student's
  line, before any check; the library's hint is printed under the red message and the other checks of
  Exercise 2 are in the cell before.
* Clarity 22: X0 stays an invited number; the figure follows it (the arctan figure, now separate, does
  not depend on it).
* Mathematics 1: the threshold is the reviewer's (0.25 on the band), not a smaller multiple of the
  rounding (the code reviewer's alternative), so the decimals case (0.7, 0.0225) is unchanged.
* Length: before the exercises 3171 words on the measure of plan 17 (3540 with each formula as one word, 4602 whitespace tokens), 2963 before this pass: within the limit of 3300, above the aim of about 3000. The added explanations (the rule of the shift, the disk, the third question, the divided equation, the complex power, the table's columns) cost about 550 words; about 350 were cut from the references, the conventions, the captions, the answers of the sliders, Example 7.3.4 and Section 7, and Hermite left the text for Exercise 3(i). The "(At home.)" cells still count, since they stand before the exercises.

**The notebook after the corrections.** 60 cells (26 code), 6 stored figures and 2 sliders; 458 958 bytes,
plain ASCII, no local path. All six figures and ten slider frames (N = 1, 2, 10, 30; 1, 5, 15, 19, 20, 30)
were rendered again from the final notebook (the old PNG files deleted first) and looked at, and the
complex plane also with the invited numbers (P = x^2 - 2x + 5, X0 = 0.5): changed after the first look,
the labels of the zeros ("0+1i" became "i"; their distance from the point now scales with the window).

**Counts.**

| Gate | As built | After the corrections |
|---|---|---|
| Library, env `ode` | 898 passed | 908 passed (87 s) |
| Library as in CI (`site/.venv`) | 684 passed, 11 skipped | 694 passed, 11 skipped (74 s) |
| Notebooks (`test_notebooks_run.py`, whole file) | 309 passed (6 min 31 s) | 320 passed (6 min 37 s) |
| `--collect-only` | 1207 | 1228 (README.md: 1228) |

**Gates after the corrections (2026-11-24).** Lint: `make_21.py` clean; no finding on the added lines of
`odelib.py`, the three test files, `make_00.py` and `build_site.py`. Additive-only proof against the
copies made before notebook 21: opcodes "equal" and "insert" only for the four files (272, 454, 166 and
194 lines added), every old line in order, line endings as they were. Name clashes: none, apart from the
two dictionaries extended on purpose. Old against new: 48 functions, 69 calls, 0 different; no old public
name missing; no constant changed; new public names: the three functions and seven constants (the import
of `Iterator` is private, `_Iterator`). `odelib` imports with NumPy alone (checked with SciPy, SymPy and
matplotlib blocked). Site build: 26 notebooks, 137 MB; `site/test_build_site.py`: 5 passed. Browser
(`check_site.py`, headless Edge): 26 of 26 notebooks without errors, 14 to 21 s each; notebook 21: 26
cells, 8 plots, 2 sliders, 14 s; by `cell_times.py` the setup cell 11.9 s, every other cell at most 0.4 s.
Trial export into a new, empty folder of the scratch area (`nb21_build/export_trial_3`): 90 files, 15.5 MB, all checks passed (allow-list, no large file, no mention of a left-out file, plain ASCII); repeated into a second new folder (`export_trial_4`) after this section was written, with the same result. The public
folder was not touched; nothing was committed or pushed. Not run, as before: the library tests at the
browser pins in an environment of their own.

### Final pass after the re-check (2026-11-24)

The re-check (`reviews_nb/recheck_nb21.md`: existing code unharmed; all seven hard points
"understandable"; 0 ERROR, 0 BLOCKING, 0 IMPORTANT, 11 MINOR) was applied in full, with its fixes. What the
parts above say about the length, the cells and the counts is superseded by this part.

* 1: the docstring of `indicial_roots` says "1e-6 sqrt((p0 - 1)^2 + 4|q0|)" (not 1.4e-6); a test pins the
  band near 1000 (gaps 0.0027 and 0.0028 double, 0.0029 and 0.0035 not).
* 2: the refusal names the size of the larger root ("the roots, up to about 3e+05 in size, ..." for
  roots +-3e5, where it said "about 0"); the comment and the docstring now say "when (p0 - 1)^2 + 4|q0|
  reaches 2.5e11, for example a double root of size about 1.8e5" (roots 4e5 and 0 are computed: a test).
* 3: one docstring sentence on the wide bands just below the limit ("two roots up to 0.5 apart are called
  a double root (170000.4 and 170000), and a difference between about 0.87 and 1.12 is called 1"), with
  a test of both examples near 1.7e5. INDICIAL_AMBIGUOUS is unchanged, as the re-check advises.
* 4: the cell of the rule with a P that has no zeros (`P_COEFFICIENTS = [1]`) prints "P has no zeros (a
  constant P): every point is ordinary, and the series solutions converge for every x; ..." and draws no
  figure (the drawing is a function called only when there are zeros); a test runs it.
* 5: the printed coefficients of the arctan cell have no "-0." (`+ 0.0`), and those of J_0 are printed
  with `np.printoptions(suppress=True)` (decimals, as 0.015625 beside 1/64).
* 6: row iii of the table: "real, unequal, not 1, 2, ... apart" (the example column shortened to fit).
* 7: "In y'' above the sum shows x^(k-2), that is m = -2: k - (-2) = k + 2 in place of k"; the two rows of
  the table of Example 7.3.2 say "(shift rule, m = 1; r stays)".
* 8: a misspelt name with `ol.` (for example `ol.INDICIAL_INTEGRAL`) still stops the answer cell (the error
  is on the student's line), but the helper cell registers once a small hint, printed under the red
  message: "the four names are ol.INDICIAL_NOT_INTEGER, ...: copy one from the table of Section 5" (a test
  runs it with allow_errors); the comment of the answer line says to copy the name; the example of the
  statement is now `ol.INDICIAL_NOT_INTEGER` "(only the form: it may or may not be an answer)".
* 9: "(inside the disk the series adds up to a finite number)" in "Why complex points matter".
* 10: "(defined below)".
* 11: ruff on the notebook's cells: the two `zip` have `strict=True`, the trailing `;` is gone; 0 findings.
* Length (re-check, Section 6): the source note of "The fact" is after the answers ("For those who read the
  sources", the reviewer's sentence unchanged), with a short citation in Section 4; "Against SciPy" is a
  comment of the Airy cell; the optional remark at x = 20 is after the answers ("Slider 2, farther out");
  Section 7 is shorter. Before the exercises: 3066 on the measure of plan 17 (3425 with each formula as one word, 4473 whitespace tokens), 3171 before this pass.

**The notebook after the final pass.** 59 cells (26 code), 6 stored figures and 2 sliders; 460 801 bytes, plain
ASCII, no local path. All six figures, the complex plane with the invited numbers and the ten slider
frames were rendered again from the final notebook (old PNGs deleted first) and looked at: no defect.

**Counts.**

| Gate | After the corrections | After the final pass |
|---|---|---|
| Library, env `ode` | 908 passed | 909 passed (115 s) |
| Library as in CI (`site/.venv`) | 694 passed, 11 skipped | 695 passed, 11 skipped (76 s) |
| Notebooks (`test_notebooks_run.py`, whole file) | 320 passed | 322 passed (6 min 38 s) |
| `--collect-only` | 1228 | 1231 (README.md: 1231) |

**Gates after the final pass (2026-11-25).** Lint: `make_21.py` clean, and ruff on the notebook itself 0
findings; no finding on the added lines of `odelib.py`, the three test files, `make_00.py`, `build_site.py`.
Additive-only proof against the copies made before notebook 21: opcodes "equal" and "insert" only (277,
475, 166 and 207 lines added), every old line in order, line endings as they were. Name clashes: none, apart
from the two dictionaries extended on purpose. Old against new: 48 functions, 69 calls, 0 different; no old
public name missing; no constant changed; new public names: the three functions and seven constants.
`odelib` imports with NumPy alone. Site build: 26 notebooks, 137 MB; `site/test_build_site.py`: 5 passed.
Browser (`check_site.py`, headless Edge): 26 of 26 notebooks without errors, 13 to 24 s each; notebook 21:
26 cells, 8 plots, 2 sliders, 14 s; by `cell_times.py` the setup cell 12.5 s, every other cell at most
0.4 s. Trial export into a new, empty folder of the scratch area (`nb21_build/export_trial_5`): 90 files, 15.5 MB, all checks passed (allow-list, no large file, no mention of a left-out file, plain ASCII); repeated into a second new folder (`export_trial_6`) after this section was written, with the same result. The
public folder was not touched; nothing was committed or pushed. Not run, as before: the library tests at the
browser pins in an environment of their own.
