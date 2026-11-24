# Notebook 19 - The Laplace transform: plan

MATH 331 - Differential Equations - Spring 2027 - phase E of the notebooks (19-22), the first notebook of
the phase

**Status (2026-11-23): built, reviewed, corrected, re-checked; exported to the public folder on 2026-11-23 at the instructor's request (the instructor commits and pushes).**

**Sessions 23 and 24, Unit 10** (lecture plan, Session 23: "Laplace transform: definition, existence,
properties, table; inverse transforms by partial fractions"; theory exercise "Prove the derivative rule";
practical exercise "Compute transforms and inverse transforms". Session 24: "Solving initial-value problems
with the Laplace transform"; theory exercise "Compare the method with undetermined coefficients"; practical
exercise "Solve an RLC circuit IVP"). About **15 minutes in class in each session** (Session 23: Sections
2 to 4; Session 24: Sections 5 and 6), the rest as homework. Notebook 18 ends with "Where to go next:
notebook 19, the Laplace transform": this notebook keeps that promise.

The instructor decided on 2026-11-23 that this notebook comes before the deck of Session 23
(`Lectures/Lecture23/Lecture23_plan.md`, pending), as the master plan `NOTEBOOKS_17-22_plan.md` says.

**References** (every item was found in the text of the book on 2026-11-23; the numbers below were
recomputed with SymPy and SciPy the same day).
* Lebl, *Notes on Diffy Qs*, **Section 6.1, The Laplace transform**:
  * the definition, F(s) = integral from 0 to infinity of e^(-st) f(t) dt; lower case for functions of
    time, upper case for their transforms; the "black box" sentence;
  * Example 6.1.1 (L{1} = 1/s, s > 0), Example 6.1.2 (L{e^(-at)} = 1/(s + a), s > -a), Example 6.1.3
    (L{t} = 1/s^2 by parts), Example 6.1.4 (the Heaviside function: not used here, Decision 4);
  * Table 6.1.1 (C, t, t^2, t^3, t^n, e^(-at), sin, cos, sinh, cosh, u(t - a));
  * Theorem 6.1.1 (linearity) and the warning that the transform of a product is not the product of the
    transforms; 1/t, tan t and e^(t^2) have no transform;
  * exponential order, Theorem 6.1.2 (existence for s > c), the limit F(s) -> 0 as s grows, Theorem 6.1.3
    (uniqueness);
  * the inverse transform; Example 6.1.5 (1/(s + 1)); Example 6.1.6 ((s^2 + s + 1)/(s^3 + s) = 1/s +
    1/(s^2 + 1), inverse 1 + sin t); the first shifting property L{e^(-at) f(t)} = F(s + a); Example 6.1.7
    (1/(s^2 + 4s + 8), inverse (1/2) e^(-2t) sin 2t); proper rational functions.
* Lebl, **Section 6.2, Transforms of derivatives and ODEs**, up to and including the procedure after
  Example 6.2.1, and Subsection "Transforms of integrals" (Example 6.2.4): Table 6.2.1 (the transforms of
  g', g'', g'''); Example 6.2.1 (x'' + x = cos 2t, x(0) = 0, x'(0) = 1; X(s) = s/((s^2 + 1)(s^2 + 4)) +
  1/(s^2 + 1); x = (1/3) cos t - (1/3) cos 2t + sin t); the remark that not every equation can be solved
  this way. The Heaviside function, the second shifting property, Example 6.2.2 and transfer functions
  belong to notebook 20.
* Lebl, **Exercises 6.E**: 6.1.1 to 6.1.11 (no answers printed), 6.1.12 (answer 8/s^3 + 8/s^2 + 4/s), 6.1.13
  (answer 2t^2 - 2t + 1 - e^(-2t)), 6.1.14 (answer 1/(s + 1)^2), 6.1.15 (answer 1/(s^2 + 2s + 2)); 6.2.1,
  6.2.3 to 6.2.5 (the damped spring with the three kinds of damping), 6.2.7 (L{-t f(t)} = F'(s)).
* Adkins and Davidson, Section 3.7 (the RLC circuit, L q'' + R q' + q/C = E(t)), as in notebook 15.
* Notebooks 06 (characteristic roots), 14 (undetermined coefficients, `forced_response`), 15 (the RLC
  circuit and its letters), 17 and 18 (roots in the plane), 13 (first-order circuits). Video R03
  (integration by parts) and R04 (partial fractions), if these are their numbers (to be confirmed against
  the index when building).

**Purpose.** One idea: **the transform turns a derivative into a multiplication by s, so an initial value
problem becomes an equation of algebra, with the initial values already inside it.** The price is a table
and partial fractions.

1. **Session 23.** The transform is an integral, hence a number for each s: the notebook computes it
   numerically, draws it, and compares it with the table. The inverse is the table read from right to
   left, after partial fractions.
2. **Session 24.** Four steps solve an initial value problem: transform, solve for X(s), split, invert.
   The same problem is solved by undetermined coefficients (notebook 14), and the two are compared. An RLC
   circuit closes the notebook: the roots of the denominator of Q(s) are the characteristic roots, and
   they decide the shape of the charge in time.

---

## 1. Contents

1. **Before you start**: what is needed (integration by parts, partial fractions, the three cases of
   notebook 06); the conventions box (Section 4 of this plan).
2. **The transform is an integral** (Session 23).
   * The definition; an integral up to infinity as a limit, with a table of the integral up to T = 1, 5,
     20 for e^(-st) with s = 1.
   * `ol.laplace_numeric(f, s)` for f = 1, t, e^(-2t) at s = 3, against 1/3, 1/9, 1/5.
   * Figure 1: f(t) = t, the weight e^(-st), and their product, for s = 1 and s = 3; the shaded area is
     F(s).
   * **Slider 1** (question first: "does F(s) grow or shrink when s grows?"): s from 0.5 to 5; left, the
     product e^(-st) f(t) and its area; right, the point (s, F(s)) on the curve 1/s^2.
3. **The table** (Session 23).
   * `ol.laplace_table(...)` printed as a table; every row checked against `ol.laplace_numeric` at three
     values of s, and once against `sp.laplace_transform` (timed).
   * Linearity on Lebl's Exercise 6.1.12, 4(t + 1)^2, with its printed answer.
   * The range of s: e^(2t) at s = 3 and at s = 1 (`laplace_numeric` raises ValueError with a sentence a
     student can read). No transform at all: e^(t^2).
   * The product warning, numerically: L{t . t} against L{t} L{t} at s = 2 (1/4 against 1/16).
4. **The inverse transform** (Session 23).
   * The table from right to left; `sp.apart` for partial fractions.
   * Lebl's Examples 6.1.6 and 6.1.7 with every step, each checked two ways: the fractions added back
     (`sp.together`), and the answer transformed forward numerically.
   * The first shifting property in a figure: 1/(s^2 + 4) and 1/((s + 2)^2 + 4).
   * A procedure in four steps for a proper rational function.
5. **Solving an initial value problem** (Session 24).
   * The derivative rule (Table 6.2.1), checked numerically on g = sin 2t.
   * The four steps on Lebl's Example 6.2.1, every line; figure 3: the solution with its three terms.
   * The same problem by undetermined coefficients (notebook 14), side by side in a table of the steps of
     each method; `ol.forced_response` as the third, independent answer. What each method needs and
     gives (the theory exercise of Session 24).
6. **An RLC circuit** (Session 24).
   * The circuit of notebook 15 with other numbers (Decision 5): L = 1 H, R = 2 ohm, C = 0.2 F, a source
     that fades, E(t) = 8 e^(-t) V, and a capacitor charged at the start, q(0) = 1 C, q'(0) = 0 A:
     q'' + 2 q' + 5 q = 8 e^(-t).
   * Q(s) = (8/(s + 1) + s + 2)/(s^2 + 2s + 5) = 2/(s + 1) - s/(s^2 + 2s + 5);
     q(t) = e^(-t) (2 - cos 2t + (1/2) sin 2t). Checked by substitution, by the initial values, and by
     `ol.forced_response`.
   * Figure 4: the charge and the current. Figure 5: the roots of the denominator, -1 and -1 +- 2i, in the
     plane.
   * **Slider 2** (corrected on 2026-11-23, after the mathematics review; the first version asked "for
     which R does the charge stop oscillating?" with the fading source still on and answered sqrt(20),
     which is false: with E(t) = 8 e^(-t) the charge stops crossing 0 at 2 ohm and shows one rise only
     from about 2.2 ohm, because the term that the source leaves, e^(-t), dies out more slowly than the
     oscillating one). The slider shows the **discharge**, the source switched off:
     q'' + R q' + 5 q = 0, q(0) = 1 coulomb, q'(0) = 0, R from 0.5 to 8 ohm; left, the roots of
     s^2 + R s + 5 in the plane; right, q(t) with its lowest value in the title. Question first: "for
     which R do the two circles meet on the axis? Compute it from R^2 = 4L/C, then check with the slider
     and with the lowest value of q." The answer: R = sqrt(20), about 4.47 ohm, the critical resistance;
     below it the charge crosses 0 (lowest value negative), from it on it never does (lowest value 0);
     the eye stops seeing the oscillation earlier, near 3.5 ohm. Figure 5 (the roots, static) was cut:
     the slider's first frame shows the roots of the circuit.
7. **What the method cannot do, and where to go next**: equations with variable coefficients or nonlinear
   terms; inputs without a transform; notebook 20 (switches, blows, convolution).
8. **Exercises**, **answers**.

Target length: that of notebook 17 (about 2500 words of text). Fall-back if it runs long: the comparison
table of Section 5 moves to an Explore item.

---

## 2. Exercises (the four steps)

Each exercise is marked with its session.

1. **Predict** (no code). (a) L{e^(3t)} and its range of s. (b) *From Lebl's warning*: true or false,
   L{t e^(-t)} = L{t} L{e^(-t)}? (c) Which form of partial fractions for 1/((s - 1)^2 (s + 1))
   (Lebl, Exercise 6.1.7)? (d) After Session 24: the transform of x'' when x(0) = 2, x'(0) = -1.
2. **Compute**. (a) *Lebl, Exercise 6.1.1, one row*: L{t^2} from the definition, by parts twice. (b) *Lebl,
   Exercise 6.1.13*: the inverse of 8/(s^3 (s + 2)), with the printed answer. (c) *Lebl, Exercise 6.1.15*:
   L{e^(-t) sin t} with the shifting property, with the printed answer. (d) After Session 24: *Lebl,
   Exercise 6.2.4 with numbers*, an underdamped spring by the four steps.
3. **Explore** (code). (i) *Lebl, Exercise 6.1.4*: L{cos^2(omega t)} through the double angle, checked
   numerically. (ii) sinh and cosh as sums of exponentials (the two rows of Table 6.1.1 left out of the
   text). (iii) *Lebl, Exercise 6.2.7*: L{-t f(t)} = F'(s), on f = sin t, numerically and with SymPy. (iv)
   *Lebl, Example 6.2.4*: the transform of an integral, 1/(s (s^2 + 1)). (v) After Session 24: the circuit
   with a battery in place of the fading source, and the limit of q(t).
4. **Explain** (two or three sentences). Why does the Laplace method need no "factor x" when the input
   duplicates the complementary solution (Session 13)? The answer: the repeated factor appears by itself
   in the denominator.

Answers are checked with the existing `ol.check`, `ol.check_formula` and `ol.check_choice`; a transform
typed as a SymPy expression is compared with `check_formula` on an interval of s.

---

## 3. New in `odelib.py` (with tests)

A new section, "The Laplace transform (notebook 19)", after the section of notebook 18. NumPy only at the
top; SciPy is imported inside `laplace_numeric`. Type hints on arguments and return values; docstrings
citing Lebl, the notebook, and the conventions (t >= 0, s real).

* `laplace_numeric(f: Callable[[float], float], s: float, *, rtol: float = LAPLACE_RTOL) -> float`: the
  defining integral by `scipy.integrate.quad` on [0, infinity). ValueError, with one sentence, when s is
  not a finite real number, when f returns a non-finite value, or when the integral does not converge
  (the estimate of quad is not small against the value, or quad warns). It is the independent check of
  every table row and of every inverse (the answer is transformed forward).
* `class LaplaceRow(NamedTuple)`: `name: str`, `time: Callable[[float], float]`,
  `transform: Callable[[float], float]`, `s_min: float` (the transform holds for s > s_min),
  `time_text: str`, `transform_text: str` (plain ASCII, for printing).
* `laplace_table(a: float = LAPLACE_DEMO_A, omega: float = LAPLACE_DEMO_OMEGA, n: int = LAPLACE_DEMO_N)
  -> tuple[LaplaceRow, ...]`: Lebl's Table 6.1.1 without the Heaviside row, for given constants (1, t, t^2,
  t^n, e^(-at), sin, cos, sinh, cosh). ValueError for omega <= 0 or n not a nonnegative integer.
* Named constants: `LAPLACE_RTOL`, `LAPLACE_DEMO_A`, `LAPLACE_DEMO_OMEGA`, `LAPLACE_DEMO_N`.
* Nothing else. In particular SymPy's `laplace_transform`, `inverse_laplace_transform` and `apart` are
  not wrapped (master plan), and no checker is added.

**Tests** (`tests/test_odelib.py`, and with SymPy in `tests/test_cross_validation.py` behind
`importorskip`):
* against the book: Examples 6.1.1 to 6.1.3, every row of Table 6.1.1, the printed answers of Exercises
  6.1.12, 6.1.14 and 6.1.15;
* against an independent method: SymPy's `laplace_transform` for every row and for seeded random
  constants (SEED = 331); the closed forms of the notebook's two inverses transformed forward;
* against invalid input: s below the range (e^(2t) at s = 1), e^(t^2), 1/t, a non-finite s, omega <= 0,
  a negative or fractional n;
* at the browser pins (numpy 2.4.6, SciPy 1.18.0) and in the CI environment (no SymPy).

`SETUP_NAMES` does not change (the setup cell is that of notebook 18, with `import sys`).

---

## 4. Conventions and checks

As for notebooks 14 to 18: `assert` for every reference value; one sentence under every plot, written
after looking at the figure; line styles as well as colors; a question before every slider and its answer
after; named constants; plain ASCII; generated by `plans/make_19.py` with `nbbuild.py`; `ruff check`
silent on the new code.

* **Letters** (rows for STYLE_SHEET Section 4, with the deck of Session 23; Decision 2):
  * t >= 0, the time; s, the variable of the transform, a real number here;
  * f(t), g(t), x(t) in lower case and F(s), G(s), X(s) their transforms (Lebl);
  * the transform written with a script L and braces; it is not the operator L[y] of Session 11, and not
    the inductance L of the circuit: Section 6 says so once, where both appear;
  * the table as Lebl prints it: e^(-at) with 1/(s + a); many books print e^(at) with 1/(s - a): compared
    once;
  * the circuit keeps the letters of notebook 15: q (C), I = q' (A), L (H), R (ohm), C (F), E(t) (V).
* **Units**: t in s, s in 1/s (said once: e^(-st) needs a pure number in the exponent); the circuit in SI.
* **Lebl's slips** (Decision 6): the opening equation prints "m x'' = c x' + k x = f(t)" (a plus sign is
  meant); equation (6.2.1) prints "s^2 X(x) - s x(0) + x'(0)" (X(s) and a minus sign are meant; the next
  line of the book is right). The notebook writes the right forms, with a parenthesis.
* **In the browser**: each SymPy transform is timed by `check_site.py`; if one cell exceeds 20 s in
  Pyodide, the symbolic call is replaced by `apart` and the table, and the numeric check stays.

---

## 5. Order of work

1. `odelib.py`: `laplace_numeric`, `LaplaceRow`, `laplace_table` and their constants; tests first; ruff on
   the new lines; the whole existing suite passes unchanged (529 library and 174 notebook tests at the
   last count, to be re-counted before starting), also in the CI environment and at the browser pins.
2. `plans/make_19.py`: Sections 1 to 4 with slider 1 (Session 23); every figure looked at.
3. Sections 5 to 7 with slider 2 (Session 24); word count; the fall-back if needed.
4. Exercises and answers; `EXERCISES` and `EXTREMES` in `tests/test_notebooks_run.py`.
5. The index row in `plans/make_00.py`, `READING_ORDER` in `site/build_site.py` (Week 12), the READMEs;
   site build and `site/test_build_site.py`; `check_site.py` with the times of the cells.
6. Independent reviews (mathematics, weakest student, code); corrections; final run of all gates.
7. Trial export with `site/export_public.py` into a scratch copy; the real export only when the instructor
   says so. No commit, no push.
8. Then the deck of Session 23: its plan gets the list of what this notebook uses, and returns for
   approval.

---

## 6. Decisions for the instructor

1. **One notebook for two sessions** (as 17 and 18). Proposed: Sections 2 to 4 for Session 23 and 5 to 6
   for Session 24, exercises marked by session.
2. **Letters**, as listed in Section 4, following Lebl: script L with braces; lower and upper case; the
   table with e^(-at) and 1/(s + a); x(t) as the unknown in Lebl's example and q(t) in the circuit. The
   alternative for the table is the more common e^(at) with 1/(s - a), which no longer matches the
   reading.
3. **How SymPy is used.** Proposed: `sp.apart` and the table for every inverse, as a student works by
   hand; `sp.laplace_transform` and `sp.inverse_laplace_transform` shown once each, as a check; every
   result also checked numerically with `ol.laplace_numeric`. The alternative, SymPy's transforms
   everywhere, hides the method and may be slow in the browser.
4. **Left to notebook 20**: the Heaviside function, the second shifting property, Lebl's Example 6.2.2,
   and transfer functions, although Lebl has them in Sections 6.1 and 6.2. This follows the lecture plan
   (Sessions 25 and 26).
5. **The circuit's numbers are the notebook's own** (Lebl gives no circuit in 6.2): L = 1 H, R = 2 ohm,
   C = 0.2 F, E(t) = 8 e^(-t) V, q(0) = 1 C, q'(0) = 0. They give an underdamped circuit with whole
   numbers, an input that is not a constant (notebook 15 had the battery), and initial values that are
   not zero, which is where the transform shows its advantage. The notebook says the numbers are its own.
6. **Lebl's slips**: write the right forms, with the parenthesis "(the text prints ...)", as in notebook
   18.
7. **Reserved for the decks.** The deck of Session 23 keeps its "three doors on the s side"
   (25/(s^2 + b s + 25), b = 26, 10, 6), which this notebook does not use. The deck of Session 24 will
   need a circuit with other numbers than Section 6, and another initial value problem than Lebl's
   Example 6.2.1.
8. **The library gains three names only** (`laplace_numeric`, `LaplaceRow`, `laplace_table`), fewer than
   notebooks 17 and 18. Proposed: no checker for transforms (the existing `check_formula` compares two
   expressions in s on an interval) and no wrapper of SymPy.

---

## 7. Files

```
ode/ode/notebooks/
    19_Laplace_Transform.ipynb                 <- to build
    plans/19_Laplace_Transform_plan.md         <- this document
    plans/make_19.py                           <- builds the notebook
    odelib.py                                  <- + laplace_numeric(), LaplaceRow, laplace_table(), constants
    tests/test_odelib.py                       <- + their tests against the book and invalid input
    tests/test_cross_validation.py             <- + against SymPy
    tests/test_notebooks_run.py                <- + NAME_19, the exercises and sliders of notebook 19
    00_Index.ipynb, plans/make_00.py           <- index row
    README.md                                  <- the list of notebooks
ode/ode/site/build_site.py                     <- READING_ORDER: Week 12 with 19
ode/ode/site/README.md                         <- counts and the table of tested notebooks
ode/ode/README.md                              <- the number of tests
```

Nothing in the public folder is touched by hand; nothing is committed or pushed.

---

## 8. As built (2026-11-23)

**The notebook.** `19_Laplace_Transform.ipynb`, 39 cells (19 code), built by `plans/make_19.py`; 505 967
bytes (the limit of `export_public.py` is 2 000 000), plain ASCII, no local path stored. Before the
exercises: **3322 words** as whitespace tokens (every LaTeX token counted), **2568** with each formula as
one word, **2302** with each formula as one word and only tokens that contain a letter or a digit (the
measure that gives 2005 for notebook 17 and 2273 for notebook 18). The fall-back of Section 1 (the
comparison table moved to an Explore item) was not needed.

**Sections.** 1 Before you start (what is needed; the conventions); 2 The transform is an integral
(the limit, Lebl's Examples 6.1.1 to 6.1.3, `ol.laplace_numeric`, Figure 1, slider 1); 3 The table
(`ol.laplace_table`, linearity on Exercise 6.1.12, the product warning, the range of s, exponential
order); 4 The inverse transform (Examples 6.1.5 to 6.1.7, the first shifting property, Figure 2, the
procedure in four steps); 5 Solving an initial value problem (the derivative rule with its proof, the
four steps on Example 6.2.1, Figure 3, the same problem by undetermined coefficients, with the table of
the two methods and `ol.forced_response`); 6 An RLC circuit (the four steps, Figures 4 and 5, slider 2);
7 What the method cannot do, and where to go next; 8 Exercises; 9 Answers.

**Figures: 7 stored** (Figure 1: f(t) = t, the weight and the product for s = 1 and s = 3; slider 1
static; Figure 2: the shift in time and in s; Figure 3: Example 6.2.1 and its three terms; Figure 4:
charge and current; Figure 5: the roots in the plane; slider 2 static). Every figure was exported to PNG
and looked at, and the two slider functions were also drawn at s = 0.5, 2.5, 5 and R = 0.5, 4.4, 4.5, 6,
8; the sentence under each plot was written after that, with its numbers recomputed (1.16 coulombs at
t = 0.64 s; 0.37 A at t = 0.27 s; -0.84 A at t = 1.48 s; the faint second rise of the charge between
t = pi and t = 3.79 s). **Sliders: 2** (`plot_area(s)`, `plot_circuit(R)`), each with its question first,
a static figure and its answer.

**Added to `odelib.py`** (a new section "The Laplace transform (notebook 19; Lebl, Sections 6.1 and 6.2)"
after the section of notebook 18; nothing existing changed except the two lists of the module docstring,
which gained the new names; `SETUP_NAMES` unchanged; numpy-only at import, checked):
* `laplace_numeric(f: Callable[[float], float], s: float, *, rtol: float = LAPLACE_RTOL) -> float`
* `class LaplaceRow(NamedTuple)`: name, time, transform, s_min, time_text, transform_text
* `laplace_table(a: float = LAPLACE_DEMO_A, omega: float = LAPLACE_DEMO_OMEGA, n: int = LAPLACE_DEMO_N)
  -> tuple[LaplaceRow, ...]`
* constants `LAPLACE_RTOL` (1e-8), `LAPLACE_HORIZON` (700.0), `LAPLACE_DEMO_A` (2.0),
  `LAPLACE_DEMO_OMEGA` (3.0), `LAPLACE_DEMO_N` (3). No private helper, no checker, no wrapper of SymPy.

**Tests added: 95.**
* `test_odelib.py`, 53: Examples 6.1.1 to 6.1.3; the table against the values of Table 6.1.1 at s = 5,
  its names, texts, ranges and array behaviour; every row against the integral at three values of s;
  other constants (e^(3t), n = 0, a NumPy integer); 20 seeded sets of random constants (seed 331); the
  printed answers of Exercises 6.1.12 to 6.1.15, Examples 6.1.6, 6.1.7, 6.2.1, 6.2.4 and the circuit,
  each transformed forward at four values of s; linearity, the product warning and a transform equal to
  0; Table 6.2.1, Exercise 6.2.7 and the shifting property; the range of s (e^(2t) at s = 3, 2.1, and
  refused at 2.01, 2, 1, 0, -1; e^(-2t) at s = 0, -1, -1.5 and refused at -2, -3); e^(t^2) (also written
  with `math.exp`), 1/t, tan t, NaN, ln(t - 1); eight kinds of s that are not a finite real number; f
  not a function or not returning a number; rtol; twelve invalid constants of the table.
* `test_cross_validation.py`, 16 (behind `importorskip("sympy")`): every row against
  `sp.laplace_transform`, with SymPy's edge of convergence against `s_min`, for the notebook's constants
  and four seeded random sets; eight inverses (the notebook's and its exercises') against `sp.apart`,
  `sp.inverse_laplace_transform` and the forward integral; Example 6.2.1, the circuit and the spring of
  Exercise 2(d) against `dsolve` and `ol.forced_response`.
* `test_notebooks_run.py`, 26: the notebook as it is, with right answers, with wrong answers, the answer
  markers; 20 particular answers (other accepted forms; a formula in the wrong letter, in quotes, with X
  in it; the question typed back; wrong signs); the sections run again after the exercises; 14 slider
  extremes with RuntimeWarning as an error (s = 0.5 and 5 as the slider gives them; R = 0.5, 8, 4.4, 4.5,
  sqrt(20) and its two neighbours, and R = 6, where s + 1 appears twice under Q(s)).

**Counts of tests, before and after.**

| Gate | Before (2026-11-23) | After |
|---|---|---|
| Library, env `ode` | 529 passed | 598 passed (41 s) |
| Library as in CI (`site/.venv`, no SymPy, no matplotlib) | 382 passed, 10 skipped | 435 passed, 10 skipped (9 s) |
| Notebooks (`test_notebooks_run.py`) | 174 passed | 200 passed (3 min 45 s) |
| `--collect-only` | 703 | 798 |

**The other gates (2026-11-23).** Lint: `ruff check --no-cache` clean on `make_19.py`, and no finding on
the lines added to `odelib.py`, the three test files, `make_00.py` and `build_site.py`. Site build: 24
notebooks, 135 MB, 26 s; `site/test_build_site.py`: 5 passed. Browser (`check_site.py`, headless Edge):
24 of 24 notebooks without errors, about 6 min; notebook 19: 19 cells, 9 plots, 2 sliders, **15 s**. Each
cell was timed in the browser: the setup cell (the start of Python and the imports) 12.1 s, every other
cell at most 0.6 s (the cell of Section 4, with `sp.apart`, `sp.inverse_laplace_transform` and Figure
2); the cell of the table, with nine `sp.laplace_transform` and 27 integrals, 0.4 s. The fall-back for a
slow SymPy cell was not needed. Trial export with `export_public.py` into a scratch folder: 84 files,
14.0 MB, all checks passed (size, ASCII, no local path); the public folder was not touched.

**Not run.** The library tests at the browser pins (numpy 2.4.6, SciPy 1.18.0) in an environment of
their own: no such environment was at hand, and making one needs a download. The browser run above
executes the new functions at those pins inside the notebook (`laplace_numeric` in fourteen cells, the
table, `rk4`), which is a partial substitute. The independent reviews (step 6 of Section 5) and the real
export are left to the coordinator.

**Departures from the plan, and why.**
1. A fifth constant, `LAPLACE_HORIZON = 700.0` (the plan lists four). `QUAD_LIMIT` is reused.
2. How `laplace_numeric` treats the infinite interval (the plan says only "by quad on [0, infinity)").
   Three things were needed to make it safe: (a) for |s| > 1 the integral is computed in u = |s| t, so
   that the weight is seen on its own scale (without it L{t} at s = 1e5 came out as 1e-98 instead of
   1e-10); (b) beyond |s| t = 700 the weight underflows while f may overflow (e^(2t) at s = 3 gave
   0 times infinity), so the integrand is taken as 0 there and the transform is accepted only if
   T |e^(-sT) f(T)| is negligible at T = 700 / |s|; (c) the accepted error is rtol max(1, |value|), so a
   transform equal to 0 is found. The price, stated in the docstring: near the edge of the range the
   function refuses (e^(ct) needs s about 3% above c; 1 and t^n need s above about 1e-3; sin(omega t)
   needs s above about omega / 50), and e^(t^2) is refused only for s below about 25 (above that no
   computation in double precision can see its growth before the weight underflows).
3. The message of a refused integral is one sentence that names both causes ("does not converge ..., or
   too slowly to be computed: s is not inside the range of the transform (or is too near to its edge),
   or f(t) has no transform"), with the time at which f stopped being finite when that is the reason.
4. `laplace_table`: the names of the rows are "1", "t", "t^2", "t^n", "e^(-at)", "sin", "cos", "sinh",
   "cosh"; the two texts keep Lebl's letters (a, omega, n), and the notebook prints the constants above
   the table; Lebl's row t^3 is the row t^n with the default n = 3; `time` and `transform` also work on
   arrays; n must be an int (3.0 is refused, as `trajectories` does for its steps) and n! must be a
   number of the computer.
5. Section 3 checks all nine rows against `sp.laplace_transform` in one loop (the plan: "once"); it is
   one cell, 0.4 s in the browser. `sp.inverse_laplace_transform` is shown once (Example 6.1.7).
6. Slider 2 computes q(t) with `ol.rk4` (400 steps), not with a closed form: at R = 6 ohm the root -1 of
   the source is also a root of the circuit and the closed form changes shape; `ol.forced_response`
   would cost one integral per plotted point. `ol.rk4` is asserted against the formula at R = 2 ohm
   (1e-6 coulomb).
7. `ol.forced_response`, the third answer of Sections 5 and 6, is compared at 41 times, not on the grid
   of the figures (time in the browser).
8. Exercise 1(c) is answered with the number of the right form (1, 2 or 3) and checked with `ol.check`;
   its expected value is computed (the form for which SymPy finds the constants). Exercise 1(a) has two
   answers (the formula and the edge of its range). Exercise 1(d) asks for "the rest" after s^2 X(s),
   because `check_formula` compares formulas in one letter.
9. Exercise 2(a) is cited as Lebl's Exercise 6.1.1 "in the text of Section 6.1" (verify the table),
   which is what the plan describes; the Exercise 6.1.1 of the list at the end of the chapter is
   another one. Exercise 2(d), the numbers of Lebl's Exercise 6.2.4: m = 1 kg, c = 2 kg/s, k = 10 N/m,
   x(0) = 2 cm, x'(0) = 1 cm/s, answer e^(-t) (2 cos 3t + sin 3t) cm. Exercise 4 gives as first step
   x' + x = e^(-t), x(0) = 0.
10. Letters and units: the unit of t is written "seconds" and that of q "coulombs", to keep the letters
    s and C free; the constants of the circuit's partial fractions are A, B, D (C is the capacitance);
    in the procedure of Section 4 a quadratic is (s + p)^2 + omega^2 (p as in notebook 07) and a
    repeated factor has A_1, A_2. The source's convention is stated (a positive E pushes charge onto the
    capacitor).
11. References of the notebook: notebooks 04, 06, 07, 14, 15 (those it uses: Runge-Kutta, the three
    cases, critical damping, undetermined coefficients, the circuit); 13, 17 and 18 are not cited.
    Videos R03 and R04 were confirmed against the list of the videos.
12. No time is printed by the notebook (the stored outputs must be the same at every build); the times
    in the browser are those above.
13. The three index-like documents: the row of `make_00.py` shows the defining integral; the week is
    "Week 12 - The Laplace transform (Sessions 23-24)" (Week 11 has no row); the READMEs give 798 tests,
    24 notebooks, the row 19 and the last browser run.

**Lebl's text.** The two slips of Decision 6 are written right, with the parenthesis. Two more were
seen in the text copy and are not used by the notebook: Example 6.1.2 prints the antiderivative with
"-s(s + a)" under it (it is -(s + a)), and the existence proof ends with M/(c - s) (it is M/(s - c)).
Claims that were recomputed and are right: Examples 6.1.1 to 6.1.7 and 6.2.1, Table 6.1.1, Table
6.2.1, Example 6.2.4, and the printed answers of Exercises 6.1.12 to 6.1.15.

**Defects left, and points for the reviewers.**
* `laplace_numeric` is a numerical integral: its limits are those of departure 2. They are documented
  and tested, but a reviewer should judge whether the refusal near the edge is the right choice.
* The rows for the reserved-letter table of the style sheet (Section 4 of this plan) are not written:
  they go with the deck of Session 23.
* The widget state stores the two slider pictures a second time, as in notebooks 17 and 18 (0.09 MB of
  the 0.51 MB here).
* The sentence "a positive E pushes charge onto the capacitor" and the sign of I = q' are the
  notebook's own statement of the convention of notebook 15; to be checked against the deck.

### Corrections after the reviews (2026-11-23)

Three reviews (`reviews_nb/math_review_nb19.md`: 3 ERROR, 9 IMPRECISE, 5 COSMETIC;
`reviews_nb/clarity_review_nb19.md`: 0 BLOCKING, 13 IMPORTANT, 17 MINOR; `reviews_nb/code_review_nb19.md`:
3 BUG, 6 RISK, 14 STYLE) and the coordinator's decisions A to F. One line per finding. What Section 8
says above about slider 2, `ol.rk4`, Figure 5, the signatures of the two functions and the counts is
superseded by this part.

**Mathematics review**
* E1 slider 2 (decision A): done. The slider shows the discharge, q'' + R q' + 5 q = 0, q(0) = 1 coulomb,
  q'(0) = 0, by `ol.free_response`; one sentence says that the source is switched off and that the roots
  of the circuit do not depend on it. The lowest value of q is in the title of the right panel
  (`lowest_charge`: for complex roots q(pi / omega_1) = -e^(-p pi / omega_1), the first place where
  q' = 0; for real roots 0, since q = (r1 e^(r2 t) - r2 e^(r1 t)) / (r1 - r2) > 0 for r2 < r1 < 0, and
  (1 + p t) e^(-p t) > 0 at the critical resistance). Asserted in the notebook for the 76 positions of
  the slider on a grid of 40 s: below sqrt(20) the charge goes below 0 and its lowest grid value agrees
  with the formula; from sqrt(20) on it stays positive. New question and answer as decided; the answer
  gives the lowest values (-0.70 at 0.5 ohm, -0.21 at 2, -0.058 at 3, -0.0019 at 4, -3.1e-8 at 4.4, 0
  from 4.5) and says that the eye stops seeing the oscillation near 3.5 ohm. The coincidence at
  R = 6 ohm is Exercise 3(vi), with its answer q = (3/4 + 2t) e^(-t) + (1/4) e^(-5t), and one
  parenthesis in "Roots and shapes"; it is tested in the library and against SymPy. Section 1, item 6
  of this plan carries a dated note.
* E2, E3, I4, I5 `laplace_numeric` (decision B): done, see "Library" below.
* I6 why e^(t^2) has no transform: done (the product e^(t(t - s)) grows without bound for every s).
* I7 "the real part, with its sign changed, is the rate of decay (1 per second) ...": done.
* I8 "a numerator Bs + C (of degree at most 1)", a quadratic twice does not occur: done.
* I9 the hypothesis on g' in the rule for g'': done.
* I10 "This is general": done with the new answer of slider 1 ("here f(t) = t is positive"); the limit
  F(s) -> 0 is stated with exponential order in Section 3.
* I11 "the smallest number c ... for every s > c": done (the checker's label is "Exercise 1a, smallest c").
* I12 exercise numbers: done, one sentence in Section 1 (Exercise 6.1.12 is cited before Section 8).
* C13, C14 invalid constants, s = True: done (decision C and S6).
* C15 Figure 2: the two curves start just after the edge of their range: done.
* C16 the table of the text shows Lebl's rows C and t^3; the sentence under it says how
  `ol.laplace_table()` holds them: done.
* C17 "(for a > 0)": done.

**Clarity review** (numbers of the review)
* 1 the antiderivative of e^(-st) and "s is a fixed number": done. 2 the integration by parts of L{t}
  with u', v, the sign and the meaning of a bracket with infinity: done. 3 the reason of s > -a: done.
* 4 the SymPy letters: done differently. They are in a two-line cell of their own right after the setup
  cell, with the proposed comment, because the setup cell must stay that of notebook 18.
* 5 S_ONE: relative comparison and the comment "(try 0.1 or -1; not 0 ...)": done (tested with -1 and 0.1).
* 6 the hard-coded last assert removed, the label built from DECAY, "# try: ..." added: done (tested).
* 7 the new question of slider 1 and its answer: done. 8 "checking code" on the table cell and on the
  cell of undetermined coefficients, the comment of S_CHECKS, the sentence under the table: done.
* 9, 10 the cell of the range of s prints a sentence in both cases (no AssertionError), and the text
  after it explains the two "no value" lines and says that `ol.laplace_numeric` is a check, not a
  proof: done (tested with S_INSIDE = 40, S_OUTSIDE = 2.5). The comment says "keep S_INSIDE below 30".
* 11 exponential order in sentences of its own with examples: done. 12 the recipe for completing the
  square and the last sentence of Example 6.1.7: done. 13 the table of the three shifted rows, and the
  letter a (not p) in the procedure: done. 14 the last line of the shifting property and "(for a > 0)":
  done. 15 the heading of the procedure and the pointer to Section 6, step 4: done. 16 the comment in
  `agrees`: done.
* 17 the proof of the derivative rule with u', v, the bracket and the substitution for g'': done.
  18 g = 1 + sin 2t (g(0) = 1) and the "try another g" comment: done (tested with e^(-t) + t^2).
* 19 "X(s) is the new unknown", step 1 written by linearity, the two misprints of the book moved to one
  line after step 4: done. 20 the three missing lines of algebra (multiplied out; times s + 1; the
  choice s = -1): done. 21 the note on SymPy's single fraction in the print of step 2: done.
  22 the letters of notebook 14 (y(x), y_c, y_p): done.
* 23 q/C = 5q, the terms without Q, the line before q(t): done. 24 see E1. 25, 29 the static figure of
  the roots is cut (the print of the roots and the two asserts stay). Note: with the discharge the
  slider's first frame shows the two roots of the circuit, not the root -1 of the source, so the three
  roots together are now only in the text and in the printed line; this was accepted to keep the cut.
* 26 the first comment of the circuit cell: done. 27 the SymPy command of Exercise 3(iii) with
  `sp.simplify` of the difference: done. 28 the template of Exercise 3(i) (Ctrl + /, the three dots):
  done. 29 "In class / At home" at the top of Sections 2 and 5 (written by content, not by cell
  number), the two checking cells marked: done; nothing else was cut. 30 Exercises 3(iii) and 3(iv)
  after Session 24: done.

**Code review**
* B1, B2, R1 to R4, S1 to S6: `laplace_numeric` rewritten, see "Library". B3, R5, R6, S7 to S9:
  `laplace_table`, see "Library".
* S10 `import math` inside the two functions: left, as the review says (the top of the file cannot
  change). S11: the numbers of the tests are named, the three-in-one test is split, the test of rtol
  counts the calls of f, `math` is imported once at the top of the Laplace block of the tests (with a
  `noqa: E402`). S12: the missing tests are added (below). S13: a comment "(the digits of the text, on
  purpose ...)" at the first such assert of Sections 3 and 6. S14: one comment "np.isclose: equal to 5
  significant digits" where it is first used. The leftover "about four minutes" of
  `notebooks/README.md`: corrected.

**Library** (`odelib.py`; still additive for everything older than notebook 19)
* `laplace_numeric(f, s, *, rtol=LAPLACE_RTOL, breaks=())`. The reviewer's prototype was read and its
  design adopted with changes: (1) the pieces double from u = 2^-20 to 2^20 (u = max(|s|, 1) t), not
  from 1, so a fast feature at the start is seen (e^(-at) is right up to a of about 1e8; the review's
  e^(-1e4 t) and 1e6 e^(-1e6 t) are tests); this replaces the "coarse grid" guard of the mathematics
  review, which a trial showed cannot see what the pieces miss; (2) `breaks` become edges; (3) two
  computations, with the edges of the second multiplied by sqrt(2), and their difference counted as
  error; (4) quad is called with `full_output`, so no warning filter is touched, and the first
  doubtful piece stops the work; (5) the guard on the growth of log |f| before the last time T
  (`LAPLACE_GROWTH_TIMES`): e^(t^2) is now refused up to s = 36 (26 before); (6) a complex value, text
  or a list returned by f is refused; an error raised inside f becomes a ValueError with one sentence
  that quotes it (KeyError, NameError, ...; a ValueError or an ArithmeticError of f is reported as
  "does not converge ... f(t) is not defined / cannot be computed at t = ..."); (7) `rtol` must lie in
  `LAPLACE_RTOL_RANGE`; s = True is refused; -0.0 is written 0. New constants: `LAPLACE_RTOL_RANGE`,
  `LAPLACE_UNIT_RATE`, `LAPLACE_EDGES`, `LAPLACE_EDGE_RATIO`, `LAPLACE_SHIFTS`, `LAPLACE_GROWTH_TIMES`,
  `LAPLACE_ROW_NAMES`; private helpers `_one_real_number` and `_laplace_row`.
* The docstring states what is promised and what is not, with measured numbers: absolute accuracy;
  refusal within about 4% of the edge (1 and t^n are now computed down to s of about 1e-5, sin and cos
  down to omega / 200); at s = 0 only functions that die out like an exponential; a number can come
  back where no transform exists (e^(t^2) from s = 37, t^t from s = 8, tan t from s = 20, 1 / (t - 1)
  from s = 30); a pulse 0.01 wide at t = 5 is missed without `breaks`.
* Measured: the scan of the code review (40 000 sines and cosines, seed 7) gives 39 998 accepted, 2
  refused, 0 beyond the promise (largest error 0.96 of the allowed one); one call costs 2 to 3 ms
  (0.1 ms before), 3.25 ms on average in the scan, and at most 65 ms in the slowest accepted case
  (sin t at s = 1/200) and 34 ms in the slowest refused one that was tried (cos 5t at s = 0.01).
* `laplace_table(*, a, omega, n)`: the constants are given by name; n = int(n) after the check; every
  row works in float arithmetic on numbers, lists and arrays (a float for a number); `transform` raises
  a ValueError at or below `s_min` (also for arrays and NumPy scalars) and when the value is not a
  number of the computer; None, text, lists, complex numbers and booleans are refused with a sentence
  that names the argument; `s_min` is 0, not -0, for a = 0. Lebl's rows C and t^3 are shown in the
  notebook's table, not added to the data (C is by linearity, t^3 is t^n with n = 3).

**Tests** (library 598 -> 627 in env `ode`; 435 -> 460 and 10 skipped in `site/.venv`; notebooks
200 -> 206; `--collect-only` 798 -> 833). New in `test_odelib.py`: pulses and steps with and without
`breaks`; the documented limits (a number for e^(t^2) at s = 40, tan t at s = 20, t^t at s = 10; a
missed narrow pulse); fast features at the start; a seeded scan of 300 sines and cosines and
s = 14.138; complex, text and failing f; rtol outside its range; the warning filters and NumPy's
settings unchanged; integer arguments of the rows; `transform` at and below the edge; every invalid
constant with its `match`. Every comparison with `laplace_numeric` uses the promise of the function
(`promised`: rtol max(1 / max(|s|, 1), |F(s)|)), not a tighter number. `test_cross_validation.py`: the
circuit with R = 6 ohm and with the battery (Exercise 3(v)) against SymPy, `dsolve` and
`forced_response`. `test_notebooks_run.py`: six runs with the numbers a student is invited to change.

**The notebook after the corrections.** 41 cells (20 code), 6 stored figures and 2 sliders; 475 898
bytes, plain ASCII. Before the exercises: 4360 whitespace tokens, 3396 with each formula as one word,
3061 on the measure of plan 17 (2302 before): the lines that the clarity review asked for were added
and, by decision E, only the static figure of the roots was cut. The changed figures (Figure 2, the
static frame of slider 2) and slider 2 at 0.5, 3.5, 4.4, 4.47, 4.48, 4.5 and 8 ohm were looked at; the
title of the right panel was shortened to "lowest q = ... coulombs" after the first look (it did not
fit).

**Gates after the corrections (2026-11-23).**

| Gate | Result |
|---|---|
| Library, env `ode` | 627 passed |
| Library as in CI (`site/.venv`) | 460 passed, 10 skipped |
| Notebooks (`test_notebooks_run.py`, whole file) | 206 passed (4 min 6 s) |
| Lint (`ruff --no-cache`) | `make_19.py` clean; 0 findings on the added lines of `odelib.py`, the three test files, `make_00.py`, `build_site.py` |
| Site build, `site/test_build_site.py` | 24 notebooks, 135 MB; 5 passed |
| Browser (`check_site.py`) | 24 of 24 notebooks without errors; notebook 19: 20 cells, 8 plots, 2 sliders, **16 s** |
| Cell times of notebook 19 in the browser | setup 12.6 s; every other cell at most 0.4 s (the table, the inverse, Example 6.2.1, the circuit) |
| Trial export into the scratch folder | 84 files, 14.0 MB, all checks passed (size, ASCII, no local path); the public folder was not touched |
| READMEs against `--collect-only` | 833 in `README.md`; `site/README.md`: 24 notebooks, row 19 with 20 code cells, 8 plots, 2 sliders |

**Not done, and departures from a proposed fix.**
* Clarity 4: the SymPy letters are in their own cell, not in the setup cell (see above).
* Mathematics 3: the fixed-grid guard was not added; the pieces that reach down to u = 2^-20 do that
  work for a feature at the start, and `breaks` for one elsewhere. A narrow pulse away from the start
  and not named in `breaks` is still missed: documented and pinned by a test.
* Code S9 (a): no `laplace_row(name, ...)` function (the library may gain three names only); the
  dictionary idiom is in the docstring and is what the notebook and its exercise cells use.
* Code R2: the errors raised inside f are wrapped (decision B), not passed through.
* At s = 0, and within about 4% of an edge, convergent integrals are refused (decision B allows it):
  1 / (1 + t)^2 at s = 0 is a test of the refusal.
* The library tests at the browser pins were again not run in an environment of their own; the browser
  run executes the new code at those pins inside the notebook.
* The second review of the corrected notebook is left to the coordinator.

### Final pass after the re-check (2026-11-23)

The re-check (`reviews_nb/recheck_nb19.md`: 0 ERROR, 0 BLOCKING, 1 IMPORTANT, 11 MINOR; all the new
mathematics recomputed and right) was applied in full, with its exact fixes.

* 1 (IMPORTANT): the checking block of slider 2 (the critical resistance, the list of lowest values, the
  asserts on the 76 positions) is a code cell of its own after the cell that holds the caption and the
  answer; the cell of `plot_circuit()` ends with that call. The notebook has 42 cells, 21 code.
* 2: "Roots and shapes" is in the "In class" part of the line at the top of Section 5; the slider's
  text says how a root is drawn in the plane. 3: the reason of R^2 = 4L/C (the discriminant of
  L s^2 + R s + 1/C). 4: for real roots "it stays positive and falls towards 0 without reaching it (the
  title then shows 0)". 5: "another DECAY (0 or more)". 6: a second printed line explains SymPy's `I`
  and the 1 after each colon. 7: "keep S_INSIDE below 36". 8: Exercise 3(vi) says how to compare with
  `ol.forced_response` (`my_q`, about 1e-15).
* 9: `_one_real_number` refuses NumPy's booleans (s, rtol, a, omega); `breaks` is refused when it is
  text and each break goes through the same helper ([True], ["5"] refused). 10: `row.transform` refuses
  a complex number, text and booleans with a ValueError ("s must be a real number, a list or an array
  of real numbers"), with no ComplexWarning, and an s so small that the formula overflows gives its
  ValueError with no RuntimeWarning.
* 11: the docstring of `laplace_numeric` has the numbers as measured again (t^t: a number from s of
  about 6; tan t refused up to about 16; 1 / (t - 1) up to about 28; e^(t^2) up to about 36) and "up to
  0.1 second of computing"; `lambda t: t ^ 2` gets the hint "(a power is written t ** 2, not t ^ 2)".
* 12: the three one-sided limits are pinned on both sides (1 / (t - 1): refused at 3, a number at 30;
  t^t: refused at 4, a number at 10; sin t: right at s = 1/100, refused at 1/300).
* Length, the two moves: the middle of the paragraph "a check, not a proof" is Exercise 3(vii), with
  its answer; the two misprints of the book are one paragraph after the answers ("For those who read
  the book"). Before the exercises: 4316 whitespace tokens, 3364 with each formula as one word, 3033
  on the measure of plan 17 (3061 before this pass).

Tests: three more in `test_odelib.py` cases (NumPy booleans for s, a and omega) and new lines in five
existing tests. Library 627 -> 630 (env `ode`), 460 -> 463 and 10 skipped (`site/.venv`); notebooks 206;
`--collect-only` 833 -> 836.

**Gates after the final pass (2026-11-23).**

| Gate | Result |
|---|---|
| Library, env `ode` | 630 passed |
| Library as in CI (`site/.venv`) | 463 passed, 10 skipped |
| Notebooks (`test_notebooks_run.py`, whole file) | 206 passed (4 min 13 s) |
| Lint (`ruff --no-cache`) | `make_19.py` clean; 0 findings on the added lines of `odelib.py`, the three test files, `make_00.py`, `build_site.py` |
| Site build, `site/test_build_site.py` | 24 notebooks, 135 MB; 5 passed |
| Browser (`check_site.py`) | 24 of 24 notebooks without errors; notebook 19: 21 cells, 8 plots, 2 sliders, **16 s** |
| Trial export into the scratch folder | 84 files, 14.0 MB, all checks passed; the public folder was not touched |
| The notebook file | 477 341 bytes, plain ASCII |
| READMEs against `--collect-only` | 836 in `README.md`; `site/README.md`: row 19 with 21 code cells, 8 plots, 2 sliders |

Not done: the library tests at the browser pins in an environment of their own (as before). Nothing
else is left from the four reports.

### Review of the limits that remained (2026-11-23, at the instructor's request)

The instructor asked whether the three limits reported at the end of the final pass matter to a student
who uses the tools, and to solve any that do. Each was tested as a student would meet it (scripts in the
scratch folder `nb19_limits`).

1. **A number for a function with no transform, when s is large: not a problem in use.** e^(t^2) is
   refused for every s up to 36, tan t up to 17, 1/t always; these are far above the values of this
   notebook (s <= 5 in the sliders and the checks). The text of Section 3 and Exercise 3(vii) say that
   the function is a check, not a proof. No change.
2. **A pulse without `breaks`: a real problem, solved.** Before, a pulse 0.1 wide at t = 5 was refused
   with the sentence "does not converge", and a pulse 0.05 wide or narrower returned 0 with no error; a
   pulse on top of a smooth function was lost in the same way. Notebook 20 (switches, blows) would have
   met this at once. `laplace_numeric` now looks at f on 256 equal steps of every piece before
   integrating, takes a step on which f changes more than 8 times what it changes on a neighboring step
   as a jump, locates the jump by halving the step, and cuts the integral there, as if `breaks` had
   named it. Measured on 1050 seeded pulses at random places, with and without e^(-t) under them: every
   pulse wider than 1/200 of its distance from t = 0 is right and none is refused (before: right only
   above about 1/10, and often refused). Narrower pulses can still be missed (about one in four at
   1/500); `breaks` computes them. The docstring says so and a test pins both sides. Side effect: more
   functions that are infinite at some time are refused (1 / (t - 1) at every s tried).
   New constants: `LAPLACE_SCAN`, `LAPLACE_JUMP_RATIO`, `LAPLACE_MAX_JUMPS`. New test:
   `test_laplace_numeric_finds_switches_and_pulses_by_itself` (60 seeded pulses, a switched sine, and
   1 + sin t at s = 0.5, which an early version of the scan refused).
   The price: one call takes about 20 ms on this computer (4 ms before). In the browser the cell that
   checks the table (27 calls) takes 2.7 s (0.4 s before); every other cell at most 1.2 s.
3. **More names than the plan's three: not a problem.** The public functions are still three. The
   constants (now thirteen) are named numbers of the library, listed in its docstring as all the others;
   a student never types them. The two private helpers are not visible in the notebook.
4. **Checks skipped after the final pass: done now.** Per-cell times in the browser re-measured (above);
   the six stored figures and ten slider frames rendered and looked at: no defect. The notebook was
   regenerated, because a stored refusal message quotes a time that the scan changed (384 to 355, 32 to
   26.7); nothing else in its outputs changed.

Gates after this change: library 631 passed (env `ode`), 464 passed and 9 skipped as in CI; notebook
tests 206 passed; `--collect-only` 837 (`README.md` updated); ruff: no finding on the new lines; site
build 24 notebooks, `test_build_site.py` 5 passed; browser 24 of 24 without errors; trial export into
the scratch folder: 84 files, all checks passed; the public folder untouched. Still not run: the library
tests at the browser pins in an environment of their own.

### A later fix of `laplace_numeric` (2026-11-24, with notebook 20)

The reviews of notebook 20 found three defects in `laplace_numeric`; the instructor approved a fix, made in two
rounds and reviewed independently:
* a step with the value 1/2 at its jump (`np.heaviside(t - a, 0.5)`, SymPy's `Heaviside`) was refused as
  "does not converge", with or without `breaks`; it is now computed;
* a steep switch without a jump (a ramp) was often refused, and about one in five returned a wrong number with
  no error; about 99 in 100 are now right;
* an integer too large for a float (10**400) raised OverflowError instead of the documented ValueError.

Notebooks 19 and 20 give the same printed results and figures with the old and the new code. In an
old-against-new comparison of about 18 600 seeded calls, no call that was right became wrong; about 2 in 100
fast oscillations (sin or cos with s below omega / 20) that were computed are now refused, as the docstring
says. 16 tests were added; no existing test was changed.
