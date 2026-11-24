# Notebook 20 - Steps, impulses, and convolution: plan

MATH 331 - Differential Equations - Spring 2027 - phase E of the notebooks (19-22), the second notebook of
the phase

**Status (2026-11-24): built, reviewed, corrected, re-checked; reviewed by the instructor and exported to
the public folder on 2026-11-24 at the instructor's request (the instructor commits and pushes).** The eight
decisions of Section 6 were approved by the instructor as proposed; Section 8 says what was built and what the
two passes changed.

**Sessions 25 and 26, Unit 11** (lecture plan, Session 25: "Heaviside step functions; shift theorems;
discontinuous forcing"; theory exercise "Prove the second shift theorem"; practical exercise "Model a
switched circuit". Session 26: "Dirac delta; impulse response; convolution; transfer functions"; theory
exercise "Express the solution through the convolution theorem"; practical exercise "Model a hammer blow on
a spring-mass system"; Quiz 6). About **15 minutes in class in each session** (Session 25: Sections 2 to
4; Session 26: Sections 5 to 7), the rest as homework. Notebook 19 ends with "Where to go next: notebook
20, steps, impulses and convolution": this notebook keeps that promise.

The instructor's requests for this notebook (2026-11-24): extra caution with the existing code when the
new code is integrated; the mathematics double-checked; the new concepts verified as understandable for
inexperienced students. Sections 4 and 5 of this plan say how each is met.

**References** (every item found in the text of the book on 2026-11-24; the numbers of this plan were
recomputed with SymPy and SciPy the same day).
* Lebl, *Notes on Diffy Qs*, **Section 6.1**, Example 6.1.4 (the Heaviside function, u(t) = 1 for t >= 0,
  and L{u(t - a)} = e^(-as)/s), and the remark after Theorem 6.1.3 that the value at the jump does not
  change the transform.
* Lebl, **Section 6.2**: "Using the Heaviside function" (the signal u(t - pi) sin t; the window
  u(t - 1) - u(t - 2); the function t on [0, 1], -t + 2 on [1, 2)); the second shifting property (6.2.2),
  L{f(t - a) u(t - a)} = e^(-as) L{f(t)}; **Example 6.2.2** (x'' + x = f(t), f = 1 for 1 <= t < 5, the
  rocket fired for 4 seconds; x = (1 - cos(t - 1)) u(t - 1) - (1 - cos(t - 5)) u(t - 5)); transfer
  functions and Example 6.2.3 (H(s) = 1/(s^2 + omega_0^2), the constant input).
* Lebl, **Section 6.3, Convolution**: the definition (6.3.1); Examples 6.3.1 (e^t * t = e^t - t - 1), 6.3.2
  (sin(omega t) * cos(omega t) = (1/2) t sin(omega t)), 6.3.3 (the inverse of 1/((s + 1) s^2) as
  e^(-t) + t - 1); the properties (6.3.3); Theorem 6.3.1; Example 6.3.4 (x'' + omega_0^2 x = f(t) from rest
  as a convolution with sin(omega_0 t)/omega_0, and resonance).
* Lebl, **Section 6.4, Dirac delta and impulse response**: the rectangular pulse and its transform; the
  unit pulse of width b and its transform (1 - e^(-bs))/(bs); the delta function as "an object that does
  not really make sense unless we integrate it"; the sifting formula; L{delta(t - a)} = e^(-as); the Note
  on the derivative of the Heaviside function; Example 6.4.2 (the impulse response sin(omega_0 t)/omega_0);
  "the solution for an arbitrary input is given as convolution with the impulse response".
* Lebl, **Exercises 6.E**: 6.2.2, 6.2.6, 6.2.9, 6.2.10 (the amplitude after the rocket stops), 6.2.12,
  6.2.13 and 6.2.14 (answers printed); 6.3.1, 6.3.2, 6.3.8, 6.3.10 to 6.3.12 (answers printed); 6.4.1 to
  6.4.3, 6.4.5, 6.4.9 to 6.4.11 (answers printed for 6.4.9 to 6.4.11).
* Notebooks 19 (the table, the four steps, `laplace_numeric` with `breaks`), 14 and 15 (`forced_response`,
  resonance, steady state), 07 (free response, the three kinds of damping).

**Left out** (Decision 7): Volterra integral equations (Lebl 6.3.3), three-point bending of a beam (6.4.4),
partial differential equations (6.5).

**Purpose.** Two ideas, one per session.
1. **Session 25: a force that switches on or off is written with the step function, and a switch at
   time a becomes a factor e^(-as) on the s side.** The four steps of notebook 19 then work unchanged.
2. **Session 26: a very short, very strong push (a blow) is the limit of pulses, the delta function; the
   response to it, the impulse response, gives the response to every other input by one integral, the
   convolution.**

---

## 1. Contents

1. **Before you start**: what is needed from notebook 19; the conventions box (Section 4 of this plan).
2. **The step function** (Session 25).
   * u(t - a): 0 before the time a, 1 from a on (Lebl: u(0) = 1); `ol.step(t, a)`; figure 1.
   * Building functions with switches: a window u(t - 1) - u(t - 2); a signal that starts late; Lebl's
     tent (t, then -t + 2, then 0), each drawn beside its formula.
   * The transform e^(-as)/s, by the definition (the integral starts at a), checked with
     `ol.laplace_numeric(..., breaks=[a])`.
3. **The second shifting property** (Session 25).
   * In words first: delay a function by a, and its transform is multiplied by e^(-as). Figure 2: f(t) and
     f(t - a) u(t - a). Checked numerically for sin t delayed by 2.
   * The usual mistake, shown: t u(t - 1) is not a delayed t; write t = (t - 1) + 1 first, which gives
     e^(-s) (1/s^2 + 1/s). Both versions are computed and compared with the integral.
   * Reading it from right to left: e^(-as) F(s) is f(t - a) u(t - a).
4. **A force that switches on and off** (Session 25): Lebl's Example 6.2.2, the rocket, with every step
   (the force as two steps; the four steps; 1/(s (s^2 + 1)) by partial fractions; the two delayed pieces).
   * Figure 3: the force and the response. The mass keeps oscillating after the rocket stops.
   * Check: `ol.forced_response` with the switched force.
   * **Slider 1** (question first: "is there a firing time after which the mass stays at rest?", Lebl's
     Exercise 6.2.10): the firing time d from 0.5 to 8; the response, and the amplitude after the stop,
     2 |sin(d/2)|; the answer is d = 2 pi (and its multiples), where the two delayed cosines cancel.
5. **From a short pulse to a blow** (Session 26).
   * A unit pulse of width b (height 1/b, area 1); its transform (1 - e^(-bs))/(bs), which tends to 1 as b
     shrinks (a table of values).
   * **Slider 2** (question first: "the pulse gets shorter and taller, with the same area: does the
     response grow without bound?"): b from 2 down to 0.05; left, the pulse; right, the response of
     x'' + x to it, which settles on sin t.
   * The delta function: not a function with values, but what the pulses do under an integral (Lebl's
     sentence quoted); the sifting formula, checked numerically with narrow pulses; L{delta(t - a)} =
     e^(-as).
   * What a blow does to a mass: m x'' + c x' + k x = P delta(t), from rest. The impulse P (in newton
     seconds) changes the velocity at once by P/m and the position not at all: the solution is the free
     motion that starts with x(0) = 0 and x'(0) = P/m (checked with `ol.free_response`).
     The notebook's own hammer: m = 1 kg, c = 2 kg/second, k = 5 N/m, P = 3 newton seconds;
     X(s) = 3/(s^2 + 2s + 5), x = (3/2) e^(-t) sin 2t. Figure 4.
   * The impulse response: the response to P = 1, `ol.impulse_response(t, a, b, c)`; Lebl's Example 6.4.2.
6. **Convolution** (Session 26).
   * The definition, read as "a weighted sum of delayed copies": figure 5 shows f(tau), g(t - tau), and the
     shaded product whose area is (f * g)(t), for one t.
   * `ol.convolution(f, g, t)`; Lebl's Examples 6.3.1 and 6.3.3 checked.
   * The theorem, numerically: L{f * g} against F(s) G(s), and the contrast with notebook 19 (the
     transform of the ordinary product is not the product).
   * **The response to any input is the input convolved with the impulse response**: X(s) = H(s) F(s)
     with H(s) = 1/(a s^2 + b s + c), Lebl's transfer function; checked against `ol.forced_response` for
     the hammer's system with a force that fades. Why: the input as a row of small blows, each answered
     by a delayed copy of the impulse response (figure 6).
   * Resonance seen again (Lebl's Example 6.3.4): the convolution gives (t / (2 omega_0)) sin(omega_0 t).
7. **Where to go next**: notebook 21 (series). What was left out, with the sections of the book.
8. **Exercises**, **answers**.

Target length: that of notebook 19 (about 3000 words before the exercises). Fall-back if it runs long: the
resonance item moves to an Explore exercise.

---

## 2. Exercises (the four steps)

Each exercise is marked with its session.

1. **Predict** (no code). (a) The value of u(t - 3) - u(t - 5) at t = 4 and at t = 6. (b) True or false:
   L{t^2 u(t - 1)} = e^(-s) 2/s^3. (c) After Session 26: the velocity of a mass of 2 kg at rest, right
   after a blow of 6 newton seconds. (d) After Session 26: L{delta(t - 2)} at s = 0.
2. **Compute**. (a) *Lebl, Exercise 6.2.13*: write a function given by cases with step functions (answer
   printed). (b) *Lebl, Exercise 6.2.6*: x'' + x = u(t - 1) from rest. (c) After Session 26: *Lebl,
   Exercise 6.4.3*: x'' + 4x = 4 delta(t - 1) from rest. (d) After Session 26: *Lebl, Exercise 6.3.10*:
   cos t * e^(-t) (answer printed).
3. **Explore** (code). (i) *Lebl, Exercise 6.2.14* checked with `ol.forced_response`. (ii) A rocket fired
   twice. (iii) After Session 26: *Lebl, Exercises 6.4.9 and 6.4.10* (the impulse responses of x'' and of
   x' + a x, answers printed) with `ol.impulse_response` where it applies. (iv) *Lebl, Exercise 6.3.12*
   by `ol.convolution`. (v) Two blows: when does the second blow stop the mass?
4. **Explain** (two or three sentences). Why does the initial value x'(0) = 0 seem to be contradicted by
   the solution of the hammer problem, and why is it not?

Answers are checked with the existing `ol.check`, `ol.check_formula` and `ol.check_choice`.

---

## 3. New in `odelib.py` (with tests)

A new section, "Steps, impulses and convolution (notebook 20)", after the section of notebook 19. NumPy
only at the top; SciPy inside the function that integrates. Type hints, docstrings citing Lebl, the
notebook, and the conventions. Arguments in the order of the existing functions (the times first).

* `step(t: ArrayLike, a: float = 0.0) -> np.ndarray`: Lebl's unit step u(t - a): 0 for t < a, 1 for
  t >= a (the value at the jump is 1, as Lebl defines it). ValueError for a that is not one finite real
  number.
* `convolution(f, g, t: ArrayLike, *, breaks: Sequence[float] = ()) -> np.ndarray`: (f * g)(t), the
  integral from 0 to t of f(tau) g(t - tau) d tau (Lebl 6.3.1), by `scipy.integrate.quad` for each t;
  `breaks` names the times at which f or g jumps (the integral is cut there and at t minus them).
  ValueError for a negative t, for f or g that is not a function of one number, for a value that is not a
  finite real number, and when quad is in doubt.
* `impulse_response(t: ArrayLike, a: float, b: float, c: float) -> np.ndarray`: the solution of
  a y'' + b y' + c y = delta(t) from rest: 0 for t < 0 and, for t >= 0, the free motion with y(0) = 0 and
  y'(0) = 1/a. **A thin wrapper of the existing `constant_coefficient_solution`**: nothing is recomputed,
  and that function is not changed. ValueError for a = 0.
* Named constants for the tolerances, defined once above their function.
* Not added (Decision 3): a pulse function (it is `step(t, a) - step(t, b)` in one line of the notebook)
  and a transfer function (one line, 1/(a s^2 + b s + c)).

**Tests** (in `tests/test_odelib.py`; with SymPy in `tests/test_cross_validation.py` behind `importorskip`):
* against the book: Examples 6.1.4, 6.2.2, 6.3.1 to 6.3.4, 6.4.2 and the printed answers of Exercises
  6.2.13, 6.2.14, 6.3.10 to 6.3.12, 6.4.9 to 6.4.11;
* against an independent method: `forced_response` (variation of parameters) for the convolution with the
  impulse response; `solve_ivp` for a narrow pulse against the impulse response; SymPy for closed forms;
  seeded random cases (SEED = 331);
* against invalid input: `pytest.raises(ValueError)` with `match`;
* the properties: f * g = g * f; the value 1 at the jump; shape and type of the results.

---

## 4. Conventions, checks, and care with the existing code

As for notebooks 14 to 19: `assert` for every reference value; one sentence under every plot, written
after looking at the figure; line styles as well as colors; a question before every slider and its answer
after; named constants; plain ASCII; generated by `plans/make_20.py` with `nbbuild.py`; the setup cell of
notebook 19 with `import sys`; `SETUP_NAMES` unchanged.

* **Letters** (rows for STYLE_SHEET Section 4, with the decks of Sessions 25 and 26; Decision 2):
  * u(t - a): the unit step (Lebl). It is not the u_1, u_2 of variation of parameters (Session 14) nor the
    unknowns of a system written to first order (Session 17): said once.
  * delta(t - a): the delta function; tau: the variable inside a convolution; (f * g)(t): the convolution,
    with a star that is not a product (said once, where it is defined).
  * b: the width of a pulse in Section 5 (Lebl); the coefficients of the equation are then written m, c, k
    as in notebook 07, so that b has one meaning in that section.
  * P: the impulse of a blow, in newton seconds; H(s): the transfer function.
* **Sign and unit conventions** in a box where each is used: x(t) in meters, positive when the spring is
  stretched; a positive force stretches it; t in seconds; s in 1/seconds.
* **The value at a jump**: u(a) = 1 (Lebl). Stated once, with Lebl's remark that it does not change any
  transform or any solution.
* **How the mathematics is double-checked** (the instructor's request):
  1. every number of this plan was recomputed with SymPy and SciPy before the plan was submitted (the
     convolutions, the rocket's response against `solve_ivp`, the amplitude 2 |sin(d/2)| with its zero at
     2 pi, the transform of t u(t - 1), the pulse limit, the hammer);
  2. in the notebook every closed form is asserted against an independent numerical method;
  3. an independent mathematics review recomputes everything again, and a fresh re-check recomputes every
     line added by the corrections.
* **How understandability is verified** (the instructor's request): a reading as the weakest student
  before and after the corrections, with these points named in advance as the hard ones: the meaning of
  u(t - a) as a switch; "delayed" against "cut off" (the mistake of Section 3); a function that is not a
  function (delta); the velocity that jumps although x'(0) = 0; the convolution integral, with its two
  variables t and tau.
* **Care with the existing code** (the instructor's request):
  1. baseline first: the whole test suite and the browser run, recorded before any change;
  2. additive only: no existing function, message, constant or test is edited; after the work, a check
     that every line of `odelib.py` that existed before is still there, unchanged (the diff shows
     additions only, apart from the list of names in the module docstring);
  3. `laplace_numeric` is used as it is; if this notebook needs a change in it, the change is proposed
     separately and not made silently;
  4. the four registries (index table, `READING_ORDER`, `EXERCISES`, `EXTREMES`) are extended, never
     reordered;
  5. every gate again after the work, including the tests of notebooks 01 to 19 and the browser run of
     all notebooks;
  6. the public folder is touched only by `site/export_public.py --update`, and only when the instructor
     says so; a trial export into a scratch folder comes first. No commit, no push.

---

## 5. Order of work

1. Baseline: library tests (two environments), notebook tests, `--collect-only`, browser run.
2. `odelib.py`: `step`, `convolution`, `impulse_response` and their constants; tests first; ruff on the
   new lines; the whole suite unchanged.
3. `plans/make_20.py`: Sections 1 to 4 with slider 1 (Session 25); every figure looked at.
4. Sections 5 to 7 with slider 2 (Session 26); word count; the fall-back if needed.
5. Exercises and answers; `EXERCISES`, `EXTREMES`, `NAME_20`.
6. Index row, `READING_ORDER` (Week 13), READMEs; site build and `site/test_build_site.py`;
   `check_site.py` with the cell times.
7. Independent reviews: mathematics, weakest student, code (including the additive-only check);
   corrections; fresh re-check; final pass; all gates again.
8. Trial export into a scratch folder. The real export waits for the instructor.

---

## 6. Decisions for the instructor

1. **One notebook for two sessions** (as 17, 18, 19): Sections 2 to 4 for Session 25, Sections 5 to 7 for
   Session 26, exercises marked by session.
2. **Letters**, as in Section 4, following Lebl: u(t - a), delta(t - a), tau, the star, H(s); P for the
   impulse of a blow; b for the width of a pulse, with m, c, k for the equation in that section.
3. **Three new library functions**: `step`, `convolution`, `impulse_response`. No pulse function and no
   transfer function (one line each in the notebook). The master plan had listed `transfer_function`;
   proposed: leave it out, because with s real it is a single division and adds nothing to check.
4. **The value at the jump is 1**, as Lebl defines it.
5. **The blow explained through the velocity**: an impulse P changes the velocity by P/m at once. This
   answers the beginner's question "how can x'(0) = 0 and the mass move?", and it gives a check with the
   existing `free_response`. The notebook's hammer (1 kg, 2 kg/second, 5 N/m, 3 newton seconds) is its own.
6. **Reserved for the decks.** Session 25's practical exercise is a switched circuit and Session 26's is
   a hammer blow: the decks will use the circuit and the door of Sessions 23 and 24, and other numbers
   for the hammer. The notebook uses Lebl's rocket (Example 6.2.2) for Session 25.
7. **Left out**: Volterra integral equations, the beam with a point load, partial differential
   equations; one line each in "Where to go next".
8. **Complex s is not used.** The link between the transfer function at s = i omega and the amplitude
   response of notebook 15 is one sentence in "Where to go next", not a computation.

---

## 7. Files

```
ode/ode/notebooks/
    20_Steps_Impulses_Convolution.ipynb          <- to build
    plans/20_Steps_Impulses_Convolution_plan.md  <- this document
    plans/make_20.py                             <- builds the notebook
    odelib.py                                    <- + step(), convolution(), impulse_response(), constants
    tests/test_odelib.py                         <- + their tests
    tests/test_cross_validation.py               <- + against SymPy
    tests/test_notebooks_run.py                  <- + NAME_20, the exercises and sliders of notebook 20
    00_Index.ipynb, plans/make_00.py             <- index row
    README.md                                    <- the list of notebooks
ode/ode/site/build_site.py                       <- READING_ORDER: Week 13 with 20
ode/ode/site/README.md                           <- counts and the table of tested notebooks
ode/ode/README.md                                <- the number of tests
```

Nothing in the public folder is touched by hand; nothing is committed or pushed.

---

## 8. As built (2026-11-24)

**The notebook.** `20_Steps_Impulses_Convolution.ipynb`, 48 cells (23 code), built by `plans/make_20.py`;
577 650 bytes (the limit of `export_public.py` is 2 000 000), plain ASCII, no local path stored. Before the
exercises: **4329 words** as whitespace tokens, **3418** with each formula as one word, **3187** with each
formula as one word and only tokens that contain a letter or a digit (the measure that gives 3033 for
notebook 19). The fall-back of Section 1 was applied (resonance is Exercise 3(vi)); the notebook is still
about 5% longer than notebook 19.

**Sections.** 1 Before you start (what is needed from notebooks 19 and 07; the conventions: u(t - a) and
what it is not, the value 1 at the jump with Lebl's remark, delta, tau, the star, h and H); 2 The step
function (values by hand and with `ol.step`, the figure with the value at the jump, three recipes for
building functions with their figure, the transform e^(-as)/s by the integral that starts at a, checked
with `ol.laplace_numeric(..., breaks=[a])`); 3 The second shifting property (in words first, the delayed
function with numbers, the proof with the substitution v = t - a, sin t delayed by 2, "cut off is not
delayed" on t u(t - 1) with both formulas against the integral, the figure, the three steps from right to
left); 4 A force that switches on and off (Lebl's Example 6.2.2 by the four steps, with the partial
fractions of 1/(s (s^2 + 1)) and the two delayed pieces, then read by cases; three checks: SymPy's own
inverse, the answer transformed forward, `ol.forced_response`; the figure; slider 1); 5 From a short pulse
to a blow (the unit pulse, its transform with a table, slider 2, the delta function with Lebl's sentence,
the sifting formula with a table of mean values, the transform e^(-as), the hammer in words and by the
four steps, its figure, "Where is x'(0) = 0?", the impulse response and `ol.impulse_response`);
6 Convolution (how to read t and tau, 1 * 1 = t, Lebl's Example 6.3.3 by hand, `ol.convolution`, the
figure of the product and its area at t = 1 and t = 3, the theorem in numbers against the ordinary
product, the response as input * impulse response with the transfer function, why without transforms, the
figure of the force as 8 blows); 7 Where to go next; 8 Exercises; 9 Answers.

**Figures: 9 stored** (u(t) and u(t - 2); the window, the late signal and Lebl's function by cases; delayed
against cut off; the rocket's force and response; slider 1 static; slider 2 static; the hammer's
displacement and velocity; the convolution as an area at two times; the force as 8 blows and their
answers). Every figure was exported to PNG and looked at, and the two slider functions were also drawn at
d = 0.5, 3.1, 6.3, 8 and b = 2, 0.5, 0.05; the sentence under each plot was written or corrected after
that, with its numbers recomputed (x = 2 at t = 1 + pi and 1.65 at the stop, then between -1.82 and 1.82;
0.771 m at t = 0.554 seconds for the hammer; areas 0.368 and 2.050; the fading force: 1.06 m at
t = 1.11 seconds, 0.37 m with 8 blows and 0.004 m with 80). Changed after the first look: the legends of
three figures (moved or given room), the caption of the rocket ("oscillates between 0 and 2" was not what
the figure shows in four seconds), the ticks of the convolution figure, the two numbers of the last
caption, which had been guessed. **Sliders: 2** (`plot_rocket(d)`, `plot_pulse(b)`), each with its
question first, a static figure, its answer, and a checking cell after the answer that asserts the
answer's formula at every position of the slider.

**Added to `odelib.py`** (a new section "Steps, impulses and convolution (notebook 20; Lebl, Sections 6.2
to 6.4)" after the section of notebook 19; 231 lines, all insertions: 5 in the module docstring, 226 of
code; `SETUP_NAMES` unchanged; numpy-only at import, checked):
* `step(t: ArrayLike, a: float = STEP_START) -> np.ndarray`
* `convolution(f, g, t: ArrayLike, *, breaks: Sequence[float] = ()) -> np.ndarray`
* `impulse_response(t: ArrayLike, a: float, b: float, c: float) -> np.ndarray`, a thin wrapper of
  `constant_coefficient_solution`, which is not changed
* constants `STEP_START` (0.0), `CONVOLUTION_RTOL` (1e-8), `CONVOLUTION_PIECES` ((4, 5)),
  `CONVOLUTION_SCAN` (256), `CONVOLUTION_WINDOW` (4); `QUAD_LIMIT`, `LAPLACE_JUMP_RATIO` and
  `LAPLACE_MAX_JUMPS` are reused; one private helper, `_real_array`. `laplace_numeric` and every other
  existing function are untouched, and no change in them is proposed.

**Tests added: 143.**
* `test_odelib.py`, 80: the step (Example 6.1.4, the value 1 at the jump for both zeros of the computer,
  shapes, the functions of Lebl's Section 6.2, Exercise 6.2.13); its transform with and without `breaks`;
  the second shifting property on seeded delays and rows of the table; t u(t - 1); the rocket (the printed
  answer by convolution with and without `breaks`, both orders, `forced_response`, the transform) and,
  against `solve_ivp`, the amplitude 2 |sin(d/2)| for five firing times; Examples 6.3.1 to 6.3.4; the
  printed answers of Exercises 6.2.14, 6.3.10 to 6.3.12 and 6.4.9 to 6.4.11; Example 6.4.2; the hammer;
  the impulse response as the limit of a pulse 1e-4 seconds long by `solve_ivp`, its transform against
  the transfer function, its slope 1/a at 0; convolution with the impulse response against
  `forced_response` on seeded equations; f * g = g * f and (c f) * g = c (f * g) on seeded functions;
  shapes, types and t = 0; jumps with and without `breaks`, two windows, a jump on an oscillation; 60
  seeded pulses found without `breaks`; the documented limit (a pulse 0.01 wide in a convolution up to
  t = 40 is missed without `breaks` and right with them); 52 kinds of invalid input, each with `match`;
  the warning filters and NumPy's settings unchanged; `SETUP_NAMES`.
* `test_cross_validation.py`, 25 (behind `importorskip("sympy")`): ten convolutions against SymPy's
  integral of the definition, with Theorem 6.3.1 checked symbolically on each; seeded functions; the
  second shifting property for three delays; L{t u(t - 1)} and L{t^2 u(t - 1)}; the rocket (partial
  fractions, SymPy's inverse, the identity after the stop) and the amplitude for five firing times by
  `dsolve`; the unit pulse and its two limits; the hammer (inverse transform, free motion by `dsolve`,
  the largest displacement); the impulse response against `dsolve` and the inverse of H(s) on seeded
  equations; the exercises of the notebook (6.2.6, 6.4.3, 6.2.14, the two blows, the two firings).
* `test_notebooks_run.py`, 38: the notebook as it is, with right answers, with wrong answers, the answer
  markers; 23 particular answers (other accepted forms, among them SymPy's own Heaviside and a Piecewise;
  the step forgotten; cut off for delayed; X(s) typed for x(t); text in quotes); the sections run again
  after the exercises; 9 runs with the numbers that the comments invite a student to change; 13 slider
  values with RuntimeWarning as an error.

**Counts of tests, before and after.**

| Gate | Before (2026-11-24) | After |
|---|---|---|
| Library, env `ode` | 631 passed (47 s) | 736 passed (102 s) |
| Library as in CI (`site/.venv`, no SymPy, no matplotlib) | 464 passed, 10 skipped | 544 passed, 10 skipped |
| Notebooks (`test_notebooks_run.py`, whole file) | 206 passed (3 min 46 s) | 244 passed (5 min 28 s) |
| `--collect-only` | 837 | 980 |

**The other gates (2026-11-24).** Lint: `ruff check --no-cache` clean on `make_20.py`, and no finding on
the lines added to `odelib.py`, the three test files and `make_00.py`. The additive-only proof (a script
with difflib on the copies made before the work): for each of `odelib.py`, `test_odelib.py`,
`test_cross_validation.py` and `test_notebooks_run.py` the opcodes are "equal" and "insert" only, nothing
deleted or replaced, every old line still there in order, the line endings as they were. Site build:
25 notebooks, 136 MB; `site/test_build_site.py`: 5 passed. Browser (`check_site.py`, headless Edge):
25 of 25 notebooks without errors, 15 to 24 s each, about 8 min in all. (In a first full run the pages of
notebooks 07E and 18 did not open within 120 s: a time-out of the page, not an error of a cell, in two
notebooks that this work does not touch. The two passed when run again, and the whole run was then
repeated: the result above.) Notebook 20 in the browser: 23 cells, 11 plots, 2 sliders, **24 s**; each
cell was timed: the setup cell (the start of Python and the imports) 14.3 s, every other cell at most
2.2 s (the cell of the rocket: SymPy's inverse, three transforms, `ol.forced_response` across the two
jumps, the figure; then 1.7 s for the cell of Section 3, 1.4 s for the table of the pulses, 1.2 s for the
hammer). Trial export with `export_public.py` into a new, empty folder of the scratch area: 87 files,
14.8 MB, all checks passed (only files of the allow-list, no large file, no mention of a file that was
left out, which includes local paths, and plain ASCII in the code); the public folder was not touched.
Nothing was committed or pushed.

**Not run.** The library tests at the browser pins (numpy 2.4.6, SciPy 1.18.0) in an environment of their
own, as for notebook 19; the browser run executes the new functions at those pins inside the notebook.
The independent reviews (step 7 of Section 5) and the real export are left to the coordinator.

**Departures from the plan, and why.**
1. The fall-back was applied: with resonance in Section 6 the text had 3171 words on the measure above.
   Lebl's Example 6.3.4 is Exercise 3(vi), with its answer, and a pointer under the last figure; it is
   tested in the library and against SymPy.
2. A step for formulas, `u`, is defined in the notebook (the cell after the setup cell, and again in the
   answer cell of Exercise 2): `sp.Heaviside(x, 1)`, Lebl's value at the jump. With it the answers of
   Exercises 2(a) to 2(c) are typed as on paper, `(t - 1) * u(t - 1)`, and checked with
   `ol.check_formula` at 7 times, none at a jump. It is not a library function. `ol.step` given a SymPy
   letter answers with one sentence that names `u`.
3. `convolution` does more than "quad for each t". The trial of the hard cases showed that at a jump that
   is not named quad's own error estimate is too small by four orders of magnitude (errors of 1e-4 with
   an estimate of 1e-9). So (a) f and g are first looked at on 256 equal steps of [0, t], a step whose
   change is more than 8 times the median change of the 4 steps on each side is taken as a jump, located
   by halving, and the integral is cut there; (b) the integral is computed twice, on 4 and on 5 equal
   pieces, and refused when the two sums or quad's estimates differ by more than 1e-8 max(1, |value|).
   The rule of the first look is not that of `laplace_numeric` (the smaller neighbor): with equal steps
   that rule marks the steps beside every maximum of a smooth function. Hence five constants, where the
   plan says "named constants for the tolerances".
4. `step`: a time that is NaN is refused, an infinite one is accepted; the result for one number is an
   array with no axes, as `forced_response` gives. `impulse_response` checks a, b and c itself (text and
   booleans are refused) and refuses times at which the response is not a number of the computer.
5. Slider 1 shows in its title the largest |x| after the stop, measured on one period, not a formula.
   Slider 2 does not draw sin t, which would answer its question: the distance from sin t is in the
   checking cell after the answer. Its left panel has a fixed scale up to 21, so that the pulse is seen to
   grow; for b = 1 the pulse is low on that scale.
6. Figures: the functions built with switches have a figure of their own, so there are seven figures and
   two static slider frames (Section 1 counts six figures). The hammer's figure also shows the velocity,
   with its jump from 0 to 3 meters per second.
7. The force that fades (Section 6) is 8 e^(-t) newtons, the right side of notebook 19's circuit on the
   same left side; from rest the answer is 2 e^(-t)(1 - cos 2t) meters.
8. Section 1 adds the letter h(t), the impulse response, to the conventions of this plan.
9. Exercises: 1(a) has two answers; 3(ii) asks for the second firing time that leaves the mass at rest
   (1 + 3 pi); 3 has a sixth item (departure 1). In the checker of 1(c) the expected velocity P/m is also
   computed from `ol.impulse_response`.
10. "L{u(t - a)} = e^(-as)/2" in the text copy of Section 6.2 is not reported in the notebook as a slip of
    the book: fractions are often broken in that copy, and Example 6.1.4 has the formula right. The
    notebook writes e^(-as)/s and cites the example. No other printed slip is corrected.
11. Registries: `NAME_20` is the first entry of `EXERCISES` and of `EXTREMES`, as `NAME_19` was (inserted;
    nothing reordered). The week is "Week 13 - Steps, impulses, and convolution (Sessions 25-26)". The row
    of `make_00.py` shows the convolution integral. Rebuilding `00_Index.ipynb` also changed its stored
    widget ids and time stamps, as every rebuild does.
12. No time is printed by the notebook (the stored outputs must be the same at every build).

**Lebl's text.** Recomputed and right: Example 6.1.4, the three functions of Section 6.2, formula (6.2.2),
Example 6.2.2, Examples 6.3.1 to 6.3.4, Theorem 6.3.1 on ten pairs, the transform of the unit pulse,
Example 6.4.2, and the printed answers of Exercises 6.2.13, 6.2.14, 6.3.10 to 6.3.12 and 6.4.9 to 6.4.11.
The quotations ("a rocket is fired for 4 seconds starting at t = 1", "an object that does not really make
sense unless we integrate it", "A pulse can come later and can be bigger", "the solution for an arbitrary
input is given as convolution with the impulse response") were found in the text.

**Defects left, and points for the reviewers.**
* `convolution` is a numerical integral. A switch or a pulse that is not named in `breaks` and lasts less
  than t / 256 can be missed, and then a wrong number comes back with no error (measured on seeded
  pulses: none wrong from t / 250 up for a pulse on 0 or on a constant; a low pulse on a changing
  function can be missed up to about t / 100; nearly half at t / 500, three in four at t / 1000; corrected
  after the mathematics review, see "Corrections after the reviews"). The docstring says so and a test
  pins it; `breaks` computes them. A reviewer should judge whether this is enough.
* It is slow: about 700 evaluations of f and of g for each t (0.01 s for each t with
  `ol.impulse_response` as g on this computer). The notebook calls it on 13 times or fewer in a cell.
* The library tests take about 100 s (47 s before): the new tests integrate many transforms.
* The length: 3187 words on the measure that gives 3033 for notebook 19.
* L{delta(t)} = 1 with the blow at the lower limit of the integral is stated with one parenthesis (the
  pulses start at t = 0, inside the integral), not discussed.
* "The velocity changes at once by P/m" is argued from Newton's law for a short pulse, not proved.
* The rows for the reserved-letter table of the style sheet (Section 4) are not written: they go with the
  decks of Sessions 25 and 26.
* The widget state stores the two slider pictures a second time, as in notebooks 17 to 19 (0.09 MB of
  the 0.58 MB here).
### Corrections after the reviews (2026-11-24)

Three reviews (`reviews_nb/math_review_nb20.md`: the mathematics of the notebook correct, 1 ERROR in
`convolution`, 4 IMPRECISE, 6 COSMETIC; `reviews_nb/clarity_review_nb20.md`: 0 BLOCKING, 10 IMPORTANT, 20
MINOR; `reviews_nb/code_review_nb20.md`: existing code unharmed, 4 BUG, 6 RISK, 9 STYLE) and the
coordinator's decisions A to F. One line per finding. What Section 8 says above about the constants of
`convolution`, the counts, the length, the figures and the step `u` is superseded by this part.

**Mathematics review**
* 1 (ERROR, with code BUG 1; decision A): done as decided. No new detector. The docstring now says that
  the search for switches is a help, not a guarantee, with the reviewers' measured facts (found when the
  jump is more than 8 times the change of the function per step, as on a flat background; a shorter pulse,
  a low pulse up to about t / 100 wide on a changing function, a steep ramp can be missed) and the two
  reproductions; both are pinned by tests as documented limits (wrong without `breaks`, right with them).
  The sentence of this plan is corrected (Defects left, first item). The suggested repair by second
  differences was not tried, by decision A.
* 2 "for a >= 0" in the transform of delta(t - a), with the reason: done. 3 what exactly has no limit:
  done (the review's sentence). 4 exponential order in the theorem: done. 5 constant coefficients for "the
  same answer, started later": done. 6 "by about f_0 b / m": done. 7 the range of s in the second shifting
  property: done. 8 the damped system for |H(i omega)|, "seldom helps": done. 9 the comment on
  `np.isclose`: done.
* 10 Lebl's Note on the derivative of the step: listed in "Left out" (Section 7) with one sentence that is
  right with the notebook's conventions (s times the transform of the step is the transform of delta; the
  derivative rule is not invoked, so the clause on x(0) is not needed). f * delta = f and Example 6.4.1
  stay unlisted.
* 11 `convolution` at t = 0 (0, f and g not evaluated) and a function that is not finite on an interval
  shorter than a step: both said in the docstring; the first is a test.

**Clarity review**
* 1 the delta function: done, the full replacement text (the order "compute with the unit pulse, then let
  b tend to 0"; delta(t - a) as the unit blow at the time a; the integral of the pulse with its line; the
  integral behind the transform), with the mathematics review's two corrections; the printed heading of
  the next cell says "(1/b) times the integral".
* 2 the force is f_0, and the line from Newton's law to f_0 b / m: done. 3 (a) "just before the blow" in
  the statement, (b) "just after" beside the formula, before any line that writes both values, (c) the
  empty dot, the dotted line at t = 0 and the two labels in the figure; "Where is x'(0) = 0?" shrunk to
  its last sentence: done (the label "just before" is under the axis: at the proposed place it lay on the
  curve).
* 4 (a) how "(f * g)(t)" is read, and that Python's star stays the product; (b) the table for t = 3: done.
  5 why the sum stops at t, and Delta tau becoming d tau: done, merged with mathematics 5.
* 6 one `u` for formulas and for numbers (`sp.Heaviside(x, 1)` for SymPy input, `ol.step(x)` otherwise),
  with its sentence in Section 2: done as proposed and tested (a notebook test: `u` on an array, a number,
  a formula, a SymPy number). The library function `step` is unchanged in what it does.
* 7 the two definitions before the student's lines: done differently. They are in a small code cell of
  their own right above the answer cell ("run this cell before the next one"), and again under the marker
  of the answer cell for the checker's own formulas. They cannot be the first lines of the answer cell:
  `nbbuild.py` puts the marker after the leading `my_... = None` lines, and an existing test requires an
  answer cell to start with them; neither may change. The template cell of Exercise 3 now says "call your
  times `times`". A test gives `t` and `u` other values and runs the two cells again.
* 8 a sentence for each taught mistake (2(a) the piece never switched off; 2(b) and 2(c) cut off for
  delayed, the step forgotten; 2(c) the 2 of the row of sin 2t), by the proposed helper; the hint under
  Exercise 1(b): done, each with a test.
* 9 no bare AssertionError: cell 11 (`CUT = 0` now runs; "keep it"), cells 13, 29 and 37 (every assert
  that a changed number of the text breaks has a sentence that says which numbers to put back), and the
  cell for the student's own m, c, k, P after the hammer figure, with its own figure and caption: done
  (tests: four cells stop with their sentence; three sets of own numbers run).
* 10 pacing: done as proposed. Session 25, at home: the third recipe and the general formula in the answer
  of slider 1. Session 26, at home: the table of the transform of the pulses, Lebl's Example 6.4.2 with
  its table, the integral by parts of Example 6.3.3, the theorem in numbers, the four-step check of the
  fading force (in a collapsed block). No step of a hard concept left the in-class path. The |H(i omega)|
  bullet is marked optional.
* 11 not needed (6 adopted). 12 no bracket notation: done. 13 "between 0 and 4": done. 14 the bracket with
  infinity: done. 15 "f(t - a) becomes f(v)" and the numbers at t = 2: done. 16 two tables, the dots at
  the jump: done. 17 G(s) named, "divide by s^2 + 1": done. 18 the force of the spring equals the force of
  the rocket: done. 19 SymPy's four terms explained: done. 20 "area 1" in the title (the word "pulse" left
  the title: it did not fit), the sentence about the fixed scale, the frames rendered again: done. 21 the
  letters a and b, four places: done (Exercise 3(iii) keeps Lebl's a and explains the "a" of the error
  sentence). 22 the unit of c as notebook 07 writes it, N s/m: done, also in the docstring of
  `impulse_response`. 23 the tangent line, once, and "as in the table above": done. 24 how the Greek
  letters are read: done. 25 the column of Example 6.3.1 dropped, headings without a second star: done
  (the example is still tested in the library and against SymPy). 26 the partial fractions of the fading
  force, in the collapsed block: done. 27 "from rest" defined: done. 28 which transform is which: done. 29
  templates for Exercise 3(ii) and 3(v): done. 30 "(Optional.)": done.

**Code review**
* BUG 1: see mathematics 1. BUG 2: done, the reviewer's tested fix (the level half way between the two
  sides is located; the step is cut there and at its two ends); a ramp 12 steps long is a test, and a ramp
  of 0.0003 in a convolution up to t = 50 is pinned as a documented limit (right with `breaks`). BUG 3:
  done (`CONVOLUTION_NARROW = 1e-12`; the four calls of the review are a test). BUG 4: done (a product
  that is not finite is a ValueError; a test).
* RISK 1: done in the new code (`_real_array`, the breaks, the values of f and g); the same gap in the old
  `_one_real_number` (`ol.step(1.0, 10 ** 400)`) is left, as the review says. RISK 2: done (the sentence
  of a refusal names both causes; the first bullet of the docstring says that it depends on the units).
  RISK 3: done (an array returned by f or g is refused, also one with a single element). RISK 4: done,
  `CHECK_T_2 = (0.15, 5.55)`; three answers with the switch at a wrong time are tests. RISK 5: done
  (ramps, a step with the value 1/2 and `breaks`, a product that overflows, the two low pulses, large
  whole numbers, arrays; 21 tests more in the library).
* RISK 6 (`laplace_numeric` with a step of value 1/2 at the jump and `breaks=[1]`): old code, NOT touched.
  The notebook never meets it: every function it gives to `ol.laplace_numeric` with `breaks` is built with
  `ol.step`, whose value at the jump is 1, and SymPy's own Heaviside (value 1/2) is only evaluated on
  arrays. Left to the coordinator to raise with the instructor, with the halving rule of BUG 2 in
  `laplace_numeric`.
* STYLE 1: the numbers of the docstring carry the names of their constants, the sentence of a refusal
  computes its "8 digits" from `CONVOLUTION_RTOL`, the cost is as measured (about 450 evaluations, 700 to
  900 with jumps). STYLE 2: `CONVOLUTION_JUMP_RATIO` and `CONVOLUTION_MAX_JUMPS`, equal to those of
  `laplace_numeric` on purpose. STYLE 3 (shared helpers with `laplace_numeric`): not done, for later, as
  the review says. STYLE 4: booleans and text are refused also inside a list. STYLE 5: the sentence of
  `impulse_response` names a too small a. STYLE 6: `rel=1e-12` where the function is exact. STYLE 7: see
  clarity 7. STYLE 8: the digits of the text keep their comment ("on purpose"); the `+ 0 * times` has one.
  STYLE 9: a SymPy number is a number for `step`.
* The proposal to shorten the new tests: applied (4 draws for the transfer function, 3 seeded SymPy
  convolutions, two values of s in the SymPy test of the shifting property, "found by itself" for one
  step). Library tests: 102 s before this pass, 73 s after it, with 21 tests more.

**The notebook after the corrections.** 51 cells (25 code), 10 stored figures and 2 sliders; 640 350
bytes, plain ASCII, no local path. Before the exercises: 5536 whitespace tokens, 4471 with each formula as
one word, **4155** on the measure of plan 17 (3187 before): the replacement texts of the clarity review,
the table of the convolution and the collapsed four-step check were added in full, and nothing was cut. By
the lines "In class / At home" the text of the two sessions' class time is smaller than before; the whole
is one third longer. Every figure and every slider frame was rendered again from the final notebook (the
old PNG files deleted first) and looked at: the label of the hammer figure and the title of slider 2 were
changed after the first look.

**Counts.**

| Gate | As built | After the corrections |
|---|---|---|
| Library, env `ode` | 736 passed (102 s) | 757 passed (73 s) |
| Library as in CI (`site/.venv`) | 544 passed, 10 skipped | 565 passed, 10 skipped (36 s) |
| Notebooks (`test_notebooks_run.py`, whole file) | 244 passed (5 min 28 s) | 259 passed (4 min 42 s) |
| `--collect-only` | 980 | 1016 |

**Gates after the corrections (2026-11-24).** Lint: `make_20.py` clean; no finding on the lines added to
`odelib.py`, the three test files and `make_00.py`. Additive-only proof against the copies made before
notebook 20: opcodes "equal" and "insert" only for the four files (266, 594, 214 and 175 lines added),
line endings as they were. Site build: 25 notebooks, 136 MB; `site/test_build_site.py`: 5 passed. Browser:
25 of 25 notebooks without errors, 11 to 20 s each, about 6 min in all. Notebook 20: 25 cells, 12 plots, 2
sliders, 20 s; the setup cell 12.5 s, every other cell at most 1.4 s (the rocket), then 1.0 s (Section 3)
and 0.8 s (the table of the pulses, the hammer). Trial export into a new, empty folder of the scratch
area: 87 files, 14.9 MB, all checks passed; the public folder was not touched; nothing committed or
pushed. READMEs: 1016 tests, the row of notebook 20 and the last browser run updated.

**Departures from a proposed fix, and what was not done.**
* Clarity 7 and code STYLE 7: a cell of its own above the answer cell, not lines at its top (see above).
* Mathematics 1: the repair by second differences was not tried (decision A); code STYLE 3: for later.
* Mathematics 10: listed in "Left out" with a shorter sentence than the review's.
* Clarity 3 (c): the place of one label; clarity 20: the wording of the title.
* The length grew to 4155 words: the instruction was to apply the full texts; a cut is a decision for the
  coordinator.
* Not run, as before: the library tests at the browser pins in an environment of their own.

### Final pass after the re-check (2026-11-24)

The re-check (`reviews_nb/recheck_nb20.md`: existing code unharmed; the five hard concepts understandable;
0 ERROR, 0 BLOCKING, 2 IMPORTANT, 11 MINOR) was applied as the coordinator decided. What the two parts
above say about the length, the number of cells, the third recipe, the proof of the shifting property and
Lebl's Example 6.4.2 in the text is superseded by this part.

* 1 (IMPORTANT, cell 28): the false reason about x' is replaced by the re-check's exact text (in so short
  a time x is almost 0 and x' is at most about P/m, so kx and cx' stay small, while f_0 = P/b is very
  large). The same wrong reason was looked for elsewhere (the answer of slider 2, the captions, the
  answers): it is not there (the answer of slider 2 speaks of the position only).
* 2 (IMPORTANT, the length): items 1 to 25 and 28 to 31 of the re-check's table are applied as listed;
  items 26 and 27 are NOT applied, by decision (the honest answer stays under each slider).
  * Deleted from the text: the bullet that named delta, tau, the star, h and H before they are known (1);
    the reason with the tangent line under the table of pulses (6) and the two sentences on the fixed
    scale (8, 31); parts of the hammer's conventions (9); the sentence on the delta function carrying the
    velocity, which is the answer of Exercise 4(a) (11); Lebl's Example 6.4.2 with its code cell (12:
    `hammer_h` is now defined in the hammer's cell, with the assert "a blow P gives P h(t)"); the bullet
    on H(i omega) and the explanation of Lebl's Note (17); the bullet on notebook 21 (25); three short
    repeats (22, 28, 30); "Lebl's examples have no units" (29).
  * Moved to the collapsed answers (Section 9): the third recipe with Lebl's function by cases, to the
    answer of Exercise 2(a) (2; its panel left the figure, which has two panels now); the general formula
    with the identity cos A - cos B, to the answer of Exercise 3(ii) (5); and, under "At home: the
    computations that the text left out", the integral behind e^(-as)/s (20), slider 2 by a formula with
    the two tangent lines (7), Example 6.3.3 by parts (13), the fading force by the four steps (15).
  * The new theory item, Exercise 4(b): the proof of the second shifting property (the theory exercise of
    Session 25; Lebl, Exercise 6.2.9), with "first step: v = t - a" and the proof in the collapsed answers
    (3). The old Exercise 4 is 4(a).
  * Shortened: "What you need" (21), the rocket's conventions (4), the two lists "At home" (18, 19), the
    two instants (10), the theorem in numbers, now one clause that points at the printed table (14; the
    constant `S_TEXT` left the cell), "the same answer, started later" (16), Lebl's quotation, now the
    reference only (23), the sentence on `breaks`, now a comment on the first call of `ol.convolution`
    (24). The exercise numbers of Lebl are said in the first lines of Section 8 (29).
  * The repeats that the re-check lists: removed with items 4, 8, 10, 11, 16, 18, 19, 31, and "and the
    initial values are 0"; left, as the re-check says: why g rises, "for a >= 0" twice, the area of a
    pulse in the answer of slider 2 (item 27 is not applied).
  * Nothing that a hard concept needs left the main text (the re-check's list "What stays" was followed).
    What moved is still tested: Lebl's function by cases, the second shifting property and Example 6.4.2
    in the library tests; the transform of a step, the formula of slider 2 (every position of the
    slider), Example 6.3.3 and the partial fractions of the fading force by asserts of the notebook.
* 3: the own-numbers cell prints "largest |x| ..., first at t = ..." (0.949 m at 0.99 seconds with no
  damper; 0.771 m at 0.55 seconds for a negative blow; two tests), and says that the mass must not be 0.
* 4: the sentence for the answer cell of Exercise 2 run alone, at the end of the exercise's text.
* 5: `_real_array` refuses a boolean, text or a date inside a list of numbers (six tests).
* 6 (a): whatever `float()` raises for a value of f or g is a refusal with the usual sentence (a test).
  6 (b), a shorter quotation of a very long number: not done (harmless, and it would also shorten the
  quotation of an array).
* 7: "Why, without transforms" is four numbered lines, no word added; the theorem in numbers: item 14.
* 8: "an instruction". 9: `S_PULSE = 0` stops with "S_PULSE must be larger than 0" (a test); the comment
  of `T_VALUES` says to keep `np.array(...)`; `OMEGA_0` no longer exists (item 12). 10: the title of the
  rocket's figure is computed from the two times (a test runs the cell with 2 and 6). 11: the second
  bullet of the docstring of `convolution` ends at "name the times in breaks", and what the search misses
  is a list of its own, with no fact changed (the rate for steep ramps is the re-check's). 12: the
  comment of `u` says "one argument". 13 (`ol.step(1.0, 10 ** 400)`, old code): nothing done, to be raised
  with RISK 6.

**The notebook after the final pass.** 50 cells (24 code), 10 stored figures and 2 sliders; 631 168 bytes,
plain ASCII, no local path. Before the exercises: 4431 whitespace tokens, 3668 with each formula as one
word, **3428** on the measure of plan 17 (4155 before this pass; the target was about 3400). All ten
figures and seven slider frames were rendered again from the final notebook (the old PNG files deleted
first) and looked at: no defect.

**Counts.**

| Gate | After the corrections | After the final pass |
|---|---|---|
| Library, env `ode` | 757 passed (73 s) | 763 passed (76 s) |
| Library as in CI (`site/.venv`) | 565 passed, 10 skipped | 571 passed, 10 skipped (49 s) |
| Notebooks (`test_notebooks_run.py`, whole file) | 259 passed (4 min 42 s) | 262 passed (6 min 6 s) |
| `--collect-only` | 1016 | 1025 |

**Gates after the final pass (2026-11-24).** Lint: `make_20.py` clean; no finding on the lines added to
`odelib.py`, the three test files and `make_00.py`. Additive-only proof against the copies made before
notebook 20: opcodes "equal" and "insert" only for the four files (273, 607, 214 and 182 lines added),
line endings as they were. Site build: 25 notebooks, 136 MB; `site/test_build_site.py`: 5 passed. Browser:
25 of 25 notebooks without errors, 15 to 22 s each, about 8 min in all. Notebook 20: 24 cells, 12 plots, 2
sliders, 22 s; the setup cell 14.9 s, every other cell at most 2.0 s (the rocket), then 1.6 s (the table
of the pulses), 1.4 s (the hammer), 1.2 s (Section 3). Trial export into a new, empty folder of the
scratch area: 87 files, 14.9 MB, all checks passed; the public folder was not touched; nothing committed
or pushed. READMEs: 1025 tests, the row of notebook 20 and the last browser run updated. Not run, as
before: the library tests at the browser pins in an environment of their own.

**For the instructor** (old code, not touched by this work): `laplace_numeric` with a step whose value
at the jump is 1/2 and `breaks=[1]` refuses a convergent integral (code review, RISK 6), and its halving
rule has the defect of BUG 2; `_one_real_number` lets an OverflowError through for 10 ** 400.
