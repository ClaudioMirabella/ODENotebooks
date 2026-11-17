# Notebook 15 - Forced vibrations and resonance: plan

MATH 331 - Differential Equations - Spring 2027 - phase D of the plan of the student site
(the second notebook of phase D)

**Status (2026-11-17): built and reviewed.** 1947 words before the exercises. The independent review found no wrong formula, answer or citation. Corrected after it: the check of the water tower was too loose (now three decimals); the cell of Section 5 used the name `amplitude`, which Exercise 3 defines as a function (running it again stopped the check); `practical_resonance` at the limit c^2 = 2mk; `cosine_forced_response` with a damping too small to matter; the title of the first slider (largest size in the window, not of the beat); the table of the analogy now starts from the three component laws; wording in ten places.

**Session 15, Units 6 and 15** (lecture plan: "Forced vibrations; beats; resonance; RLC
circuits"; theory exercise "Establish the mechanical-electrical analogy"; practical exercise
"Plot an amplitude-response curve"). About **15 minutes in class** (sections 2 and 5), the rest as
homework.

**References** (every item was found in the text of the book and recomputed on 2026-11-17; the
text copy of Adkins and Davidson loses minus signs, so their formulas were read with the control
characters visible).
* Lebl, *Notes on Diffy Qs*, **Section 2.6, Forced oscillations and resonance**: Example 2.6.1
  (0.5 x'' + 8x = 10 cos(pi t) from rest, with Figure 2.6.2 and the "beating"); pure resonance,
  x_p = F0 t sin(omega t) / (2 m omega), with Figure 2.6.3 (F0 = 2, m = 1, omega = pi); the damped
  case with x_tr and x_sp, the amplitude C(omega) and the phase; Figure 2.6.4 (k = 1, m = 1,
  F0 = 1, c = 0.7, omega = 1.1, different initial conditions); practical resonance at
  omega = sqrt(omega0^2 - 2 p^2); Figure 2.6.5 (k = m = F0 = 1; c = 0.4, 0.8, 1.6); the remark
  "do not memorize the formula"; the footnote on the Tacoma Narrows bridge. Exercises 2.6.2,
  2.6.3, 2.6.4, 2.6.5 (the water tower), 2.6.6, 2.6.8 (list at the end of the chapter; the book
  prints the answers of 2.6.6 and 2.6.8).
* Adkins and Davidson, *Ordinary Differential Equations* (Springer, 2012): **Section 3.6**, the
  part "Undamped Forced Motion" (their formula (18), both cases; beats in an unbalanced washing
  machine and in tuning a piano); **Section 3.7**, equations (4) and (5), Table 3.4 (which
  includes the pair forcing function - applied voltage), Example 2 (q'' + 9q' + 14q = 14 from
  rest). They obtain (18) with Laplace transforms; the notebook uses the guesses of notebook 14.
* Notebooks 14 (particular solutions; the factor x and where it comes from; transient and
  steady part), 07 (omega0, p, the three kinds of damping, the table spring - circuit), 07E
  (energy), 13 (first-order circuits).

**Purpose.** One equation, m x'' + c x' + k x = F0 cos(omega t), and one question: **how large is
the response, as a function of the frequency of the force?** Four ideas.

1. Without damping and away from omega0 the response is a sum of two cosines; when the two
   frequencies are close they produce **beats**.
2. At omega = omega0 the guess must be multiplied by t (notebook 14, Section 4): the amplitude
   grows without bound, **pure resonance**. The slider shows beats getting longer and longer
   until they become that growth: resonance is the limit of beats.
3. With damping everything that depends on the initial data dies out; what stays is a cosine of
   the frequency of the force, with an amplitude C(omega) and a delay. The curve C(omega) is the
   **amplitude-response curve**: its maximum is practical resonance, and it disappears when the
   damping is large.
4. The same curve describes a circuit with a coil, a resistor and a capacitor driven by an
   alternating voltage: this is how a radio is tuned.

---

## 1. Contents

| # | Section | What the student does | Values shown (all checked) |
|---|---|---|---|
| 0 | Title, session, references, what you will learn | reads | - |
| 1 | **Theory in brief** | the equation and its letters (omega0 = sqrt(k/m), p = c/(2m) as in notebook 07; omega is the frequency of the force); the three results in one box: off resonance, x_p = F0 cos(omega t) / (m (omega0^2 - omega^2)); at resonance, F0 t sin(omega0 t) / (2 m omega0); with damping, x_sp = C cos(omega t - gamma) with C = F0 / (m sqrt((2 omega p)^2 + (omega0^2 - omega^2)^2)). Lebl's advice: remember the method, not the formula | - |
| 2 | **A force on an undamped spring** (Lebl, Example 2.6.1) | by hand with the guess A cos(omega t) (why no sine is needed here: there is no x'), then the initial conditions on the sum (the warning of notebook 14). Plot of the solution: it does not look like a cosine at all | omega0 = 4; A = 20 / (16 - pi^2) = 3.262; x = 3.262 (cos(pi t) - cos 4t) |
| 3 | **Beats** | the identity cos B - cos A = 2 sin((A - B)/2) sin((A + B)/2): a fast oscillation whose amplitude is a slow sine. Same plot with the envelope drawn. Adkins-Davidson's two pictures in words: the washing machine and the piano tuner. Question before the plot: how long is one beat? | envelope 6.52 sin(0.429 t): one beat lasts 2 pi / (4 - pi) = 7.32 s; the fast oscillation has period 1.76 s |
| 4 | **Pure resonance** | omega = omega0: the guess A cos(omega0 t) is a solution of the homogeneous equation, so (notebook 14) try t (A cos + B sin): A = 0, B = F0 / (2 m omega0). Lebl's Figure 2.6.3. Then the limit: (cos(omega t) - cos(omega0 t)) / (omega0^2 - omega^2) tends to t sin(omega0 t) / (2 omega0), checked with `sp.limit`. **Slider 1**: omega from 2.5 to 4 for the spring of Example 2.6.1: the beats get longer and larger, and at omega = 4 the envelope is a straight line. One sentence on the Tacoma Narrows bridge (Lebl's footnote: a common but wrong example) | Figure 2.6.3: (1/pi) t sin(pi t); for Example 2.6.1 at omega = 4 the amplitude grows by F0 / (2 m omega0) = 2.5 per second |
| 5 | **With damping: what stays** | the guess A cos + B sin (now the sine is needed), solved with SymPy as in notebook 14; amplitude C and phase gamma; transient x_tr and steady periodic solution x_sp (Lebl's names). Lebl's Figure 2.6.4 redrawn: five initial conditions, one final curve. Question first: after which time is the transient below 1 %? | k = m = F0 = 1, c = 0.7, omega = 1.1: C = 1.253, gamma = 1.837 rad; p = 0.35, so 1 % after ln(100)/0.35 = 13.2 |
| 6 | **The amplitude-response curve** (Session 15, practical exercise) | C as a function of omega for Lebl's three dampings (Figure 2.6.5), with the maximum marked. Where the maximum is: omega_r = sqrt(omega0^2 - 2 p^2), below omega0; none if c^2 >= 2 m k. Three readings: C(0) = F0 / k (a slow force just stretches the spring); C tends to 0 for fast forces (Lebl's swing); small damping gives a tall narrow peak. **Slider 2** (c and omega): left, the motion from rest with the lines at +C and -C; right, the curve C(omega) with a dot at the chosen omega and a vertical line at omega_r | c = 0.4: omega_r = 0.959, C = 2.55; c = 0.8: 0.825, 1.36; c = 1.6: no maximum, C(0) = 1; the limit case is c = sqrt(2) = 1.414 |
| 7 | **The same curve in a circuit** (Session 15, theory exercise) | the analogy established, not only stated: Newton's law and Kirchhoff's voltage law written side by side, term by term (force of the spring - voltage of the capacitor, and so on), giving L q'' + R q' + q/C = E(t). The table of notebook 07 with two more lines (force - voltage; omega0 = 1/sqrt(LC), p = R/(2L)). A constant voltage: Adkins-Davidson 3.7 Example 2, where the steady part is a constant (the capacitor ends with 1 C and no current). An alternating voltage: the response curve of the charge is the curve of Section 6 with other letters; the **current** I = q' has amplitude omega C(omega), largest exactly at omega0, whatever R: tuning a radio means choosing the capacitor so that 1/sqrt(LC) is the frequency of the station | q = 1 + (2/5) e^(-7t) - (7/5) e^(-2t); current maximum at omega0 (checked with SymPy) |
| 8 | **Exercises** | the four steps below | - |
| 9 | **Answers** (collapsed), where to go next (notebook 16; Unit 11, where a force that is switched on and off is treated with Laplace transforms; Fourier series, for a force that is periodic but not a cosine) | opens after trying | - |

Length: at most 2100 words before the exercises, and this time the limit is kept: if the draft is
longer, section 7 loses the alternating-voltage paragraph to an optional notebook 15E. Eight
figures; two sliders, each with its static picture and its question first.

## 2. Exercises (the four steps)

1. **Predict** (no code) - *Lebl, Exercise 2.6.8*: m = 5, k = 20, F0 = 5, no damping. Which
   omega gives resonance (2)? By how much does the amplitude grow each second (0.25)? What is it
   at t = 100 (25, the answer of the book)? Then two True/False: with damping, the practical
   resonance frequency is below omega0 (True); with more damping, the peak of the response curve
   is higher (False).
2. **By hand, then check** - (a) *Lebl, Exercise 2.6.6*: m = 4, k = 4, c = 1, F0 = 2: the
   frequency of practical resonance and the amplitude there (sqrt(31/32) = 0.984 and
   16 / (3 sqrt 7) = 2.016, the answers of the book). (b) *Lebl, Exercise 2.6.5*, the water
   tower: m = 10 000 kg, k = 1000 N/m, no friction, force m A omega^2 cos(omega t) with A = 1 and
   0.5 cycles per second: the natural frequency (0.316 rad/s) and the largest displacement
   (2 A omega^2 / |omega0^2 - omega^2| = 2.02 m, with omega = pi rad/s: more than the 1.5 m the tower
   can stand). The book prints no answer for this one: the value is mine, checked against the
   numerical solution. (c) *Adkins-Davidson 3.7, Example 2*: the charge q(t), typed as a formula
   (unique: checked with `ol.check_solution` and its initial data).
3. **Fill in** - Complete `amplitude(omega, m, c, k, F0)`, the height of the response curve. The
   next cell compares it with `ol.steady_state` on Lebl's three dampings and on Exercise 2.6.6,
   and with the amplitude **measured** on a numerical solution after the transient (so a formula
   that is wrong by a factor is caught by the physics, not only by another formula); then it
   draws the student's own response curve, which is the practical exercise of the session.
4. **Explore** - (i) *Lebl, Exercises 2.6.3 and 2.6.4*: for which c (given m, k), and for which
   m (given c, k), is there no practical resonance? (c >= sqrt(2 m k); m <= c^2 / (2k).) Check
   with slider 2. (ii) *Lebl, Exercise 2.6.2*: two forces at once, F0 cos(omega t) +
   F1 cos(3 omega t): the steady solution is the sum of the two (superposition, notebook 14).
   (iii) With slider 1, choose omega so that one beat lasts 10 seconds. (iv) For the circuit of
   Section 7, plot the amplitude of the current against omega for three resistances: where is the
   maximum?

Answers (final values only) in a collapsed block; full solutions after the session.

## 3. New in `odelib.py` (with tests)

* `steady_state(m, c, k, F0, omega)`: the pair (C, gamma) of the steady periodic solution
  C cos(omega t - gamma) of m x'' + c x' + k x = F0 cos(omega t). With c > 0 it is Lebl's formula,
  gamma between 0 and pi from `arctan2` (the quadrant matters: gamma passes pi/2 at omega0).
  With c = 0 and omega different from omega0: C = F0 / (m |omega0^2 - omega^2|) and gamma = 0 or pi.
  With c = 0 and omega = omega0: ValueError that says "pure resonance: there is no steady
  solution".
* `practical_resonance(m, c, k)`: the frequency sqrt(omega0^2 - 2 p^2) at which C(omega) is
  largest, or 0.0 when c^2 >= 2 m k (the largest response is then to a constant force).
* `cosine_forced_response(t, m, c, k, F0, omega, x0=0.0, v0=0.0)`: the exact solution of the
  initial value problem, in closed form (the pictures need long times and many points, where the
  numerical integral of `forced_response` would be slow in the browser). Steady part plus the
  homogeneous solution with the remaining initial data; for c = 0 the forced part is written
  with the product of sines, which stays accurate when omega is close to omega0, and becomes
  F0 t sin(omega0 t) / (2 m omega0) at omega = omega0 (same tolerance as `damping_regime`).

Tests:
* `tests/test_odelib.py`: Lebl's Example 2.6.1 (3.262 (cos(pi t) - cos 4t)); Figure 2.6.3
  ((1/pi) t sin(pi t)); Figure 2.6.4 (C = 1.253); Figure 2.6.5 (0.959 and 2.55; 0.825 and 1.36;
  none); Exercise 2.6.6 (0.984 and 2.016); Exercise 2.6.8 (25 at t = 100); Adkins-Davidson's
  formula (18) in both cases; C(0) = F0 / k; C is largest at `practical_resonance` (compared with
  a fine grid); gamma = pi/2 at omega0; invalid arguments (m or k not positive, c negative,
  omega not positive, non-finite values; pure resonance in `steady_state`).
* `tests/test_cross_validation.py`: `cosine_forced_response` against `forced_response` and
  against `solve_ivp` for random parameters in the four kinds of damping, at omega far from,
  close to (relative distance 1e-6) and equal to omega0; continuity in omega across omega0; the
  steady part of SymPy's `dsolve` solution has the amplitude and phase of `steady_state`.

## 4. Conventions and checks

As for notebook 14: same setup cell, `assert` for every reference value, one sentence under
every plot (written after looking at the figure and recomputing every number), line styles as
well as colours, a question before every slider and its answer after, named constants, plain
ASCII, generated by `plans/make_15.py` with `nbbuild.py`.

* Letters: x(t), m, c, k, omega0, p as in notebook 07 and in Lebl; omega for the force; F0;
  amplitude C and phase gamma of the steady solution (Lebl), with one sentence saying that it is
  the A of notebook 07 (which here is taken by the coefficients A, B of the guess) and that C is
  also the capacitance in Section 7, where the amplitude is therefore written in words.
  Adkins-Davidson write beta for the natural frequency and y for the displacement: their formula
  is quoted in the notebook's letters, and the text says so.
* Lessons of the review of notebook 14, applied from the start: every formula of Adkins and
  Davidson is read from the copy with visible control characters; no quotation marks around
  words that are not literally in the book; tables of rules are tested on a case that would
  break them (here: the formula for omega_r when omega0^2 < 2 p^2, the phase when omega > omega0,
  the amplitude at omega = 0); cross-references to sections of other notebooks are checked
  against their `make_NN.py`.
* `tests/test_notebooks_run.py`: the notebook as it is; with the correct answers; with wrong
  answers (omega_r = omega0, the amplitude without the square root, the phase in the wrong
  quadrant, an `amplitude` wrong by the factor m); `EXTREMES` for both sliders (omega = omega0
  exactly, omega0 plus and minus one step, both ends; c = 0 with omega = omega0 in slider 2,
  c at its maximum, omega at both ends).
* The site is rebuilt and the notebook is run in the browser with `check_site.py`.
* Index notebook: one more row (15, U6/U15, Session 15, Lebl 2.6); home page: a line "Week 8"
  with 15; notebooks 07 and 14 ("where to go next") point to 15 by number.
* After the build: an independent review as for notebook 14 (mathematics, captions, citations,
  exercises and checks, the three library functions, clarity, length), and its corrections.

## 5. Order of work

1. `odelib.py`: the three functions, tests first, from the values above; all tests pass.
2. `plans/make_15.py`: sections 1-4 with slider 1; look at the figures.
3. Sections 5-6 with slider 2.
4. Section 7; word count; decide whether the alternating-voltage paragraph stays.
5. Exercises and answers; `EXERCISES` and `EXTREMES` in the tests.
6. Index, home page, links from 07 and 14; rebuild; `check_site.py`.
7. Independent review; corrections; final run of all tests.

## 6. Decisions for the instructor

1. **Sliders.** Two (proposed): the frequency of the force without damping (beats becoming
   resonance), and damping with frequency on the response curve. A third one for the circuit is
   left out: it would be the second with other letters.
2. **The phase.** Proposed: computed and shown in Section 5 (the response is late), but the
   phase curve gamma(omega) is not drawn in Section 6, to keep the practical exercise on the
   amplitude. It can go to 15E with the energy absorbed per cycle.
3. **Optional notebook 15E** (proposed: later, together with 14E): the phase curve; the power
   absorbed and why it is largest at omega0; the width of the peak and the quality factor; the
   response to a square wave as a sum of cosines (a preview of Fourier series).
4. **The water tower** (Exercise 2b) has no printed answer in Lebl. Proposed: keep it, because it
   is the only exercise with a real structure and a real decision (does it collapse?), and say in
   the notebook that the value is not from the book.

## 7. Files

```
ode/ode/notebooks/
    15_Forced_Vibrations.ipynb                 <- to build
    plans/15_Forced_Vibrations_plan.md         <- this document
    plans/make_15.py                           <- builds the notebook
    odelib.py                                  <- + steady_state(), practical_resonance(), cosine_forced_response()
    tests/test_odelib.py                       <- + their tests against the books
    tests/test_cross_validation.py             <- + against forced_response, solve_ivp and dsolve
    tests/test_notebooks_run.py                <- + the exercises and sliders of notebook 15
    00_Index.ipynb, plans/make_07.py, make_14.py   <- index row; one line each
ode/ode/site/build_site.py                     <- READING_ORDER: Week 8 with 15
```
