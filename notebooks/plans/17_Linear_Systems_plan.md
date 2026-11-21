# Notebook 17 - Linear systems and the eigenvalue method: plan

MATH 331 - Differential Equations - Spring 2027 - phase D of the notebooks (14-18), the fourth
notebook of the phase

**Status (2026-11-20): planned; the seven decisions of Section 6 approved by the instructor as proposed; being built.**

**Sessions 17 and 18, Unit 8** (lecture plan, Session 17: "Eigenvalue refresher; rewriting an
nth-order ODE as a system; matrix form"; theory exercise "Prove the equivalence of the conversion
to a first-order system"; practical exercise "Model a pair of coupled tanks". Session 18:
"Eigenvalue method for real, complex, and repeated eigenvalues; fundamental matrix"; theory
exercise "Find generalized eigenvectors"; practical exercise "Solve a two-mass oscillator"). About
**15 minutes in class in each session** (Session 17: sections 3-4; Session 18: sections 6-7), the
rest as homework. Notebook 16 already ends with "notebook 17, where an equation of order n becomes
a system of n equations of order 1": this notebook keeps that promise in Section 3.

**References** (every item was found in the text of the book and recomputed on 2026-11-20 with
SymPy and NumPy; exercise numbers of Lebl are those of the list at the end of Chapter 3, "3.E";
in the text copy the numbers of Section 3.4 are printed as page-index placeholders and are
counted here in their order: Examples 3.4.1-3.4.4, Theorems 3.4.1 and 3.4.2).
* Lebl, *Notes on Diffy Qs*, **Section 3.1, Introduction to systems of ODEs**: Example 3.1.2 (two
  brine tanks, x1' = (r/V)(x2 - x1), x2' = (r/V)(x1 - x2), the total salt is constant and both
  amounts tend to the average); the change to first order (3.1.8) and Example 3.1.4
  (x''' = 2x'' + 8x' + x + t with u1 = x, u2 = x', u3 = x''); the remark that k equations of
  order n become nk equations of order 1; Example 3.1.6 (x' = 2y - x, y' = x, reduced to
  y'' + y' - 2y = 0; x = (2e^t + e^(-2t))/3, y = (e^t - e^(-2t))/3; the values x(2) = 2.475 and
  y(2) = 2.457 printed with Figure 3.1.4); Theorem 3.1.1 (Picard for systems, and the remark that
  it also gives existence and uniqueness for equations of order n).
* Lebl, **Sections 3.2 and 3.3**: Theorem 3.2.1 (invertible if and only if det is not 0); Theorem
  3.3.1 (superposition), the fundamental matrix X(t), X(t0) c = b for the initial conditions;
  Example 3.3.2 (the system of Example 3.1.1 in matrix form, x(0) = (1, 2): x1 = e^t,
  x2 = e^t/2 + 3e^(-t)/2).
* Lebl, **Section 3.4, Eigenvalue method**: Example 3.4.1 ([[2, 1], [0, 1]], lambda = 2, v = (1, 0));
  Examples 3.4.2-3.4.3 ([[2, 1, 1], [1, 2, 0], [0, 0, 2]]: eigenvalues 1, 2, 3; eigenvector
  (1, 1, 0) for 3); the questions in the text after Example 3.4.3 (are eigenvectors unique? the
  2 x 2 shortcut without row reduction); Theorem 3.4.1 and Example 3.4.4 (general solution
  c1 (1, -1, 0) e^t + c2 (0, 1, -1) e^(2t) + c3 (1, 1, 0) e^(3t)); the note that for an equation
  of order n written as a system, det(P - lambda I) = 0 is the characteristic equation; the part
  "Complex eigenvalues" ([[1, 1], [-1, 1]], lambda = 1 +- i, v = (i, 1) for 1 - i, real solutions
  (e^t sin t, e^t cos t) and (e^t cos t, -e^t sin t)); Theorem 3.4.2.
* Lebl, **Section 3.6, Second order systems and applications**: M x'' = K x, A = M^(-1) K;
  Theorem 3.6.1 (natural frequencies omega_i with -omega_i^2 the eigenvalues; the zero eigenvalue
  gives v (a + b t)); Example 3.6.1 (m1 = 2, m2 = 1, k1 = 4, k2 = 2: A = [[-3, 1], [2, -2]],
  eigenvalues -1 and -4, modes (1, 2) and (1, -1), x(0) = (1, -1), x'(0) = (0, 6):
  x = (1, 2) 2 sin t + (1, -1)(cos 2t - sin 2t); Figures 3.6.3 and 3.6.4); Example 3.6.2 (two rail
  cars: impact after 5.22 s at 3.85 m/s; a safe distance 4 pi / sqrt(3) = 7.26 m).
* Lebl, **Section 3.7, Multiple eigenvalues**: algebraic and geometric multiplicity, defect;
  Theorem 3.7.1; Example 3.7.1 ([[3, 1], [0, 3]], v1 = (1, 0), generalized eigenvector v2 = (0, 1)
  from (A - 3I) v2 = v1, solution (v2 + v1 t) e^(3t)); the two solutions (3.7.1); chains (3.7.2)
  and (3.7.3).
* Lebl, **Section 3.8, Matrix exponentials**: Theorem 3.8.1 (x = e^(tA) x(0)); the simple case
  A = [[5, 4], [-1, 1]] = 3I + B with B^2 = 0 and e^(tA) = e^(3t) [[1 + 2t, 4t], [-t, 1 - 2t]];
  the exercise in the text of that section (a 2 x 2 matrix with a single eigenvalue:
  (A - lambda I)^2 = 0; not Exercise 3.8.1 of the list);
  Example 3.8.1 ([[1, 2], [2, 1]], eigenvalues 3 and -1, x(0) = (4, 2):
  x = 3e^(3t) + e^(-t), y = 3e^(3t) - e^(-t)); Section 3.8.4 (e^(tA) = X(t) X(0)^(-1)).
* Lebl, **Exercises (list 3.E)**: 3.1.4, 3.1.12, 3.4.1, 3.4.3, 3.4.4, 3.4.7, 3.6.4, 3.6.7 (answer
  printed: x2 = (2/5) cos(t / sqrt(6)) - (2/5) cos t), 3.7.1, 3.7.10 (answer printed: eigenvalue 2
  with defect 2), 3.8.14 (answer printed: x = ((1 - t) e^(2t), (2 - t) e^(2t))).
* Notebooks 08 (a 2 x 2 linear system with `A @ x`), 06 (characteristic roots; the slider of two
  roots coming together), 07 (damping regimes), 07E (the plane of states), 02E (a decay chain is a
  system), 16 (Example 2.3.3; `higher_order_solution` already works through the system), 04 (Euler
  and Runge-Kutta). Videos R05 (complex numbers), R07 (determinants and eigenvalues), R08 (linear
  systems), R09 (Taylor series, for e^(tA)).

**Purpose.** One idea, met three times: **a linear system x' = A x is solved by the eigenvalues
and eigenvectors of A**, exactly as a single equation is solved by the roots of its characteristic
equation.

1. **Systems are everywhere, and an equation of order n is one of them.** Two tanks that exchange
   brine, two carts joined by springs, and every equation of order n (Lebl's u1 = y, u2 = y', ...)
   are systems of first order. The conversion is an equivalence, proved in the notebook: the
   eigenvalues of the matrix are the roots of the characteristic equation of notebook 06.
2. **Try x = v e^(lambda t).** Then A v = lambda v: an eigenvalue problem. Three cases for a 2 x 2
   matrix, the same three as notebook 06: two real eigenvalues, a complex pair (take real and
   imaginary parts), one repeated eigenvalue with one eigenvector (a generalized eigenvector w,
   and the solution (w + v t) e^(lambda t), the "times t" of notebook 06).
3. **The fundamental matrix.** The n solutions side by side form X(t); the initial conditions are
   the linear system X(0) C = x(0) (video R08), and e^(tA) = X(t) X(0)^(-1) is one formula for every
   case: the computer's check of every hand computation.

---

## 1. Contents

| # | Section | What the student does | Values shown (all checked) |
|---|---|---|---|
| 0 | Title, sessions, references, what you will learn | reads | - |
| 1 | **Theory in brief** | the vector x(t) with components x1, x2, ..., the matrix A, x' = A x; eigenvalue lambda and eigenvector v (v not zero) of A v = lambda v, found from det(A - lambda I) = 0; for 2 x 2: lambda^2 - (tr A) lambda + det A = 0; table of the three cases with their real solutions; superposition and X(t), X(0) C = x(0); x(t) = e^(tA) x(0) (Lebl, Theorem 3.8.1) | - |
| 2 | **Eigenvalues and eigenvectors: a refresher** (Session 17) | Lebl, Example 3.4.1 by hand (A v = 2 v, one line). Examples 3.4.2-3.4.3: det(A - lambda I) = (2 - lambda)((2 - lambda)^2 - 1), eigenvector for 3 by row reduction; `sp.Matrix(...).eigenvects()` agrees; `np.linalg.eig` gives (0.707, 0.707, 0): the same direction, scaled to length 1 (Lebl's question "are eigenvectors unique?"). The 2 x 2 shortcut: for the first row (a - lambda, b), the vector (b, lambda - a) is an eigenvector when it is not zero. `ol.two_by_two_case` on Example 3.8.1's matrix | 1, 2, 3; (1, 1, 0); [[1, 2], [2, 1]]: trace 2, det -3, eigenvalues -1 and 3, eigenvectors (2, -2) and (2, 2) |
| 3 | **An equation of order n as a system** (Session 17, theory exercise) | Lebl's u1 = y, u2 = y', ..., un = y^(n-1); Example 3.1.4 (quoted with its letter x). The **proof of the equivalence**, both ways, with the initial conditions (in Section 3 of this plan, below the table). The matrix of the system (`ol.companion_matrix`); its characteristic polynomial is the characteristic polynomial of the equation divided by a_n (SymPy, for a general third-order equation). Notebook 16's Example 2.3.3 as a system: eigenvalues -1, 1, 3, the roots found there; `ol.rk4` on the system (the dependent variable is a vector, Lebl 3.1) against `ol.higher_order_solution`. The way back (Lebl, Example 3.1.6): a system becomes one equation of order 2; rk4 at t = 2 against the two values printed in the book | Example 3.1.4: [[0, 1, 0], [0, 0, 1], [1, 8, 2]]; y(1) = 7.64770 (rk4, h = 0.01, error 1e-7); x(2) = 2.47523, y(2) = 2.45691 (book: 2.475, 2.457) |
| 4 | **Two tanks: a model in matrix form** (Session 17, practical exercise) | convention box (below); Lebl, Example 3.1.2 with numbers chosen for the notebook and said to be so: V = 100 L, Q = 1 L/s, x(0) = (20, 0) g. Rate of change = salt in - salt out, tank by tank; written as x' = A x with A = (Q/V) [[-1, 1], [1, -1]]. Question first: what happens after a long time? Numerical solution (rk4) and picture; the total x1 + x2 stays 20 g, because each column of A adds up to 0. One sentence: Section 5 explains the curve with two numbers, the eigenvalues | both amounts tend to 10 g; the total is 20 g at every step |
| 5 | **The eigenvalue method: real eigenvalues** (Session 18) | x = v e^(lambda t) gives A v = lambda v (Lebl 3.4). The tanks by hand: lambda = 0 with v = (1, 1) and lambda = -2Q/V with v = (1, -1); X(t), X(0) C = x(0); x = 10 (1, 1) + 10 (1, -1) e^(-t/50). Reading it: the eigenvalue 0 is the total that never changes, -0.02 /s the speed of mixing. `ol.eigen_solution` and `expm` agree with the hand result and with the rk4 curve of Section 4. Lebl's Example 3.4.4 (3 x 3) checked by substitution in one SymPy line. Example 3.8.1: e^(tA) = X(t) X(0)^(-1), the book's solution | within 1 g of the average after ln(10) / 0.02 = 115 s; Example 3.8.1: x = 3e^(3t) + e^(-t), y = 3e^(3t) - e^(-t) |
| 6 | **Complex and repeated eigenvalues** (Session 18, theory exercise) | Complex: Lebl's [[1, 1], [-1, 1]]: lambda = 1 - i, v = (i, 1); Euler's formula (video R05) splits v e^(lambda t) into real and imaginary parts, each a real solution (checked by substitution); a pair a +- ib needs only one of the two eigenvalues (Theorem 3.4.2). Repeated: Lebl, Example 3.7.1 ([[3, 1], [0, 3]]): one eigenvector, defect 1; the generalized eigenvector w from (A - 3I) w = v; the second solution (w + v t) e^(3t), checked by substitution. Why it always works for 2 x 2: (A - lambda I)^2 = 0 (the exercise in the text of Lebl's Section 3.8, not the one of the list), so any vector w with (A - lambda I) w not zero gives v = (A - lambda I) w. Lebl's simple case A = 3I + B: e^(tA) by hand, against `expm`. **Slider 1** (below) | 1 +- i; (e^t sin t, e^t cos t) and (e^t cos t, -e^t sin t); v = (1, 0), w = (0, 1) (the book's vectors); e^(3t) [[1 + 2t, 4t], [-t, 1 - 2t]] |
| 7 | **Two masses on springs** (Session 18, practical exercise) | convention box (below); Lebl, Example 3.6.1: Newton's law for each cart with the sign of each spring force written out; M x'' = K x, A = M^(-1) K = [[-3, 1], [2, -2]]; eigenvalues -1 and -4, so natural frequencies 1 and 2 (Theorem 3.6.1); the two modes drawn (Figure 3.6.3: together; in opposite directions); the solution for x(0) = (1, -1), x'(0) = (0, 6) by hand (constants from two 2 x 2 systems), the book's result. The same problem as four equations of order 1 (Section 3): the 4 x 4 matrix has eigenvalues +- i, +- 2i, and `expm` gives the same curves. **Slider 2** (below) | x1 = 2 sin t + cos 2t - sin 2t, x2 = 4 sin t - cos 2t + sin 2t; at t = 3: 1.52183 and -0.67511 by both routes |
| 8 | **Exercises** | the four steps below | - |
| 9 | **Answers** (collapsed), where to go next (notebook 18: the pictures of 2 x 2 systems in the plane, Lebl 3.5, and nonlinear systems; Lebl 3.9 for systems with a forcing term) | opens after trying | - |

**The proof of Section 3** (the theory exercise of Session 17, written in the notebook in full).
Take a_n y^(n) + ... + a_1 y' + a_0 y = 0 with a_n not 0, and the system u' = A u, where u1' = u2,
..., u_(n-1)' = u_n and u_n' = -(a_0 u1 + a_1 u2 + ... + a_(n-1) u_n) / a_n.
* If y solves the equation, put u_k = y^(k-1). Then u_k' = y^(k) = u_(k+1) for k < n, and
  u_n' = y^(n), which the equation gives as the last row. So u solves the system.
* If u solves the system, put y = u1. The first n - 1 rows give, one after the other, u2 = y',
  u3 = y'', ..., u_n = y^(n-1); then the last row, multiplied by a_n, is the equation for y.
* The initial conditions match: u(t0) = (y(t0), y'(t0), ..., y^(n-1)(t0)). So the two problems
  have the same solutions, and Picard's theorem for systems (Lebl, Theorem 3.1.1) gives existence
  and uniqueness for the equation of order n.
* det(lambda I - A) = (a_n lambda^n + ... + a_0) / a_n: the eigenvalues of A are the roots of the
  characteristic equation (Lebl's note after Example 3.4.4; checked by SymPy for n = 3).

**Slider 1 (Section 6): three cases in one family.** The state (x, x') of x'' + c x' + x = 0
(notebook 07 with m = k = 1) solves u' = A u with A = [[0, 1], [-1, -c]]. Question before moving
it: "at c = 2 the two eigenvalues are equal. Does the motion change suddenly when c passes 2?"
Left: x(t) and x'(t) for 0 <= t <= 15 from x(0) = 1, x'(0) = 0, by `ol.eigen_solution`; right: the
two eigenvalues in the complex plane, with the circle |lambda| = 1 on which they travel while they
are complex. The title says the case (`ol.two_by_two_case`) and the eigenvalues. Range c from 0 to
4, step 0.1; static figure at c = 0.5 (complex pair -0.25 +- 0.968i). Answer after it: no jump; at
c = 2 the solution is x = (1 + t) e^(-t), the "times t" bridge between the two formulas, as in
notebook 06's slider of merging roots; at c = 0 the eigenvalues are +- i and nothing decays.

**Slider 2 (Section 7): the two modes.** Cart 1 starts at x1(0) = 1 m, cart 2 at a position chosen
with the slider, both at rest. Question before moving it: "is there a start from which both carts
swing with one frequency only?" The plot shows x1 and x2 for 0 <= t <= 4 pi, and the title the
amounts of the two modes, (1 + x2(0))/3 of the slow one and (2 - x2(0))/3 of the fast one (from
the 2 x 2 system for the constants). Range x2(0) from -2 to 2 m, step 0.1; static figure at
x2(0) = 0 (amounts 1/3 and 2/3). Answer after it: x2(0) = 2 gives only the slow mode (the carts
move together, cos t); x2(0) = -1 only the fast one (mirror images, cos 2t).

**Conventions** (rule 6 of the course notation, written in a box where each problem starts, on
every figure, and in one line above each exercise that uses them):
* *Tanks* (Section 4 and Exercise 3). x1(t), x2(t): grams of salt in tank 1 and tank 2; t in
  seconds from the start; V in litres, the same in both tanks and constant; flow rates in L/s,
  each pipe with the direction of its arrow in the figure; the concentration of a tank is x/V in
  g/L; x' > 0 means the salt in that tank is increasing. Closed tanks (Lebl, Example 3.1.2): a
  pump moves Q L/s from tank 1 to tank 2 and another Q L/s back. Open tanks (Exercise 3, built
  from Lebl's Exercise 3.1.12): clean water enters tank 1 at Q_w L/s, a pump moves Q + Q_w L/s from
  tank 1 to tank 2, another moves Q L/s from tank 2 back to tank 1, and Q_w L/s leaves tank 2 for
  the drain, so that both volumes stay V; one sentence says that Lebl states only the clean water
  and the drain, and that these pipes are the notebook's choice.
* *Carts* (Section 7, slider 2, Exercise 2(d)). x1, x2: displacements of the carts in metres from
  their rest positions (springs neither stretched nor compressed), **positive away from the wall**
  (to the right in the figure); x' is the velocity, positive to the right; masses in kg, spring
  constants in N/m, t in seconds. Lebl gives Example 3.6.1 without units: the units are the
  notebook's, and it says so. A spring stretched by d pulls each end towards the other with force
  k d; the sign of every force is written out once for each cart.

Length: at most 2100 words before the exercises (two sessions, but the limit is kept). If the
draft is longer, the way back of Section 3 (Example 3.1.6) and Lebl's Example 3.4.4 move to the
exercises, and Example 3.8.1 is reduced to its formula. Nine figures; two sliders, each with its
static picture and its question first.

## 2. Exercises (the four steps)

1. **Predict** (no code) - (a) *Lebl, Exercise 3.4.1*: a 3 x 3 matrix A has the eigenvalue 3 with
   eigenvector (1, -1, 3): what is A v? ((3, -3, 9).) (b) How many equations of order 1 replace the
   system of *Lebl, Exercise 3.1.4* (two equations of order 2)? (4.) (c) *From Lebl, Exercise
   3.4.7*: the eigenvalues of [[2, 7, -1], [0, -3, 5], [0, 0, 4]], in increasing order
   ([-3, 2, 4]: a triangular matrix shows them on its diagonal). (d) *Lebl, Exercise 3.6.4*, with
   the carts of Section 7 and m1 = m2 = k1 = k2 = 1: True or False, the first cart can move while
   the second stays at rest? (False: then x2'' = 0 forces x1 = x2 = 0.)
2. **By hand, then check** - (a) *Lebl, Exercise 3.4.3*: x1' = 3x1 + x2, x2' = 2x1 + 4x2: the
   eigenvalues in increasing order (2, 5) and one eigenvector for each (any nonzero multiple of
   (1, -1) and of (1, 2) is accepted). (b) *Lebl, Exercise 3.4.4*: x1' = x1 - 2x2, x2' = 2x1 + x2:
   the eigenvalue with positive imaginary part (1 + 2i, typed `1 + 2j`), an eigenvector for it
   (any complex multiple of (i, 1)), and the real solution with x(0) = (1, 0), typed as two SymPy
   formulas (x1 = e^t cos 2t, x2 = e^t sin 2t; a value not printed in the book, checked by
   substitution and by `expm`). (c) *Lebl, Exercise 3.7.1* (the theory exercise of Session 18):
   A = [[5, -3], [3, -1]]: the repeated eigenvalue (2), an eigenvector v (a multiple of (1, 1)),
   and a generalized eigenvector w with (A - 2I) w = v **for the student's own v** (for v = (1, 1):
   (1/3, 0), or (0, -1/3), or any of the others). (d) *Lebl, Exercise 3.6.7* (the practical
   exercise of Session 18): a 2 kg cart joined by a spring k = 1 to a 3 kg cart, which is joined to
   the wall by a spring k = 1; the 2 kg cart starts 1 m from rest, away from the wall, the 3 kg cart
   at rest position, both let go: the position of the 3 kg cart as a SymPy formula in `t` (the
   answer of the book, (2/5) cos(t / sqrt(6)) - (2/5) cos t). The book's letters are kept: x1 for
   the 2 kg cart (its "first cart", the one away from the wall), x2 for the 3 kg cart, both
   positive away from the wall; the convention line says so.
3. **Fill in** (the practical exercise of Session 17) - Complete `tank_matrix(Q, Qw, V)`, the
   matrix of the open tanks of the convention box. The next cell does not compare matrices: it
   computes "salt in - salt out" for each tank from the list of pipes (flow times the
   concentration of the tank the pipe leaves; 0 for clean water), at several states x and three
   sets of (Q, Qw, V), and compares it with `tank_matrix(Q, Qw, V) @ x`; the first mismatch is
   reported in grams per second, with the tank and the numbers. Then it checks the closed case
   (Qw = 0: each column adds up to 0, the total salt is constant), solves the open tanks of the
   notebook (V = 100 L, Q = 1 L/s, Qw = 3 L/s, x(0) = (20, 0) g) with `ol.eigen_solution`, and
   draws the salt in both tanks: eigenvalues -0.02 and -0.06 /s with eigenvectors (1, 2) and
   (1, -2); tank 2 holds at most 7.70 g, at t = 50 ln(sqrt(3)) = 27.5 s, the moment when both tanks
   hold the same amount (x2' = (Q + Qw)(x1 - x2)/V); everything is washed out.
4. **Explore** - (i) With slider 2, find the two starts that excite one mode only, then predict
   x1 and x2 for each without the slider. (ii) *Lebl, Exercise 3.8.14*: e^(tA) for
   A = [[3, -1], [1, 1]] and the solution with x(0) = (1, 2) (the answer of the book); compare with
   `expm` and with `ol.eigen_solution`. (iii) *Lebl, Exercise 3.7.10*: A = [[2, 0, 0], [-1, -1, 9],
   [0, -1, 5]] has the eigenvalue 2 three times and defect 2: find a chain w3, w2 = (A - 2I) w3,
   v = (A - 2I) w2 (for w3 = (1, 0, 0): (0, -1, 0) and (0, 3, 1), the book's vectors). (iv) *Lebl,
   Example 3.6.2*: two rail cars with the zero eigenvalue; when does car 2 reach the wall 10 m
   away, and how fast? (5.22 s, 3.85 m/s, the book's values; `scipy.optimize.brentq`.)

Answers (final values only) in a collapsed block; full solutions after the sessions.

## 3. New in `odelib.py` (with tests)

A new section of the library, "Linear systems x' = A x (notebook 17)", placed after the section of
notebook 16. NumPy only at the top; `scipy.linalg` is used only in the tests. Every function has
type hints on its arguments and its return value, a docstring citing Lebl, the name of this
notebook, and its units statement.

* `companion_matrix(coefficients: Sequence[float]) -> np.ndarray`: the n x n matrix A of the
  system u' = A u for u = (y, y', ..., y^(n-1)) of a_n y^(n) + ... + a_1 y' + a_0 y = 0.
  `coefficients` are a_n, ..., a_0, the highest derivative first, as in `numpy.roots` and
  `higher_order_solution`. Rows 1 to n-1 hold the ones above the diagonal; the last row is
  -(a_0, ..., a_(n-1)) / a_n. Units: the entries of row n have the units of y^(n) / y^(k-1); the
  function is unit-free. ValueError (same conditions as `higher_order_solution`, own message):
  not a flat list, fewer than two coefficients, a value that is not finite, a_n = 0.
  `higher_order_solution` keeps its own code unchanged (additive rule).
* `two_by_two_case(matrix: ArrayLike, rtol: float = REPEATED_EIGENVALUE_RTOL) -> TwoByTwoCase`:
  what a student computes by hand for a real 2 x 2 matrix, as a `typing.NamedTuple` (readable
  fields, prints in one line):
  * `case`: "real distinct", "complex pair", "repeated, two eigenvectors" or "repeated, one
    eigenvector";
  * `trace`, `determinant`, `discriminant` (trace^2 - 4 det), floats;
  * `eigenvalues`: (lambda1, lambda2), real and increasing; for a pair a +- ib, (a + ib, a - ib)
    with b > 0, as Python complex numbers; for a repeated one, (lambda, lambda);
  * `eigenvectors`: real distinct: by the shortcut of Section 2, (b, lambda - a) from the first row
    or (lambda - d, c) from the second, whichever is larger (never the zero vector: a triangular or
    diagonal matrix is the case that breaks the shortcut, and it is tested); complex: (v, conj v);
    repeated with A = lambda I: (e1, e2); repeated, one eigenvector: (v,) with v the larger column
    of A - lambda I;
  * `generalized`: None, or, in the last case, the unit vector e_j of that column, so that
    (A - lambda I) w = v holds exactly (for [[3, 1], [0, 3]] this gives Lebl's v = (1, 0) and
    w = (0, 1)).
  The case is "repeated" when |discriminant| <= rtol * s^2, with s the largest size of an entry
  (`REPEATED_EIGENVALUE_RTOL = 1e-12`, a named constant with its reason in a comment: near a
  repeated eigenvalue the error of using the repeated formula grows like the discriminant times
  t^3, so it stays below 1e-9 on the windows of the notebook); "two eigenvectors" only when
  A - lambda I is exactly zero (a nearly diagonal defective matrix is still handled exactly by the
  one-eigenvector formulas). ValueError: not 2 x 2, complex or non-finite entries, rtol not
  positive.
* `eigen_solution(t: ArrayLike, matrix: ArrayLike, x0: Sequence[float],
  rtol: float = REPEATED_EIGENVALUE_RTOL) -> np.ndarray`: the solution of x' = A x, x(0) = x0, for a
  real 2 x 2 matrix, by the formula of its case (Lebl, Theorems 3.4.1, 3.4.2 and formula (3.7.1)):
  C1 v1 e^(lambda1 t) + C2 v2 e^(lambda2 t); C1 Re(v e^(lambda t)) + C2 Im(v e^(lambda t));
  e^(lambda t) x0; C1 v e^(lambda t) + C2 (w + v t) e^(lambda t). The constants come from
  X(0) C = x0 by `np.linalg.solve`. Returns a real array of shape `t.shape + (2,)` (column j is
  x_(j+1), the same layout as `rk4`); x(0) = x0 to rounding. Units: t in the time unit of the rates
  in A (A has units 1/time), x in the units of x0. ValueError: a matrix that is not 2 x 2, with the
  advice "for a larger system use scipy.linalg.expm(t * A) @ x0"; x0 not of length 2; values that
  are not finite; a solution too large to compute at the requested times (the message of
  `higher_order_solution`). Closed forms and vectorised NumPy only: no `expm` in a loop, so a
  slider redraw costs well under a millisecond (0.07 ms for 801 points on the desktop).
* `check_vector(label: str, value, expected, *, up_to_scale: bool = False,
  tol: float = CHECK_TOLERANCE) -> bool`: the exercise checker for vectors and matrices, with the
  one-line messages of `check`. With `up_to_scale=False` it is `check` itself (same rule, same
  messages; a matrix is typed as a list of rows). With `up_to_scale=True` (eigenvectors, which are
  never unique) the answer is accepted when it is a nonzero multiple of `expected`, real or complex:
  the best multiple k = (conj(e) . value) / (conj(e) . e) is found and the answer is right when
  |value - k e| <= tol * max(1, |value|). Messages: no answer yet (None); without quotes (text);
  the decimal comma (as `check`); "n values expected, got m"; "the zero vector is never an
  eigenvector"; "not yet (your vector does not point along an eigenvector, try again)". It never
  reveals the expected vector and never raises for a wrong answer; it raises ValueError only for a
  mistake of the notebook (an expected vector that is zero or not one-dimensional with
  `up_to_scale=True`). It replaces the list check hand-coded in notebook 16's Exercise 1 for new
  notebooks; notebook 16 is not changed.

Tests:
* `tests/test_odelib.py` (they need only NumPy, SciPy and pytest, so they also run in the site CI):
  * `companion_matrix`: Lebl's Example 3.1.4 ([[0, 1, 0], [0, 0, 1], [1, 8, 2]] from
    (1, -2, -8, -1)); notebook 16's Example 2.3.3 (eigenvalues -1, 1, 3); `np.poly` of the matrix
    equals the coefficients divided by a_n and its eigenvalues equal `np.roots`, for seeded random
    equations of order 1 to 5; order 1 ([[-a0/a1]]); `expm(t A) u0` (first entry) equals
    `higher_order_solution`; invalid input (a_n = 0, one coefficient, NaN, a nested list).
  * `two_by_two_case`: Example 3.8.1 (-1 and 3; vectors along (1, -1) and (1, 1)); the complex
    example of Lebl 3.4 (1 +- i); Exercises 3.4.3 (2, 5), 3.4.4 (1 +- 2i), 3.4.9 (the book's
    eigenvectors (-2, 1 -+ sqrt(3) i) up to a complex factor); Example 3.7.1 (v = (1, 0),
    w = (0, 1), exactly the book's); the simple case of 3.8 ([[5, 4], [-1, 1]]: 3, one
    eigenvector along (2, -1)); Exercises 3.7.1, 3.8.9, 3.8.14 (2, 2; -2, -2; 2, 2, one
    eigenvector each); 3I (two eigenvectors); the tanks and the rail cars (a zero eigenvalue);
    Example 3.6.1's A (-4, -1); the cases that break the shortcut: [[2, 0], [0, 3]], [[1, 0], [1, -1]]
    (Example 3.3.2), [[3, 1e-8], [0, 3]]; for every result: A v = lambda v with v nonzero and
    (A - lambda I) w = v; trace and determinant; invalid input (3 x 3, a row of length 3, NaN,
    complex entries, rtol <= 0).
  * `eigen_solution`: the book's solutions: Example 3.3.2 (x(0) = (1, 2)), Example 3.8.1
    (x(0) = (4, 2)), Example 3.1.6 (2.475 and 2.457 at t = 2), Exercise 3.8.14, the complex example
    of Lebl 3.4 against its two real solutions with fitted constants, Example 3.7.1; the closed
    tanks (10 +- 10 e^(-t/50)); x(0) = x0; the shape for a scalar t, a list and a 2-D array of
    times; against `scipy.linalg.expm` for seeded random matrices (SEED = 331, entries of size
    0.1, 1 and 10, relative error below 1e-9); **continuity across the cases**: the family
    [[0, 1], [-1, -c]] at c = 2 +- 10^(-k) for k = 3, ..., 14 and at c = 2, and the family
    [[-1, 1], [e, -1]] at e = +- 10^(-k) and 0, all within 1e-9 of `expm` on 0 <= t <= 10 (the
    prototype written for this plan stayed below 3e-12 at these boundaries and on 2000 seeded
    random matrices);
    invalid input (a 3 x 3 matrix, with the advice in the message; x0 of length 3; NaN; overflow).
  * `check_vector`: accepts (2, 2, 0), (-1, -1, 0) and the unit vector of `np.linalg.eig` for
    Lebl's (1, 1, 0); accepts (1, -i) for the eigenvector (i, 1) (a complex multiple) and refuses
    its conjugate (-i, 1); refuses (1, 0, 0), the zero vector (with its message), a vector of the
    wrong length, (1, 1 + 1e-3); None, text and "3,5" get the messages of `check`; a matrix with
    `up_to_scale=False` behaves exactly as `check` (same printed line); ValueError for an expected
    zero vector with `up_to_scale=True`.
* `tests/test_cross_validation.py` (skipped in CI without SymPy, as now): `eigen_solution` against
  `solve_ivp` (rtol 1e-11) for seeded random matrices of the three cases; the characteristic
  polynomial of `companion_matrix` against SymPy's `charpoly` for random rational coefficients;
  `two_by_two_case` against SymPy's `eigenvects` (eigenvalues equal, eigenvectors parallel,
  geometric multiplicity equal) for random integer matrices, including built defective ones
  (lambda I + rank-one nilpotent).

## 4. Conventions and checks

As for notebooks 14-16: same setup cell, `assert` for every reference value, one sentence under
every plot (written after looking at the figure and recomputing every number), line styles as well
as colours, a question before every slider and its answer after, named constants, plain ASCII,
generated by `plans/make_17.py` with `nbbuild.py`; `ruff check` (the course rules in
`notebooks/ruff.toml`) reports nothing on `make_17.py`, the new lines of `odelib.py` and the new
tests.

* **Setup cell.** The same required lines as notebook 16 (`import sys`, `import odelib as ol`, the
  version print), with the names already in `SETUP_NAMES` (np, sp, plt, ol, solve_ivp, interact,
  FloatSlider, IntSlider) and without `solve_bvp`. `from scipy.linalg import expm` goes in the first
  cell of Section 3 that uses it, with a comment, so `SETUP_NAMES` does not change. The existing
  hint for a bare `i` (it suggests `1j`) covers the complex answers of Exercise 2(b).
* **Letters** (course notation, Section 4; rows to add to the reserved-letter table when the
  instructor approves, before the decks of Sessions 17-18):
  * x(t) the vector unknown, with components x1, x2, ... (rule 1: no quantity named; Lebl); in the
    tanks x1, x2 are grams of salt, in the carts metres, as in the book, said once in each box.
  * A the matrix of x' = A x, as in notebook 08 and Lebl 3.6-3.8 (Lebl writes P in 3.3-3.4: one
    sentence says so). A as the ambient temperature and A, B as undetermined coefficients do not
    occur in this notebook. Lebl's A and B for the initial salt in Example 3.1.2 become x1(0),
    x2(0) (an initial value is never A), said once.
  * lambda, lambda1, lambda2 the eigenvalues of A (as the text). The style sheet reserves lambda
    for the parameter of x'' + lambda x = 0 (Session 16) and, announced, for a characteristic
    unknown (Session 12): the new row says "the eigenvalues of a matrix (Session 17 on)", and
    Section 3 says once that for a companion matrix they are the roots r of notebook 06.
  * v, v1, v2 eigenvectors (as the text). v is reserved for a substitution and a velocity, and v0
    for an initial slope: in this notebook a velocity is always written x' (never v), and no
    substitution occurs.
  * w the generalized eigenvector (Lebl writes v2 in Section 3.7; w avoids "v2" meaning a second
    eigenvector in Section 5 and a generalized one in Section 6; said once). w as a load belongs to
    Session 16's beam only. Chains in Explore (iii): w3, w2, v.
  * C1, C2, ... and the column C for the constants (rule 3; Lebl writes c1, c2 and a vector c).
  * X(t) the fundamental matrix, I the identity matrix, e^(tA) the matrix exponential (Lebl).
  * u1, ..., un the variables of the conversion (Lebl 3.1.2), announced; u as a substitution does
    not occur here. In Section 3 the equation is written for y(t) with coefficients a_n, ..., a_0
    (Session 16's letters, with t in place of x, said once); Example 3.1.4 keeps its x.
  * Q for the flow between the tanks and Q_w for the clean water (`Qw` in code). Lebl writes r and
    s; r is reserved for characteristic roots and s for the variable of an integral, while Q is
    the course's flow rate: one sentence compares.
  * M, K the mass and stiffness matrices of Section 7 (Lebl 3.6), announced there; M as a
    population and K1, ..., Kn as test constants do not occur.
  * omega1 = 1 and omega2 = 2, the natural frequencies of the two modes (Lebl, Theorem 3.6.1). The
    style sheet reserves omega1 for the pseudo-frequency of a damped motion (Session 14), which
    does not occur here; Decision 2 below.
* **Lessons of the reviews of 14-16, applied from the start:** no global name that an exercise
  defines as a function (`tank_matrix` is used nowhere else; Section 4's matrix is called
  `closed_tanks`); tolerances of `ol.check` and `ol.check_vector` chosen by looking at the nearest
  plausible wrong answer (an eigenvector of the other eigenvalue; the conjugate eigenvector; w = v;
  the formula of the 2 kg cart for the 3 kg one); every number in a title is what the title says
  it is; quotations only when literal; tables of rules tested on the case that would break them
  (the shortcut of Section 2 on triangular and diagonal matrices; the three cases at their
  boundary); a single number typed where a list is expected does not stop the notebook (the
  `check_vector` messages); cross-references to sections of other notebooks checked against their
  `make_NN.py`.
* `tests/test_notebooks_run.py`:
  * `NAME_17 = "17_Linear_Systems.ipynb"`; the notebook as it is (every checker prints "no answer
    yet" or "complete tank_matrix first");
  * with the correct answers, including the other accepted forms (eigenvectors (-2, 2) and (3, 6);
    (1, -1j) for (b); w = (0, -1/3) for v = (1, 1); the matrix written with `np.array` or as a list
    of lists);
  * with wrong answers: an eigenvector of the other eigenvalue, the zero vector, the conjugate
    eigenvector, w = v, the 2 kg cart's formula in place of the 3 kg cart's, and three wrong tank
    matrices (transposed; without the division by V; the matrix with Q - Qw that keeps neither the
    volumes nor the salt balance); each caught with its message;
  * a test that Section 6 runs again after Exercise 3 (no clash of names);
  * `EXTREMES[NAME_17]`: `plot_cases(c=0.0)`, `plot_cases(c=4.0)`, `plot_cases(c=2.0)`,
    `plot_cases(c=0.0 + 20 * 0.1)` (2.0 as the slider gives it), `plot_cases(c=2.0 - 1e-10)`,
    `plot_cases(c=2.0 + 1e-10)`, `plot_modes(x2_start=-2.0)`, `plot_modes(x2_start=2.0)`,
    `plot_modes(x2_start=-1.0)`, `plot_modes(x2_start=-2.0 + 10 * 0.1)` (-1.0 as the slider gives
    it), `plot_modes(x2_start=0.0)`; with RuntimeWarning turned into an error, as for every
    notebook.
* **Time.** Library tests: about 45 new tests, under 2 s. Notebook tests: about 10-15 s more (three
  executions of the notebook and the extremes). In the browser (`check_site.py`): estimated 15-20 s,
  inside the present 12-19 s per notebook: the costly steps are the SymPy calls (about ten small
  `eigenvects`, `charpoly`, `simplify` and one symbolic 2 x 2 exponential, which is replaced by the
  hand formula checked against `expm` if it takes more than 2 s in the browser), `expm` on at most
  201 times per figure, rk4 on 4 x 4 with at most 1000 steps, and nine figures. Each cell is timed
  in the browser, as for notebook 16.
* **Registries** (all in the private working copy):
  * `plans/make_00.py`, one row after 16:
    `| 17 | [Linear systems and the eigenvalue method](17_Linear_Systems.ipynb) | $\vec{x}\,' = A\vec{x}$ | 17-18 | 3.1-3.4, 3.6-3.8 |`;
    then `00_Index.ipynb` is rebuilt;
  * `site/build_site.py`, `READING_ORDER`: a line
    `("Week 9 - Linear systems and the eigenvalue method (Sessions 17-18)", ("17",))` after Week 8;
    `START_HERE`, `GUIDE_NOTEBOOK`, `PYODIDE_PACKAGES` unchanged;
  * `tests/test_notebooks_run.py`: `NAME_17`, `EXERCISES[NAME_17]`, `EXTREMES[NAME_17]`;
  * `odelib.SETUP_NAMES`: unchanged (no new name in the setup cell);
  * `notebooks/README.md` (the list) and `site/README.md` (counts and the table of tested
    notebooks); the Contents list of the odelib docstring gets the four names under "Linear
    systems" in the housekeeping step of the notebooks plan, not here;
  * no other notebook changes: notebook 16 already points to 17 by number.
* The site is rebuilt and the notebook is run in the browser with `check_site.py`; the export is
  checked for ASCII, size and stored local paths.
* After the build: an independent review of the mathematics, then a reading as the weakest student,
  as for 14-16, and their corrections.

## 5. Order of work

1. `odelib.py`: the four functions and `REPEATED_EIGENVALUE_RTOL`, tests first, from the values
   above; ruff on the new lines; the whole existing suite passes unchanged (295 + 111), and the
   library tests also in the CI environment (NumPy, SciPy and pytest only).
2. `plans/make_17.py`: sections 1-4 (Session 17); look at the figures.
3. Sections 5-6 with slider 1.
4. Section 7 with slider 2; word count; decide what moves to the exercises.
5. Exercises and answers; `EXERCISES` and `EXTREMES` in the tests.
6. Index row, `READING_ORDER`, READMEs; rebuild; `check_site.py` with the times of the cells.
7. Independent review; reading as the weakest student; corrections; final run of all tests.

## 6. Decisions for the instructor

1. **One notebook for two sessions** (as the notebooks plan fixes). Proposed: Sections 2-4 for
   Session 17 and 5-7 for Session 18, with the exercises marked by session (Exercise 3 and 1(a)-(c)
   after Session 17, the rest after Session 18). The alternative, two notebooks (17 and 17E), would
   put the eigenvalue method in an optional notebook, which it is not.
2. **Letters.** Proposed: A, lambda, v, w, X(t), C1, C2, u1, ..., un, Q and Q_w, M and K, as in
   Section 4 of this plan, added to the reserved-letter table before the decks of Sessions 17-18.
   For the natural frequencies, proposed: omega1 and omega2 as in Lebl's Theorem 3.6.1, announced
   (the pseudo-frequency omega1 of Session 14 does not occur in Unit 8); the alternative is to name
   them in words ("the slow frequency 1, the fast frequency 2").
3. **The open tanks.** Lebl's Exercise 3.1.12 gives only the clean water in and the drain out. As
   the answer reads in the text copy (x2' = (r/V) x1 - ((r - s)/V) x2), the volumes do not stay
   constant and its determinant -rs/V^2 is negative, so the amounts do not go to zero, against part
   (c) of the same exercise. Proposed: the notebook states its own pipes (Q + Q_w from tank 1 to
   tank 2, Q back), says in one sentence that they are its choice, and does not quote the book's
   answer. Please confirm, or tell me if you prefer the book's statement with a remark on it.
4. **`eigen_solution` for 2 x 2 only** (the cases done by hand). Proposed: larger systems
   (Example 3.4.4, the 4 x 4 form of the carts, Exercise 3.7.10) use `scipy.linalg.expm` in the
   cells, and the error message of `eigen_solution` says so. The alternative is an n x n version for
   diagonalizable matrices, which adds code without adding a case the students compute.
5. **Units in Example 3.6.1** (Lebl gives none). Proposed: kg, N/m, m and s, said to be the
   notebook's; the tank numbers (V = 100 L, Q = 1 L/s, Q_w = 3 L/s, 20 g) are also the notebook's.
6. **Left out, for notebook 18 or an optional 17E later** (proposed: decide after notebook 18):
   the pictures in the plane (Lebl 3.5, the practical exercise of Session 19); forced systems
   (Lebl's forced oscillations in 3.6, dynamic damping of Exercise 3.6.2: m2 = 2/25 kg; Section
   3.9); the full rail-car story of Example 3.6.2 beyond Explore (iv).
7. **`expm` outside the setup cell.** Proposed: imported in Section 3 with a comment, so that
   `SETUP_NAMES` is unchanged. The alternative is the setup cell plus one name in `SETUP_NAMES`, so
   that a cell run out of order gets the usual hint.

## 7. Files

```
ode/ode/notebooks/
    17_Linear_Systems.ipynb                    <- to build
    plans/17_Linear_Systems_plan.md            <- this document
    plans/make_17.py                           <- builds the notebook
    odelib.py                                  <- + companion_matrix(), two_by_two_case() with TwoByTwoCase,
                                                  eigen_solution(), check_vector(), REPEATED_EIGENVALUE_RTOL
    tests/test_odelib.py                       <- + their tests against the book and expm
    tests/test_cross_validation.py             <- + against solve_ivp and SymPy
    tests/test_notebooks_run.py                <- + NAME_17, the exercises and sliders of notebook 17
    00_Index.ipynb, plans/make_00.py           <- index row
    README.md                                  <- the list of notebooks
ode/ode/site/build_site.py                     <- READING_ORDER: Week 9 with 17
ode/ode/site/README.md                         <- counts and the table of tested notebooks
```

## 8. As built (2026-11-20)

**The notebook.** `17_Linear_Systems.ipynb`, 40 cells (18 code), built by `plans/make_17.py`; 785 KB,
plain ASCII, no local path stored. Before the exercises: **2309 words** counted as for plans 14-16
(text split on spaces, every LaTeX token counted), **1572** when each formula counts as one word. The
fall-back of Section 1 was applied in full (Example 3.1.6 and Example 3.4.4 moved to Exercise 4 (v)
and (vi); Example 3.8.1 reduced to its formula and its check), and every text cell was shortened; the
remaining excess over 2100 is the LaTeX of 21 matrices (about 250 tokens). Not cut further, because
what is left is the content of Section 1 of this plan: the instructor decides.

**Figures: 9 stored** before the exercises (the vectors $\vec{v}$ and $A\vec{v}$ of Example 3.8.1; Example
2.3.3 by rk4 against notebook 16; the closed tanks with their drawing; the excess of salt on a log
scale; the two real solutions of the complex example; slider 1 static; the carts' drawing and the two
modes; Figure 3.6.4 by hand against `expm` of the 4 x 4 system; slider 2 static), plus one in Exercise 3
(the open tanks, drawn once `tank_matrix` is right). **Sliders: 2** (`plot_cases(c)`, `plot_modes(x2_start)`),
each with its question first, a static figure and its answer.

**Functions added to `odelib.py`** (new section "Linear systems x' = A x (notebook 17)" after notebook 16;
`check_vector` after `check_solution`; nothing existing changed; `SETUP_NAMES` unchanged):
* `companion_matrix(coefficients: Sequence[float]) -> np.ndarray`
* `class TwoByTwoCase(NamedTuple)`: case, trace, determinant, discriminant, eigenvalues, eigenvectors, generalized
* `two_by_two_case(matrix: ArrayLike, rtol: float = REPEATED_EIGENVALUE_RTOL) -> TwoByTwoCase`
* `eigen_solution(t: ArrayLike, matrix: ArrayLike, x0: Sequence[float], rtol: float = REPEATED_EIGENVALUE_RTOL) -> np.ndarray`
* `check_vector(label: str, value, expected, *, up_to_scale: bool = False, tol: float = CHECK_TOLERANCE) -> bool`
* constants `REPEATED_EIGENVALUE_RTOL = 1e-12`, `CASE_REAL`, `CASE_COMPLEX`, `CASE_REPEATED_TWO`,
  `CASE_REPEATED_ONE`, `SYSTEM_SIZE`, `LARGER_SYSTEMS`; private helpers `_real_two_by_two`, `_shortcut_eigenvector`;
  `from typing import NamedTuple` at the top. The module docstring "Contents" now lists every public name.

**Tests added.** `test_odelib.py`: 92 (book values of Examples 3.1.4, 3.3.2, 3.1.6, 3.4.x, 3.6.1-3.6.2,
3.7.1, 3.8.1 and Exercises 3.4.3, 3.4.4, 3.4.9, 3.7.1, 3.8.9, 3.8.14; 18 matrices incl. the ones that break
the shortcut; `expm` on 150 seeded random matrices; continuity at 25 points across the case boundaries
of two families, worst gap 8e-11 against the 1e-9 required; invalid input; `check_vector`).
`test_cross_validation.py`: 10 (solve_ivp, SymPy `charpoly`, SymPy `eigenvects` incl. built defective
matrices; skipped without SymPy). `test_notebooks_run.py`: 24 (the notebook as it is, right answers,
wrong answers, 18 particular answers: other accepted forms, each wrong answer with its message, a number
where a list is expected; Section 6 run again after Exercise 3; 11 slider extremes).

**Departures from the plan, and why.**
1. Section 1 gives Example 3.1.6 as $x = (2e^t + e^{-2t})/3$: wrong (it gives $x(2) = 4.93$). The book's
   and the notebook's is $x = (e^t + 2e^{-2t})/3$, which gives 2.475.
2. Word count above 2100 (see above).
3. `check_vector(up_to_scale=True)` accepts when $|value - k e| \le tol\,|value|$, not $tol \max(1, |value|)$:
   with the plan's rule any vector shorter than `tol` would be accepted. A list inside a list gets its
   own message ("one list of n numbers ... without brackets inside it").
4. `eigen_solution`, case "repeated, one eigenvector": evaluated as $e^{\lambda t}(x_0 + t(A - \lambda I)x_0)$, the
   same function as $C_1\vec{v}e^{\lambda t} + C_2(\vec{w} + \vec{v}t)e^{\lambda t}$ with $X(0)C = x_0$, but exact also when
   rounding leaves $(A - \lambda I)^2$ slightly nonzero (then the $C$ form divides by a tiny eigenvector).
5. `two_by_two_case` computes the discriminant as $(a - d)^2 + 4bc$ (the same number, less rounding) and
   the real eigenvalues without cancellation; $-0.0$ is printed as 0.
6. Section 7: the constants are $C_1, \dots, C_4$ (rule 3), with "Lebl: $a_i$, $b_i$" said once.
7. Exercise 3 prints the eigenvalues in increasing order: $-0.06$, $(1, -2)$ and $-0.02$, $(1, 2)$.
8. Exercise 2(c): the message for a wrong $\vec{w}$ says "(A - lambda I) w", so it does not reveal the eigenvalue.
9. A figure of $\vec{v}$ and $A\vec{v}$ in Section 2 (not listed in Section 1) keeps the count of nine figures
   after Example 3.1.6 left Section 3.
10. The setup cell imports `solve_ivp` with `noqa`, as notebook 15 (it is in `SETUP_NAMES`; unused here).
11. `plans/make_00.py`: the new row (fixed by Section 4) is 118 characters, one E501 more in that file
    (a Markdown table row cannot be wrapped; the other 16 rows have the same finding).

**Registries and documents.** `make_00.py` row and `00_Index.ipynb` rebuilt; `READING_ORDER` Week 9;
`EXERCISES`, `EXTREMES`, `NAME_17`; `notebooks/README.md` (the library list); `site/README.md` (rows
14-17 of the tested table, last browser run, site size); `ode/ode/README.md` ("40 tests" -> 532).

**Gates (2026-11-20).**

| Gate | Result | Time |
|---|---|---|
| Library, env `ode` | 397 passed (295 + 102) | 15 s |
| Library as in CI (`site/.venv`, no SymPy) | 274 passed, 8 skipped (182 + 92) | 11 s |
| Notebooks (`test_notebooks_run.py`, 8 kernels) | 135 passed (111 + 24) | 2 min 50 s |
| Lint (`ruff`, `notebooks/ruff.toml`) | `make_17.py` clean; no new finding in the changed files except item 11 | seconds |
| Site build | 22 notebooks, 133 MB, Pyodide 314.0.6 with 29 packages | 18 s |
| `site/test_build_site.py` | 5 passed | 1 s |
| Browser (`check_site.py`, headless Edge) | 22 of 22 without errors; notebook 17: 18 cells, 11 plots, 2 sliders, **18 s** | 7.4 min |

Not done (by instruction): export, the independent review of the mathematics and the reading as the
weakest student (Section 5, step 7).

### Corrections after the reviews (2026-11-20)

Three independent reviews (`reviews_nb/17_math_review.md`, `17_beginner_reading.md`,
`17_code_review.md`) and the coordinator's decisions 1-6. One line per finding.

**Mathematics review** (0 ERROR, 5 WARNING, 16 NOTE)
* W1 "What you will learn" without hypotheses: fixed with the reviewer's sentence (linear, constant
  coefficients; the carts are x'' = Ax, or x' = Ax with four unknowns).
* W2 citation of X(t)X(0)^(-1): fixed (Theorem 3.8.1 for e^(tA) x(0), Section 3.8.4 for X(t)X(0)^(-1)).
* W3 Exercise 3 and Lebl's Exercise 3.1.12: fixed with the coordinator's wording ("Lebl keeps the pipes
  of Example 3.1.2 (r L/s each way) ...; this notebook changes the pipe from tank 1 to tank 2").
* W4 no row for A = lambda I: fixed (new row of the table of Section 1).
* W5 Exercise 4(iv) direction: fixed ("positive towards the wall, as in the book: the opposite of
  Section 7's direction").
* N6 "a fundamental matrix", any two independent solutions: fixed.
* N7 "n distinct real eigenvalues": fixed.
* N8 the other eigenvalue gives the same real part and minus the imaginary part: fixed.
* N9 a w exists for every eigenvector (k w for k v): fixed, in the "quick way" paragraph.
* N10 "ol.two_by_two_case takes the longer of the two": fixed.
* N11 coinciding curves of the complex figure: fixed by the new figure (5b below).
* N12 the theorem says "correspond one to one": fixed.
* N13 "v an eigenvector": fixed.
* N14 alpha + i beta, beta > 0 in TwoByTwoCase, two_by_two_case and eigen_solution: fixed.
* N15 nearly equal diagonal labelled "one eigenvector": fixed; inside the band a12 = a21 = 0 is
  CASE_REPEATED_TWO, and eigen_solution evaluates both repeated cases as e^(lambda t)(x0 + t(A - lambda I)x0),
  so it stays accurate (test added).
* N16 comment above REPEATED_EIGENVALUE_RTOL: "s t at most about 20".
* N17 w-check of Exercise 2(c): fixed; complex arrays, |(A - lambda I)w - v| <= CHECK_TOLERANCE |v|
  (also code review M9; a complex v and its w are now accepted, tested).
* N18 right values in the wrong order: fixed; 1(c) and 2(a) say "right values: write them in increasing
  order" (tested).
* N19 "From Lebl" for 1(b), 2(b), 2(c): fixed.
* N20 "(Lebl, (3.1.8), there for any equation)" and "Lebl writes P for the system matrix there" (moved to
  Section 5): fixed.
* N21 (a) "clean water" and "drain" on the drawings of the open tanks: fixed; (b) the entries of a 2 x 2
  matrix are now a11, a12, a21, a22 everywhere (coordinator, decision 3; STYLE_SHEET rows), so the
  slider's damping c does not clash; one clause "c is the damping" added; (c) letters: the coordinator
  added the rows to STYLE_SHEET Section 4.

**Beginner reading** (all BLOCKING and CONFUSING applied with the reviewer's wording; POLISH as below)
* 0.1 R09 in the references: fixed (references in one line).
* 1.1 BLOCKING identity matrix: fixed. 1.2 the 2 x 2 determinant shown: fixed (with a11 ... a22).
* 1.3 BLOCKING e^(tA): defined by its series (video R09) and X(0)^(-1) said; fixed.
* 1.4 C = (C1, C2) as a vector, notebook 08: fixed. 1.5 x' of a vector, row k: fixed. 1.6 the
  "eigenvalue" of notebook 16: fixed, one sentence.
* 3.1 the 3 x 3 expansion, 3.2 the row reduction shown, 3.4 why the second row holds, 3.5 complex
  multiples: fixed. 3.3 BLOCKING triangular matrices: fixed (taught before Exercise 1(c)).
* 4.1 comments on eigenvects and np.linalg.eig: fixed; 4.2 readable output (`report()` in place of the
  TwoByTwoCase print, also in Sections 5 and 6), and the worked shortcut for [[1, 2], [2, 1]]: fixed.
* 5.1 legend "A v, same colour" and caption: fixed. 5.2 "drawing code: you can skip reading it": fixed.
* 7.1 "solved for y^(n)", 7.2 det(lambda I - A) against det(A - lambda I), 7.3 several and nonlinear
  equations, 7.4 Example 3.1.4 in one sentence (details in the code comment): fixed.
* 8.1-8.3 comments (rk4 with a vector, states, coefficients, charpoly): fixed.
* 10.1 "figure below the next cell", 10.2 how A is read off, 11.1 "V litres", 11.2 sum(axis): fixed.
* 12.1 the column sums give x1' + x2' = 0: fixed (in Section 4, answer shortened).
* 13.1 the shortcut with its numbers, 13.2 the order 0, -0.02, 13.3 the constants, 13.4 Example 3.8.1
  with its steps: fixed. 14.1 np.outer replaced by np.column_stack, 14.2, 14.3 sp.expand: fixed.
* 15.1 logarithmic axis wording: fixed.
* 16.1 1 - i and (1, -i) = -i (i, 1), 16.2 i (-i sin t) = sin t, 16.3 "defect" defined: fixed.
  16.4 "A quick way to the pair" replaces "Why w always exists"; Lebl's simple case moved to the statement
  of Exercise 4(ii), its check kept in the code of Section 6: fixed.
* 17.1 the complex figure: fixed (5b below). 17.2 comments on applyfunc, re, im: fixed.
* 18.1 eigenvalues = roots of r^2 + c r + 1, 18.2 why the circle, 18.3 the slider's units: fixed;
  19.1 labels: fixed (and decision 5a).
* 22.1 Lebl's K carries the signs, 22.2 the step to A v = -omega^2 v, 22.3 rad/s, 22.4 the factor 2:
  fixed. 22.5 the block matrix: moved to the code comment, one sentence in the text.
* 23.1 comments in the carts' code: fixed. 26.1 shorter caption: fixed.
* 29.1 the slider answer no longer gives Exercise 4(i) away: fixed.
* 30.2/30.3 first steps for 1(b), 1(d): fixed. 31.1 the comments that gave the answers away: removed.
* 32.1-32.3 first steps for 2(b), 2(c), 2(d): fixed.
* 34.1 BLOCKING Exercise 3: the drawing of the open tanks before the fill-in (`draw_tanks`), its
  caption, a "First step", what to return and what the plot will show: fixed.
* 36.1 the exception with its text ("NameError: name 'q' is not defined"): fixed (tested).
* 37.2 BLOCKING Exercise 4(ii) with B^2 = 0 and the stopped series: fixed. 37.3 "two eigenvectors
  missing", 37.4 the direction (math W5) and the first step with brentq, 37.5 y(0), y'(0) and the call,
  37.6 the hint for (vi): fixed.

**Code review** (1 IMPORTANT, M1-M15)
* I1 NaN in the Exercise 3 checker: fixed ("not <=" and NaN-safe argmax); test added (`np.full((2, 2), np.nan)`).
* M1 the case decided on A / s: fixed; tests with 1e-200 and 1e155 entries.
* M2 no RuntimeWarning before the ValueError: fixed (trace and determinant inside errstate; tested with
  warnings as errors).
* M3 companion_matrix with a tiny a_n: ValueError, tested.
* M4 0 * inf: x0 = 0 returns zeros; a term with constant 0 stays 0 (`_scaled_by`); tested.
* M5 check_vector(label, value, expected, tol=..., *, up_to_scale=...): tol positional, as check; tested.
* M6 "value(s) expected" in check_vector, as check; tests updated.
* M7 a SymPy column vector is read as a list: fixed, tested (NumPy column).
* M8 type hints: TwoByTwoCase fields typed; `MatrixLike = Sequence[Sequence[float]] | np.ndarray` for the
  matrix arguments.
* M9 w-check: fixed with N17.
* M10 E501 of make_00.py's table row: `[lint.per-file-ignores] "plans/make_00.py" = ["E501"]` in
  notebooks/ruff.toml.
* M11 literals: LOG_MARKS, SQUARE_FIGURE, CHECK_WINDOW_344, CHECK_WINDOW_367 and the drawing geometry
  (PIPE_LEVELS, LABEL_GAP, CART_HALF_WIDTH, COIL_SIZE, ARROW, ...) named.
* M12 expm in the setup cell: NOT applied (coordinator: decision 7 of this plan stands).
* M13 ruff.toml comment: reworded (no `--diff`; `--no-cache`; keep the findings on added lines).
* M14 NUMERATOR_RANGE and DENOMINATOR named in test_cross_validation.py; the other points need no change.
* M15 notebooks/README.md: "about three minutes".

**Coordinator's decisions 5 and 6**
* 5a every figure is drawn at the width it is shown (8 in): the tank drawing above its plot, the slider-1
  motion above the eigenvalue plane, the carts' drawing above the two modes; no figure is 11-12 in wide
  any more, so the labels have the size of the other figures'.
* 5b the complex figure shows the two real solutions divided by e^t in two panels (real part, imaginary
  part), 0 <= t <= 4 pi; the caption says so and what to see.
* 6 length: see the count below.

**Length (decision 6).** After the corrections: **2858 words** before the exercises counted as in plans
14-16 (text split on spaces, every LaTeX token counted), **2032** with each formula counted as one word
(before the reviews: 2309 and 1572). The reviewer's cuts and moves were taken (references in one line,
Example 3.1.4 and the 4 x 4 form of the carts into code comments, the "quick way" paragraph, Lebl's simple
case into Exercise 4(ii), shorter answers and captions); the added explanations are those the reviews
require (I, e^(tA), det of a 2 x 2 and of a 3 x 3, the triangular fact, first steps, the a11 ... a22
notation, which alone costs many LaTeX tokens). The count is above the 2250-2350 the coordinator
accepted; reaching it would mean removing explanations the beginner reading marked BLOCKING or
CONFUSING, so the instructor decides (clarity before length).

**Gates after the corrections (2026-11-20).**

| Gate | Result | Time |
|---|---|---|
| Library, env `ode` | 402 passed (295 + 107) | 17 s |
| Library as in CI (`site/.venv`, no SymPy) | 279 passed, 8 skipped | 8 s |
| Notebooks (`test_notebooks_run.py`) | 139 passed (111 + 28) | 2 min 54 s |
| Lint (`ruff --no-cache`) | make_17.py, make_00.py clean; odelib.py and the three test files: no finding on new lines (old counts unchanged) | seconds |
| Site build | 22 notebooks, 133 MB | 18 s |
| `site/test_build_site.py` | 5 passed | 1 s |
| Browser (`check_site.py`) | 22 of 22 without errors; notebook 17: 19 cells, 12 plots, 2 sliders, **17 s** | 7.4 min |

**Re-check (`reviews_nb/17_recheck.md`: ready to publish; 10 MINOR, beginner 18.3 PARTLY), applied with its fixes**
* P1 (STYLE_SHEET row for M, K): fixed by the coordinator, not in the notebook.
* P2 triangular wording: "the product of the diagonal entries of A - lambda I, (a11 - lambda)(a22 - lambda)...".
* P3 Exercise 4(ii): B = A - 2I only (Lebl's matrix written as [[5, 4], [-1, 1]] - 3I); "since 2I commutes
  with B (2I B = B 2I), e^(tA) = e^(2t) e^(tB)".
* P4 Exercise 4(iv): `brentq` with `from scipy.optimize import brentq`, as notebooks 10 and 12.
* P5 "the next vector": "(or (0, 1) if (A - lambda I)(1, 0) is the zero vector, as for Example 3.7.1)".
* P6 "a_ij is in row i, column j", once in Section 1.
* P7 complex figure: legends on top (`COMPLEX_WINDOW`), off the curves.
* P8 w-check: w flattened to 1-D (a SymPy column works; test added); the complex value shown in full.
* P9 literals: `N_POINTS` defined in Section 3 and used for the complex figure; `SIMPLE_CHECK_TIMES`;
  `TANK_DRAWING_HEIGHT`.
* P10 / beginner 18.3: slider 1 says "x > 0 in the positive direction of notebook 07"; answers 4(iv)
  "towards the wall" and 2(d) "in m, > 0 away from the wall"; 1(b) and 4(v) say "the unknowns are x(t),
  y(t), the independent variable t".
* P11 (a) "the cases of notebook 06 (a repeated eigenvalue splits in two)"; (b) "(Section 3)" dropped in
  the slider-1 answer; (c) STYLE_SHEET rows: the coordinator's.
* Optional cuts taken: the last paragraph of Section 3 (the code comments say it) and the shorter caption
  of the Example 2.3.3 figure; the others kept (they carry explanation).
* Length after the re-check: **2850 words** (whitespace tokens), **2013** with each formula as one word.

**Gates after the re-check (2026-11-20).**

| Gate | Result | Time |
|---|---|---|
| Library, env `ode` | 402 passed | 17 s |
| Library as in CI (`site/.venv`) | 279 passed, 8 skipped | 7 s |
| Notebooks (`test_notebooks_run.py`) | 140 passed (111 + 29) | 2 min 50 s |
| Lint (`ruff --no-cache`) | make_17.py clean; no new finding in odelib.py and the tests | seconds |
| Site build | 22 notebooks, 133 MB | 16 s |
| `site/test_build_site.py` | 5 passed | 1 s |
| Browser (`check_site.py`) | 22 of 22 without errors; notebook 17: 19 cells, 12 plots, 2 sliders, **18 s** | 7.2 min |
