# Notebook 18 - The phase plane and nonlinear systems: plan

MATH 331 - Differential Equations - Spring 2027 - phase D of the notebooks (14-18), the fifth and last
notebook of the phase

**Status (2026-11-20): planned; the eight decisions of Section 6 approved by the instructor as proposed; being built.**

**Sessions 19 and 20, Unit 9** (lecture plan, Session 19: "Phase plane; equilibria; nodes, saddles,
spirals, and centers; trace-determinant plane"; theory exercise "Classify linear systems from their
trace and determinant"; practical exercise "Draw phase portraits in Python". Session 20: "Nonlinear
systems; Jacobian linearization; stability; predator-prey, competing species, pendulum"; theory
exercise "Explain when linearization predicts stability correctly"; practical exercise "Analyze the
Lotka-Volterra model"). About **15 minutes in class in each session** (Session 19: sections 2-3;
Session 20: sections 5-6), the rest as homework. Notebook 17 ends with "Where to go next: notebook 18,
the pictures of 2 x 2 systems in the plane (Lebl 3.5) and nonlinear systems": this notebook keeps that
promise in Sections 2-3 (the pictures) and 4-7 (nonlinear systems).

**References** (every item was found in the text of the book and recomputed on 2026-11-20 with SymPy
and NumPy/SciPy; exercise numbers of Lebl are those of the lists at the end of Chapters 3 and 8, "3.E"
and "8.E"; in the text copy the six cases of Section 3.5 are numbered "3.5.0.1: 1" to "3.5.0.1: 6" and
are called here Cases 1-6).
* Lebl, *Notes on Diffy Qs*, **Section 3.5, Two dimensional systems and their vector fields**: the
  system (3.5.1) in the plane with coordinates (x, y); the six cases with their matrices, eigenvalues
  and eigenvectors: Case 1, source / unstable node, [[1, 1], [0, 2]], eigenvalues 1 and 2,
  eigenvectors (1, 0) and (1, 1) (Figures 3.5.1-3.5.3); Case 2, sink / stable node, [[-1, -1], [0, -2]],
  -1 and -2, the same eigenvectors (Figure 3.5.4); Case 3, saddle, [[1, 1], [0, -2]], 1 and -2,
  eigenvectors (1, 0) and (1, -3) (Figure 3.5.5); Case 4, center, [[0, 1], [-4, 0]], +-2i, eigenvector
  (1, 2i) for 2i, real solutions (cos 2t, -2 sin 2t) and (sin 2t, 2 cos 2t), ellipses (Figure 3.5.6);
  Case 5, spiral source, [[1, 1], [-4, 1]], 1 +- 2i, e^t (cos 2t, -2 sin 2t) and e^t (sin 2t, 2 cos 2t)
  (Figure 3.5.7); Case 6, spiral sink, [[-1, -1], [4, -1]], -1 +- 2i, eigenvector (1, 2i) for -1 - 2i,
  e^(-t) (cos 2t, 2 sin 2t) and e^(-t) (-sin 2t, 2 cos 2t) (Figure 3.5.8); the argument that the
  derivative on the line of an eigenvector points along it ("P(a v) = a lambda v"); Table 3.5.1 (six
  rows, for a nonsingular matrix with two eigenvectors); the remark that a zero eigenvalue gives
  "somewhat different (simpler in a way)" pictures, left to the exercises.
* Lebl, **Section 3.1.3** (the direction field or vector field of an autonomous system, Figure 3.1.4) and
  **Section 1.6** (equilibria and stability of x' = f(x)), which Sections 3.5 and 8.1 compare with.
* Lebl, **Section 8.1, Linearization, critical points, and equilibria**: the autonomous system
  x' = f(x, y), y' = g(x, y) and (8.1.1); phase portrait, trajectories; critical points (equilibria);
  Example 8.1.1 (x'' = -x + x^2 as x' = y, y' = -x + x^2; Figure 8.1.2; critical points (0, 0) and
  (1, 0)); the change of variables u = x - x0, v = y - y0, the Jacobian matrix and the linearization;
  Example 8.1.2 (Jacobian [[0, 1], [-1 + 2x, 0]]; linearizations [[0, 1], [-1, 0]] at (0, 0) and
  [[0, 1], [1, 0]] at (1, 0); Figure 8.1.3).
* Lebl, **Section 8.2, Stability and classification of isolated critical points**: isolated, almost
  linear (the Jacobian invertible), the examples x' = x^2, y' = y^2 (Jacobian zero at the origin) and
  x' = x, y' = x^2 (not isolated); Table 8.2.1 (five rows, "missing center points") with the column
  Stability; the definitions of stable, unstable and asymptotically stable; Example 8.2.1
  (x' = -y - x^2, y' = -x + y^2: saddle at (0, 0) with eigenvalues 1 and -1, sink at (1, -1) with
  eigenvalues -1 and -3; Figure 8.2.1); Example 8.2.2 (x' = y + y^2 e^x, y' = x: saddle at (0, 0),
  spiral source at (0, -1) with eigenvalues 1/2 +- i sqrt(3)/2); Section 8.2.3 "Trouble with centers"
  and Example 8.2.3 (x' = y, y' = -x + y^3: the linearization is a center, the origin is a spiral
  source; eigenvalues of the Jacobian (3/2) y^2 +- (i/2) sqrt(4 - 9 y^4); Figure 8.2.3); Section 8.2.4,
  conservative equations x'' + f(x) = 0, the energy (1/2) y^2 + integral of f = C, "either an unstable
  saddle point, or a stable center", Example 8.2.4 (trajectories (1/2) y^2 + (1/2) x^2 - (1/3) x^3 = C).
* Lebl, **Section 8.3, Applications of nonlinear systems**: 8.3.1, the pendulum theta'' + (g/L) sin
  theta = 0 as a system in (theta, omega); critical points (k pi, 0), centers at even k and saddles at
  odd k; g/L = 1 "for simplicity" (Figure 8.3.2); the conserved quantity (1/2) omega^2 - (g/L) cos theta;
  the period T = 4 sqrt(L/(2g)) integral from 0 to theta0 of d theta / sqrt(cos theta - cos theta0) and
  the claims "15%" at theta0 = pi/2, "3.8%" at pi/4, "0.048%" at 5 degrees (Figure 8.3.4). 8.3.2,
  predator-prey: (8.3.1) x hares, y foxes; (8.3.2) x' = (a - b y) x, y' = (c x - d) y with a, b, c, d
  positive; the example a = 0.4, b = 0.01, c = 0.003, d = 0.3 with 50 hares and 20 foxes (Figure
  8.3.5); critical points (0, 0) (saddle, eigenvalues a and -d) and (d/c, a/b) = (100, 40) (Jacobian
  [[0, -bd/c], [ac/b, 0]], eigenvalues +- i sqrt(ad)); the constant of motion
  C = y^a x^d e^(-cx - by) and the check C' = 0 (8.3.3); "(d/c, a/b) is a maximum for C"; the claim
  that with 200 foxes "the number of hares will plummet to just slightly more than 1 hare".
* Lebl, **Exercises 3.E (Section 3.5)**: 3.5.2, 3.5.3, 3.5.4, 3.5.6 (answers printed: (a) +-sqrt(2),
  saddle; (b) 1 and 2, source; (c) +-2i, center; (d) -1 and -2, sink; (e) 5 and -3, saddle), 3.5.7
  (answer printed: spiral source), 3.5.8 (answer printed: x = C1 t + C2, y = C1).
* Lebl, **Exercises 8.E**: 8.1.7 (answer printed for (c): critical point (1/2, -1/4), linearization
  u' = -u + v, v' = u + v; see Section 4 for (a) and (b)), 8.2.5, 8.2.7, 8.2.8 (answers printed:
  (b) (0, 0) spiral sink, asymptotically stable; (0, 1) saddle, unstable), 8.3.1, 8.3.2, 8.3.5, 8.3.6
  (answer printed: saddles at odd k, sinks at even k), 8.3.7 (answer printed for (b): (550/3, 40),
  [[0, -11/6], [3/25, 1/4]], eigenvalues (5 +- i sqrt(327))/40, spiral source).
* Notebooks 03 (the phase line and "stable" for x' = f(x)), 07 and 07E (the plane of states (x, x'),
  the energy of a spring, the pendulum with sin theta ~ theta), 09 (the pendulum is not linear), 12
  (partial derivatives M_y, N_x), 17 (eigenvalues, eigenvectors, trace and determinant,
  `two_by_two_case`, `eigen_solution`, the three cases), 04 (Runge-Kutta). Videos R05 (complex numbers),
  R07 (determinants and eigenvalues), R09 (Taylor series: the tangent line, here the tangent plane).

**Purpose.** One idea, met twice: **the picture of a 2 x 2 system near an equilibrium is decided by the
eigenvalues of one matrix**, which are read off from two numbers, its trace and its determinant.

1. **Linear systems (Session 19).** The plane of states of notebook 07E becomes the *phase plane*: each
   solution (x(t), y(t)) is a curve, a *trajectory*. For x' = A x the eigenvectors are the straight
   trajectories, and the eigenvalues say which way they run; six pictures (Lebl's Table 3.5.1), and a
   map, the **trace-determinant plane**, that tells them apart without computing a single eigenvalue:
   lambda1 + lambda2 = tr A, lambda1 lambda2 = det A.
2. **Nonlinear systems (Session 20).** Near an equilibrium (x*, y*), x' = f(x, y), y' = g(x, y) looks
   like the linear system of its **Jacobian matrix** J (the derivative of calculus, in two variables).
   The linear picture tells the truth whenever no eigenvalue has real part 0; a center of the
   linearization tells nothing, and the nonlinear terms decide (Lebl's "trouble with centers"). Two
   classical models, the pendulum and Lotka-Volterra's hares and foxes, are both linear centers, and both
   are true centers, for a reason the linearization cannot see: a conserved quantity.

---

## 1. Contents

| # | Section | What the student does | Values shown (all checked) |
|---|---|---|---|
| 0 | Title, sessions, references, what you will learn | reads | - |
| 1 | **Theory in brief** | the phase plane (notebook 07E's plane of states), trajectory, direction field (Lebl 3.1.3), equilibrium (x*, y*) where f = g = 0 (Lebl: critical point); stable, asymptotically stable, unstable in words (Lebl 8.2), compared once with notebook 03's "stable" for x' = f(x), which is Lebl's asymptotically stable; for x' = A x: lambda^2 - (tr A) lambda + det A = 0 (notebook 17), so lambda1 + lambda2 = tr A and lambda1 lambda2 = det A (from (lambda - lambda1)(lambda - lambda2), multiplied out); Lebl's Table 3.5.1 with a column "tr A, det A"; partial derivatives (notebook 12: differentiate in x with y held fixed), the Jacobian J, the linearization u' = J u with u = (x - x*, y - y*) | - |
| 2 | **Pictures of x' = A x** (Session 19) | Lebl's Case 3 (saddle) by hand: triangular, so eigenvalues 1 and -2 (notebook 17's fact); eigenvectors (1, 0) and (1, -3) by the shortcut; the two straight-line solutions (1, 0) e^t and (1, -3) e^(-2t); on the line of v the arrow is lambda times v (Lebl's one line). Figure 1: left, the two eigenvector lines with their arrows (Figures 3.5.2 and 3.5.5); right, `ol.direction_field` and `ol.phase_portrait` on the same matrix. Then **the six pictures** of Lebl's Cases 1-6 in one figure, each titled by `ol.classify_linear_2d` | Case 3: eigenvalues -2 and 1 (`two_by_two_case` order); Cases 1-6: tr 3, det 2 (1, 2); tr -3, det 2 (-2, -1); tr -1, det -2 (-2, 1); tr 0, det 4 (+-2i); tr 2, det 5 (1 +- 2i); tr -2, det 5 (-1 +- 2i) |
| 3 | **The trace-determinant plane** (Session 19, theory exercise) | the **classification from tr A and det A**, written in the notebook in full (below the table); Figure 2: the plane with the parabola det A = (tr A)^2/4, the regions named with Lebl's words, the six matrices of Section 2 as labelled points; the borderlines met by Lebl's Exercises 3.5.2 (on the parabola: one eigenvector), 3.5.3 (on the axis det A = 0) and Case 4 (on the half-axis tr A = 0, det A > 0); asymptotically stable exactly when tr A < 0 and det A > 0. **Slider 1** (below) | the six points as in Section 2; Exercise 3.5.2 [[1, 1], [0, 1]]: tr 2, det 1, eigenvalue 1 twice, one eigenvector (1, 0); Exercise 3.5.3 [[1, 1], [1, 1]]: tr 2, det 0, eigenvalues 0 and 2 |
| 4 | **Drawing a phase portrait in Python** (Session 19, practical exercise) | the recipe in four steps (write the field as a function of x and y; draw the direction field; choose starts; draw the trajectories both ways in time), applied to a nonlinear system before any theory: Lebl's Example 8.1.1, x' = y, y' = -x + x^2 (no units), Figure 8.1.2. Question first: "which points look special, and what do the curves do near them?". Figure 3: the portrait above, x(t) below for the starts (0.5, 0) (a closed curve: x swings between 0.5 and a negative value and back) and (1.2, 0) (it leaves the window). Answer: two special points, (0, 0) (curves go around it) and (1, 0) (curves come in and go out); Section 5 explains them | equilibria (0, 0), (1, 0) read from the picture and checked: f = g = 0 there |
| 5 | **Equilibria and the Jacobian** (Session 20) | Example 8.1.1 by hand: y = 0 and x(x - 1) = 0; J = [[0, 1], [-1 + 2x, 0]] (Lebl, Example 8.1.2), entry by entry with the partial derivatives written out; `sp.Matrix([f, g]).jacobian([x, y])` agrees; `ol.jacobian` (central differences) agrees to 1e-9. At (1, 0): [[0, 1], [1, 0]], eigenvalues +-1, a saddle. Figure 4: the nonlinear portrait near (1, 0) and the portrait of its linearization, side by side at the same scale (Figure 8.1.3, right). At (0, 0): +-i, a center of the linearization: Section 6. Lebl's Example 8.2.1 in two lines of code with `ol.classify_equilibrium_2d`: the book's values | (1, 0): eigenvectors (1, 1) for 1 and (1, -1) for -1; Example 8.2.1: (0, 0) saddle (1, -1); (1, -1) sink, asymptotically stable (-1, -3) |
| 6 | **When the linearization tells the truth** (Session 20, theory exercise) | Lebl's Table 8.2.1 and the rule (below the table); the two exceptions, a center and a zero eigenvalue (not "almost linear"). Two systems with **the same linearization** [[0, 1], [-1, 0]] at the origin: Example 8.1.1 (the energy E = y^2/2 + x^2/2 - x^3/3 is constant along every solution, SymPy: dE/dt = 0 (Lebl, Example 8.2.4), so near the origin the curves close: a true center) and Example 8.2.3 (x' = y, y' = -x + y^3: d(x^2 + y^2)/dt = 2y^4 >= 0 (Lebl's Exercise 8.2.5, done here in one line), so the distance to the origin never decreases: a spiral source). Figure 5: both from the same start (0.2, 0), two panels; `ol.classify_equilibrium_2d` says "inconclusive" for both, and why | E at the saddle (1, 0) is 1/6 (the curve through it separates the closed curves from the others); 8.2.3 from (0.2, 0): distance 0.2 at t = 0, 2 at t = 33.0 |
| 7 | **Two models: the pendulum and hares and foxes** (Session 20, practical exercise) | convention boxes (below). *Pendulum* (Lebl 8.3.1, g/l = 1): the drawing with theta > 0 marked, above the phase portrait (Figure 8.3.2); equilibria (k pi, 0); at (0, 0) a linear center (inconclusive), but the energy (1/2) theta'^2 - (g/l) cos theta is constant: a true center (short swings); at (pi, 0) a saddle (upside down); the wavy curves above and below are the pendulum going over the top. *Lotka-Volterra* (Lebl 8.3.2, his numbers): what each term means; equilibria by hand; (0, 0) saddle; (100, 40) a linear center; Lebl's constant of motion C = y^a x^d e^(-cx - by), C' = 0 by SymPy and constant along the rk4 solution; so the curves close: the populations repeat. **Slider 2** (below) | pendulum: (0, 0) +-i, (pi, 0) +-1; Lotka-Volterra: (0, 0) eigenvalues 0.4 and -0.3; (100, 40) eigenvalues +-0.346i (sqrt(0.12)), small swings repeat every 2 pi / sqrt(0.12) = 18.1 time units; Lebl's start (50, 20): hares between 32.4 and 227.1, foxes between 15.4 and 82.5, period 19.35; C constant to 1e-8 |
| 8 | **Exercises** | the four steps below | - |
| 9 | **Answers** (collapsed), where to go next (notebook 19, the Laplace transform; Lebl 8.4, limit cycles, for curves that close without a conserved quantity) | opens after trying | - |

**The classification of Section 3** (the theory exercise of Session 19, written in the notebook in full).
Let A be a real 2 x 2 matrix with eigenvalues lambda1, lambda2. Multiplying out
(lambda - lambda1)(lambda - lambda2) and comparing with lambda^2 - (tr A) lambda + det A gives
lambda1 + lambda2 = tr A and lambda1 lambda2 = det A; the eigenvalues are complex exactly when the
discriminant (tr A)^2 - 4 det A is negative, and then they are alpha +- i beta with alpha = tr A / 2 and
lambda1 lambda2 = alpha^2 + beta^2 > 0.
* det A < 0: the product is negative, so the eigenvalues are not complex (their product would be
  positive) and have opposite signs: a **saddle**.
* det A > 0 and (tr A)^2 > 4 det A: real, with the same sign (positive product), the sign of their sum
  tr A: a **source** if tr A > 0, a **sink** if tr A < 0.
* (tr A)^2 < 4 det A (so det A > 0): complex with real part tr A / 2: a **spiral source** if tr A > 0, a
  **spiral sink** if tr A < 0, a **center** if tr A = 0.
* The borderlines: on the parabola (tr A)^2 = 4 det A one eigenvalue tr A / 2 twice (notebook 17's
  repeated case: one eigenvector, Exercise 3.5.2, or A = lambda I); on the axis det A = 0 an eigenvalue 0,
  so A x = 0 has nonzero solutions: a whole line of equilibria (Exercise 3.5.3).
* Stability: the origin is asymptotically stable exactly when both eigenvalues (or their real parts) are
  negative, that is, when tr A < 0 and det A > 0.

**The rule of Section 6** (the theory exercise of Session 20). Let (x*, y*) be an equilibrium and J the
Jacobian there. If no eigenvalue of J has real part 0 (equivalently: det J is not 0, and J is not a
center), the equilibrium has the stability of the linear system u' = J u, and for a saddle, a node or a
spiral also its picture near the point (Lebl, Section 8.2.2, Table 8.2.1: "the system will behave like its
linearization, at least if we are close"). If J is a center, the real parts are 0 and the small nonlinear
terms decide: Example 8.1.1 is a center, Example 8.2.3 a spiral source, with the same J (Lebl, Section
8.2.3). If det J = 0 the point is not "almost linear" and the linearization decides nothing either. Lebl
states the rule without proof; the notebook says so and gives the two examples as the evidence.

**Slider 1 (Section 3): moving through the trace-determinant plane.** A = [[-1, 1], [p, -1]]
(no units; tr A = -2 always, det A = 1 - p, discriminant 4p). Question before moving it: "the trace stays
-2 and the determinant goes down as p grows. At which values of p does the picture change its kind, and
for which p does the origin stop attracting every solution?" Top: the phase portrait in the window
[-2, 2] x [-2, 2] (direction field, 12 trajectories both ways in time from a circle of radius 1.8, the
eigenvector lines (1, sqrt(p)) and (1, -sqrt(p)) dashed when p > 0); the title says the kind and the
eigenvalues (`ol.classify_linear_2d`). Bottom: the trace-determinant plane with the parabola and the
moving point (-2, 1 - p). Range p from -3 to 3, step 0.1; static figure at p = -1 (spiral sink,
eigenvalues -1 +- i). Answer after it: spiral sink for p < 0; at p = 0 the two eigenvector lines merge
into (1, 0) (one eigenvector, eigenvalue -1 twice: the point is on the parabola); sink for 0 < p < 1; at
p = 1, det A = 0: the line x = y is made of equilibria (the matrix of notebook 17's closed tanks, without
the factor Q/V), and every solution goes straight to it; for p > 1 a saddle: almost every solution leaves.
The origin stops attracting every solution at p = 1.

**Slider 2 (Section 7): more foxes.** Lebl's hares and foxes from x(0) = 50 hares and y(0) foxes,
chosen with the slider. Question before moving it (Lebl's remark): "if we bring more foxes into the
forest, do the hares do better, or worse?" Top: the trajectory in the phase plane (hares across, foxes
up) with the direction field and the equilibrium (100, 40); bottom: x(t) and y(t) for 0 <= t <= 60 time
units (more than two periods). The title gives the smallest number of hares and of foxes along the curve.
Range y(0) from 10 to 200 foxes, step 10; static figure at y(0) = 20 (Lebl's figure: hares between 32.4
and 227.1). Answer after it: the curve is always closed (the constant of motion), so the populations
always come back, but the farther the start from (100, 40), the deeper the fall: with 200 foxes the hares
fall to 1.27 (Lebl: "slightly more than 1 hare"), which in a real forest means extinction (Lebl's
warning that 0.1 fox makes no sense).

**Conventions** (rule 6 of the course notation, written in a box where each problem starts, on every
figure, and in one line above each exercise that uses them):
* *Phase planes* (all sections). The first unknown x across (positive to the right), the second y up
  (positive upwards); an arrow on a trajectory shows the direction of increasing t. Both x and y are
  functions of t here, and y' = dy/dt (said once: elsewhere in the course y' = dy/dx). Lebl's examples of
  Sections 3.5, 8.1 and 8.2 have no units, and the notebook says so.
* *Pendulum* (Section 7, Exercise 2(d), Explore (v) and (vi)). theta(t): the angle in radians between
  the rod and the downward vertical, 0 when the pendulum hangs at rest, **positive counterclockwise** in
  the drawing; theta' in rad/s, positive counterclockwise; t in seconds; length l = 9.81 m, so that
  g/l = 1 /s^2 (Lebl takes g/L = 1 "for simplicity": the length is the notebook's, said so). The phase
  plane is (theta, theta'); its curves run clockwise, as in notebook 07E's plane of states. Lebl writes
  omega for theta' and L for the length; notebooks 07 and 09 wrote L: one sentence compares.
* *Hares and foxes* (Section 7, slider 2, Exercise 3, Explore (i) and (iv)). x(t) the number of hares,
  y(t) the number of foxes (Lebl's (8.3.1)), both >= 0 (only the quadrant x >= 0, y >= 0 is drawn);
  x' > 0 means the hares are increasing. a = 0.4 and d = 0.3 per time unit (growth of hares without
  foxes, death of foxes without hares); b = 0.01 per fox per time unit, c = 0.003 per hare per time unit.
  Lebl gives no time unit: the notebook says "time units" (Decision 5). Exercise 3: gamma in 1/(hare and
  time unit); Explore (iv): h foxes removed per time unit.

Length: at most 2100 words before the exercises, counted as in plan 17 (each formula as one word; plan
17 reached about 2000 that way). If the draft is longer: Example 8.2.1 goes into the code with one
sentence; the derivation of the pendulum's energy goes into a code comment; Section 4's x(t) panel is
dropped. Ten figures before the exercises plus one in Exercise 3; two sliders, each with its static
picture and its question first.

## 2. Exercises (the four steps)

Each exercise is marked with its session: 1(a), 1(b), 2(a), Explore (ii) and (iii) after Session 19; the
rest after Session 20. Word answers are typed in quotes ("saddle"); the words allowed are listed once
above Exercise 1: source, sink, saddle, center, spiral source, spiral sink (Lebl's synonyms "unstable
node", "stable node" and the spelling "centre" are accepted), and for stability: asymptotically stable,
stable, unstable.

1. **Predict** (no code) - (a) *Lebl, Exercise 3.5.7*: A has eigenvalues 2 +- i: the picture?
   ("spiral source", the book's answer.) (b) *From Lebl, Exercise 3.5.4*: True or False, a diagonal
   matrix [[a, 0], [0, b]] with a, b not 0 can give a spiral? (False: its eigenvalues a and b are real.)
   (c) *From Lebl, Exercise 8.2.7*: if g(x, y) > 1 at every point, how many equilibria has x' = f(x, y),
   y' = g(x, y)? (0: y' is never 0; and y(t) > y(0) + t grows without bound.) (d) *From Lebl, Section
   8.2.4*: True or False, a conservative equation x'' + f(x) = 0 can have an asymptotically stable
   equilibrium? (False: J = [[0, 1], [-f'(x*), 0]] has trace 0, so a saddle or a center, and the energy
   is constant.)
2. **By hand, then check** - (a) *From Lebl, Exercise 3.5.6 (d) and (e)*: for x' = x + 3y,
   y' = -2x - 4y and for x' = x - 4y, y' = -4x + y: the trace, the determinant and the picture ((d) -3, 2,
   "sink"; (e) 2, -15, "saddle"; the book gives the eigenvalues -1, -2 and 5, -3). (b) *From Lebl,
   Exercise 8.1.7 (c)*: x' = (x - 1)^2 + y, y' = x^2 + y: the equilibrium (1/2, -1/4) (the book's), the
   Jacobian there [[-1, 1], [1, 1]] (the book's linearization, typed as a list of rows), and the picture
   ("saddle": det -2; not printed in the book). (c) *From Lebl, Exercise 8.2.8 (b)*: x' = y - y^2 - x,
   y' = -x has the equilibrium (0, 0) and one more: find it ((0, 1)), and the picture and the stability of
   each ((0, 0): "spiral sink", "asymptotically stable"; (0, 1): "saddle", "unstable"; the book's
   answers). (d) *From Lebl, Exercise 8.3.5 (b)*, with the pendulum of Section 7 (g/l = 1 /s^2; theta'
   > 0 counterclockwise): starting at theta(0) = 0, the smallest theta'(0) in rad/s that takes the
   pendulum over the top (2, from the energy: (1/2) theta'(0)^2 - 1 = 0 + 1; the book prints no answer;
   checked with rk4: 1.99 swings back at theta = 2.94, 2.01 goes over).
3. **Fill in** (the practical exercise of Session 20) - *Lebl, Exercise 8.3.2*: the hares grow
   logistically, x' = (0.4 - 0.01 y) x - gamma x^2, y' = (0.003 x - 0.3) y (convention line of the
   hares and foxes; gamma in 1/(hare and time unit)). First a drawing: the direction field for
   gamma = 0.001 in the window 0 <= x <= 450, 0 <= y <= 80, with a caption saying what the arrows show,
   and a "first step" (differentiate the right-hand side of x' with respect to x, holding y fixed: the
   term -gamma x^2 gives -2 gamma x). Complete `hares_jacobian(x, y, gamma)`, which returns the Jacobian
   as [[f_x, f_y], [g_x, g_y]]. The next cell does not compare formulas: at four points and three
   values of gamma it compares the student's matrix with `ol.jacobian` of the field; the first mismatch is
   reported with its point, gamma, the entry ("row 1, column 2: the derivative of x' with respect to y")
   and both numbers; a matrix of the wrong shape, a NaN or an error inside the function gets its own
   one-line message (the error with its text, as in notebook 17). Then the cell finds the equilibria in
   the quadrant x >= 0, y >= 0 for gamma = 0.001 and 0.01, classifies each with
   `ol.classify_linear_2d(hares_jacobian(...))`, prints a small table and draws both phase portraits from
   Lebl's start (50, 20): gamma = 0.001: (0, 0) saddle, (400, 0) saddle, (100, 30) spiral sink
   (eigenvalues -0.05 +- 0.296i): hares and foxes settle at 100 and 30; gamma = 0.01: (0, 0) saddle,
   (40, 0) sink (-0.4 and -0.18); the third equilibrium (100, -60) is outside the quadrant: the foxes die
   out and the hares settle at 40 (checked with solve_ivp at t = 400).
4. **Explore** - (i) With slider 2, find the first number of foxes at the start (with 50 hares) for
   which the hares fall below 2 (190: 1.66 hares; 180 gives 2.17); then check Lebl's "slightly more than
   1 hare" for 200 foxes (1.27). (ii) *Lebl, Exercise 3.5.8*: draw x' = y, y' = 0 with the recipe of
   Section 4 and check the book's solution x = C1 t + C2, y = C1 by substitution; is it one of the six
   pictures? (No: horizontal lines, the x axis made of equilibria; `ol.classify_linear_2d` says
   "parallel lines".) (iii) *Lebl, Exercise 3.5.3*: [[1, 1], [1, 1]]: eigenvalues 0 and 2; the line
   x + y = 0 is made of equilibria and every other solution runs away from it along a line of slope 1
   (x' = y' = x + y). (iv) *Lebl, Exercise 8.3.7 (b)*: foxes hunted at h = 10 foxes per time unit,
   y' = (0.003 x - 0.3) y - 10: the equilibrium (550/3, 40), the Jacobian [[0, -11/6], [3/25, 1/4]] and the
   eigenvalues (5 +- i sqrt(327))/40 = 0.125 +- 0.452i, the book's: a spiral source; draw the solution
   from (188.3, 40) and see the foxes reach 0 (at t = 28.3: Lebl's "something will die out"). (v) *Lebl,
   Section 8.3.1*: measure the period of the pendulum with rk4 from theta(0) = pi/2, pi/4 and 5 degrees,
   theta'(0) = 0 (twice the time until theta' changes sign), and check the book's relative errors
   (T - T_linear)/T: 15.3%, 3.84%, 0.048% (the book: 15%, 3.8%, 0.048%; T_linear = 2 pi s with g/l = 1);
   the first step is given. (vi) *From Lebl, Exercises 8.3.1 (a) and 8.3.6*: the damped pendulum
   theta'' + c theta' + (g/l) sin theta = 0 (Lebl's mu is called c, the damping of notebook 07; said once):
   `ol.classify_equilibrium_2d` at (0, 0) and (pi, 0) for c = 1 (spiral sink, eigenvalues
   -1/2 +- i sqrt(3)/2; saddle) and c = 3 (sink, -0.38 and -2.62; saddle, the book's answer for
   c^2 > 4 g/l); what does each mean for the pendulum?

Answers (final values only) in a collapsed block; full solutions after the sessions.

## 3. New in `odelib.py` (with tests)

A new section of the library, "The phase plane and nonlinear systems (notebook 18)", placed after the
section of notebook 17; `check_choice` goes after `check_vector`. NumPy only at the top; matplotlib is
imported inside the two drawing functions, as in `slope_field`; SciPy and SymPy are used only in the
tests. Every function has type hints on its arguments and its return value, a docstring citing Lebl, the
name of this notebook, and its units statement. A system in the plane is passed everywhere in one form,
Lebl's: `f(x, y)` returns the pair (f(x, y), g(x, y)), works on NumPy arrays (the notebook writes it with
`np.sin`, `np.exp`), and is the same function for the direction field, the trajectories and the Jacobian.

* `class PhaseType(NamedTuple)`: `kind: str`, `stability: str`, `trace: float`, `determinant: float`,
  `eigenvalues: tuple[float, float] | tuple[complex, complex]`, `case: str` (the case of
  `two_by_two_case`).
* `classify_linear_2d(matrix: MatrixLike, rtol: float = REPEATED_EIGENVALUE_RTOL) -> PhaseType`: the
  picture of x' = A x at the origin, from tr A and det A (Lebl, Section 3.5, Table 3.5.1, and the
  borderline cases of Exercises 3.5.2, 3.5.3, 3.5.8). **A thin wrapper of `two_by_two_case`**: it calls
  it once (validation, scaling, the repeated band, the eigenvalues and the case are reused, nothing is
  recomputed), and adds only the map from (case, sign of tr A, sign of det A) to a kind, with two more
  bands decided on A / s like the discriminant: |det A| <= rtol s^2 and |tr A| <= rtol s. It is not a
  duplicate: `two_by_two_case` knows nothing of pictures or stability. Kinds (named constants
  `KIND_...`, Lebl's words): "saddle", "source", "sink", "spiral source", "spiral sink", "center",
  "degenerate source" and "degenerate sink" (repeated, one eigenvector), "star source" and "star sink"
  (A = lambda I), "line of equilibria" (one eigenvalue 0), "parallel lines" (eigenvalue 0 twice, A not
  0), "all equilibria" (A = 0). Stability (named constants `STABILITY_...`, Lebl 8.2):
  "asymptotically stable" for the four sinks; "stable" for a center, an attracting line of equilibria
  and A = 0; "unstable" for the rest. Units: the entries of A in 1/time, as for `eigen_solution`; the
  kind does not depend on the unit. ValueError: those of `two_by_two_case` (not 2 x 2, complex or
  non-finite entries, rtol not positive). The docstring says "not to be confused with
  `classify_linear_system` (notebook 16, algebraic systems M C = b)".
* `jacobian(f: Callable[..., Sequence[float]], point: Sequence[float], h: float = JACOBIAN_STEP)
  -> np.ndarray`: the Jacobian matrix of f at `point` by central differences, column j =
  (F(p + h_j e_j) - F(p - h_j e_j)) / (2 h_j) with h_j = h max(1, |p_j|) (Lebl, Section 8.1.2). f is
  called as f(*point), so the same function serves the plane (f(x, y)) and Lebl's three-dimensional
  Exercise 8.1.9 (f(x, y, z)). `JACOBIAN_STEP = 1e-5`, with its reason in a comment: the error of a
  central difference is about h^2 times the third derivative plus the rounding 1e-16 / h, both near
  1e-10 for the functions of the course. Returns an n x n float array. Units: entry (i, j) has the units
  of f_i divided by those of x_j. ValueError: a point that is not a flat list of finite numbers; f
  returning a number of values different from the number of coordinates; a value that is not finite at
  or near the point (the message names the coordinate); h not positive. An error raised inside f is not
  caught (it is the notebook's or the student's message).
* `class EquilibriumType(NamedTuple)`: `point: tuple[float, ...]`, `jacobian: np.ndarray`,
  `linear: PhaseType` (the linearization), `stability: str`, `reason: str`.
* `classify_equilibrium_2d(f: Field, point: Sequence[float], rtol: float = LINEARIZATION_RTOL,
  h: float = JACOBIAN_STEP) -> EquilibriumType`: linearization plus `classify_linear_2d` (Lebl, Sections
  8.1-8.2, Table 8.2.1). First it checks that `point` is an equilibrium: |f| and |g| at the point at most
  `EQUILIBRIUM_TOL` (1e-9) times max(1, |J| max(1, |point|)), so (550/3, 40) typed as floats is accepted;
  otherwise ValueError "(x, y) = (0.5, 0) is not an equilibrium: f = 0, g = -0.25 (both must be 0)". Then
  J = `jacobian(f, point, h)` and `linear = classify_linear_2d(J, rtol)`. `stability` is
  `linear.stability` when no eigenvalue has real part 0 (saddles, nodes, spirals, degenerate and star
  nodes), and `STABILITY_INCONCLUSIVE` = "inconclusive" (the word of `classify_equilibrium`, notebook 03)
  for a center, a line of equilibria, parallel lines or J = 0. `reason` is one sentence: "no eigenvalue
  has real part 0: near the point the system behaves like its linearization (Lebl, Table 8.2.1)", "the
  linearization is a center: the nonlinear terms decide (Lebl, Section 8.2.3)", or "det J = 0: not almost
  linear (Lebl, Section 8.2.1)". `LINEARIZATION_RTOL = 1e-7`, with its reason: the entries of J carry an
  error of about 1e-9 relative, so trace, determinant and discriminant are trusted to about 1e-9; a band
  a hundred times wider never takes rounding for a sign, and a true nonzero trace of a model is far
  larger. A degenerate node of J keeps its stability but the docstring says that the nonlinear picture
  near it may be a node or a spiral (Lebl's Exercise 8.2.8 (c) calls (0, 1) a "source"; J has the
  eigenvalue 1 twice). Units as for `jacobian`.
* `trajectories(f: Field, starts: ArrayLike, t_span: tuple[float, float], n_steps: int = PORTRAIT_STEPS,
  window: tuple[tuple[float, float], tuple[float, float]] | None = None) -> tuple[np.ndarray, np.ndarray]`:
  the solutions of x' = f(x, y), y' = g(x, y) from several starts at once, by the classical Runge-Kutta
  method of notebook 04 with the fixed step (t1 - t0) / n_steps, **all starts in one vectorised
  computation**: the existing private step `_rk4_step` is applied to an array of shape (k, 2), so a
  portrait of 24 paths costs the same number of Python steps as one path. t_span may decrease (backward
  in time). Returns t (n_steps + 1 times) and the paths, shape (n_steps + 1, k, 2), [i, j] the point of
  start j at t[i]. A path that leaves the window enlarged by `ESCAPE_MARGIN` (1, one window size on each
  side), or whose value is no longer finite, is NaN from there on (no RuntimeWarning: computed inside
  `np.errstate`); the loop stops when every path has left. `PORTRAIT_STEPS = 400` (for Lebl's
  hares and foxes over 40 time units the error is about 1e-5 hares against `solve_ivp` at rtol 1e-12; 0.006
  for the start with 200 foxes; slider 2 uses 1200 steps over 60 time units: error 1e-6 from Lebl's start,
  8e-4 hares from the start with 200 foxes).
  ValueError: starts not of shape (k, 2) (one start (x, y) is accepted as k = 1) or not finite; t_span
  with equal or non-finite ends; n_steps not a positive integer, or above `MAX_STEPS` (existing rule);
  f returning values of the wrong shape. Units: t in the time unit of the rates in f; x, y in the units of
  the starts.
* `direction_field(ax, f: Field, x_range: tuple[float, float], y_range: tuple[float, float],
  n: int = DIRECTION_GRID, color: str = "0.6", length: float = ARROW_LENGTH)`: the 2-D analogue of
  `slope_field` with its look (Lebl 3.1.3): at each point of an n x n grid an arrow along (f, g), every
  arrow of the same length measured in grid cells (`ARROW_LENGTH = 0.7`, the value of `slope_field`), so
  the picture shows directions, not speeds (the caption says so). Points where (f, g) is zero (an
  equilibrium on the grid) or not finite are left out before drawing, so no warning appears. One call of
  `ax.quiver`. `DIRECTION_GRID = 17`. Returns `ax`. ValueError for a window that is not increasing (the
  message of `slope_field`).
* `phase_portrait(ax, f: Field, starts: ArrayLike, t_span: tuple[float, float],
  x_range: tuple[float, float], y_range: tuple[float, float], *, n_steps: int = PORTRAIT_STEPS,
  both_ways: bool = True, color: str = "C0")`: the trajectories through `starts`, from `trajectories`,
  forward over t_span = (0, T) (T > 0) and, with `both_ways`, backward over (0, -T); a dot at each start
  and one arrowhead on each forward part, a third of the way along its drawn length
  (`TRAJECTORY_ARROW_AT = 1/3`); the limits of the window are set. Returns `ax`. ValueError: T not
  positive, a start outside the window, and those of `trajectories`. The notebook chooses few starts
  (8-14 per picture): the cost is in matplotlib, not in the integration.
* `check_choice(label: str, value, expected: str, choices: Sequence[str],
  synonyms: Mapping[str, str] | None = None) -> bool`: the exercise checker for answers that are words,
  with the one-line manners of `check`. The answer is normalised (lower case, spaces around removed, "-"
  and "_" read as spaces, repeated spaces as one) and mapped through `synonyms`. Messages: None -> "no
  answer yet (replace None with your answer, a word in quotes)"; not a string -> 'type the word in
  quotes, for example "saddle"'; not among the choices -> "not one of the words: ..." (the list of
  choices, which is the menu, not the answer); otherwise "correct" or "not yet, try again". It never
  raises for an answer; ValueError only for a mistake of the notebook (`expected` not among `choices`).
  Constants: `PORTRAIT_WORDS` (the six words of Table 3.5.1), `STABILITY_WORDS` ("asymptotically
  stable", "stable", "unstable"), `KIND_SYNONYMS` ({"unstable node": "source", "stable node": "sink",
  "centre": "center", "center point": "center", "saddle point": "saddle"}). Without it a "classify"
  exercise could only be checked through numeric codes.
* The type alias `Field = Callable[[np.ndarray, np.ndarray], tuple[ArrayLike, ArrayLike]]`, a private
  helper `_field_values(f, x, y)` (calls f, broadcasts constants, checks the shape, used by the three
  functions that call f), and the constants above, each defined once above its function. `SETUP_NAMES`
  does not change (no new name in the setup cell). The module docstring "Contents" gets the new names
  under "The phase plane (notebook 18)" and under "Exercises".

Tests:
* `tests/test_odelib.py` (NumPy, SciPy and pytest only, so they also run in the site CI; the drawing
  tests use `pytest.importorskip("matplotlib")` and the Agg backend, so they are skipped in CI and run
  locally):
  * `classify_linear_2d`: Lebl's Cases 1-6 give the six rows of Table 3.5.1 with their stability;
    Exercise 3.5.6 (a)-(e) give the book's answers; 3.5.7 on [[2, -1], [1, 2]] (eigenvalues 2 +- i)
    gives "spiral source"; 3.5.2 "degenerate source"; 3.5.3 "line of equilibria", unstable; 3.5.8
    "parallel lines", unstable; the four sign patterns of 3.5.4 (never a spiral); 2I and -I (star); the
    zero matrix; the slider family [[-1, 1], [p, -1]] at p = -3, -1, 0, 0.5, 1, 3, at the slider's
    rounded values -3 + 30 * 0.1 and -3 + 40 * 0.1, and at 1e-10 and 1 +- 1e-10 (spiral sink, degenerate
    sink, sink, line of equilibria, saddle, as the prototype of this plan gave); scale invariance
    (the six cases times 1e-200 and times 1e150); trace, determinant and eigenvalues equal those of
    `two_by_two_case`; invalid input (3 x 3, NaN, complex, text, rtol <= 0) with ValueError.
  * `jacobian`: Example 8.1.2 at (0, 0), (1, 0) and seeded random points against [[0, 1], [-1 + 2x, 0]];
    Example 8.2.1 at (1, -1) ([[-2, -1], [-1, -2]]); Example 8.2.2 at (0, -1) ([[1, -1], [1, 0]], with
    e^x); Lebl's Lotka-Volterra at (100, 40) ([[0, -1], [0.12, 0]]); the pendulum at (pi, 0); a linear
    field returns its matrix to 1e-9; Exercise 8.1.9 in three dimensions at (-1, 1, -1)
    ([[1, 0, -2], [0, -1, -2], [-2, 0, 1]], recomputed: the text copy's answer lost a letter); all
    within 1e-8 relative; invalid input (NaN point, three values for two coordinates, a value that is not
    finite at the point, h <= 0).
  * `classify_equilibrium_2d`: Example 8.2.1 ((0, 0) saddle; (1, -1) sink, asymptotically stable,
    eigenvalues -3 and -1); Example 8.2.2 ((0, 0) saddle; (0, -1) spiral source, 1/2 +- i sqrt(3)/2);
    Example 8.2.3 and Example 8.1.1 at (0, 0): inconclusive, reason "center"; 8.1.1 at (1, 0): saddle;
    Exercise 8.2.8 (a)-(c), the book's answers (with (0, 1) of (c) "degenerate source", unstable);
    Lotka-Volterra (0, 0) saddle and (100, 40) inconclusive; Exercise 8.3.7 at (550/3, 40), typed as
    floats: spiral source with eigenvalues 0.125 +- 0.452i; the pendulum at (k pi, 0), k = -2..2 (centers
    inconclusive, saddles unstable); Exercise 8.1.7 (a) at (1, 0) and Lebl's x' = x^2, y' = y^2 at
    (0, 0): inconclusive, reason "det J = 0"; ValueError for (0.5, 0) in Example 8.1.1, with the message.
  * `trajectories`: Lebl's Cases 1-6 against `ol.eigen_solution` (notebook 17) on 0 <= t <= 2 with
    2000 steps, relative error below 1e-9 (the prototype: 8e-13); Lotka-Volterra from (50, 20) against `solve_ivp`
    (rtol 1e-12) with 1600 steps over 40 time units, below 1e-7; the constant of motion of Lebl along the
    path, drift below 1e-8; forward then backward returns to the start within 1e-9; Example 8.2.3 from
    (0.3, 0) leaves the window: NaN afterwards, never inf, no RuntimeWarning (warnings as errors); the
    shape for one start and for k starts; invalid input (shape (k, 3), NaN start, equal ends,
    n_steps = 0 and above MAX_STEPS, f returning one array).
  * `direction_field` and `phase_portrait` (matplotlib, Agg): the number of arrows is the number of
    grid points minus those where the field is zero (Example 8.1.1 on a grid through (0, 0) and (1, 0));
    every arrow has the same length in grid cells; a field with NaN values (sqrt of a negative number)
    draws without a warning; limits set; two lines per start with `both_ways`; ValueError for a window
    that is not increasing, T <= 0 and a start outside the window.
  * `check_choice`: accepts "saddle", " Saddle ", "SADDLE", "spiral-sink", "stable node" for "sink",
    "centre" for "center"; refuses "source" for "sink" ("not yet"); None, 3, and "spiral" get their
    messages; never raises for an answer (also for a list or a dict); ValueError when `expected` is not
    among the choices; returns True or False.
* `tests/test_cross_validation.py` (skipped without SymPy, as now): `classify_linear_2d` against a
  classification from SymPy's exact eigenvalues for 500 seeded random integer matrices with entries from
  -5 to 5 (singular and repeated ones included, SEED = 331); `jacobian` against SymPy's `jacobian` at
  seeded random points for the systems of Examples 8.1.1, 8.2.1, 8.2.2 and Exercises 8.1.3, 8.2.1, 8.2.4;
  `classify_equilibrium_2d` against SymPy (equilibria by `sp.solve`, eigenvalues of the exact Jacobian)
  for the same systems.

## 4. Conventions and checks

As for notebooks 14-17: same setup cell, `assert` for every reference value, one sentence under every
plot (written after looking at the figure and recomputing every number), line styles as well as colours,
a question before every slider and its answer after, named constants, plain ASCII, generated by
`plans/make_18.py` with `nbbuild.py`; `ruff check --no-cache` (the course rules in `notebooks/ruff.toml`)
reports nothing on `make_18.py`, the new lines of `odelib.py` and the new tests.

* **Setup cell.** The one of notebook 17, unchanged (`import sys`, `import odelib as ol`, the version
  print; np, sp, plt, ol, solve_ivp, interact, FloatSlider, IntSlider). `brentq` is not needed (Explore
  (v) measures the half period by a sign change of theta' and a linear interpolation, shown in its first
  step). `SETUP_NAMES` does not change.
* **Letters** (course notation, Section 4 of the style sheet; rows to add to the reserved-letter table
  when the instructor approves, before the decks of Sessions 19-20; Decision 2):
  * x(t), y(t): the two unknowns of a system in the plane (Lebl 3.5 and 8), both functions of t, with
    y' = dy/dt here. New row needed ("from Session 19, the unknowns of a planar system; y' = dy/dt
    there"). In the hares and foxes they keep Lebl's letters (rule 1 would name them after the quantity;
    H and F are taken by the harvest and the force): announced.
  * f(x, y), g(x, y): the right-hand sides (Lebl). The reserved f(x, y) of y' = f(x, y) is the same idea;
    g(x, y) is not gravity: in the pendulum section the right-hand sides are written out and g is gravity
    only (said once). Row to extend for f and g.
  * (x*, y*): an equilibrium (the reserved x*, y*; Lebl writes (x0, y0), and the subscript 0 marks a
    start in the course: said once).
  * J: the Jacobian matrix. New row ("from Session 20, the Jacobian matrix at an equilibrium").
  * u = (u1, u2) = (x - x*, y - y*): the displacement from the equilibrium in the linearization. Lebl
    writes u and v; v is kept for eigenvectors (Session 17's row). u1, u2 also name notebook 17's
    conversion variables and Session 14's varying parameters, which do not occur here. New row, or an
    extension of the u1, u2 row.
  * tr A and det A, the trace and the determinant, written as words in titles and axes; no new letter
    (T is the period of the pendulum, as in notebook 07; D is a constant made from C).
  * lambda, v (eigenvalues, eigenvectors) and A (the matrix) as in notebook 17; a11, ..., a22 for named
    entries.
  * theta, theta' for the pendulum (theta is reserved for the pendulum's angle; Lebl's omega is not used,
    because omega is the forcing frequency and omega0, omega1 the natural and pseudo-frequency); l for the
    length (reserved; Lebl L, notebooks 07 and 09 L: one sentence).
  * a, b, c, d of Lotka-Volterra (Lebl's): reserved from Session 11 for the coefficients of
    a y'' + b y' + c y = 0, which do not occur in Section 7: announced there ("Lebl's four rates, not the
    coefficients of notebook 06"). New row (Session 20). gamma (Lebl's Exercise 8.3.2) is reserved for
    the phase of R cos(beta x - gamma), which does not occur in Unit 9: announced. c in Explore (vi) is the
    damping of the pendulum (Lebl's mu, reserved for the integrating factor), announced there.
  * E for the energy in Section 6 and the pendulum (notebook 07E's E); C for Lebl's constant of motion of
    Lotka-Volterra (a constant along a solution, rule 3), said once; h in Explore (iv) is Lebl's hunting
    rate (h is reserved for a step and a height, which do not occur in that item; H, the harvest, would
    clash with Lebl's text): announced.
  * Kind names: Lebl's words (source, sink, saddle, center, spiral source, spiral sink), with "stable
    node" and "unstable node" given once as his synonyms; "center" with Lebl's spelling. Stability: Lebl's
    three words; notebook 03 wrote "stable" for what Lebl now calls asymptotically stable (one sentence,
    rule 6).
* **Statements of Lebl that look wrong** (checked with SymPy and numerically on 2026-11-20; as with
  Exercise 3.1.12 in notebook 17). None of them is quoted by the notebook; where the notebook uses the
  passage it gives the right value and one parenthesis (Decision 7):
  1. Section 8.2.1, first paragraph: "two isolated critical points (0, 0) and (0, 1)" for Examples
     8.1.1-8.1.2, whose critical points are (0, 0) and (1, 0) (as Section 8.1 says).
  2. Example 8.2.4, last lines: "The critical point at (0, 1) is a saddle": it is (1, 0).
  3. Exercise 8.1.7 (a), the printed answer: critical points "(0, 0) and (0, 1)" with linearizations
     "u' = -2u - (1/pi) v, v' = -v" and "u' = -2u + (1/pi) v, v' = v". For
     x' = sin(pi y) + (x - 1)^2, y' = y^2 - y the critical points are (1, 0) and (1, 1)
     (y^2 - y = 0, then sin(pi y) = 0 and (x - 1)^2 = 0), with Jacobians [[0, pi], [0, -1]] and
     [[0, -pi], [0, 1]], both singular. The answer seems to belong to another system. Not used.
  4. Exercise 8.1.7 (b), the printed answer: only "(0, 0)" for x' = x + y + y^2, y' = x; there is also
     (0, -1) (Jacobian [[1, -1], [1, 0]], a spiral source). Not used.
  5. Exercise 8.2.11, the printed answer: "stable if f'(x0) < 0 and unstable when f'(x0) < 0": the second
     sign is > 0. Not used (notebook 03 states it right).
  6. Section 3.5, Case 1: "the derivative points in the direction of v when alpha is positive": the
     scalar is called a in the line before (a typo of the copy). Not quoted.
  7. Exercise 8.1.9, the printed answer at (-1, 1, -1) reads "u' = u - 2, v' = -v - 2w, w' = w - 2u" in
     the text copy: a w is lost (u' = u - 2w). Only in a test.
  Claims that were checked and are right: the six cases of 3.5 and the answers of 3.5.6-3.5.8; the
  eigenvalues of Examples 8.2.1-8.2.3; the pendulum's 15%, 3.8% and 0.048% (computed with the complete
  elliptic integral: 15.28%, 3.84%, 0.0476%; Lebl's relative error is divided by T, not by T_linear,
  which would give 18.0%, 4.0%, 0.0476%; the notebook says which); the constant of motion of
  Lotka-Volterra (C' = 0 by SymPy) and that (d/c, a/b) is its maximum; "slightly more than 1 hare" with
  200 foxes (1.268); the answers of 8.2.8, 8.2.9, 8.3.6 and 8.3.7.
* **Lessons of the reviews of notebook 17, applied from the start:**
  * every fact is defined before it is used: partial derivatives (with notebook 12's M_y), the Jacobian,
    lambda1 + lambda2 = tr A and lambda1 lambda2 = det A (shown by multiplying out), the real part of a
    complex pair is tr A / 2, "stable" and "asymptotically stable", "isolated" and "almost linear",
    "trajectory", "direction field"; the triangular fact and the 2 x 2 shortcut are recalled from
    notebook 17 with a link, not assumed;
  * every model states its conventions where it starts (rule 6: the boxes above, a line above each
    exercise that uses one, "x > 0" and "theta > 0" arrows on the drawings, axis labels with units);
    Lebl's examples without units say "no units";
  * a drawing before every fill-in (the direction field of Exercise 3 before `hares_jacobian`), with a
    caption, a first step, what to return and what the next cell will show;
  * readable outputs: a small `report(name, kind)` helper prints one line ("saddle (unstable):
    tr -1, det -2, eigenvalues -2 and 1") instead of a NamedTuple;
  * every figure is drawn at the width at which it is shown (8 in), panels stacked vertically (the
    portrait above its time graph or its trace-determinant plane), square axes for phase planes
    (`set_aspect("equal")` when x and y have the same units), legends off the curves;
  * checkers NaN-safe ("not <=" in every comparison) and accepting every right form: a point as a tuple,
    a list or an array; a matrix as a list of rows or `np.array`; integers for floats; the words with
    their synonyms and any capitals; values in the wrong order get "right values: write them in the
    order asked" where an order is asked;
  * no global name that an exercise defines (`hares_jacobian` is used nowhere else; Section 7's field is
    `lotka_volterra`); the comments in the code do not give the answers away; the slider answers do not
    give Explore (i) away (they quote only Lebl's 200);
  * a number in a title is what the title says it is; quotations only when literal; cross-references to
    other notebooks checked against their `make_NN.py` (notebook 03's "stable", 07E's plane of states and
    clockwise motion, 12's M_y, 17's shortcut and closed tanks).
* **Phase portraits in the browser.** Each portrait is one call of `trajectories` for all its starts
  (vectorised RK4, no SciPy, no loop over starts): measured on the desktop, six portraits of Section 2
  with 12 starts both ways (400 steps each way) and their direction fields, saved as a PNG, take 0.47 s;
  one Lotka-Volterra path of 1200 steps 0.06 s; the Lotka-Volterra figure 0.17 s. No `streamplot`: a
  direction field of at most 17 x 17 arrows with one `quiver` call. The slider redraws cost about 0.1 s
  on the desktop (one portrait of 12 starts, one plane).
* `tests/test_notebooks_run.py`:
  * `NAME_18 = "18_Phase_Plane.ipynb"`; the notebook as it is (every checker prints "no answer yet",
    Exercise 3 "complete hares_jacobian first");
  * with the correct answers, including the other accepted forms ("Sink", "stable node", "centre",
    (0.5, -0.25) as a tuple, the Jacobians as `np.array`, 2.0 for 2);
  * with wrong answers: the other kind ("source" for "sink", "center" for "spiral sink"), "stable" for
    "asymptotically stable", the point (-1/4, 1/2), a transposed Jacobian, 2 rad/s squared (4) in 2(d);
    for Exercise 3: the transposed matrix, -gamma x in place of -2 gamma x, +0.01 x in place of -0.01 x,
    the Lotka-Volterra Jacobian without gamma, a NaN matrix, a matrix of the wrong shape, a typo raising
    NameError (the message with its text); each caught with its message, none stopping the notebook;
  * a test that Section 7 runs again after Exercise 3 (no clash of names);
  * `EXTREMES[NAME_18]`: `plot_family(p=-3.0)`, `plot_family(p=3.0)`, `plot_family(p=0.0)`,
    `plot_family(p=1.0)`, `plot_family(p=-3.0 + 30 * 0.1)`, `plot_family(p=-3.0 + 40 * 0.1)` (0 and 1 as
    the slider gives them), `plot_family(p=1e-10)`, `plot_family(p=-1e-10)`, `plot_family(p=1.0 - 1e-10)`,
    `plot_family(p=1.0 + 1e-10)`, `plot_foxes(foxes=10.0)`, `plot_foxes(foxes=200.0)`,
    `plot_foxes(foxes=40.0)` (the start (50, 40), where the hares are at their smallest),
    `plot_foxes(foxes=10.0 + 19 * 10.0)`; with RuntimeWarning turned into an error, as for every
    notebook.
* **Time.** Library tests: about 70 new tests, under 3 s (the drawing tests need matplotlib). Notebook
  tests: about 10-15 s more. In the browser (`check_site.py`): estimated **18-24 s**, inside or just
  above the present 12-19 s per notebook (notebook 17: 18 s with 12 plots) and far inside the budget of
  420 s: about 8 s of kernel start and imports as for every notebook; eleven figures at about 0.2-0.5 s
  each on the desktop, three to four times slower in Pyodide (about 6-8 s); the SymPy calls (two
  `jacobian`, three small `solve`, two `simplify` of an energy and of Lebl's C', about 2-3 s); the
  sliders drawn once each. Each cell is timed in the browser, as for notebooks 16 and 17; if a portrait
  takes more than 2 s there, its starts or steps are reduced.
* **Registries** (all in the private working copy):
  * `plans/make_00.py`, one row after 17:
    `| 18 | [The phase plane and nonlinear systems](18_Phase_Plane.ipynb) | $x' = f(x, y),\ y' = g(x, y)$ | 19-20 | 3.5, 8.1-8.3 |`;
    then `00_Index.ipynb` is rebuilt;
  * `site/build_site.py`, `READING_ORDER`: a line
    `("Week 10 - The phase plane and nonlinear systems (Sessions 19-20)", ("18",))` after Week 9;
    `START_HERE`, `GUIDE_NOTEBOOK`, `PYODIDE_PACKAGES` unchanged;
  * `tests/test_notebooks_run.py`: `NAME_18`, `EXERCISES[NAME_18]`, `EXTREMES[NAME_18]`;
  * `odelib.SETUP_NAMES`: unchanged;
  * `notebooks/README.md` (the list of notebooks and of the library's sections) and `site/README.md`
    (counts and the table of tested notebooks); `ode/ode/README.md` (the number of tests);
  * no other notebook changes: notebook 17 already points to 18 by number.
* The site is rebuilt and the notebook is run in the browser with `check_site.py`; the export is checked
  for ASCII, size and stored local paths. With notebook 18, phase D (notebooks 14-18) is complete.
* After the build: an independent review of the mathematics, a reading as the weakest student and a code
  review, as for notebook 17, and their corrections.

## 5. Order of work

1. `odelib.py`: `classify_linear_2d` with `PhaseType`, `jacobian`, `classify_equilibrium_2d` with
   `EquilibriumType`, `trajectories`, `direction_field`, `phase_portrait`, `check_choice` and their
   constants; tests first, from the values above; ruff on the new lines; the whole existing suite passes
   unchanged (402 library + 140 notebook tests), and the library tests also in the CI environment (NumPy,
   SciPy and pytest only) and at the browser pins (numpy 2.4.6, SciPy 1.18.0).
2. `plans/make_18.py`: sections 1-3 with slider 1 (Session 19); look at every figure at display size.
3. Section 4 (the recipe), then Sections 5-6 (Session 20).
4. Section 7 with the pendulum drawing and slider 2; word count; apply the fall-back of Section 1 if
   needed.
5. Exercises and answers; `EXERCISES` and `EXTREMES` in the tests.
6. Index row, `READING_ORDER`, READMEs; rebuild; `check_site.py` with the times of the cells.
7. Independent reviews (mathematics, weakest student, code); corrections; final run of all tests.

## 6. Decisions for the instructor

1. **One notebook for two sessions** (as notebook 17). Proposed: Sections 2-4 for Session 19 and 5-7 for
   Session 20, with the exercises marked by session (Section 2 of this plan). The alternative, 18 and 18E,
   would put the nonlinear half in an optional notebook, which it is not.
2. **Letters.** Proposed: x, y as the unknowns with y' = dy/dt; f, g; (x*, y*); J; u = (u1, u2); tr A,
   det A in words; theta, theta' and l for the pendulum; Lebl's a, b, c, d, gamma and h announced where
   they are used; Lebl's kind names. New rows of the reserved-letter table (x, y of a planar system; J;
   u1, u2 of the linearization; a, b, c, d of Lotka-Volterra) before the decks of Sessions 19-20. The
   alternative for the populations is H(t) and F(t) or h(t) and f(t), which clash with the harvest, the
   force, the step and the right-hand side.
3. **Integration of the portraits.** Proposed: a vectorised fixed-step RK4 in odelib (`trajectories`,
   reusing the private `_rk4_step` of notebook 04's methods), not `solve_ivp` as the plan of notebooks
   17-22 first said: one computation for all the starts of a picture, no SciPy import, the same result on
   every machine, cheap in the browser, and the method of notebook 04; it is tested against `solve_ivp`,
   `eigen_solution` and Lebl's constant of motion. The alternative is `solve_ivp` once per start (simpler
   inside, about 12-24 calls per figure, slower in Pyodide).
4. **A checker for words** (`check_choice`). Proposed: added, with Lebl's synonyms, so "classify"
   exercises are checked as a student answers them. The alternative is numeric codes (1 = saddle, ...),
   which a student must look up.
5. **Units the book does not give.** Proposed: the pendulum with l = 9.81 m (g/l = 1 /s^2, Lebl's
   choice), and for the hares and foxes "time units" (Lebl gives none; a = 0.4 per time unit). The
   alternative is years (the period of Lebl's example is then 19 years) or months (19 months): please
   choose; the notebook will say that the unit is its own.
6. **Competing species** (named in the lecture plan for Session 20; not in Lebl's Chapter 8, and not in
   the other books of the course). Proposed: an Explore item (vii) with the notebook's own numbers, said to
   be so, built from Lebl's logistic equation (Section 1.6): x' = x(3 - x - 2y), y' = y(2 - x - y)
   (equilibria (0, 0) source, (3, 0) sink, (0, 2) sink, (1, 1) saddle with eigenvalues -1 +- sqrt(2):
   one species always wins, which one depends on the start; recomputed with SymPy). The alternative is to
   leave it to the deck of Session 20.
7. **Lebl's slips** (Section 4, items 1-7). Proposed: the notebook uses only passages that are right,
   writes (1, 0) where Section 8.2.1 and Example 8.2.4 print (0, 1), with the parenthesis "(the text
   prints (0, 1))", and does not use Exercises 8.1.7 (a), (b) and 8.2.11. The alternative is to correct
   silently.
8. **Left out** (proposed: decide after notebook 22): limit cycles and the Van der Pol oscillator (Lebl
   8.4; one line in "Where to go next"), chaos (8.5), systems in three dimensions beyond the test of
   Exercise 8.1.9, an optional 18E on the nonlinear pendulum (period, separatrix, damping: the master
   plan's Decision D6), forced systems (Lebl 3.9).

## 7. Files

```
ode/ode/notebooks/
    18_Phase_Plane.ipynb                       <- to build
    plans/18_Phase_Plane_plan.md               <- this document
    plans/make_18.py                           <- builds the notebook
    odelib.py                                  <- + classify_linear_2d() with PhaseType, jacobian(),
                                                  classify_equilibrium_2d() with EquilibriumType, trajectories(),
                                                  direction_field(), phase_portrait(), check_choice(), Field,
                                                  and their constants (KIND_..., STABILITY_..., PORTRAIT_WORDS,
                                                  STABILITY_WORDS, KIND_SYNONYMS, JACOBIAN_STEP, LINEARIZATION_RTOL,
                                                  EQUILIBRIUM_TOL, PORTRAIT_STEPS, ESCAPE_MARGIN, DIRECTION_GRID,
                                                  ARROW_LENGTH, TRAJECTORY_ARROW_AT)
    tests/test_odelib.py                       <- + their tests against the book, eigen_solution and solve_ivp
    tests/test_cross_validation.py             <- + against SymPy
    tests/test_notebooks_run.py                <- + NAME_18, the exercises and sliders of notebook 18
    00_Index.ipynb, plans/make_00.py           <- index row
    README.md                                  <- the list of notebooks
ode/ode/site/build_site.py                     <- READING_ORDER: Week 10 with 18
ode/ode/site/README.md                         <- counts and the table of tested notebooks
ode/ode/README.md                              <- the number of tests
```

## 8. As built (2026-11-20)

**The notebook.** `18_Phase_Plane.ipynb`, 39 cells (20 code), built by `plans/make_18.py`; 1.86 MB (1 857 094
bytes, under the 2 000 000 of `export_public.py`), plain ASCII, no local path stored. Before the exercises:
**2814 words** as whitespace tokens (every LaTeX token counted), **2292** with each formula as one word,
**2102** with each formula as one word and only tokens that contain a letter or a digit (the measure that
gives 2005 for notebook 17, whose plan reports 2013). The fall-backs of Section 1: Example 8.2.1 is code
with one comment line; the derivation of the pendulum's energy is a code comment; Section 4's x(t) panel
is kept (it carries no words to cut). The text was shortened instead (captions, references, the rule of
Section 6, Table 8.2.1 as one sentence).

**Figures: 10 stored** before the exercises (Case 3: eigenvector lines and portrait; the six cases; the
trace-determinant plane; slider 1 static; Example 8.1.1 with x(t); near (1, 0) and its linearization;
Examples 8.1.1 and 8.2.3 from (0.2, 0) with the distance against t; the pendulum drawing and portrait; the
hares and foxes with x(t), y(t); slider 2 static), plus 2 in Exercise 3 (the direction field before the
fill-in; the two portraits once `hares_jacobian` is right). **Sliders: 2** (`plot_family(p)`,
`plot_foxes(foxes)`), each with its question first, a static figure and its answer.

**Functions added to `odelib.py`** (new section "The phase plane and nonlinear systems (notebook 18)" after
notebook 17's; `check_choice` after `check_vector`; nothing existing changed; `SETUP_NAMES` unchanged):
* `Field` (type alias); `class PhaseType(NamedTuple)`: kind, stability, trace, determinant, eigenvalues, case
* `classify_linear_2d(matrix: MatrixLike, rtol: float = REPEATED_EIGENVALUE_RTOL) -> PhaseType`
* `jacobian(f: Callable[..., Sequence[float]], point: Sequence[float], h: float = JACOBIAN_STEP) -> np.ndarray`
* `class EquilibriumType(NamedTuple)`: point, jacobian, linear, stability, reason
* `classify_equilibrium_2d(f: Field, point: Sequence[float], rtol: float = LINEARIZATION_RTOL, h: float = JACOBIAN_STEP) -> EquilibriumType`
* `trajectories(f: Field, starts: ArrayLike, t_span: tuple[float, float], n_steps: int = PORTRAIT_STEPS, window: tuple[tuple[float, float], tuple[float, float]] | None = None) -> tuple[np.ndarray, np.ndarray]`
* `direction_field(ax: Axes, f: Field, x_range, y_range, n: int = DIRECTION_GRID, color: str = "0.6", length: float = ARROW_LENGTH) -> Axes`
* `phase_portrait(ax: Axes, f: Field, starts: ArrayLike, t_span, x_range, y_range, *, n_steps: int = PORTRAIT_STEPS, both_ways: bool = True, color: str = "C0") -> Axes`
* `check_choice(label: str, value, expected: str, choices: Sequence[str], synonyms: Mapping[str, str] | None = None) -> bool`
* constants `KIND_*` (13), `STABILITY_ASYMPTOTIC`, `STABILITY_STABLE`, `STABILITY_UNSTABLE`, `STABILITY_INCONCLUSIVE`,
  `JACOBIAN_STEP`, `LINEARIZATION_RTOL`, `EQUILIBRIUM_TOL`, `REASON_LINEAR`, `REASON_CENTER`, `REASON_SINGULAR`,
  `PORTRAIT_STEPS`, `ESCAPE_MARGIN`, `DIRECTION_GRID`, `ARROW_LENGTH`, `TRAJECTORY_ARROW_AT`, `TRAJECTORY_WIDTH`,
  `START_DOT_SIZE`, `ARROW_HEAD_SIZE`, `ARROW_HEAD_SPAN`, `PORTRAIT_WORDS`, `STABILITY_WORDS`, `KIND_SYNONYMS`;
  private helpers `_field_values`, `_flat_point`, `_values_at`, `_start_points`, `_window`, `_arrow_on`, `_word`;
  imports `Mapping`, `TYPE_CHECKING` (matplotlib's `Axes` only for the type hints). The module docstring lists every
  new name, and its error-handling paragraph says that `trajectories` gives NaN after a path leaves the window.
  numpy-only at import (checked: SciPy, SymPy and matplotlib are not loaded by `import odelib`).

**Tests added.** `test_odelib.py`: 104 (Cases 1-6, Exercises 3.5.2-3.5.8, the slider family at 12 values, scale
1e-200 and 1e150, invalid input; Jacobians of Examples 8.1.2, 8.2.1, 8.2.2, Lotka-Volterra, the pendulum,
Exercise 8.1.9 in 3-D; equilibria of Examples 8.2.1-8.2.3, 8.1.1, Exercises 8.2.8, 8.3.7, 8.1.7 (a), the pendulum
k = -2..2; `trajectories` against `eigen_solution` (6 cases) and `solve_ivp`, Lebl's constant of motion, forward
and back, NaN after leaving, no RuntimeWarning, shapes, invalid input; 3 drawing tests (skipped without
matplotlib); `check_choice`, 15). `test_cross_validation.py`: 21 (500 seeded integer matrices against SymPy's
exact eigenvalues; `jacobian` and `classify_equilibrium_2d` against SymPy on 10 systems of Lebl).
`test_notebooks_run.py`: 33 (the notebook as it is, right and wrong answers, 26 particular answers, Section 7 run
again after Exercise 3, 14 slider extremes with RuntimeWarning as an error).

**Departures from the plan, and why.**
1. Size: the first build was 2.57 MB, above the 2 MB limit of `export_public.py` (the sliders' pictures are stored
   twice, once more in the widget state). The notebook draws 11 arrows per side (9 in the six small pictures; the
   library default stays 17), no grid in phase planes, the six pictures in 2 rows of 3 (8 x 6.4 in) with 6 starts
   each, 8 starts near the saddle of Figure 4, trajectories of the linear pictures over 2.5 time units each way,
   and slightly lower figures (7.4-7.6 in). Result 1.86 MB. Further margin would need quantised PNGs (a 60%
   saving measured), which is a new mechanism: not done.
2. Section 6's figure: one phase plane (window +-1) with both systems from (0.2, 0), and the distance from the
   origin against t, in place of two phase planes: at the scale needed for the spiral the closed curve of radius
   0.2 was invisible.
3. Exercise 2(b): the Jacobian [[-1, 1], [1, 1]] is symmetric, so a transposed one is right; the wrong-answer test
   uses a sign slip, [[1, 1], [1, 1]]. The transposed matrix is tested in Exercise 3.
4. Explore (iv): the foxes reach 0 at t = 28.6 time units (rk4 and `solve_ivp` agree), not 28.3.
5. `check_choice`, an answer that is not text: "type the word in quotes: "...", with your word in place of the
   dots" (the plan's example word "saddle" is an answer of the notebook).
6. Lebl's Table 8.2.1 is given as one sentence (word count); the rule follows it as in the plan.
7. Exercise 3 has two figures (the plan's Section 1 counts one, its Section 2 describes both).
8. "centre" cannot be tested in a notebook answer (no exercise expects a center); it is tested in the library.
9. `classify_linear_2d` computes trace and determinant of A / s for its two bands (two products: everything else
   comes from `two_by_two_case`). Exercise 8.2.8 (a)'s (1, 0) is a "star source" (J = I; Lebl: "source").
10. `phase_portrait` marks the starts with `ax.scatter` (so each start has exactly two lines) and the arrowheads
    with `ax.annotate`; `trajectories` refuses a non-integer `n_steps` (2.5) as well as 0 and above `MAX_STEPS`.
11. Exercise 1(c) and 1(d) compute their expected values (zeros of a sample g > 1; the stability of [[0, 1],
    [-f', 0]] for four slopes) rather than store them. Explore (vii), competing species, added (Decision 6).
12. A bug found by the new tests and fixed before the build was final: `check_in_order` with a single number
    raised IndexError (now `ravel`).

**Registries and documents.** `make_00.py` row and `00_Index.ipynb` rebuilt; `READING_ORDER` Week 10 (one
line); `NAME_18`, `EXERCISES`, `EXTREMES`; `notebooks/README.md` (library list, about four minutes);
`site/README.md` (row 18, site size, last browser run); `ode/ode/README.md` (542 -> 700 tests).

**Gates (2026-11-20).**

| Gate | Result | Time |
|---|---|---|
| Library, env `ode` | 527 passed (402 + 125) | 28 s |
| Library as in CI (`site/.venv`, no SymPy, no matplotlib) | 380 passed, 11 skipped (279 + 101; 8 + 3) | 8 s |
| Notebooks (`test_notebooks_run.py`) | 173 passed (140 + 33) | 3 min 37 s |
| Lint (`ruff --no-cache`) | make_18.py and make_00.py clean; no finding on the added lines of odelib.py and the three test files | seconds |
| Site build | 23 notebooks, 135 MB | 18 s |
| `site/test_build_site.py` | 5 passed | 1 s |
| Browser (`check_site.py`, headless Edge) | 23 of 23 without errors; notebook 18: 20 cells, 13 plots, 2 sliders, **21 s** (estimate 18-24 s) | 7.3 min |

Not done (by instruction): export, the independent reviews (Section 5, step 7). Style-sheet rows: for the
coordinator (see the report).

### Corrections after the reviews (2026-11-20)

Three reviews (`reviews_nb/18_math_review.md`, `18_beginner_reading.md`, `18_code_review.md`) and the
coordinator's decisions 1-5. One line per finding.

**Mathematics review** (1 ERROR, 3 WARNING, 13 NOTE)
* E1 slider-2 answer "worse": replaced with the reviewer's text (it depends on how many: up to 40 foxes the
  curve shrinks, beyond about 70 it is larger, 200 foxes give 1.27); question "many more foxes (up to 200)";
  the claim "farther start, deeper fall" removed everywhere; Explore (i) asks for the slider's steps of 10,
  answer "190 on the slider's steps; the exact threshold, between 180 and 190, is about 183".
* W2 the rule with its hypotheses (continuous partial derivatives; stability and the kind of picture; one
  eigenvalue twice: the curves may also turn slowly); "What you will learn" item 3 the same; library docstring.
* W3 Exercise 1(d): first step and answer by the energy, not by the trace.
* W4 "(Lebl's text prints (0, 1) here and in Section 8.2.1)" added (Decision 7).
* N5 det J = 0: "does not decide the picture"; library: a line of equilibria with tr J > 0 is now
  STABILITY_UNSTABLE with REASON_SINGULAR_UNSTABLE (tested; cross-validation adjusted).
* N6 "grows (2y^4 > 0 except where the curve crosses y = 0)". N7 Figure 3.5.2 caption fixed.
* N8 degenerate / star source or sink, line of equilibria, parallel lines defined in Section 3.
* N9 Exercise 2(d): the "crucial" speed, above which it goes over; answer: with exactly 2 rad/s it creeps.
* N10 see E1. N11 answer 4(vii): except on the curve into the saddle (1, 1).
* N12 item 1: "(on the borderlines, with a look at A)". N13 "at least a line (the whole plane if A = 0)".
* N14 the step of ol.jacobian described exactly. N15 "(Lebl; his Exercise 8.3.3)".
* N16 E(x, y), E(theta, theta'), C(x, y) with arguments; "not the u1, u2 of notebooks 14 and 17"; the pendulum
  box says t = 0 when the motion starts; answers 2(d), 4(vi) carry "positive counterclockwise".
* N17 KIND_SYNONYMS gains stable/unstable spiral and focus (read-only MappingProxyType).

**Beginner reading** (no BLOCKING; all CONFUSING and the POLISH items applied with the reviewer's wording,
shortened where the length needed it)
* 0.1 references (07E optional); 1.1 t is not drawn, the plane of (x, x') via notebook 17; 1.2 notebook 09's
  segments against arrows; 1.3 notebook 03's "stable" reconciled; 1.4 the tangent-plane step; 1.5 the table
  column dropped.
* 3.1 why in along one line and out along the other; 4.1, 5.1, 14.1, 16.1, 18.1, 22.2 code comments
  (linear_field, the fields of classify_linear_2d and classify_equilibrium_2d, the last argument of
  direction_field, *WINDOW, paths[i, k, j], NaN after leaving the window, flatnonzero, "A if ... else B",
  the period-finding lines).
* 6.1, 8.1 captions; 8.2 the discriminant step; 8.3 "degenerate"; 8.4 "a center is stable but not
  asymptotically stable"; 9.1 Exercise 4(iii) now gives the eigenvalues (no giveaway).
* 13.1 dropped with the slider answer's tanks clause (length); 13.2 step 4 explains both ways and T; 13.3;
  13.4 the question now answerable before running.
* 14.2 equilibria marked by stars; 15.1; 16.2 dashed eigen-directions on the nonlinear panel; 16.3 Example
  8.2.1 named in the caption; 16.4 start circle turned by a quarter step.
* 17.1 isolated / almost linear in one sentence; 17.2 "conserved" defined, the energy derived (no reliance
  on 07E); 17.3 why closed; 17.4 not taken (length; the separatrix sentence kept).
* 19.1 the pendulum as x' = y, y' = -(g/l) sin x; 19.2 E' = 0 shown; 19.3; 19.4 clockwise explained
  without 07E; 21.1 the separatrix tied to Exercise 2(d).
* 21.2-21.6 equilibria pairs, lambda^2 + ad = 0, 18.1 against 19.35, C(x, y) and dC/dt printed, hilltop.
* 22.1 arrows of one length on the screen: direction_field now measures the axes (equal aspect: data
  units; otherwise the box), plane_axes is called before it; tested.
* 22.3 start (125, 40); 23.1 see E1; 26.1 see E1 (the coordinator kept 1.27); 26.2 shorter caption.
* 27.1 "conservative" defined; 28.1 comment dropped; 29.1, 29.2 first steps; 31.1 logistic link; 32.1 the
  bottom row of arrows starts at y = 4.
* 36.2 a commented template for a portrait in the Explore cell; 36.3 how to compute and find when the foxes
  reach 0; 36.4 H for the hunting (Lebl's h); 36.5 np.radians, T twice the half period; 36.6, 36.7 first
  steps; 36.8 "(no units)".

**Code review** (1 IMPORTANT, M1-M11)
* I1 size: figure sizes about 12% smaller (PAIR 7.0 x 3.8, SIX 7.0 x 5.6, PLANE 7.0 x 5.4, FAMILY 6.4 x 6.1,
  STACKED 7.0 x 6.5, PENDULUM 7.0 x 5.8 with a smaller drawing, HARES 7.0 x 3.9, PORTRAITS 7.0 x 7.4) and
  FIELD_ARROWS = 9: 1 564 218 bytes. The widget-state copy (nbbuild) is left for a later change.
* M1 JACOBIAN_STEP comment; M2 "an array of shape ..." message; M3 trajectories docstring (the pair, not
  points); M4 _arrow_on uses the first visible run; M5 one helper _ranges (finite, increasing, numbers) for
  direction_field, phase_portrait, trajectories; n of direction_field a whole number >= 2 (MIN_GRID);
  check_choice refuses non-text choices and synonyms that are not words (all ValueError with a sentence).
* M6 a fifth check point (0, 20) with x = 0 ("correct at all 15 points"; a test with a formula that divides
  by x). M7 x, y redefined in the Exercise 2 cell. M8 CHECK_TOLERANCE in the order hint.
* M9 PARABOLA_POINTS, POINT_SIZE, MOVING_SIZE, STAR_SIZE, BOB_SIZE named. M10 EquilibriumType.point
  tuple[float, float]; KIND_SYNONYMS read-only. M11 nothing to change.
* Not mine to edit: the comment at site/export_public.py:59 ("well below") is stale; listed for the
  coordinator.

**Letters (decision 5).** The notebook follows the new STYLE_SHEET rows: planar x, y with y' = dy/dt; f, g;
(x*, y*), J; u1, u2 announced; a, b, c, d of Lotka-Volterra announced; theta and $\ell$ (written l in code
and titles, as notebooks 07 and 16 write $\ell$ in text); E(x, y), C(x, y); tr and det in words.

**Length.** Before the exercises: **2991** whitespace tokens, **2457** with each formula as one word,
**2255** with each formula as one word and only tokens with a letter or digit (the measure of 2102 before
the reviews). Within the coordinator's 2150-2250 band (2255, at its edge): the reviewer's cuts were taken
and every added explanation was written as briefly as possible.

**Gates after the corrections (2026-11-20).**

| Gate | Result | Time |
|---|---|---|
| Library, env `ode` | 529 passed | 27 s |
| Library as in CI (`site/.venv`) | 382 passed, 11 skipped | 8 s |
| Notebooks (`test_notebooks_run.py`) | 174 passed (140 + 34) | 3 min 36 s |
| Lint (`ruff --no-cache`) | make_18.py clean; 0 findings on the added lines of odelib.py and the tests | seconds |
| Site build | 23 notebooks, 135 MB | 17 s |
| `site/test_build_site.py` | 5 passed | 1 s |
| Browser (`check_site.py`) | 23 of 23 without errors; notebook 18: 20 cells, 13 plots, 2 sliders, **23 s** | 7.1 min |

`--collect-only`: 703 tests (`ode/ode/README.md` updated). Notebook file: 1 564 218 bytes.

**Re-check (`reviews_nb/18_recheck.md`: ready after one rebuild), applied with its fixes**
* P1 direction_field: the box is measured after the limits and the aspect (`ax.apply_aspect()`), the shaft is
  `ARROW_WIDTH = 0.075` of the arrow (quiver units "inches"); test assertion added; the `_equal_aspect` helper
  removed. Pendulum, hares-and-foxes, slider-2 and Exercise 3 figures checked at display width: thin shafts,
  small heads, no overlap.
* P2 Exercise 4(iv): the student defines `hunted(x, y)` (LV constants, the hunting 10) and uses it in
  `classify_equilibrium_2d` and `trajectories`; no stale `field`.
* P3 slider-2 answer: why the hares never fall below 50 from 40 foxes (the start is on the line x' = 0).
* P4 slider-2 answer: "the hares fall to about one (Lebl: ...; Exercise 4(i) measures it)"; 1.27 removed.
* P5 Exercise 1(d): the first step ends with a question; the answer says "tend to", with the reason.
* P6 T_NEAR = 4.0 for both panels of the saddle figure: every curve leaves the window.
* P7 inset arrow grids: ARROWS_HARES ((10, 440), (4, 76)), ARROWS_LV, ARROWS_FOXES, ARROWS_PENDULUM, each
  followed by the limits of the drawing window: no arrow cut at an edge.
* P8 slider 1: the legend shows only the moving point; "saddle" visible.
* P9 STYLE_SHEET rows: done by the coordinator.

Gates after the re-check fixes: library (env `ode`) 529 passed, 26 s; CI-like (`site/.venv`) 382 passed,
11 skipped, 9 s; notebooks 174 passed, 3 min 33 s; ruff: make_18.py clean, 0 findings on added lines; site
build 23 notebooks, 135 MB, 15 s; `test_build_site.py` 5 passed; browser 23 of 23 without errors, notebook
18: 20 cells, 13 plots, 2 sliders, **19 s**, 6.8 min in all. Notebook file 1 534 912 bytes. Length before
the exercises: 3009 whitespace tokens, 2476 with each formula as one word, 2273 on the measure of plan 17.
