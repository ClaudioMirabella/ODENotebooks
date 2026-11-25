"""
odelib - small, tested helpers for the MATH 331 notebooks on ordinary differential equations.

The notebooks keep the mathematics visible: every formula implemented here is derived
in the notebooks, with references to the course textbook,

    J. Lebl, *Notes on Diffy Qs: Differential Equations for Engineers* (CC BY-SA),
    free on LibreTexts (section numbers as in that edition).

Contents
--------
Analytical solutions
    exponential_solution          P(t) = P0 e^{k (t - t0)}                   (Lebl 0.2, 1.1)
    growth_rate_from_two_values   k = ln(P2/P1) / (t2 - t1)                   (notebook 01)
    decay_constant, half_life     lambda = ln 2 / tau                                (decay law)
    age_from_fraction             t = -ln(fraction) / lambda                       (radiocarbon dating)
    logistic_solution             x' = k x (M - x)                            (Lebl 1.6)
    logistic_blowup_time          finite-time blow-up when x0 < 0
    harvesting_equilibria         x' = k x (M - x) - h                        (Lebl 1.6)
    torricelli_height,            h' = -(a/A) sqrt(2 g h): draining of a tank  (notebook 10)
    torricelli_drain_time
    linear_first_order            y' + p(x) y = f(x), y(x0) = y0, by numerical integrals (Lebl 1.4)
    relaxation                    x' = (x_inf - x)/tau: cooling, mixing, drag, RC, RL (notebook 13)
    quadratic_drag_velocity       m v' = m g - c v^2 from rest: v_T tanh(g t / v_T)
Numerical methods (fixed step)
    euler, heun, rk4              one-step methods for y' = f(t, y)           (Lebl 1.7)
    observed_order                order of convergence from successive errors
    solution_curve                the solution through a point, both ways, inside a window
Plotting and first-order stability
    slope_field                   short segments of slope f(t, y)             (Lebl 1.2)
    phase_line                    critical points and arrows of x' = f(x)     (Lebl 1.6)
    classify_equilibrium          stable / unstable / inconclusive from f'(x*) (Lebl 1.6)
Second-order linear equations
    characteristic_roots          roots of a r^2 + b r + c = 0                (Lebl 2.2)
    constant_coefficient_solution exact y(t) of a y'' + b y' + c y = 0, any a, b, c (Lebl 2.2)
    damping_regime                over-, critically, under-damped, undamped   (Lebl 2.4)
    free_response                 exact x(t) of m x'' + c x' + k x = 0         (Lebl 2.2, 2.4)
    forced_response               a y'' + b y' + c y = f(t), variation of parameters (Lebl 2.5; notebook 14)
    root_multiplicity             how often alpha + i beta is a characteristic root (Lebl 2.5)
    steady_state,                 the steady periodic solution and its peak    (Lebl 2.6; notebook 15)
    practical_resonance
    cosine_forced_response        m x'' + c x' + k x = F0 cos(omega t), beats and resonance (Lebl 2.6)
    wronskian_values              W = y1 y2' - y1' y2 from sampled values     (notebook 05)
Equations of order n, boundary value problems (notebook 16)
    higher_order_solution         a_n y^(n) + ... + a_0 y = 0 with n initial values (Lebl 2.3)
    classify_linear_system        one, none or infinitely many solutions of M C = b (Lebl 4.1)
    beam_deflection               EI y'''' = w(x) with two conditions at each end
Linear systems x' = A x (notebook 17)
    companion_matrix              the system u' = A u of an equation of order n (Lebl 3.1)
    two_by_two_case, TwoByTwoCase the case, eigenvalues and eigenvectors of a 2 x 2 matrix (Lebl 3.4, 3.7)
    eigen_solution                x(t) for a 2 x 2 matrix by the eigenvalue method (Lebl 3.4, 3.7)
The phase plane (notebook 18)
    classify_linear_2d, PhaseType the picture of x' = A x at the origin, from tr A and det A (Lebl 3.5)
    jacobian                      the Jacobian matrix of f at a point, by central differences (Lebl 8.1)
    classify_equilibrium_2d,      an equilibrium of x' = f(x, y), y' = g(x, y) by its linearization
      EquilibriumType             (Lebl 8.2)
    trajectories                  the solutions from several starts at once, by RK4 (notebook 04)
    direction_field               arrows along (f, g) on a grid, all of one length (Lebl 3.1)
    phase_portrait                the trajectories through given points, both ways in time (Lebl 3.5, 8.1)
The Laplace transform (notebook 19)
    laplace_numeric               F(s) at one s, by the integral that defines it (Lebl 6.1)
    laplace_table, LaplaceRow     Lebl's Table 6.1.1 as data: f(t), F(s) and the range of s (Lebl 6.1)
Steps, impulses and convolution (notebook 20)
    step                          Lebl's unit step u(t - a): 0 before the time a, 1 from a on (Lebl 6.1, 6.2)
    convolution                   (f * g)(t), the integral of f(tau) g(t - tau) from 0 to t (Lebl 6.3)
    impulse_response              a y'' + b y' + c y = delta(t) from rest (Lebl 6.4)
Series solutions (notebook 21)
    series_coefficients           a_0, ..., a_(n-1) of a power series, from a recurrence (Lebl 7.2, 7.3)
    series_value                  the partial sum of a power series at x, by Horner's rule (Lebl 7.1)
    indicial_roots                the roots of r(r - 1) + p0 r + q0 = 0 and their case (Lebl 7.3)
Numerical methods, part 2 (notebook 22)
    local_error, global_error     the error of one step; the error at the end for several N (Lebl 1.7)
    backward_euler                the implicit Euler method, by Newton's method at each step
    embedded_step                 Heun's step and the estimate |Heun - Euler| of Euler's error
Exercises
    check                         compare an answer found by hand with the expected one
    check_formula                 the same for a formula typed as a SymPy expression
    check_solution                a formula checked by substituting it in the equation
    check_vector                  a vector or a matrix; an eigenvector up to a factor
    check_choice                  a word, such as "saddle", with Lebl's synonyms
Help for beginners
    guide                         the student guide (also on the home page of the site)
    hint_for                      one sentence of advice under a beginner's error message
Named constants (each defined once, above the function that uses it): GRAVITY, MAX_STEPS, CURVE_STEP,
STEEP_FRACTION, QUAD_LIMIT, ROOT_ROUNDING, NEGLIGIBLE_DAMPING, SUPPORTS, BEAM_MARGIN,
REPEATED_EIGENVALUE_RTOL, CASE_REAL, CASE_COMPLEX, CASE_REPEATED_TWO, CASE_REPEATED_ONE, SYSTEM_SIZE,
LARGER_SYSTEMS, KIND_SADDLE, KIND_SOURCE, KIND_SINK, KIND_SPIRAL_SOURCE, KIND_SPIRAL_SINK, KIND_CENTER,
KIND_DEGENERATE_SOURCE, KIND_DEGENERATE_SINK, KIND_STAR_SOURCE, KIND_STAR_SINK, KIND_LINE_OF_EQUILIBRIA,
KIND_PARALLEL_LINES, KIND_ALL_EQUILIBRIA, STABILITY_ASYMPTOTIC, STABILITY_STABLE, STABILITY_UNSTABLE,
STABILITY_INCONCLUSIVE, JACOBIAN_STEP, LINEARIZATION_RTOL, EQUILIBRIUM_TOL, REASON_LINEAR, REASON_CENTER,
REASON_SINGULAR, REASON_SINGULAR_UNSTABLE, PORTRAIT_STEPS, ESCAPE_MARGIN, DIRECTION_GRID, ARROW_LENGTH,
MIN_GRID, ARROW_WIDTH, TRAJECTORY_ARROW_AT,
TRAJECTORY_WIDTH, START_DOT_SIZE, ARROW_HEAD_SIZE, ARROW_HEAD_SPAN, LAPLACE_RTOL, LAPLACE_RTOL_RANGE,
LAPLACE_HORIZON, LAPLACE_UNIT_RATE, LAPLACE_EDGES, LAPLACE_EDGE_RATIO, LAPLACE_SHIFTS,
LAPLACE_GROWTH_TIMES, LAPLACE_SCAN, LAPLACE_JUMP_RATIO, LAPLACE_MAX_JUMPS, LAPLACE_NARROW,
LAPLACE_DEMO_A, LAPLACE_DEMO_OMEGA, LAPLACE_DEMO_N, LAPLACE_ROW_NAMES,
STEP_START, CONVOLUTION_RTOL, CONVOLUTION_PIECES, CONVOLUTION_SCAN, CONVOLUTION_WINDOW,
CONVOLUTION_JUMP_RATIO, CONVOLUTION_MAX_JUMPS, CONVOLUTION_NARROW,
SERIES_CENTER, INDICIAL_NOT_INTEGER, INDICIAL_DOUBLE, INDICIAL_INTEGER, INDICIAL_COMPLEX,
INDICIAL_TOLERANCE, INDICIAL_AMBIGUOUS,
NEWTON_TOLERANCE, NEWTON_MAX_ITERATIONS, BACKWARD_MAX_STEPS,
CHECK_TOLERANCE, ROUNDED, FORMULA_SAMPLES,
RESIDUAL_ROUNDING, PORTRAIT_WORDS, STABILITY_WORDS, KIND_SYNONYMS; the guide and the hints:
GUIDE_START, GUIDE_TYPING, GUIDE_TYPING_RULE, GUIDE_ERRORS, GUIDE_HOW, GUIDE_WORDS, GUIDE_TOPICS,
GUIDE_WIDTH, HINT_LABEL, FUNCTION_NAMES, CONSTANT_NAMES, WORD_NAMES, SETUP_NAMES, GUIDE_PARTS_TYPED,
RUN_ALL, SAMPLE, HINTS_INSTALLED.

Error handling: invalid arguments raise ``ValueError`` (never a silent NaN), with one
documented exception: ``logistic_solution`` returns NaN for times at or beyond the
blow-up time, where the solution does not exist (and, by design, ``trajectories`` gives NaN for
the points of a path after it has left the window of the picture).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from collections.abc import Iterator as _Iterator      # (notebook 21: a recurrence can loop over a)
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:                       # for the type hints only: matplotlib is not imported at the top
    from matplotlib.axes import Axes

ArrayLike = float | Sequence[float] | np.ndarray
RHS = Callable[[float, np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def _require_positive(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return value


def _require_finite(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# Exponential growth and decay
# ---------------------------------------------------------------------------
def exponential_solution(t: ArrayLike, P0: float, k: float, t0: float = 0.0) -> np.ndarray:
    """Unique solution of P' = k P, P(t0) = P0:  P(t) = P0 exp(k (t - t0)).

    k > 0 gives growth (Malthus), k < 0 decay (k = -lambda). Defined for every real t.
    """
    P0, k, t0 = _require_finite("P0", P0), _require_finite("k", k), _require_finite("t0", t0)
    return P0 * np.exp(k * (np.asarray(t, dtype=float) - t0))


def growth_rate_from_two_values(t1: float, P1: float, t2: float, P2: float) -> float:
    """k such that P(t) = C e^{kt} passes through (t1, P1) and (t2, P2): k = ln(P2/P1)/(t2 - t1).

    Requires P1, P2 of the same (nonzero) sign and t1 != t2.
    """
    if t1 == t2:
        raise ValueError("the two times must be different")
    if P1 == 0 or P2 == 0 or np.sign(P1) != np.sign(P2):
        raise ValueError("P1 and P2 must be nonzero and of the same sign (P never changes sign)")
    return float(np.log(P2 / P1) / (t2 - t1))


def decay_constant(half_life: float) -> float:
    """Decay constant lambda = ln 2 / tau from the half-life tau > 0."""
    return float(np.log(2.0) / _require_positive("half_life", half_life))


def half_life(decay_const: float) -> float:
    """Half-life tau = ln 2 / lambda from the decay constant lambda > 0 (it does not depend on N0)."""
    return float(np.log(2.0) / _require_positive("decay_const", decay_const))


def age_from_fraction(fraction: float, half_life_value: float) -> float:
    """Elapsed time after which a fraction 0 < f <= 1 of the isotope remains: t = -ln(f) / lambda."""
    fraction = float(fraction)
    if not 0 < fraction <= 1:
        raise ValueError(f"the remaining fraction must be in (0, 1], got {fraction}")
    return float(-np.log(fraction) / decay_constant(half_life_value))


# ---------------------------------------------------------------------------
# Logistic equation  x' = k x (M - x)   (notation of Lebl, Section 1.6)
# ---------------------------------------------------------------------------
def logistic_blowup_time(x0: float, k: float, M: float, t0: float = 0.0) -> float:
    """Time t* at which the solution with x(t0) = x0 < 0 tends to -infinity.

    From the closed form (see ``logistic_solution``) the denominator
    x0 + (M - x0) e^{-kM(t - t0)} vanishes at  t* = t0 + ln((M - x0)/(-x0)) / (kM).
    Returns +inf when x0 >= 0 (the solution then exists for all t >= t0).
    """
    k, M = _require_positive("k", k), _require_positive("M", M)
    x0 = _require_finite("x0", x0)
    if x0 >= 0:
        return float("inf")
    return float(t0 + np.log((M - x0) / (-x0)) / (k * M))


def logistic_solution(t: ArrayLike, x0: float, k: float, M: float, t0: float = 0.0) -> np.ndarray:
    """Solution of the logistic IVP  x' = k x (M - x),  x(t0) = x0  (k, M > 0).

        x(t) = M x0 / (x0 + (M - x0) e^{-k M (t - t0)}).

    Valid for t >= t0 in all cases. For x0 < 0 the solution exists only up to the blow-up
    time t* (``logistic_blowup_time``); for t >= t* NaN is returned on purpose, since no
    solution exists there. Times t < t0 (backward in time) are also allowed, but then a
    solution with x0 > M blows up to +infinity in the past; such points are returned as NaN.
    """
    k, M = _require_positive("k", k), _require_positive("M", M)
    x0, t0 = _require_finite("x0", x0), _require_finite("t0", t0)
    t = np.asarray(t, dtype=float)
    denominator = x0 + (M - x0) * np.exp(-k * M * (t - t0))
    with np.errstate(divide="ignore", invalid="ignore"):
        x = M * x0 / denominator
    # The denominator equals M > 0 at t = t0 and is monotonic in t, so the formula holds
    # exactly on the interval where it stays positive; elsewhere the solution does not exist.
    return np.where(denominator > 0, x, np.nan)


def harvesting_equilibria(k: float, M: float, h: float) -> tuple[float, ...]:
    """Critical points of x' = k x (M - x) - h  (h >= 0 constant harvesting rate).

    Roots of k x^2 - k M x + h = 0:  x = (kM +- sqrt((kM)^2 - 4hk)) / (2k)  (Lebl, Section 1.6).
    Returns (B, A) with B <= A, a single value when h = kM^2/4 (saddle-node), or () when
    h > kM^2/4 (no equilibria: the population always collapses).
    """
    k, M = _require_positive("k", k), _require_positive("M", M)
    h = float(h)
    if h < 0:
        raise ValueError("the harvesting rate h must be >= 0")
    discriminant = (k * M) ** 2 - 4 * h * k
    if np.isclose(discriminant, 0.0, atol=1e-12 * (k * M) ** 2):
        return (M / 2,)
    if discriminant < 0:
        return ()
    root = np.sqrt(discriminant)
    return ((k * M - root) / (2 * k), (k * M + root) / (2 * k))


GRAVITY = 9.81                  # m/s^2: the rounded value used throughout the course


def torricelli_drain_time(h0: float, area_ratio: float, g: float = GRAVITY) -> float:
    """Time to empty a tank of constant cross-section through a hole at the bottom.

    Torricelli's law, h' = -(a/A) sqrt(2 g h) with ``area_ratio`` = a/A (hole area over
    cross-section of the tank), is separable: T = (A/a) sqrt(2 h0 / g)  (notebook 10;
    the law is stated in Adkins-Davidson, Section 1.1, Exercise 3).
    """
    h0, area_ratio, g = (_require_positive(name, value) for name, value in
                         (("h0", h0), ("area_ratio", area_ratio), ("g", g)))
    return np.sqrt(2 * h0 / g) / area_ratio


def torricelli_height(t: ArrayLike, h0: float, area_ratio: float,
                      g: float = GRAVITY) -> np.ndarray:
    """Height of the liquid in the tank of ``torricelli_drain_time``, with h(0) = h0.

    h(t) = (sqrt(h0) - (a/A) sqrt(g/2) t)^2 up to the draining time, and 0 afterwards
    (the tank stays empty), so the function can be evaluated and plotted for any t >= 0.
    """
    drain_time = torricelli_drain_time(h0, area_ratio, g)         # validates the arguments
    t = np.asarray(t, dtype=float)
    if np.any(t < 0):
        raise ValueError("t must be >= 0 (the tank starts draining at t = 0)")
    root = np.sqrt(h0) - area_ratio * np.sqrt(g / 2) * np.minimum(t, drain_time)
    return root ** 2


def relaxation(t: ArrayLike, x0: float, x_inf: float, tau: float) -> np.ndarray:
    """Solution of x' = (x_inf - x) / tau with x(0) = x0:  x_inf + (x0 - x_inf) e^{-t/tau}.

    The common form of the first-order models of notebook 13: Newton's cooling (tau = 1/k),
    a mixing tank of constant volume (tau = V/r), a body with linear drag (tau = m/k), the
    RC circuit (tau = RC) and the RL circuit (tau = L/R). ``tau`` > 0 is the time constant:
    63 % of the gap x_inf - x0 is closed after one tau, 99 % after five.
    """
    x0, x_inf = _require_finite("x0", x0), _require_finite("x_inf", x_inf)
    tau = _require_positive("tau", tau)
    t = np.asarray(t, dtype=float)
    return x_inf + (x0 - x_inf) * np.exp(-t / tau)


def quadratic_drag_velocity(t: ArrayLike, terminal_velocity: float,
                            g: float = GRAVITY) -> np.ndarray:
    """Velocity of a body released from rest with drag proportional to v^2.

    m v' = m g - c v^2 has the terminal velocity v_T = sqrt(m g / c) and, for v(0) = 0,
    the solution v(t) = v_T tanh(g t / v_T) (separable; notebook 13).
    """
    v_t, g = _require_positive("terminal_velocity", terminal_velocity), _require_positive("g", g)
    t = np.asarray(t, dtype=float)
    return v_t * np.tanh(g * t / v_t)


def linear_first_order(p: Callable[[float], float], f: Callable[[float], float],
                       x0: float, y0: float, x: ArrayLike) -> np.ndarray:
    """Solution of the linear problem y' + p(x) y = f(x), y(x0) = y0, at the points ``x``.

    The definite-integral form of the integrating-factor method (Lebl, formula (1.4.5)):

        y(x) = exp(-P(x)) * (y0 + integral from x0 to x of exp(P(t)) f(t) dt),
        P(x) = integral from x0 to x of p(s) ds,

    with the integrals computed numerically, so it also works when they have no closed form
    (Lebl, Exercise 1.4.3). ``p`` and ``f`` take one number and must be continuous between
    x0 and every point of ``x`` (to the left or to the right of x0). SciPy is imported here,
    not at the top: the rest of odelib needs NumPy only.
    """
    from scipy.integrate import quad

    x0, y0 = _require_finite("x0", x0), _require_finite("y0", y0)
    points = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(points)):
        raise ValueError("x must contain finite numbers")

    def exponent(s: float) -> float:                     # P(s)
        return quad(p, x0, s)[0]

    def value(xi: float) -> float:
        integral = quad(lambda s: np.exp(exponent(s)) * f(s), x0, xi)[0]
        return np.exp(-exponent(xi)) * (y0 + integral)

    import warnings

    from scipy.integrate import IntegrationWarning

    try:
        with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            warnings.simplefilter("error", IntegrationWarning)    # a doubtful integral is an error here
            values = np.array([value(xi) for xi in points.ravel()])
    except (IntegrationWarning, ZeroDivisionError, OverflowError) as problem:
        raise ValueError("the integrals could not be computed: p and f must be continuous between "
                         f"x0 and every point of x, and the interval not too long ({problem})") from None
    if not np.all(np.isfinite(values)):
        raise ValueError("the integrating factor is too large or too small on this interval: "
                         "use points closer to x0")
    return values.reshape(points.shape)


# ---------------------------------------------------------------------------
# Fixed-step one-step methods for y' = f(t, y),  y(t0) = y0
# ---------------------------------------------------------------------------
MAX_STEPS = 1_000_000           # a larger request is a typing mistake: refuse it, do not freeze


def _require_few_steps(n_steps: float) -> None:
    """Stop at once when a step size would need an unreasonable number of steps.

    In the browser version of the course a loop of 10^9 steps freezes the tab; an error
    message that names the cause is safer than a computation that never ends.
    """
    if n_steps > MAX_STEPS:
        raise ValueError(f"too many steps ({n_steps:.3g}, the limit is {MAX_STEPS}): use a larger step h")


def _grid(t_span: tuple[float, float], h: float) -> np.ndarray:
    """Uniform grid t0, t0 + h, ..., t1. The interval must contain a whole number of steps."""
    t0, t1 = (_require_finite("t0", t_span[0]), _require_finite("t1", t_span[1]))
    h = _require_positive("h", h)
    if t1 <= t0:
        raise ValueError("t_span must satisfy t0 < t1")
    n_steps = (t1 - t0) / h
    _require_few_steps(n_steps)
    n = int(round(n_steps))
    if n < 1 or not np.isclose(n_steps, n, rtol=1e-9, atol=1e-12):
        raise ValueError(f"(t1 - t0)/h = {n_steps} must be a positive integer")
    return t0 + h * np.arange(n + 1)


def _integrate(step, f: RHS, t_span, y0, h):
    t = _grid(t_span, h)
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    y = np.empty((t.size, y0.size))
    y[0] = y0
    for i in range(t.size - 1):
        with np.errstate(over="ignore", invalid="ignore"):   # overflow is detected just below
            y[i + 1] = step(f, t[i], y[i], h)
        if not np.all(np.isfinite(y[i + 1])):
            raise FloatingPointError(
                f"the numerical solution overflowed at t = {t[i + 1]:.6g} "
                "(blow-up of the exact solution, or an unstable step size)")
    return t, (y[:, 0] if y.shape[1] == 1 else y)


def _euler_step(f, t, y, h):
    return y + h * np.asarray(f(t, y), dtype=float)


def _heun_step(f, t, y, h):
    k1 = np.asarray(f(t, y), dtype=float)
    k2 = np.asarray(f(t + h, y + h * k1), dtype=float)
    return y + h * (k1 + k2) / 2


def _rk4_step(f, t, y, h):
    k1 = np.asarray(f(t, y), dtype=float)
    k2 = np.asarray(f(t + h / 2, y + h / 2 * k1), dtype=float)
    k3 = np.asarray(f(t + h / 2, y + h / 2 * k2), dtype=float)
    k4 = np.asarray(f(t + h, y + h * k3), dtype=float)
    return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def euler(f: RHS, t_span: tuple[float, float], y0: ArrayLike, h: float):
    """Euler's method  y_{i+1} = y_i + h f(t_i, y_i)  (Lebl 1.7). Order 1. Returns (t, y)."""
    return _integrate(_euler_step, f, t_span, y0, h)


def heun(f: RHS, t_span: tuple[float, float], y0: ArrayLike, h: float):
    """Improved Euler (Heun): average of the slopes at both ends of the step. Order 2."""
    return _integrate(_heun_step, f, t_span, y0, h)


def rk4(f: RHS, t_span: tuple[float, float], y0: ArrayLike, h: float):
    """Classical fourth-order Runge-Kutta method. Order 4."""
    return _integrate(_rk4_step, f, t_span, y0, h)


def observed_order(errors: Sequence[float], ratio: float = 2.0) -> np.ndarray:
    """Observed orders log(e_i / e_{i+1}) / log(ratio) for step sizes divided by ``ratio``."""
    e = np.asarray(errors, dtype=float)
    if np.any(e <= 0):
        raise ValueError("errors must be positive")
    return np.log(e[:-1] / e[1:]) / np.log(ratio)


CURVE_STEP = 0.01               # default step of solution_curve
STEEP_FRACTION = 0.1            # a step that would cross this fraction of the window is "vertical"


def _follow(f, t0: float, y0: float, t_end: float, y_range: tuple[float, float], h: float):
    """RK4 from (t0, y0) towards t_end (either side of t0), while the curve stays in y_range."""
    direction = 1.0 if t_end > t0 else -1.0
    ts, ys = [], []
    t, y = t0, y0
    height = y_range[1] - y_range[0]
    while direction * (t_end - t) > 0:
        step = direction * min(h, abs(t_end - t))            # the last step lands on t_end
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            slope = float(f(t, y))
            if not np.isfinite(slope) or abs(slope) * h > STEEP_FRACTION * height:
                break                                        # vertical tangent: the solution ends here
            y = float(_rk4_step(f, t, y, step))
        t = t + step
        if not np.isfinite(y) or not y_range[0] <= y <= y_range[1]:
            break                                            # left the window (or blew up)
        ts.append(t)
        ys.append(y)
    return ts, ys


def solution_curve(f: Callable[[float, float], float], point: tuple[float, float],
                   t_range: tuple[float, float], y_range: tuple[float, float],
                   h: float = CURVE_STEP) -> tuple[np.ndarray, np.ndarray]:
    """The solution of y' = f(t, y) through ``point``, for drawing on a slope field.

    The curve is followed with RK4 (step ``h``) to the left and to the right of the point,
    and stops where it leaves the window ``t_range`` x ``y_range``: a solution that blows up
    (y' = y^2, Lebl Example 1.2.3) simply ends near the edge of the picture. It also stops
    where the slope becomes so steep that one step would cross a tenth of the window: at a
    vertical tangent (the circle y' = -t/y at y = 0) the solution ends, and continuing
    would draw nonsense. Returns
    ``(t, y)`` with t increasing; every returned point lies inside the window.
    """
    t0, y0 = (_require_finite("t", point[0]), _require_finite("y", point[1]))
    h = _require_positive("h", h)
    t_lo, t_hi = (_require_finite("t_range", v) for v in t_range)
    y_lo, y_hi = (_require_finite("y_range", v) for v in y_range)
    if not (t_lo < t_hi and y_lo < y_hi):
        raise ValueError("t_range and y_range must be increasing intervals")
    if not (t_lo <= t0 <= t_hi and y_lo <= y0 <= y_hi):
        raise ValueError(f"the point {point!r} is outside the window")
    _require_few_steps((t_hi - t_lo) / h)
    left_t, left_y = _follow(f, t0, y0, t_lo, (y_lo, y_hi), h)
    right_t, right_y = _follow(f, t0, y0, t_hi, (y_lo, y_hi), h)
    t = np.array(left_t[::-1] + [t0] + right_t)
    y = np.array(left_y[::-1] + [y0] + right_y)
    return t, y


# ---------------------------------------------------------------------------
# Plotting helpers (matplotlib is imported lazily: odelib works without it)
# ---------------------------------------------------------------------------
def slope_field(ax, f: Callable[[np.ndarray, np.ndarray], np.ndarray], t_range: tuple[float, float],
                y_range: tuple[float, float], n: int = 21, color: str = "0.6", length: float = 0.7):
    """Draw the slope field of y' = f(t, y): at each grid point a short segment of slope f.

    All segments have the same length measured in grid cells (``length``), whatever the
    slope, so steep regions stay readable (Lebl, Section 1.2). ``f`` must accept NumPy
    arrays (it is evaluated on the whole grid at once).
    """
    from matplotlib.collections import LineCollection

    if not (t_range[0] < t_range[1] and y_range[0] < y_range[1]):
        raise ValueError(f"the window must have t_range and y_range increasing, got {t_range!r} and {y_range!r}")
    t = np.linspace(*t_range, n)
    y = np.linspace(*y_range, n)
    T, Y = np.meshgrid(t, y)
    S = np.asarray(f(T, Y), dtype=float) * np.ones_like(T)     # slopes (broadcast if constant)
    dt = (t_range[1] - t_range[0]) / (n - 1)                   # grid spacing in t
    dy = (y_range[1] - y_range[0]) / (n - 1)                   # grid spacing in y
    # Direction (1, S) measured in grid cells: (1/dt, S/dy). Normalise it to half the
    # requested length, then convert back to data units.
    u, v = 1.0 / dt, S / dy
    scale = (length / 2) / np.hypot(u, v)
    half_t, half_y = u * scale * dt, v * scale * dy
    segments = np.stack([np.stack([T - half_t, Y - half_y], axis=-1),
                         np.stack([T + half_t, Y + half_y], axis=-1)], axis=-2).reshape(-1, 2, 2)
    ax.add_collection(LineCollection(segments, colors=color, linewidths=1.0))
    ax.set_xlim(*t_range)
    ax.set_ylim(*y_range)
    return ax


def phase_line(ax, f: Callable[[float], float], critical_points: Sequence[float],
               x_range: tuple[float, float], x_position: float = 0.0):
    """Vertical phase line of the autonomous equation x' = f(x) (Lebl, Section 1.6).

    Critical points are marked with a dot (filled = stable, open = unstable, half = semi-stable,
    decided from the sign of f just below and just above each point); arrows show the
    direction of motion between them.
    """
    lo, hi = x_range
    points = sorted(c for c in critical_points if lo < c < hi)
    ax.axvline(x_position, color="0.3", lw=1.5)
    edges = [lo, *points, hi]
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        mid = (a + b) / 2
        sign = np.sign(f(mid))
        if sign != 0:
            ax.annotate("", xy=(x_position, mid + sign * (b - a) * 0.15),
                        xytext=(x_position, mid - sign * (b - a) * 0.15),
                        arrowprops={"arrowstyle": "-|>", "color": "C0", "lw": 2})
    eps = 1e-6 * max(1.0, hi - lo)
    for c in points:
        below, above = np.sign(f(c - eps)), np.sign(f(c + eps))
        if below > 0 and above < 0:
            style = {"markerfacecolor": "k"}                     # stable: arrows point in
        elif below < 0 and above > 0:
            style = {"markerfacecolor": "white"}                 # unstable: arrows point out
        else:
            style = {"markerfacecolor": "0.6"}                   # semi-stable
        ax.plot([x_position], [c], "o", color="k", markersize=9, **style)
    ax.set_ylim(lo, hi)
    ax.set_xticks([])
    return ax


def classify_equilibrium(fprime_at_point: float) -> str:
    """Linear stability of an equilibrium x* of x' = f(x) from f'(x*):
    'stable' if f'(x*) < 0, 'unstable' if f'(x*) > 0, 'inconclusive' if f'(x*) = 0."""
    if fprime_at_point < 0:
        return "stable"
    if fprime_at_point > 0:
        return "unstable"
    return "inconclusive"


# ---------------------------------------------------------------------------
# Second-order linear equations with constant coefficients (notebooks 05-07, 14)
# ---------------------------------------------------------------------------
def characteristic_roots(a: float, b: float, c: float) -> tuple[complex, complex]:
    """Roots r1, r2 of the characteristic equation a r^2 + b r + c = 0 (Lebl, Section 2.2).

    Obtained by substituting y = e^{rt} into a y'' + b y' + c y = 0. Returned as complex
    numbers (imaginary part 0 for real roots), ordered so that Re(r1) >= Re(r2) and, for a
    complex pair, Im(r1) > 0. Requires a != 0 (otherwise the equation is not second order).
    """
    a, b, c = _require_finite("a", a), _require_finite("b", b), _require_finite("c", c)
    if a == 0:
        raise ValueError("a must be nonzero: the equation must be of second order")
    disc = complex(b * b - 4 * a * c)
    root = np.sqrt(disc)
    r1, r2 = (-b + root) / (2 * a), (-b - root) / (2 * a)
    if (r1.real, r1.imag) < (r2.real, r2.imag):
        r1, r2 = r2, r1
    return complex(r1), complex(r2)


def constant_coefficient_solution(t: ArrayLike, a: float, b: float, c: float, y0: float, v0: float,
                                  rtol: float = 1e-9) -> np.ndarray:
    """Exact solution of a y'' + b y' + c y = 0 with y(0) = y0, y'(0) = v0 (Lebl, Section 2.2).

    Three cases, according to the roots r1, r2 of a r^2 + b r + c = 0:
      * real, distinct:   y = C1 e^{r1 t} + C2 e^{r2 t},   C1 = (v0 - r2 y0)/(r1 - r2), C2 = y0 - C1;
      * real, repeated r: y = (y0 + (v0 - r y0) t) e^{r t};
      * complex a +- ib:  y = e^{a t} (y0 cos(b t) + (v0 - a y0)/b sin(b t)).
    The constants come from solving C1 + C2 = y0, r1 C1 + r2 C2 = v0: a linear system whose
    determinant is the Wronskian of the two basis solutions at t = 0 (notebook 05). The root is
    treated as repeated when |b^2 - 4ac| <= rtol b^2.
    """
    t = np.asarray(t, dtype=float)
    r1, r2 = characteristic_roots(a, b, c)
    disc = b * b - 4 * a * c
    if abs(disc) <= rtol * b * b:                         # repeated root r = -b/(2a)
        r = -b / (2 * a)
        return (y0 + (v0 - r * y0) * t) * np.exp(r * t)
    if disc > 0:                                          # two real roots
        r1, r2 = r1.real, r2.real
        c1 = (v0 - r2 * y0) / (r1 - r2)
        return c1 * np.exp(r1 * t) + (y0 - c1) * np.exp(r2 * t)
    alpha, beta = r1.real, r1.imag                        # complex pair alpha +- i beta, beta > 0
    return np.exp(alpha * t) * (y0 * np.cos(beta * t) + (v0 - alpha * y0) / beta * np.sin(beta * t))

def damping_regime(m: float, c: float, k: float, rtol: float = 1e-9) -> str:
    """Regime of m x'' + c x' + k x = 0 (m, k > 0, c >= 0), from the discriminant c^2 - 4mk.

    'undamped' (c = 0), 'underdamped' (c^2 < 4mk), 'critically damped' (c^2 = 4mk within the
    relative tolerance ``rtol``) or 'overdamped' (c^2 > 4mk)  (Lebl, Section 2.4).
    """
    m, k = _require_positive("m", m), _require_positive("k", k)
    c = float(c)
    if c < 0:
        raise ValueError("the damping coefficient c must be >= 0")
    if c == 0:
        return "undamped"
    c_crit = 2 * np.sqrt(m * k)
    if abs(c - c_crit) <= rtol * c_crit:
        return "critically damped"
    return "underdamped" if c < c_crit else "overdamped"


def free_response(t: ArrayLike, m: float, c: float, k: float, x0: float, v0: float,
                  rtol: float = 1e-9) -> np.ndarray:
    """Exact solution of m x'' + c x' + k x = 0 with x(0) = x0, x'(0) = v0.

    With p = c/(2m) and w0 = sqrt(k/m) the equation is x'' + 2p x' + w0^2 x = 0 (Lebl 2.4):
      * p > w0 (overdamped):  x = C1 e^{r1 t} + C2 e^{r2 t},  r1,2 = -p +- sqrt(p^2 - w0^2);
      * p = w0 (critical):    x = (x0 + (v0 + p x0) t) e^{-p t};
      * p < w0 (underdamped): x = e^{-p t} (x0 cos(w1 t) + (v0 + p x0)/w1 sin(w1 t)),
                               w1 = sqrt(w0^2 - p^2)  (p = 0: undamped).
    Near the critical value the two-exponential formula suffers from cancellation, so the
    critical formula is used when |c - c_crit| <= rtol c_crit (same test as damping_regime).
    """
    t = np.asarray(t, dtype=float)
    regime = damping_regime(m, c, k, rtol)
    p, w0 = c / (2 * m), np.sqrt(k / m)
    if regime == "critically damped":
        return (x0 + (v0 + p * x0) * t) * np.exp(-p * t)
    if regime == "overdamped":
        s = np.sqrt(p * p - w0 * w0)
        r1, r2 = -p + s, -p - s
        c1 = (v0 - r2 * x0) / (r1 - r2)          # from C1 + C2 = x0, r1 C1 + r2 C2 = v0
        return c1 * np.exp(r1 * t) + (x0 - c1) * np.exp(r2 * t)
    w1 = np.sqrt(w0 * w0 - p * p)
    return np.exp(-p * t) * (x0 * np.cos(w1 * t) + (v0 + p * x0) / w1 * np.sin(w1 * t))


QUAD_LIMIT = 200                # subintervals allowed to scipy.integrate.quad in forced_response


def forced_response(t: ArrayLike, a: float, b: float, c: float, f: Callable[[float], float],
                    y0: float = 0.0, v0: float = 0.0) -> np.ndarray:
    """Solution of a y'' + b y' + c y = f(t) with y(0) = y0, y'(0) = v0, at the points ``t``.

    Variation of parameters (Lebl, Section 2.5; Adkins-Davidson, Section 5.6) written once and
    for all for constant coefficients:

        y(t) = yh(t) + integral from 0 to t of g(t - s) f(s) / a ds,

    where yh is the solution of the homogeneous equation with the given initial data and g
    the one with g(0) = 0, g'(0) = 1 (both from ``constant_coefficient_solution``). The integral
    alone is the particular solution that starts from rest. It is computed numerically, so
    ``f`` may be any function of one number that is continuous between 0 and every point of
    ``t`` (to the left or to the right of 0); no closed form is needed (Lebl, Exercise 2.5.7).
    The result is an array with the shape of ``t`` (for a single number, an array with no axes).
    SciPy is imported here, not at the top: the rest of odelib needs NumPy only.
    """
    import warnings

    from scipy.integrate import IntegrationWarning, quad

    y0, v0 = _require_finite("y0", y0), _require_finite("v0", v0)
    characteristic_roots(a, b, c)                         # checks a, b, c (finite, a not zero)
    if not callable(f):
        raise ValueError("f must be a function of one number, for example lambda s: np.cos(2 * s)")
    points = np.asarray(t, dtype=float)
    if not np.all(np.isfinite(points)):
        raise ValueError("t must contain finite numbers")

    def particular(ti: float) -> float:
        if ti == 0:
            return 0.0
        return quad(lambda s: float(constant_coefficient_solution(ti - s, a, b, c, 0.0, 1.0)) * f(s) / a,
                    0.0, ti, limit=QUAD_LIMIT)[0]

    try:
        with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            warnings.simplefilter("error", IntegrationWarning)    # a doubtful integral is an error here
            forced = np.array([particular(ti) for ti in points.ravel()])
    except (IntegrationWarning, ZeroDivisionError, OverflowError) as problem:
        raise ValueError("the integral could not be computed: f must be continuous between 0 and "
                         f"every point of t, and the interval not too long ({problem})") from None
    except TypeError as problem:
        raise ValueError(f"f must take one number and return one number ({problem})") from None
    values = constant_coefficient_solution(points.ravel(), a, b, c, y0, v0) + forced
    if not np.all(np.isfinite(values)):
        raise ValueError("the solution is too large on this interval, or f is not finite there: "
                         "use points closer to 0")
    return values.reshape(points.shape)


ROOT_ROUNDING = 1e-12           # relative size of q(z) below which z is taken as a root


def root_multiplicity(a: float, b: float, c: float, alpha: float, beta: float = 0.0,
                      rtol: float = 1e-9) -> int:
    """How many times alpha + i beta is a root of a r^2 + b r + c = 0: 0, 1 or 2.

    This is the power of x by which the usual guess of the method of undetermined coefficients
    must be multiplied when the right-hand side is (a polynomial times) e^{alpha x} cos(beta x)
    or e^{alpha x} sin(beta x) (Lebl, Section 2.5: "multiply by x until there is no duplication").
    The number is a root when the polynomial q vanishes there (up to rounding), and a double
    root when the discriminant vanishes too, with the same test as
    ``constant_coefficient_solution``: |b^2 - 4ac| <= rtol b^2 (at a root |q'|^2 = |b^2 - 4ac|).
    """
    a, b, c = _require_finite("a", a), _require_finite("b", b), _require_finite("c", c)
    if a == 0:
        raise ValueError("a must be nonzero: the equation must be of second order")
    rtol = _require_positive("rtol", rtol)
    z = complex(_require_finite("alpha", alpha), _require_finite("beta", beta))
    value, slope = a * z * z + b * z + c, 2 * a * z + b
    size = abs(a) * abs(z) ** 2 + abs(b) * abs(z) + abs(c)
    if abs(value) > ROOT_ROUNDING * size:                 # only rounding is forgiven: near a double root
        return 0                                          # q is of the order of the square of the distance
    return 2 if abs(slope) ** 2 <= rtol * b * b else 1    # |q'|^2 = |b^2 - 4ac| at a root


# ---------------------------------------------------------------------------
# A spring driven by the force F0 cos(omega t) (notebook 15; Lebl, Section 2.6)
# ---------------------------------------------------------------------------
def _cosine_coefficients(m: float, c: float, k: float, F0: float, omega: float, rtol: float):
    """A and B of the particular solution A cos(omega t) + B sin(omega t), after checking the arguments."""
    m, k = _require_positive("m", m), _require_positive("k", k)
    omega, F0, c = _require_positive("omega", omega), _require_finite("F0", F0), _require_finite("c", c)
    if c < 0:
        raise ValueError("the damping coefficient c must be >= 0")
    stiffness, friction = k - m * omega * omega, c * omega       # coefficients of the 2 x 2 system for A, B
    if c == 0 and abs(stiffness) <= rtol * k:
        raise ValueError("pure resonance (no damping, and omega is the natural frequency): "
                         "there is no steady solution, the amplitude grows without bound")
    size = stiffness * stiffness + friction * friction
    if size == 0:                                         # c omega so small that its square is zero
        raise ValueError("the damping is too small to be told from zero at the natural frequency: use c = 0")
    return F0 * stiffness / size, F0 * friction / size


def steady_state(m: float, c: float, k: float, F0: float, omega: float, rtol: float = 1e-9) -> tuple[float, float]:
    """Amplitude C and phase gamma of the steady periodic solution C cos(omega t - gamma).

    For m x'' + c x' + k x = F0 cos(omega t) (Lebl, Section 2.6.2): substituting
    A cos(omega t) + B sin(omega t) gives (k - m omega^2) A + c omega B = F0 and
    -c omega A + (k - m omega^2) B = 0, so

        C = sqrt(A^2 + B^2) = |F0| / sqrt((k - m omega^2)^2 + (c omega)^2),

    which is Lebl's F0 / (m sqrt((2 omega p)^2 + (omega0^2 - omega^2)^2)), and gamma is the angle
    of the point (A, B): between 0 and pi for F0 > 0 and c > 0, equal to pi/2 at the natural
    frequency. Without damping it is 0 below the natural frequency and pi above it; at the
    natural frequency there is no steady solution (ValueError).
    """
    A, B = _cosine_coefficients(m, c, k, F0, omega, rtol)
    if A == 0 and B == 0:                                 # no force: no oscillation, and no phase to speak of
        return 0.0, 0.0
    return float(np.hypot(A, B)), float(np.arctan2(B, A))


def practical_resonance(m: float, c: float, k: float, rtol: float = 1e-9) -> float:
    """The frequency at which the amplitude of the steady solution is largest (Lebl, Section 2.6.2).

    sqrt(omega0^2 - 2 p^2), with omega0^2 = k/m and p = c/(2m), when this is a positive number;
    0.0 when c^2 >= 2 m k (within the relative tolerance ``rtol``): then the amplitude decreases
    with the frequency and the largest response is to a constant force (no practical resonance).
    """
    m, k = _require_positive("m", m), _require_positive("k", k)
    c = _require_finite("c", c)
    if c < 0:
        raise ValueError("the damping coefficient c must be >= 0")
    inside = k / m - 2 * (c / (2 * m)) ** 2
    return float(np.sqrt(inside)) if inside > rtol * k / m else 0.0


NEGLIGIBLE_DAMPING = 1e-9       # c omega / k below which cosine_forced_response uses the undamped formula


def cosine_forced_response(t: ArrayLike, m: float, c: float, k: float, F0: float, omega: float,
                           x0: float = 0.0, v0: float = 0.0) -> np.ndarray:
    """Exact solution of m x'' + c x' + k x = F0 cos(omega t) with x(0) = x0, x'(0) = v0.

    With damping: the steady solution A cos(omega t) + B sin(omega t) plus the solution of the
    homogeneous equation that carries the rest of the initial data (the transient).
    Without damping the part forced from rest is

        F0 (cos(omega t) - cos(omega0 t)) / (m (omega0^2 - omega^2))
            = (2 F0 / m) sin((omega0 - omega) t / 2) sin((omega0 + omega) t / 2) / (omega0^2 - omega^2)

    (Adkins-Davidson, Section 3.6, formulas (18) and (19); Lebl, Example 2.6.1: beats). It is
    computed from the product, which stays accurate when omega is close to omega0 and becomes
    F0 t sin(omega0 t) / (2 m omega0) when they are equal (pure resonance). A damping so small
    that c omega <= 1e-9 k is treated as no damping: there the damped formula would subtract two
    huge numbers.
    """
    t = np.asarray(t, dtype=float)
    if not np.all(np.isfinite(t)):
        raise ValueError("t must contain finite numbers")
    x0, v0 = _require_finite("x0", x0), _require_finite("v0", v0)
    c = _require_finite("c", c)
    if 0 <= c * _require_positive("omega", omega) <= NEGLIGIBLE_DAMPING * _require_positive("k", k):
        m, k = _require_positive("m", m), _require_positive("k", k)
        omega, F0 = _require_positive("omega", omega), _require_finite("F0", F0)
        omega_0 = np.sqrt(k / m)
        slow, fast = (omega_0 - omega) / 2, (omega_0 + omega) / 2
        # sin(slow t) / slow, also right at slow = 0: numpy's sinc(u) is sin(pi u) / (pi u)
        forced = F0 / m * t * np.sinc(slow * t / np.pi) * np.sin(fast * t) / (2 * fast)
        return forced + free_response(t, m, 0.0, k, x0, v0)
    A, B = _cosine_coefficients(m, c, k, F0, omega, rtol=1e-9)
    steady = A * np.cos(omega * t) + B * np.sin(omega * t)
    with np.errstate(over="ignore", invalid="ignore"):
        values = steady + free_response(t, m, c, k, x0 - A, v0 - omega * B)
    if not np.all(np.isfinite(values)):
        raise ValueError("the solution is too large to be computed at these times (far in the past, "
                         "the transient of a damped system is enormous)")
    return values


# ---------------------------------------------------------------------------
# Equations of order n and boundary value problems (notebook 16)
# ---------------------------------------------------------------------------
def higher_order_solution(t: ArrayLike, coefficients: Sequence[float], initial: Sequence[float]) -> np.ndarray:
    """Solution of a_n y^(n) + ... + a_1 y' + a_0 y = 0 with y(0), y'(0), ..., y^(n-1)(0) given.

    ``coefficients`` are a_n, ..., a_1, a_0 (the highest derivative first, as for
    ``numpy.roots``) and ``initial`` the n initial values (Lebl, Theorem 2.3.2: exactly one
    solution). The equation is rewritten as a system of n equations of first order for the
    vector (y, y', ..., y^(n-1)), whose solution is the matrix exponential applied to the
    initial vector, so repeated and complex roots need no special case. SciPy is imported
    here, not at the top: the rest of odelib needs NumPy only.
    """
    from scipy.linalg import expm

    a = np.asarray(coefficients, dtype=float)
    start = np.asarray(initial, dtype=float)
    if a.ndim != 1 or a.size < 2 or not np.all(np.isfinite(a)) or a[0] == 0:
        raise ValueError("coefficients must be a_n, ..., a_0 with n >= 1, finite, and a_n not zero")
    order = a.size - 1
    if start.shape != (order,) or not np.all(np.isfinite(start)):
        raise ValueError(f"an equation of order {order} needs {order} finite initial values")
    points = np.asarray(t, dtype=float)
    if not np.all(np.isfinite(points)):
        raise ValueError("t must contain finite numbers")
    system = np.zeros((order, order))
    system[:-1, 1:] = np.eye(order - 1)                   # (y^(j))' = y^(j+1)
    system[-1, :] = -a[:0:-1] / a[0]                      # y^(n) = -(a_0 y + ... + a_(n-1) y^(n-1)) / a_n
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.array([(expm(system * ti) @ start)[0] for ti in points.ravel()])
    if not np.all(np.isfinite(values)):
        raise ValueError("the solution is too large to be computed at these times")
    return values.reshape(points.shape)


def classify_linear_system(matrix: ArrayLike, right_side: ArrayLike, rtol: float = 1e-9):
    """How many solutions the square system M C = b has, and one of them when there is one.

    Returns ``("one solution", C)``, ``("infinitely many solutions", C)`` with one particular
    C, or ``("no solution", None)``. The n conditions of an initial or boundary value problem
    are such a system for the n constants of the general solution; with conditions at two
    points M may be singular, and then the problem has no solution or infinitely many (Lebl,
    Examples 4.1.1 and 4.1.2, Theorem 4.1.2). Each equation is first divided by its largest
    coefficient, so that multiplying an equation by a number does not change the answer. Then
    M is taken as singular when its smallest singular value is below ``rtol`` times the
    largest (numbers such as sin(pi) are not exactly zero in a computer), and a singular system
    as solvable when the best candidate satisfies every equation within ``rtol`` times the size
    of its terms.
    """
    M = np.asarray(matrix, dtype=float)
    b = np.asarray(right_side, dtype=float)
    rtol = _require_positive("rtol", rtol)
    if M.ndim != 2 or M.shape[0] == 0 or M.shape[0] != M.shape[1] or b.shape != (M.shape[0],):
        raise ValueError("a square matrix and a right-hand side of the same size are expected")
    if not (np.all(np.isfinite(M)) and np.all(np.isfinite(b))):
        raise ValueError("the matrix and the right-hand side must contain finite numbers")
    largest = np.max(np.abs(M), axis=1)
    scale = np.where(largest > 0, largest, 1.0)           # an equation 0 = b_i is left as it is
    M, b = M / scale[:, None], b / scale
    sizes = np.linalg.svd(M, compute_uv=False)
    cutoff = rtol * sizes[0] if sizes[0] > 0 else 0.0
    if sizes[0] > 0 and sizes[-1] > cutoff:
        return "one solution", np.linalg.solve(M, b)
    candidate = np.linalg.lstsq(M, b, rcond=rtol)[0]      # the best that can be done
    misfit = np.abs(M @ candidate - b)                    # equation by equation: "0 = 1" is never forgiven
    if np.all(misfit <= rtol * (np.abs(b) + np.abs(M) @ np.abs(candidate))):
        return "infinitely many solutions", candidate
    return "no solution", None


SUPPORTS = {"supported": (0, 2), "fixed": (0, 1), "free": (2, 3)}    # the derivatives that are zero at that end
BEAM_MARGIN = 1e-9              # x may leave the beam by this fraction of its length (rounding)


def beam_deflection(x: ArrayLike, load: Sequence[float], left: str, right: str,
                    length: float = 1.0, rigidity: float = 1.0) -> np.ndarray:
    """Deflection y(x) of a beam: EI y^(4) = w(x) for 0 <= x <= length, two conditions at each end.

    ``load`` holds the coefficients of the polynomial w(x), the highest power first as for
    ``numpy.polyval`` (``[w0]`` is a uniform load, ``[w0 / L, 0]`` the load w0 x / L of
    Kiusalaas, Example 8.4). ``left`` and ``right`` are "supported" (y = y'' = 0), "fixed"
    (y = y' = 0) or "free" (y'' = y^(3) = 0). The deflection is positive in the direction of the
    load. Method: with xi = x / length, y = (length^4 / EI) Y(xi) where Y^(4) = w(length xi);
    four integrations give a polynomial plus a cubic, and the four conditions are a 4 x 4
    system for the cubic. If that system is singular the beam is not held (for example two free
    ends) and a ValueError says so.
    """
    length, rigidity = _require_positive("length", length), _require_positive("rigidity", rigidity)
    w = np.atleast_1d(np.asarray(load, dtype=float))
    if w.ndim != 1 or w.size == 0 or not np.all(np.isfinite(w)):
        raise ValueError("load must be the list of the coefficients of a polynomial")
    for name in (left, right):
        if not isinstance(name, str) or name not in SUPPORTS:
            raise ValueError(f"a support must be one of {sorted(SUPPORTS)}, got {name!r}")
    points = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(points)):
        raise ValueError("x must contain finite numbers")
    margin = BEAM_MARGIN * length
    if np.any(points < -margin) or np.any(points > length + margin):
        raise ValueError(f"x must be between 0 and the length of the beam, {length:g}")
    scaled = w * length ** np.arange(w.size - 1, -1, -1)  # the load as a polynomial in xi
    particular = np.polyint(scaled, 4)                    # integrated four times, constants zero
    rows, right_side = [], []
    for position, support in ((0.0, left), (1.0, right)):
        for order in SUPPORTS[support]:
            rows.append([np.polyval(np.polyder([0.0] * (3 - power) + [1.0] + [0.0] * power, order), position)
                         if power >= order else 0.0 for power in (3, 2, 1, 0)])
            right_side.append(-np.polyval(np.polyder(particular, order), position))
    kind, cubic = classify_linear_system(rows, right_side)
    if kind != "one solution":
        raise ValueError(f"a beam that is {left} on the left and {right} on the right is not held in place: "
                         f"the four conditions have {kind}")
    xi = points / length
    return length ** 4 / rigidity * (np.polyval(particular, xi) + np.polyval(cubic, xi))


def wronskian_values(y1: ArrayLike, dy1: ArrayLike, y2: ArrayLike, dy2: ArrayLike) -> np.ndarray:
    """Wronskian W = y1 y2' - y1' y2 from sampled values of two functions and their derivatives.

    Geometrically, W(t) is the signed area of the parallelogram spanned by the two state
    vectors (y1, y1') and (y2, y2') in the phase plane (notebook 05).
    """
    y1, dy1, y2, dy2 = (np.asarray(v, dtype=float) for v in (y1, dy1, y2, dy2))
    return y1 * dy2 - dy1 * y2


# ---------------------------------------------------------------------------
# Linear systems x' = A x (notebook 17; Lebl, Sections 3.1-3.4, 3.7, 3.8)
# ---------------------------------------------------------------------------
# A 2 x 2 matrix has a repeated eigenvalue when its discriminant trace^2 - 4 det is zero; in a computer
# it is taken as zero when |discriminant| <= REPEATED_EIGENVALUE_RTOL * s^2, s the largest size of an
# entry. Near a repeated eigenvalue the error made by using the repeated formula grows like the
# discriminant times t^2 (times t^3 for the part along (A - lambda I) x0), so with this value it stays
# below 1e-9 when s t is at most about 20 (the windows of the notebook), while the formulas of the
# other two cases, used just outside the band, lose at most about 1e-10 to cancellation.
REPEATED_EIGENVALUE_RTOL = 1e-12
CASE_REAL = "real distinct"
CASE_COMPLEX = "complex pair"
CASE_REPEATED_TWO = "repeated, two eigenvectors"
CASE_REPEATED_ONE = "repeated, one eigenvector"
SYSTEM_SIZE = 2                 # the size of the systems solved by the formulas of eigen_solution
LARGER_SYSTEMS = "for a larger system use scipy.linalg.expm(t * A) @ x0"
MatrixLike = Sequence[Sequence[float]] | np.ndarray


class TwoByTwoCase(NamedTuple):
    """What a student computes by hand for a real 2 x 2 matrix A (see ``two_by_two_case``)."""

    case: str                       # CASE_REAL, CASE_COMPLEX, CASE_REPEATED_TWO or CASE_REPEATED_ONE
    trace: float                    # a11 + a22
    determinant: float              # a11 a22 - a12 a21
    discriminant: float             # trace^2 - 4 determinant
    eigenvalues: tuple[float, float] | tuple[complex, complex]   # complex: alpha + i beta first, beta > 0
    eigenvectors: tuple[np.ndarray, ...]                          # one array for each independent eigenvector
    generalized: np.ndarray | None  # w with (A - lambda I) w = v, only in the case CASE_REPEATED_ONE


def companion_matrix(coefficients: Sequence[float]) -> np.ndarray:
    """The matrix A of the system u' = A u that is the equation a_n y^(n) + ... + a_1 y' + a_0 y = 0.

    Lebl, Section 3.1 (the change to first order, (3.1.8) and Example 3.1.4), used in notebook 17:
    with u1 = y, u2 = y', ..., un = y^(n-1), the linear homogeneous equation of order n with constant
    coefficients is the same problem as the system of n equations of order 1
        u1' = u2,  ...,  u_(n-1)' = u_n,  u_n' = -(a_0 u1 + a_1 u2 + ... + a_(n-1) u_n) / a_n.
    ``coefficients`` are a_n, ..., a_1, a_0, the highest derivative first, as for ``numpy.roots`` and
    ``higher_order_solution``. Rows 1 to n - 1 hold a 1 just above the diagonal; the last row is
    -(a_0, ..., a_(n-1)) / a_n. Its eigenvalues are the roots of the characteristic equation
    a_n r^n + ... + a_0 = 0 (Lebl's note after Example 3.4.4). The function is unit-free: an entry
    of row n has the units of y^(n) divided by those of y^(k-1).
    """
    try:
        a = np.asarray(coefficients, dtype=float)
    except (TypeError, ValueError):
        a = None
    if a is None or a.ndim != 1 or a.size < 2 or not np.all(np.isfinite(a)) or a[0] == 0:
        raise ValueError("coefficients must be one flat list a_n, ..., a_1, a_0 of at least two finite "
                         "numbers, with a_n (the coefficient of the highest derivative) not zero")
    order = a.size - 1
    matrix = np.zeros((order, order))
    matrix[:-1, 1:] = np.eye(order - 1)                   # u_k' = u_(k+1)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        matrix[-1, :] = -a[:0:-1] / a[0]                  # u_n' = -(a_0 u1 + ... + a_(n-1) u_n) / a_n
    if not np.all(np.isfinite(matrix)):
        raise ValueError("a_n is too small compared with the other coefficients: "
                         "divide the equation by a larger number")
    return matrix


def _real_two_by_two(matrix: MatrixLike, advice: str) -> np.ndarray:
    """The matrix as a 2 x 2 array of finite real numbers, or a ValueError that says what is wrong."""
    try:
        raw = np.asarray(matrix)
    except (TypeError, ValueError):                       # rows of different lengths
        raw = None
    if raw is not None and np.iscomplexobj(raw):
        raise ValueError("the entries of the matrix must be real numbers (no complex entries)")
    try:
        A = None if raw is None else raw.astype(float)
    except (TypeError, ValueError):                       # text, or a symbol
        A = None
    if A is None:
        raise ValueError("a 2 x 2 matrix of numbers is expected, typed as two rows: "
                         f"[[a11, a12], [a21, a22]]; {advice}")
    if A.shape != (SYSTEM_SIZE, SYSTEM_SIZE):
        raise ValueError("a 2 x 2 matrix is expected, typed as two rows: [[a11, a12], [a21, a22]] "
                         f"(got shape {A.shape}); {advice}")
    if not np.all(np.isfinite(A)):
        raise ValueError("the entries of the matrix must be finite numbers")
    return A


def _shortcut_eigenvector(A: np.ndarray, eigenvalue: complex | float) -> np.ndarray:
    """An eigenvector of A for ``eigenvalue`` without row reduction (notebook 17, Section 2).

    From the first row (a11 - lambda) v1 + a12 v2 = 0 the vector (a12, lambda - a11) works, and from
    the second row (lambda - a22, a21); the longer of the two is returned, so a triangular or diagonal
    matrix (where one of them is the zero vector) is handled too.
    """
    (a11, a12), (a21, a22) = A
    first = np.array([a12, eigenvalue - a11])
    second = np.array([eigenvalue - a22, a21])
    return second if np.linalg.norm(second) > np.linalg.norm(first) else first


def two_by_two_case(matrix: MatrixLike, rtol: float = REPEATED_EIGENVALUE_RTOL) -> TwoByTwoCase:
    """The case of a real 2 x 2 matrix A = [[a11, a12], [a21, a22]] and what is computed for it by hand.

    Lebl, Section 3.4 (eigenvalue method, the 2 x 2 shortcut after Example 3.4.3, complex eigenvalues)
    and Section 3.7 (Example 3.7.1), used in notebook 17. The eigenvalues are the roots of
    lambda^2 - (trace) lambda + det = 0, and the case is decided by the discriminant
    trace^2 - 4 det (computed as (a11 - a22)^2 + 4 a12 a21, the same number with less rounding):
      * CASE_REAL, "real distinct": eigenvalues lambda1 < lambda2, one eigenvector each, from the
        shortcut (a12, lambda - a11) of the first row or (lambda - a22, a21) of the second, the longer;
      * CASE_COMPLEX, "complex pair": eigenvalues (alpha + i beta, alpha - i beta) with beta > 0, as
        complex numbers, and the eigenvectors (v, conj v);
      * CASE_REPEATED_TWO, "repeated, two eigenvectors": A = lambda I (a12 = a21 = 0), eigenvectors
        (1, 0) and (0, 1);
      * CASE_REPEATED_ONE, "repeated, one eigenvector": v is the longer column j of A - lambda I, and
        ``generalized`` is w = e_j (1 in place j), so that (A - lambda I) w = v holds exactly; for
        [[3, 1], [0, 3]] this gives Lebl's v = (1, 0) and w = (0, 1).
    The eigenvalue is taken as repeated when |discriminant| <= rtol * s^2, s the largest size of an
    entry (REPEATED_EIGENVALUE_RTOL explains the value); the test is made on A / s, so that very small
    or very large entries are judged alike (the eigenvectors are those of the shortcut on A itself).
    Inside that band a diagonal matrix (a12 = a21 = 0) is "two eigenvectors", even if a11 and a22
    differ by rounding. The entries may have any units of 1/time: the eigenvalues have the same units.
    """
    rtol = _require_positive("rtol", rtol)
    A = _real_two_by_two(matrix, "for a larger matrix use np.linalg.eig or sp.Matrix(...).eigenvects()")
    (a11, a12), (a21, a22) = A
    with np.errstate(over="ignore", invalid="ignore"):
        trace, determinant = float(a11 + a22), float(a11 * a22 - a12 * a21)
        discriminant = float((a11 - a22) ** 2 + 4 * a12 * a21)
    size = float(np.max(np.abs(A)))
    scaled = A / size if size > 0 else A                  # the case is decided on entries of size 1
    (p11, p12), (p21, p22) = scaled
    trace_s, determinant_s = p11 + p22, p11 * p22 - p12 * p21
    discriminant_s = (p11 - p22) ** 2 + 4 * p12 * p21
    if abs(discriminant_s) <= rtol:
        eigenvalue = float(trace_s / 2 * size) + 0.0
        if a12 == 0 and a21 == 0:
            return TwoByTwoCase(CASE_REPEATED_TWO, trace, determinant, discriminant, (eigenvalue, eigenvalue),
                                (np.array([1.0, 0.0]), np.array([0.0, 1.0])), None)
        shifted = A - eigenvalue * np.eye(SYSTEM_SIZE)
        column = int(np.argmax(np.linalg.norm(shifted, axis=0)))
        return TwoByTwoCase(CASE_REPEATED_ONE, trace, determinant, discriminant, (eigenvalue, eigenvalue),
                            (shifted[:, column].copy(),), np.eye(SYSTEM_SIZE)[column])
    root = np.sqrt(abs(discriminant_s))
    if discriminant_s > 0:
        larger = (trace_s + root) / 2 if trace_s >= 0 else (trace_s - root) / 2   # no cancellation
        smaller = determinant_s / larger                                         # the product is det
        low, high = sorted((larger, smaller))
        vectors = (_shortcut_eigenvector(scaled, low) * size, _shortcut_eigenvector(scaled, high) * size)
        low, high = (float(value * size) + 0.0 for value in (low, high))      # + 0.0: no -0.0
        return TwoByTwoCase(CASE_REAL, trace, determinant, discriminant, (low, high), vectors, None)
    eigenvalue = complex(trace_s / 2, root / 2)
    vector = _shortcut_eigenvector(scaled, eigenvalue) * size     # the shortcut of A itself
    eigenvalue = eigenvalue * size
    return TwoByTwoCase(CASE_COMPLEX, trace, determinant, discriminant, (eigenvalue, eigenvalue.conjugate()),
                        (vector, vector.conj()), None)


def _scaled_by(exponential: np.ndarray, amount: np.ndarray) -> np.ndarray:
    """exponential * amount, with 0 where amount is 0 (so an overflow times 0 is 0, not NaN)."""
    with np.errstate(over="ignore", invalid="ignore"):
        return np.where(amount == 0, 0.0, exponential * amount)


def eigen_solution(t: ArrayLike, matrix: MatrixLike, x0: Sequence[float],
                   rtol: float = REPEATED_EIGENVALUE_RTOL) -> np.ndarray:
    """The solution of x' = A x, x(0) = x0, for a real 2 x 2 matrix A, by the formula of its case.

    Lebl, Theorems 3.4.1 and 3.4.2 and formula (3.7.1), used in notebook 17 (the case and the
    vectors come from ``two_by_two_case``):
      * real distinct:   x = C1 v1 e^(lambda1 t) + C2 v2 e^(lambda2 t);
      * complex pair:    x = C1 Re(v e^(lambda t)) + C2 Im(v e^(lambda t)), lambda = alpha + i beta,
                         beta > 0;
      * A = lambda I:    x = e^(lambda t) x0;
      * one eigenvector: x = C1 v e^(lambda t) + C2 (w + v t) e^(lambda t).
    The constants come from X(0) C = x0, X(0) the matrix whose columns are the two solutions at t = 0
    (``np.linalg.solve``). In the last case C1 v + C2 w = x0 and C2 v = (A - lambda I) x0, so the
    formula is evaluated as e^(lambda t) (x0 + t (A - lambda I) x0): the same function, which stays
    exact when rounding leaves (A - lambda I)^2 slightly different from zero; the case A = lambda I is
    evaluated the same way (there A - lambda I = 0, or is zero up to rounding). A term whose constant
    is 0 is 0, also where its exponential overflows.
    Returns a real array of shape ``t.shape + (2,)``: the last index j holds x_(j+1), the layout of
    ``rk4``. Units: t in the time unit of the rates in A (A has units 1/time), x in the units of x0.
    """
    A = _real_two_by_two(matrix, LARGER_SYSTEMS)
    found = two_by_two_case(A, rtol)
    try:
        start = np.asarray(x0, dtype=float)
    except (TypeError, ValueError):
        start = None
    if start is None or start.shape != (SYSTEM_SIZE,) or not np.all(np.isfinite(start)):
        raise ValueError(f"x0 must be a list of {SYSTEM_SIZE} finite numbers, "
                         "the values of x1 and x2 at t = 0")
    times = np.asarray(t, dtype=float)
    if not np.all(np.isfinite(times)):
        raise ValueError("t must contain finite numbers")
    column = times[..., np.newaxis]                       # one row of the result for each time
    if not np.any(start):
        return np.zeros(times.shape + (SYSTEM_SIZE,))     # x0 = 0 stays 0
    with np.errstate(over="ignore", invalid="ignore"):
        if found.case == CASE_REAL:
            first, second = found.eigenvectors
            C = np.linalg.solve(np.column_stack([first, second]), start)
            values = (_scaled_by(np.exp(found.eigenvalues[0] * column), C[0] * first)
                      + _scaled_by(np.exp(found.eigenvalues[1] * column), C[1] * second))
        elif found.case == CASE_COMPLEX:
            real_part, imaginary_part = found.eigenvectors[0].real, found.eigenvectors[0].imag
            alpha, beta = found.eigenvalues[0].real, found.eigenvalues[0].imag
            C = np.linalg.solve(np.column_stack([real_part, imaginary_part]), start)
            cosine, sine = np.cos(beta * column), np.sin(beta * column)
            values = _scaled_by(np.exp(alpha * column), C[0] * (real_part * cosine - imaginary_part * sine)
                                + C[1] * (real_part * sine + imaginary_part * cosine))
        else:                                             # repeated: one eigenvector, or A = lambda I
            eigenvalue = found.eigenvalues[0]
            along = (A - eigenvalue * np.eye(SYSTEM_SIZE)) @ start       # C2 v
            values = _scaled_by(np.exp(eigenvalue * column), start + column * along)
    if not np.all(np.isfinite(values)):
        raise ValueError("the solution is too large to be computed at these times")
    return values


# ---------------------------------------------------------------------------
# The phase plane and nonlinear systems (notebook 18; Lebl, Sections 3.5, 8.1-8.3)
# ---------------------------------------------------------------------------
# A system in the plane is passed everywhere in one form, Lebl's: f(x, y) returns the pair
# (f(x, y), g(x, y)) of the right-hand sides of x' = f(x, y), y' = g(x, y), and works on NumPy arrays.
Field = Callable[[np.ndarray, np.ndarray], tuple[ArrayLike, ArrayLike]]

# The pictures of x' = A x at the origin, with Lebl's words (Section 3.5, Table 3.5.1) and the
# borderline cases of his Exercises 3.5.2, 3.5.3 and 3.5.8.
KIND_SADDLE = "saddle"
KIND_SOURCE = "source"
KIND_SINK = "sink"
KIND_SPIRAL_SOURCE = "spiral source"
KIND_SPIRAL_SINK = "spiral sink"
KIND_CENTER = "center"
KIND_DEGENERATE_SOURCE = "degenerate source"      # eigenvalue > 0 twice, one eigenvector
KIND_DEGENERATE_SINK = "degenerate sink"          # eigenvalue < 0 twice, one eigenvector
KIND_STAR_SOURCE = "star source"                  # A = lambda I, lambda > 0
KIND_STAR_SINK = "star sink"                      # A = lambda I, lambda < 0
KIND_LINE_OF_EQUILIBRIA = "line of equilibria"    # one eigenvalue 0
KIND_PARALLEL_LINES = "parallel lines"            # the eigenvalue 0 twice, A not 0
KIND_ALL_EQUILIBRIA = "all equilibria"            # A = 0
# Stability of an equilibrium (Lebl, Section 8.2.2) and the word of classify_equilibrium (notebook 03)
STABILITY_ASYMPTOTIC = "asymptotically stable"
STABILITY_STABLE = "stable"
STABILITY_UNSTABLE = "unstable"
STABILITY_INCONCLUSIVE = "inconclusive"
_SINKS = (KIND_SINK, KIND_SPIRAL_SINK, KIND_DEGENERATE_SINK, KIND_STAR_SINK)


class PhaseType(NamedTuple):
    """The picture of x' = A x at the origin (see ``classify_linear_2d``)."""

    kind: str                       # one of the KIND_... words
    stability: str                  # STABILITY_ASYMPTOTIC, STABILITY_STABLE or STABILITY_UNSTABLE
    trace: float                    # a11 + a22 = lambda1 + lambda2
    determinant: float              # a11 a22 - a12 a21 = lambda1 lambda2
    eigenvalues: tuple[float, float] | tuple[complex, complex]   # as in two_by_two_case
    case: str                       # the case of two_by_two_case (CASE_REAL, CASE_COMPLEX, ...)


def classify_linear_2d(matrix: MatrixLike, rtol: float = REPEATED_EIGENVALUE_RTOL) -> PhaseType:
    """The picture of x' = A x near the origin, for a real 2 x 2 matrix A, from tr A and det A.

    Lebl, Section 3.5 (Table 3.5.1 and the six cases) and the borderline cases of Exercises 3.5.2,
    3.5.3 and 3.5.8, used in notebook 18. Not to be confused with ``classify_linear_system``
    (notebook 16, algebraic systems M C = b). A thin wrapper of ``two_by_two_case``: it is called once
    (the validation, the repeated band, the eigenvalues and the case come from it) and only the map
    to a picture is added. With lambda1 + lambda2 = tr A and lambda1 lambda2 = det A:
      * det A < 0: "saddle" (real eigenvalues of opposite signs);
      * det A > 0, real eigenvalues: "source" if tr A > 0, "sink" if tr A < 0; the same eigenvalue
        twice: "degenerate source" / "degenerate sink" (one eigenvector) or "star source" /
        "star sink" (A = lambda I);
      * det A > 0, complex eigenvalues alpha +- i beta, alpha = tr A / 2: "spiral source" (tr A > 0),
        "spiral sink" (tr A < 0), "center" (tr A = 0);
      * det A = 0, an eigenvalue 0: "line of equilibria" (tr A not 0), "parallel lines" (tr A = 0,
        A not 0, Lebl's Exercise 3.5.8), "all equilibria" (A = 0).
    The stability (Lebl, Section 8.2.2): STABILITY_ASYMPTOTIC for the four sinks; STABILITY_STABLE for a
    center, a line of equilibria that attracts (tr A < 0) and A = 0; STABILITY_UNSTABLE for the rest.
    As for the discriminant in ``two_by_two_case``, det A and tr A are taken as 0 when
    |det A| <= rtol s^2 and |tr A| <= rtol s, s the largest size of an entry (decided on A / s, so the
    unit does not matter). Units: the entries of A in 1/time, as for ``eigen_solution``; the kind does
    not depend on the unit. ValueError: those of ``two_by_two_case`` (not 2 x 2, complex or non-finite
    entries, rtol not positive).
    """
    found = two_by_two_case(matrix, rtol)
    A = np.asarray(matrix, dtype=float)                   # accepted by two_by_two_case just above
    size = float(np.max(np.abs(A)))
    if size == 0:
        return PhaseType(KIND_ALL_EQUILIBRIA, STABILITY_STABLE, found.trace, found.determinant,
                         found.eigenvalues, found.case)
    (p11, p12), (p21, p22) = A / size                     # entries of size at most 1
    trace_s, determinant_s = p11 + p22, p11 * p22 - p12 * p21
    no_trace = abs(trace_s) <= rtol
    if abs(determinant_s) <= rtol:                        # an eigenvalue 0; the other is tr A
        kind = KIND_PARALLEL_LINES if no_trace else KIND_LINE_OF_EQUILIBRIA
    elif determinant_s < 0:
        kind = KIND_SADDLE
    elif found.case == CASE_COMPLEX:
        kind = KIND_CENTER if no_trace else KIND_SPIRAL_SOURCE if trace_s > 0 else KIND_SPIRAL_SINK
    elif found.case == CASE_REAL:                         # det A > 0: here |tr A| > 2 sqrt(det A) > 0
        kind = KIND_SOURCE if trace_s > 0 else KIND_SINK
    elif found.case == CASE_REPEATED_ONE:
        kind = KIND_DEGENERATE_SOURCE if trace_s > 0 else KIND_DEGENERATE_SINK
    else:                                                 # CASE_REPEATED_TWO: A = lambda I
        kind = KIND_STAR_SOURCE if trace_s > 0 else KIND_STAR_SINK
    if kind in _SINKS:
        stability = STABILITY_ASYMPTOTIC
    elif kind == KIND_CENTER or (kind == KIND_LINE_OF_EQUILIBRIA and trace_s < 0):   # towards the line
        stability = STABILITY_STABLE
    else:
        stability = STABILITY_UNSTABLE
    return PhaseType(kind, stability, found.trace, found.determinant, found.eigenvalues, found.case)


# A central difference (F(p + h) - F(p - h)) / (2 h) has an error of about h^2 times the third
# derivative, plus the rounding of F, about 1e-16 / h: with h = 1e-5 both are near 1e-10 for functions
# that vary on the scale max(1, |point|), such as the polynomials, exponentials and the pendulum of the
# course near the origin; far from the origin an oscillating f loses accuracy like (h |p|)^2 (the step
# is h max(1, |p_j|)): about 1e-7 relative for sin(3x) at |p| = 20.
JACOBIAN_STEP = 1e-5


def _flat_point(point: Sequence[float], size: int | None = None) -> np.ndarray:
    """The point as a flat float array of finite numbers (of ``size`` coordinates), or a ValueError."""
    try:
        p = np.asarray(point, dtype=float)
    except (TypeError, ValueError):                       # text, a symbol, rows of different lengths
        p = None
    wanted = "coordinates" if size is None else f"{size} coordinates"
    if p is None or p.ndim != 1 or p.size == 0 or (size is not None and p.size != size):
        raise ValueError(f"the point must be one flat list of {wanted}, for example (1.0, 0.0)")
    if not np.all(np.isfinite(p)):
        raise ValueError("the coordinates of the point must be finite numbers")
    return p


def _values_at(f: Callable[..., Sequence[float]], p: np.ndarray, near: str) -> np.ndarray:
    """f(*p) as a float array of p.size finite numbers, or a ValueError that says which value is wrong."""
    with np.errstate(all="ignore"):                       # a value that is not finite is reported below
        result = f(*p)
    try:
        values = np.asarray(result, dtype=float)
    except (TypeError, ValueError):
        values = None
    if values is None or values.shape != p.shape:
        got = ("something else" if values is None else f"{values.size} value(s)" if values.size != p.size
               else f"an array of shape {values.shape}")
        raise ValueError(f"f must return {p.size} numbers, one for each coordinate of the point; got {got}")
    bad = np.flatnonzero(~np.isfinite(values))
    if bad.size:
        raise ValueError(f"value number {bad[0] + 1} of f is not a finite number {near}")
    return values


def jacobian(f: Callable[..., Sequence[float]], point: Sequence[float],
             h: float = JACOBIAN_STEP) -> np.ndarray:
    """The Jacobian matrix of f at ``point`` by central differences (Lebl, Section 8.1.2; notebook 18).

    Entry (i, j) is the partial derivative of f_i with respect to the coordinate j: the derivative in
    that coordinate with the others held fixed. Column j is (F(p + h_j e_j) - F(p - h_j e_j)) / (2 h_j),
    with h_j = h max(1, |p_j|) (JACOBIAN_STEP explains the value) and e_j the unit vector of coordinate j.
    f is called as f(*point): f(x, y) returning (f(x, y), g(x, y)) in the plane, f(x, y, z) in three
    dimensions (Lebl's Exercise 8.1.9). Returns an n x n float array. Units: entry (i, j) has the units of
    f_i divided by those of the coordinate j (1/time for a system x' = f(x, y) whose x and y have the
    same units). ValueError: a point that is not a flat list of finite numbers; f returning a number of
    values different from the number of coordinates; a value that is not finite at or near the point
    (the message names the coordinate); h not positive. An error raised inside f is not caught: it is
    the message of the function that was passed.
    """
    h = _require_positive("h", h)
    p = _flat_point(point)
    _values_at(f, p, "at the point")
    columns = []
    for j in range(p.size):
        step = h * max(1.0, abs(p[j]))
        plus, minus = p.copy(), p.copy()
        plus[j] += step
        minus[j] -= step
        near = f"near the point (coordinate {j + 1} moved by {step:.1g})"
        columns.append((_values_at(f, plus, near) - _values_at(f, minus, near)) / (2 * step))
    return np.column_stack(columns)


# The entries of a central-difference Jacobian carry an error of about 1e-9 relative, so its trace,
# determinant and discriminant are trusted to about 1e-9: a band a hundred times wider never takes
# rounding for a sign, and a true nonzero trace of a model of the course is far larger.
LINEARIZATION_RTOL = 1e-7
EQUILIBRIUM_TOL = 1e-9          # |f| and |g| allowed at an equilibrium, relative to |J| max(1, |point|)
REASON_LINEAR = ("no eigenvalue has real part 0: near the point the system has the stability of its "
                 "linearization (Lebl, Table 8.2.1)")
REASON_CENTER = "the linearization is a center: the nonlinear terms decide (Lebl, Section 8.2.3)"
REASON_SINGULAR = "det J = 0: not almost linear, the linearization decides nothing (Lebl, Section 8.2.1)"
REASON_SINGULAR_UNSTABLE = ("det J = 0, but the other eigenvalue tr J is positive: unstable; the picture is "
                            "not decided (not almost linear, Lebl, Section 8.2.1)")


class EquilibriumType(NamedTuple):
    """An equilibrium of x' = f(x, y), y' = g(x, y) and its linearization (``classify_equilibrium_2d``)."""

    point: tuple[float, float]      # (x*, y*)
    jacobian: np.ndarray            # J at the point, by central differences
    linear: PhaseType               # the picture of the linearization u' = J u
    stability: str                  # linear.stability, STABILITY_UNSTABLE or STABILITY_INCONCLUSIVE
    reason: str                     # one of the REASON_... sentences


def classify_equilibrium_2d(f: Field, point: Sequence[float], rtol: float = LINEARIZATION_RTOL,
                            h: float = JACOBIAN_STEP) -> EquilibriumType:
    """An equilibrium of x' = f(x, y), y' = g(x, y), classified by its linearization (notebook 18).

    Lebl, Sections 8.1-8.2 (Table 8.2.1, "Trouble with Centers"). First the point is checked to be an
    equilibrium: |f| and |g| there at most EQUILIBRIUM_TOL max(1, |J| max(1, |point|)), |J| the largest
    entry of J (so a point such as (550/3, 40) typed as floats is accepted). Then J = ``jacobian(f, point,
    h)`` and ``linear = classify_linear_2d(J, rtol)`` (LINEARIZATION_RTOL explains the value).
    ``stability`` is ``linear.stability`` when no eigenvalue of J has real part 0 (saddles, nodes,
    spirals, degenerate and star nodes): near the point the system has the stability, and the kind of
    picture, of its linearization, when f and g have continuous partial derivatives (Lebl states this
    without proof). It is STABILITY_INCONCLUSIVE ("inconclusive", the word of ``classify_equilibrium``,
    notebook 03) for a center of J (the nonlinear terms decide: Lebl's Example 8.2.3 is a spiral source
    with the same J as the center of Example 8.1.1) and when det J = 0 (a line of equilibria, parallel
    lines or J = 0: the point is not "almost linear"), except that det J = 0 with tr J > 0 (an
    eigenvalue with positive real part) is STABILITY_UNSTABLE. ``reason`` says which, in one sentence.
    A degenerate or star node of J (one eigenvalue twice) keeps its stability, but the nonlinear curves
    near it may also turn slowly around the point (Lebl's Exercise 8.2.8 (c) calls such a point (0, 1)
    a "source").
    Units as for ``jacobian``: x and y in the units of the point, J in 1/time. ValueError: the point is
    not an equilibrium (the message gives f and g there), and those of ``jacobian`` and
    ``classify_linear_2d``.
    """
    p = _flat_point(point, SYSTEM_SIZE)
    J = jacobian(f, p, h)
    values = _values_at(f, p, "at the point")
    allowed = EQUILIBRIUM_TOL * max(1.0, float(np.max(np.abs(J))) * max(1.0, float(np.max(np.abs(p)))))
    if not np.all(np.abs(values) <= allowed):
        raise ValueError(f"(x, y) = ({p[0]:g}, {p[1]:g}) is not an equilibrium: f = {values[0]:.4g}, "
                         f"g = {values[1]:.4g} (both must be 0)")
    linear = classify_linear_2d(J, rtol)
    if linear.kind == KIND_LINE_OF_EQUILIBRIA and linear.stability == STABILITY_UNSTABLE:   # tr J > 0
        stability, reason = STABILITY_UNSTABLE, REASON_SINGULAR_UNSTABLE
    elif linear.kind in (KIND_LINE_OF_EQUILIBRIA, KIND_PARALLEL_LINES, KIND_ALL_EQUILIBRIA):
        stability, reason = STABILITY_INCONCLUSIVE, REASON_SINGULAR
    elif linear.kind == KIND_CENTER:
        stability, reason = STABILITY_INCONCLUSIVE, REASON_CENTER
    else:
        stability, reason = linear.stability, REASON_LINEAR
    return EquilibriumType((float(p[0]), float(p[1])), J, linear, stability, reason)


def _field_values(f: Field, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """f(x, y) as an array of shape x.shape + (2,): (f, g) at every point, constants broadcast."""
    result = f(x, y)
    try:
        first, second = result                            # exactly two parts: f(x, y) and g(x, y)
        parts = [np.broadcast_to(np.asarray(part, dtype=float), np.shape(x)) for part in (first, second)]
    except (TypeError, ValueError):
        raise ValueError("f(x, y) must return two values, (f(x, y), g(x, y)), each a number or an array "
                         "of the shape of x") from None
    return np.stack(parts, axis=-1)


PORTRAIT_STEPS = 400            # RK4 steps of a trajectory (plan 18: about 1e-5 hares over 40 time units)
ESCAPE_MARGIN = 1.0             # a path is followed until it leaves the window enlarged by this many sizes


def _start_points(starts: ArrayLike) -> np.ndarray:
    """The starts as a float array of shape (k, 2) of finite numbers (one start (x, y): k = 1)."""
    try:
        points = np.asarray(starts, dtype=float)
    except (TypeError, ValueError):                       # text, or points of different lengths
        points = None
    if points is not None and points.shape == (SYSTEM_SIZE,):
        points = points[np.newaxis, :]
    if points is None or points.ndim != 2 or points.shape[1] != SYSTEM_SIZE or points.shape[0] == 0:
        raise ValueError("starts must be a list of points (x, y), for example [(1.0, 0.0), (0.5, 0.5)]")
    if not np.all(np.isfinite(points)):
        raise ValueError("the starts must be finite numbers")
    return points


def trajectories(f: Field, starts: ArrayLike, t_span: tuple[float, float], n_steps: int = PORTRAIT_STEPS,
                 window: tuple[tuple[float, float], tuple[float, float]] | None = None,
                 ) -> tuple[np.ndarray, np.ndarray]:
    """The solutions of x' = f(x, y), y' = g(x, y) from several starts at once (notebook 18).

    The classical Runge-Kutta method of notebook 04 (Lebl, Section 1.7) with the fixed step
    (t1 - t0) / n_steps, applied to all the starts together: the step of ``rk4`` works on an array of
    shape (k, 2), so a portrait of 24 paths costs the same number of Python steps as one path. t_span =
    (t0, t1) may decrease (backward in time). Returns t (n_steps + 1 times, from t0 to t1) and the
    paths, an array of shape (n_steps + 1, k, 2): [i, j] is the point (x, y) of start j at t[i]. f
    returns the pair (f(x, y), g(x, y)), not an array of points (with exactly two starts an array of
    points, shape (2, 2), cannot be told apart from the pair: write the pair). With
    ``window = ((x_lo, x_hi), (y_lo, y_hi))`` a path that leaves the window enlarged by ESCAPE_MARGIN
    times its size on each side is NaN from there on, and so is a path whose value is no longer finite
    (no RuntimeWarning: computed inside ``np.errstate``); the loop stops when every path has left.
    Units: t in the time unit of the rates in f; x and y in the units of the starts. ValueError: starts
    not of shape (k, 2) (one start (x, y) is accepted as k = 1) or not finite; t_span with equal or
    non-finite ends; n_steps not a positive integer, or above MAX_STEPS; a window that is not
    increasing; f returning values of the wrong shape.
    """
    points = _start_points(starts)
    t0, t1 = _require_finite("t0", t_span[0]), _require_finite("t1", t_span[1])
    if t0 == t1:
        raise ValueError("t_span must have two different ends (t0, t1)")
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError(f"n_steps must be a positive whole number, got {n_steps!r}")
    if n_steps > MAX_STEPS:
        raise ValueError(f"too many steps ({n_steps}, the limit is {MAX_STEPS}): use fewer steps")
    low = np.array([-np.inf, -np.inf])
    high = np.array([np.inf, np.inf])
    if window is not None:
        try:
            x_range, y_range = window
        except (TypeError, ValueError):
            raise ValueError(f"the window must be ((x_lo, x_hi), (y_lo, y_hi)), got {window!r}") from None
        (x_lo, x_hi), (y_lo, y_hi) = _ranges(x_range, y_range)
        sizes = np.array([x_hi - x_lo, y_hi - y_lo])
        low = np.array([x_lo, y_lo]) - ESCAPE_MARGIN * sizes
        high = np.array([x_hi, y_hi]) + ESCAPE_MARGIN * sizes

    def field(s, state):                                  # the right-hand side for rk4: (k, 2) -> (k, 2)
        return _field_values(f, state[:, 0], state[:, 1])

    times = np.linspace(t0, t1, n_steps + 1)
    h = (t1 - t0) / n_steps
    paths = np.full((n_steps + 1,) + points.shape, np.nan)
    paths[0] = points
    alive = np.arange(points.shape[0])                    # the starts whose paths are still followed
    state = points.copy()
    with np.errstate(all="ignore"):                       # overflow and NaN are handled just below
        for i in range(n_steps):
            state = _rk4_step(field, times[i], state, h)
            inside = np.all(np.isfinite(state) & (state >= low) & (state <= high), axis=1)
            alive, state = alive[inside], state[inside]
            if alive.size == 0:
                break
            paths[i + 1, alive] = state
    return times, paths


DIRECTION_GRID = 17             # arrows per side of a direction field
ARROW_LENGTH = 0.7              # length of every arrow, in shorter sides of a grid cell (as in slope_field)
MIN_GRID = 2                    # a grid needs at least two points per side
ARROW_WIDTH = 0.075             # shaft width of a direction-field arrow, as a part of its length


def _ranges(x_range: tuple[float, float],
            y_range: tuple[float, float]) -> tuple[tuple[float, float], tuple[float, float]]:
    """The two ranges of a window as floats, or a ValueError unless both are finite and increasing."""
    message = (f"the window must have x_range and y_range increasing, got {x_range!r} and {y_range!r} "
               "(each range is two finite numbers, the smaller first)")
    try:
        (x_lo, x_hi), (y_lo, y_hi) = [(float(lo), float(hi)) for lo, hi in (x_range, y_range)]
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if not (np.isfinite([x_lo, x_hi, y_lo, y_hi]).all() and x_lo < x_hi and y_lo < y_hi):
        raise ValueError(message)
    return (x_lo, x_hi), (y_lo, y_hi)


def _window(x_range: tuple[float, float], y_range: tuple[float, float]) -> None:
    """A ValueError unless both ranges of the window are finite and increasing."""
    _ranges(x_range, y_range)


def direction_field(ax: Axes, f: Field, x_range: tuple[float, float], y_range: tuple[float, float],
                    n: int = DIRECTION_GRID, color: str = "0.6", length: float = ARROW_LENGTH) -> Axes:
    """Draw the direction field of x' = f(x, y), y' = g(x, y): at each point of a grid an arrow along (f, g).

    The 2-D analogue of ``slope_field``, with its look (Lebl, Section 3.1.3, the vector field; notebook
    18). Every arrow has the same length and the same shape on the screen, ``length`` times the shorter
    side of a grid cell, with a shaft ARROW_WIDTH times as wide, whatever the speed |(f, g)|: the picture
    shows directions, not speeds. The screen is measured on the axes box as drawn when the function is
    called, after the limits and the aspect (``ax.set_aspect("equal")``) are applied; a later change of the
    box, such as ``tight_layout``, changes the lengths only slightly. Points of the n x n grid where
    (f, g) is zero (an equilibrium on the grid) or not finite are left out before drawing, so no warning
    appears. One call of ``ax.quiver``; the limits of the window are set. Units: x and y in the units of
    the window; the arrows have none. Returns ``ax``. ValueError for a window that is not finite and
    increasing, and for n not a whole number of at least 2.
    """
    (x_lo, x_hi), (y_lo, y_hi) = _ranges(x_range, y_range)
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < MIN_GRID:
        raise ValueError(f"n, the number of arrows per side, must be a whole number of at least {MIN_GRID}, "
                         f"got {n!r}")
    x = np.linspace(x_lo, x_hi, n)
    y = np.linspace(y_lo, y_hi, n)
    X, Y = np.meshgrid(x, y)
    dx, dy = (x_hi - x_lo) / (n - 1), (y_hi - y_lo) / (n - 1)       # grid spacing in x and in y
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.apply_aspect()                                              # the box as drawn (equal aspect included)
    box = ax.get_position()                                        # the axes, in parts of the figure
    width, height = ax.figure.get_size_inches()
    sx, sy = box.width * width / (x_hi - x_lo), box.height * height / (y_hi - y_lo)   # inches per unit
    arrow = length * min(dx * sx, dy * sy)                         # the length of every arrow, in inches
    with np.errstate(all="ignore"):
        values = _field_values(f, X, Y)
        u, v = values[..., 0] * sx, values[..., 1] * sy            # the direction on the screen
        size = np.hypot(u, v)
    keep = np.isfinite(size) & (size > 0)
    if np.any(keep):
        U = u[keep] / size[keep] * arrow / sx                      # back to data units
        V = v[keep] / size[keep] * arrow / sy
        ax.quiver(X[keep], Y[keep], U, V, angles="xy", scale_units="xy", scale=1, pivot="mid", color=color,
                  units="inches", width=ARROW_WIDTH * arrow)       # the shaft in proportion to the arrow
    return ax


TRAJECTORY_ARROW_AT = 1 / 3     # where the arrowhead of a trajectory sits, as a part of its drawn length
TRAJECTORY_WIDTH = 1.5          # line width of a trajectory (points)
START_DOT_SIZE = 16             # area of the dot at each start (points^2)
ARROW_HEAD_SIZE = 14            # size of the arrowhead (points)
ARROW_HEAD_SPAN = 0.02          # the arrowhead points along this part of the window, from its place


def _arrow_on(ax: Axes, path: np.ndarray, x_range: tuple[float, float], y_range: tuple[float, float],
              color: str) -> None:
    """One arrowhead on a path (rows (x, y), NaN after it leaves), TRAJECTORY_ARROW_AT along its drawn part.
    """
    with np.errstate(invalid="ignore"):                   # NaN rows compare as False: not visible
        visible = (np.all(np.isfinite(path), axis=1) & (path[:, 0] >= x_range[0]) & (path[:, 0] <= x_range[1])
                   & (path[:, 1] >= y_range[0]) & (path[:, 1] <= y_range[1]))
    if not visible.any():
        return
    first = int(np.argmax(visible))                       # only the first visible run: no arrow across a gap
    end = first + (int(np.argmin(visible[first:])) if not visible[first:].all() else len(visible) - first)
    shown = path[first:end]
    if len(shown) < 2:
        return
    sizes = np.array([x_range[1] - x_range[0], y_range[1] - y_range[0]])
    steps = np.hypot(*(np.diff(shown, axis=0) / sizes).T)          # lengths measured in window sizes
    along = np.concatenate([[0.0], np.cumsum(steps)])
    if along[-1] <= ARROW_HEAD_SPAN:
        return                                                      # a path that hardly moves: no arrow
    tail = int(np.searchsorted(along, TRAJECTORY_ARROW_AT * along[-1]))
    head = int(np.searchsorted(along, along[min(tail, len(along) - 1)] + ARROW_HEAD_SPAN))
    tail, head = min(tail, len(shown) - 2), min(max(head, tail + 1), len(shown) - 1)
    ax.annotate("", xy=tuple(shown[head]), xytext=tuple(shown[tail]),
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": TRAJECTORY_WIDTH,
                            "mutation_scale": ARROW_HEAD_SIZE, "shrinkA": 0, "shrinkB": 0})


def phase_portrait(ax: Axes, f: Field, starts: ArrayLike, t_span: tuple[float, float],
                   x_range: tuple[float, float], y_range: tuple[float, float], *,
                   n_steps: int = PORTRAIT_STEPS, both_ways: bool = True, color: str = "C0") -> Axes:
    """Draw the trajectories of x' = f(x, y), y' = g(x, y) through ``starts`` (Lebl, Sections 3.5 and 8.1).

    Used in notebook 18. The paths come from ``trajectories`` (all the starts in one computation):
    forward over t_span = (t0, t1), t1 > t0, and, with ``both_ways``, backward over the same length of
    time, from t0 to t0 - (t1 - t0), so each start lies on the whole curve drawn through it. A dot marks
    each start and one arrowhead on each forward part, at TRAJECTORY_ARROW_AT of its drawn length,
    shows the direction of increasing t. The paths are followed until they leave the window enlarged by
    ESCAPE_MARGIN; the limits of the window are set. Units: t in the time unit of the rates in f; x and
    y in the units of the window. Returns ``ax``. ValueError: t1 not after t0, a start outside the
    window, a window that is not increasing, and those of ``trajectories``.
    """
    x_range, y_range = _ranges(x_range, y_range)
    t0, t1 = _require_finite("t0", t_span[0]), _require_finite("t1", t_span[1])
    if not t1 > t0:
        raise ValueError(f"t_span = (t0, t1) must have t1 > t0 (the paths are drawn forward and backward), "
                         f"got {t_span!r}")
    points = _start_points(starts)
    outside = ~((points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])
                & (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]))
    if np.any(outside):
        x0, y0 = points[np.argmax(outside)]               # the first start outside
        raise ValueError(f"the start (x, y) = ({x0:g}, {y0:g}) is outside the window")
    window = (tuple(x_range), tuple(y_range))
    forward = trajectories(f, points, (t0, t1), n_steps, window)[1]
    parts = [forward]
    if both_ways:
        parts.append(trajectories(f, points, (t0, 2 * t0 - t1), n_steps, window)[1])
    for path in range(forward.shape[1]):
        for part in parts:
            ax.plot(part[:, path, 0], part[:, path, 1], "-", color=color, linewidth=TRAJECTORY_WIDTH)
        _arrow_on(ax, forward[:, path], x_range, y_range, color)
    ax.scatter(forward[0, :, 0], forward[0, :, 1], s=START_DOT_SIZE, color=color, zorder=3)
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    return ax


# ---------------------------------------------------------------------------
# The Laplace transform (notebook 19; Lebl, Sections 6.1 and 6.2)
# ---------------------------------------------------------------------------
# Conventions: t >= 0 is the time and s, the variable of the transform, is a real number. With t in
# seconds s is in 1/s, so that the exponent s t of the weight e^(-st) is a pure number.
LAPLACE_RTOL = 1e-8             # accuracy asked of the integral: absolute below 1, see laplace_numeric
LAPLACE_RTOL_RANGE = (1e-12, 1e-2)      # an rtol outside is refused (rounding decides below, nothing above)
# e^(-700) is about 1e-304, near the smallest positive number of the computer, and e^(700) is near the
# largest. Beyond |s| t = LAPLACE_HORIZON the weight e^(-st) cannot be trusted: laplace_numeric takes
# the integrand as 0 there, after checking that it is already negligible at that time.
LAPLACE_HORIZON = 700.0
LAPLACE_UNIT_RATE = 1.0         # for |s| above it the variable is u = |s| t; below it u = t
# The integral is the sum of the integrals over pieces. In u their edges double from the first number to
# the second (about 1e-6 and 1e6), so that a feature of f is seen on every time scale in between; each
# piece is one call of scipy's quad, with the QUAD_LIMIT of forced_response.
LAPLACE_EDGES = (2.0 ** -20, 2.0 ** 20)
LAPLACE_EDGE_RATIO = 2.0
LAPLACE_SHIFTS = (1.0, 2.0 ** 0.5)      # the edges of the two computations are multiplied by these numbers
LAPLACE_GROWTH_TIMES = (0.8, 0.9, 1.0)  # the parts of the last time T at which the growth of |f| is measured
# Before integrating, f is looked at on LAPLACE_SCAN equal steps of every piece. A step on which f changes
# more than LAPLACE_JUMP_RATIO times what it changes on a neighboring step is taken as a jump (a switch,
# the end of a pulse); the jump is located by halving the step, and the integral is cut there, as if
# ``breaks`` had named it. A cut that was not needed costs time, and now and then a refusal where a
# number was possible (the accuracy asked of each piece grows with their number). A piece with more than
# LAPLACE_MAX_JUMPS such steps is a fast oscillation, not a list of switches, and is left whole.
LAPLACE_SCAN = 256
LAPLACE_JUMP_RATIO = 8.0
LAPLACE_MAX_JUMPS = 8
LAPLACE_NARROW = 1e-12          # two cuts nearer than this part of their place are one (as in convolution)
LAPLACE_DEMO_A = 2.0            # the constants of the table printed in notebook 19: e^(-2t),
LAPLACE_DEMO_OMEGA = 3.0        # sin(3t), cos(3t), sinh(3t), cosh(3t)
LAPLACE_DEMO_N = 3              # and t^3 (a row of its own in Lebl's Table 6.1.1)
LAPLACE_ROW_NAMES = ("1", "t", "t^2", "t^n", "e^(-at)", "sin", "cos", "sinh", "cosh")   # LaplaceRow.name


def _one_real_number(name: str, value: float) -> float:
    """The value as a finite float, or a ValueError that names the argument (None, text, a list, ...)."""
    try:
        refused = isinstance(value, (bool, np.bool_, str, bytes)) or np.iscomplexobj(value)
        number = float("nan") if refused else float(value)
    except (TypeError, ValueError, OverflowError):        # OverflowError: an integer too large for a float
        number = float("nan")
    if not np.isfinite(number):
        raise ValueError(f"{name} must be one finite real number, got {value!r}")
    return number


def laplace_numeric(f: Callable[[float], float], s: float, *, rtol: float = LAPLACE_RTOL,
                    breaks: Sequence[float] = ()) -> float:
    """The Laplace transform F(s) of f at one real number s, straight from its definition.

    Lebl, Section 6.1.1, used in notebook 19: F(s) is the integral from 0 to infinity of e^(-st) f(t) dt,
    computed here by ``scipy.integrate.quad`` and by nothing else, so it is an independent check of a
    table row, of a symbolic transform, and of an inverse transform (transform the answer forward and
    compare). f is a function of one number t >= 0 (only t >= 0 is used) that returns one real number;
    s is one real number. Units: with t in seconds, s is in 1/s and F(s) is in the units of f times
    seconds. ``breaks`` are the times at which f jumps or has a corner (a switch, the ends of a pulse):
    give them, and the integral is cut there.

    How it is computed. With u = max(|s|, 1) t the interval is cut into pieces whose edges double, from
    about 1e-6 to about 1e6 in u (LAPLACE_EDGES), at the breaks, and at the jumps of f that a first look
    at f finds by itself (LAPLACE_SCAN equal steps on every piece; a jump is located by halving its
    step, and the integral is cut there and at the two ends of that step, so that a switch that closes
    steeply without a jump is computed too; two cuts at one place are one cut, whatever value f has at
    the jump itself); beyond the time
    T = LAPLACE_HORIZON / |s| the weight e^(-st) is not a number of the computer any more and the
    integrand is taken as 0. The sum over the pieces is computed twice, on two different sets of
    pieces. The result is accepted when (a) scipy's quad is in doubt on no piece, (b) the errors it
    estimates and the difference between the two sums are at most rtol max(1, |F(s)|), the rule of
    ``check`` (for |s| > 1: rtol max(1 / |s|, |F(s)|), since transforms become small like 1 / s),
    (c) T |e^(-sT) f(T)| is negligible by the same rule, and (d) |f| is not seen to grow faster and
    faster than e^(st) just before T. Otherwise the function says that the integral does not
    converge, or too slowly: that is the safe answer.

    What is promised, and what is not (measured with the default rtol; "about", and not for every s):
      * The accuracy is absolute, not relative: when |F(s)| is much smaller than rtol max(1, 1 / |s|)
        (t^10 at s = 30, for example) the digits returned mean nothing.
      * Near the edge of the range of the transform the answer is a refusal, not a number: for
        f(t) = e^(ct), s must be about 4% of |s| above the edge c; 1 and t^n (edge 0) are computed
        for s above about 1e-5; sin(omega t) and cos(omega t) for s above about omega / 200 (slowly:
        up to 0.1 second of computing). At s = 0 only functions that die out like an exponential are
        computed
        (1 / (1 + t)^2 is refused).
      * A numerical integral cannot prove convergence. A function that grows faster than every
        exponential is refused only when its growth shows before the time T: e^(t^2) is refused for
        s up to about 36, and for larger s a number is returned (about 1 / s, the integral up to T)
        although the transform does not exist; t^t gives a number from s of about 6. In the same way
        a function that is infinite at a time t0 > 0 (tan t at pi / 2, 1 / (t - 1.3) at 1.3) is
        refused only while the weight e^(-s t0) does not hide the infinity (tan t: s up to about 17;
        1 / (t - 1.3): s up to about 28) or when the first look at f falls on t0 itself; for larger s
        a number may be returned. Use this function to check a transform that is known to exist, not
        to decide whether one exists.
      * Switches and pulses are found without help when they are wider than about 1/200 of their
        distance from t = 0 (a pulse 0.03 wide at t = 5, on top of any smooth function): the integral
        is cut at their ends. A narrower one (a pulse 0.001 wide at t = 5) can be missed, and then
        a wrong number is returned with no error, the transform of f without the pulse: name its
        ends in ``breaks`` and it is computed. e^(-at) is right up to a of about 1e8.
      * A switch that closes steeply without a jump (a ramp from 0 to 1) is computed in about 99
        cases in 100 when it lasts 1/10000 of its starting time or more; the others are refused or,
        a few in 1000, wrong by a little with no error. A steeper one (1/100000 of its starting time)
        is wrong, by a little and with no error, about one time in four. Name the two ends of a ramp
        in ``breaks`` and it is computed.
      * A fast oscillation (sin or cos of omega t with s below about omega / 20) and a function that
        repeats (a square wave) are sometimes refused although the integral converges: the refusal
        is the safe answer, and a larger s or ``breaks`` at the switches gives the number.

    ValueError, with one sentence: s that is not a finite real number; f that is not a function, that
    does not take one number, that returns something that is not one real number (text, a complex
    number), or that stops with an error of its own (the sentence quotes it); rtol outside
    LAPLACE_RTOL_RANGE; breaks that are not finite times >= 0; an integral that does not converge (s
    at or below the edge of the range of the transform, as e^(2t) at s = 1; a function without a
    transform, as 1/t for every s, or e^(t^2) and tan t for moderate s, see above; f not finite at
    some t that the integral reaches). SciPy is imported here, not at the top: the rest of odelib
    needs NumPy only.
    """
    import math

    from scipy.integrate import quad

    rtol = _one_real_number("rtol", rtol)
    if not LAPLACE_RTOL_RANGE[0] <= rtol <= LAPLACE_RTOL_RANGE[1]:
        raise ValueError(f"rtol must be between {LAPLACE_RTOL_RANGE[0]:g} and {LAPLACE_RTOL_RANGE[1]:g}, "
                         f"got {rtol!r}")
    if not callable(f):
        raise ValueError("f must be a function of one number, for example lambda t: np.exp(-2 * t)")
    rate = _one_real_number("s", s) + 0.0                 # + 0.0: -0.0 is written 0
    try:
        jumps = [math.nan] if isinstance(breaks, (str, bytes)) else sorted(
            {_one_real_number("a break", point) for point in breaks})
    except (TypeError, ValueError):                       # not a list, or a point that is not a number
        jumps = [math.nan]
    if not all(math.isfinite(point) and point >= 0 for point in jumps):
        raise ValueError(f"breaks must be a list of finite times, each 0 or more, got {breaks!r}")
    scale = max(abs(rate), LAPLACE_UNIT_RATE)             # the integral is computed in u = scale t
    horizon = LAPLACE_HORIZON / abs(rate) if rate != 0 else math.inf     # the time T; none for s = 0
    diverges = (f"the integral of e^(-st) f(t) from 0 to infinity does not converge for s = {rate:g}, or "
                "too slowly to be computed: s is not inside the range of the transform (or is too near "
                "to its edge), or f(t) has no transform")

    def value_at(t: float) -> float:
        """f(t) as a finite float, or a ValueError with the sentence for what went wrong."""
        try:
            raw = f(t)
        except ArithmeticError as problem:                # a division by 0, an overflow
            raise ValueError(f"{diverges} (f(t) cannot be computed at t = {t:.3g}: {problem})") from None
        except TypeError as problem:
            hint = " (a power is written t ** 2, not t ^ 2)" if "^" in str(problem) else ""
            raise ValueError(f"f must take one number and return one number ({problem}){hint}") from None
        except ValueError as problem:                     # math.log of a negative number, for example
            raise ValueError(f"{diverges} (f(t) is not defined at t = {t:.3g}: {problem})") from None
        except Exception as problem:                      # whatever else f does: said in one sentence
            raise ValueError(f"f stopped with an error at t = {t:.3g} "
                             f"({type(problem).__name__}: {problem})") from None
        try:
            refused = isinstance(raw, (str, bytes)) or np.iscomplexobj(raw)
            value = math.nan if refused else float(raw)
        except (TypeError, ValueError):
            refused = True
        if refused:
            raise ValueError(f"f must return one real number, got {raw!r} at t = {t:.3g}")
        if math.isnan(value):
            raise ValueError(f"{diverges} (f(t) is not a number at t = {t:.3g})")
        if math.isinf(value):
            raise ValueError(f"{diverges} (f(t) is too large for the computer at t = {t:.3g})")
        return value

    def integrand(u: float) -> float:
        t = u / scale
        return math.exp(-rate * t) * value_at(t)

    end = scale * horizon                                 # in u; infinite for s = 0
    last = min(end, LAPLACE_EDGES[1])

    def seen_jumps() -> set[float]:
        """The places (in u) where a scan of f sees a jump: cuts that the caller did not name."""
        found: set[float] = set()
        left = LAPLACE_EDGES[0]
        while left < last:
            right = min(left * LAPLACE_EDGE_RATIO, last)
            points = [left + (right - left) * index / LAPLACE_SCAN for index in range(LAPLACE_SCAN + 1)]
            values = [value_at(point / scale) for point in points]
            changes = [abs(after - before) for before, after in zip(values[:-1], values[1:], strict=True)]
            floor = rtol * max(abs(value) for value in values)
            beside = [min(changes[max(index - 1, 0):index] + changes[index + 1:index + 2])
                      for index in range(len(changes))]   # the smaller change of the neighboring steps
            marked = [index for index, change in enumerate(changes)
                      if change > LAPLACE_JUMP_RATIO * beside[index] and change > floor]
            if len(marked) <= LAPLACE_MAX_JUMPS:
                for index in marked:                      # halve the step: where is f half way between
                    low, high = points[index], points[index + 1]            # its values on the two sides?
                    below, above = values[index], values[index + 1]
                    middle = (low + high) / 2
                    while low < middle < high:
                        between = value_at(middle / scale)
                        if abs(between - below) <= abs(between - above):
                            low = middle
                        else:
                            high = middle
                        middle = (low + high) / 2
                    # low and high are now neighboring numbers. At a jump f differs across them, and
                    # one cut is enough. Otherwise f closes steeply without a jump: the step is also cut
                    # at its two ends, so that the steep part lies in pieces of its own.
                    if 2 * abs(value_at(high / scale) - value_at(low / scale)) >= abs(above - below):
                        found.add(high)
                    else:
                        found.update((points[index], high, points[index + 1]))
            left = right
        return found

    def by_pieces(shift: float, cuts: set[float]) -> tuple[float, float]:
        """The sum of the integrals over the pieces and of the errors that quad estimates."""
        edges = {0.0, last} | cuts
        edge = LAPLACE_EDGES[0] * shift
        while edge < last:
            edges.add(edge)
            edge *= LAPLACE_EDGE_RATIO
        named = {scale * point for point in jumps if 0.0 < scale * point < last}
        ordered = [0.0]
        for edge in sorted((edges | named) - {0.0, last}):
            if last - edge <= LAPLACE_NARROW * last:
                continue                                  # at the last edge, which is a cut already
            if edge - ordered[-1] > LAPLACE_NARROW * edge:
                ordered.append(edge)
            elif edge in named and 0.0 < ordered[-1] and ordered[-1] not in named:
                ordered[-1] = edge                        # of two cuts at one place, the named break stays
        ordered.append(last)
        pieces = list(zip(ordered[:-1], ordered[1:], strict=True))
        if last < end:
            pieces.append((last, end))                    # s = 0, or a very small s: the rest at once
        share = rtol / len(pieces)
        total = error = 0.0
        for left, right in pieces:
            part, estimate, _, *doubt = quad(integrand, left, right, epsabs=share, epsrel=share,
                                             limit=QUAD_LIMIT, full_output=True)
            if doubt:                                     # quad adds a message when it is in doubt
                raise ValueError(diverges)
            total += part
            error += estimate
        return total, error

    with np.errstate(all="ignore"):
        cuts = seen_jumps()
        (value, estimate), (other, other_estimate) = (by_pieces(shift, cuts) for shift in LAPLACE_SHIFTS)
        tail, growing = 0.0, False
        if rate != 0:
            tail = scale * horizon * abs(integrand(end))
            sizes = [abs(value_at(part * horizon)) for part in LAPLACE_GROWTH_TIMES]
            if all(size > 0 for size in sizes):           # the slopes of log |f| between these times
                step = (LAPLACE_GROWTH_TIMES[-1] - LAPLACE_GROWTH_TIMES[-2]) * horizon
                before = (math.log(sizes[1]) - math.log(sizes[0])) / (
                    (LAPLACE_GROWTH_TIMES[1] - LAPLACE_GROWTH_TIMES[0]) * horizon)
                after = (math.log(sizes[2]) - math.log(sizes[1])) / step
                growing = after > rate and after - before > rtol * abs(after)
    error = max(estimate, other_estimate, abs(value - other))
    allowed = max(rtol, rtol * abs(value))                # value = scale F(s): rtol max(1, scale |F(s)|)
    if growing or not (math.isfinite(value) and error <= allowed and tail <= allowed):
        raise ValueError(diverges)
    return float(value / scale)


class LaplaceRow(NamedTuple):
    """One row of the table of Laplace transforms (see ``laplace_table``)."""

    name: str                               # a short name for the row: one of LAPLACE_ROW_NAMES
    time: Callable[[ArrayLike], np.ndarray | float]         # f(t) for t >= 0: a number, a list or an array
    transform: Callable[[ArrayLike], np.ndarray | float]    # F(s); a ValueError for s at or below s_min
    s_min: float                            # the transform holds for s > s_min
    time_text: str                          # f(t) with Lebl's letters, in plain ASCII: "e^(-a t)"
    transform_text: str                     # F(s) the same way: "1/(s + a)"


def _laplace_row(name: str, time: Callable[[np.ndarray], np.ndarray],
                 transform: Callable[[np.ndarray], np.ndarray], s_min: float, time_text: str,
                 transform_text: str) -> LaplaceRow:
    """A row whose two functions take numbers, lists or arrays, in float arithmetic, and refuse s <= s_min."""
    def result(values: np.ndarray, points: np.ndarray) -> np.ndarray | float:
        return float(values) if points.ndim == 0 else values

    def of_time(t: ArrayLike) -> np.ndarray | float:
        points = np.asarray(t, dtype=float)
        return result(time(points), points)

    def of_s(s: ArrayLike) -> np.ndarray | float:
        try:
            refused = isinstance(s, (bool, np.bool_, str, bytes)) or np.iscomplexobj(s)
            points = None if refused else np.asarray(s, dtype=float)
        except (TypeError, ValueError):
            refused = True
        if refused:
            raise ValueError(f"s must be a real number, a list or an array of real numbers, got {s!r}")
        if not np.all(points > s_min):
            raise ValueError(f"the transform {transform_text} of {time_text} holds only for s > {s_min:g} "
                             f"(the row {name!r} of the table), got s = {s!r}")
        with np.errstate(over="ignore", divide="ignore"):
            values = transform(points)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"the transform {transform_text} is too large for the computer at s = {s!r}")
        return result(values, points)

    return LaplaceRow(name, of_time, of_s, s_min, time_text, transform_text)


def laplace_table(*, a: float = LAPLACE_DEMO_A, omega: float = LAPLACE_DEMO_OMEGA,
                  n: int = LAPLACE_DEMO_N) -> tuple[LaplaceRow, ...]:
    """Lebl's Table 6.1.1 of Laplace transforms as data, for given constants a, omega and n.

    Lebl, Section 6.1.1 (Table 6.1.1; Examples 6.1.1 to 6.1.3 compute the rows 1, e^(-at) and t), used
    in notebook 19 and in its tests. Nine rows, in the order of the book, with the names of
    LAPLACE_ROW_NAMES ("1", "t", "t^2", "t^n", "e^(-at)", "sin", "cos", "sinh", "cosh"):

        f(t)             F(s)                       holds for
        1                1/s                        s > 0
        t                1/s^2                      s > 0
        t^2              2/s^3                      s > 0
        t^n              n!/s^(n + 1)               s > 0
        e^(-a t)         1/(s + a)                  s > -a
        sin(omega t)     omega/(s^2 + omega^2)      s > 0
        cos(omega t)     s/(s^2 + omega^2)          s > 0
        sinh(omega t)    omega/(s^2 - omega^2)      s > omega
        cosh(omega t)    s/(s^2 - omega^2)          s > omega

    The book's first row is a constant C with C/s: by linearity it is C times the row 1. Its row t^3
    is t^n with n = 3 (the default), and its row for the Heaviside function u(t - a) is left to
    notebook 20. The table is printed as Lebl prints it, e^(-a t) with 1/(s + a); many books print
    e^(a t) with 1/(s - a), which is this row with -a in place of a. Each row holds f(t) and F(s) as
    functions of a number, a list or a NumPy array (computed in float arithmetic: a float for a
    number, an array otherwise), the number s_min, and the two formulas as text with the letters a,
    omega, n. F(s) raises a ValueError for an s at or below s_min, where the formula is not the
    transform. To take a row by its name: ``rows = {row.name: row for row in ol.laplace_table()}``,
    then ``rows["sin"]``. The constants are given by name, as ``ol.laplace_table(a=-3)``.
    Conventions: t >= 0 and s real; with t in seconds, a, omega and s are in 1/s. ValueError: a that
    is not one finite real number (None, text, a list, a complex number); omega the same, or not
    positive (sinh and cosh need s > omega > 0); n not a whole number, 0 or more (3.0 is refused), or
    so large that n! is not a number of the computer.
    """
    import math

    a = _one_real_number("a", a)
    omega = _one_real_number("omega", omega)
    if omega <= 0:
        raise ValueError(f"omega must be a positive number, got {omega!r}")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError(f"n must be a whole number such as 3 (not 3.0), 0 or more, got {n!r}")
    n = int(n)                                            # a plain Python integer: no overflow in n + 1
    try:
        factorial = float(math.factorial(n))
    except OverflowError:
        raise ValueError(f"n = {n} is too large: n! is not a number of the computer") from None
    return (
        _laplace_row("1", np.ones_like, lambda s: 1.0 / s, 0.0, "1", "1/s"),
        _laplace_row("t", lambda t: t, lambda s: 1.0 / s ** 2, 0.0, "t", "1/s^2"),
        _laplace_row("t^2", lambda t: t ** 2, lambda s: 2.0 / s ** 3, 0.0, "t^2", "2/s^3"),
        _laplace_row("t^n", lambda t: t ** n, lambda s: factorial / s ** (n + 1), 0.0,
                     "t^n", "n!/s^(n + 1)"),
        _laplace_row("e^(-at)", lambda t: np.exp(-a * t), lambda s: 1.0 / (s + a), 0.0 - a,
                     "e^(-a t)", "1/(s + a)"),
        _laplace_row("sin", lambda t: np.sin(omega * t), lambda s: omega / (s ** 2 + omega ** 2), 0.0,
                     "sin(omega t)", "omega/(s^2 + omega^2)"),
        _laplace_row("cos", lambda t: np.cos(omega * t), lambda s: s / (s ** 2 + omega ** 2), 0.0,
                     "cos(omega t)", "s/(s^2 + omega^2)"),
        _laplace_row("sinh", lambda t: np.sinh(omega * t), lambda s: omega / (s ** 2 - omega ** 2), omega,
                     "sinh(omega t)", "omega/(s^2 - omega^2)"),
        _laplace_row("cosh", lambda t: np.cosh(omega * t), lambda s: s / (s ** 2 - omega ** 2), omega,
                     "cosh(omega t)", "s/(s^2 - omega^2)"),
    )


# ---------------------------------------------------------------------------
# Steps, impulses and convolution (notebook 20; Lebl, Sections 6.2 to 6.4)
# ---------------------------------------------------------------------------
# Conventions: t is the time, and t = 0 is when the problem starts. At the jump itself the unit step is 1,
# u(0) = 1, as Lebl defines it (Example 6.1.4); no transform and no solution depends on that one value.
STEP_START = 0.0                # the default a of step: u(t), the switch that closes at t = 0
CONVOLUTION_RTOL = 1e-8         # accuracy asked of the integral: rtol max(1, |value|), the rule of check
# The integral from 0 to t is computed twice, as a sum over equal pieces of [0, t]: so many in the first
# computation and so many in the second. The two sums must agree, or convolution refuses to answer.
CONVOLUTION_PIECES = (4, 5)
# Before integrating, f and g are looked at on CONVOLUTION_SCAN equal steps of [0, t]. A step on which a
# function changes more than CONVOLUTION_JUMP_RATIO times the typical change (the median) of the
# CONVOLUTION_WINDOW steps on each side of it is taken as a jump; the place where the function is half way
# between its two sides is located by halving the step, and the integral is cut there and at the two ends
# of the step, as if ``breaks`` had named them. More than CONVOLUTION_MAX_JUMPS such steps are a fast
# oscillation, not a list of switches: then nothing is cut. The two numbers are those of laplace_numeric,
# on purpose, under names of their own: a later change of that function does not reach this one unseen.
CONVOLUTION_SCAN = 256
CONVOLUTION_WINDOW = 4
CONVOLUTION_JUMP_RATIO = LAPLACE_JUMP_RATIO
CONVOLUTION_MAX_JUMPS = LAPLACE_MAX_JUMPS
# Two cuts nearer to each other than this part of t are one cut: a piece that narrow adds nothing that
# can be seen at CONVOLUTION_RTOL, and quad cannot integrate over two neighboring numbers.
CONVOLUTION_NARROW = 1e-12


def _real_array(name: str, value: ArrayLike) -> np.ndarray:
    """The value as an array of floats without NaN, or a ValueError that names the argument."""
    try:
        raw = np.asarray(value)
        items = np.asarray(value, dtype=object).ravel()   # the items as they were typed
        refused = raw.dtype.kind in "bcUSMm" or any(      # booleans, complex numbers, text, dates
            isinstance(item, (bool, np.bool_, str, bytes)) for item in items)
        points = None if refused else raw.astype(float)
    except (TypeError, ValueError, OverflowError):        # None, a ragged list, 10 ** 400, a SymPy letter
        refused = True
    if refused or bool(np.isnan(points).any()):
        raise ValueError(f"{name} must be a real number, a list or an array of real numbers, got {value!r}")
    return points


def step(t: ArrayLike, a: float = STEP_START) -> np.ndarray:
    """Lebl's unit step (Heaviside) function u(t - a): 0 for t < a and 1 for t >= a.

    Lebl, Example 6.1.4 and Section 6.2 ("Using the Heaviside function"), used in notebook 20. It is a
    switch that closes at the time a: f(t) * step(t, a) is f from the time a on and 0 before, and
    step(t, a) - step(t, b) is 1 for a <= t < b and 0 elsewhere (a window). ``step(t)`` is u(t). The
    value at the jump itself is 1, as Lebl defines it; the transform e^(-as)/s and every solution are the
    same with any other value there. t is a number, a list or an array (in seconds, say), a is one
    number in the same unit, and the result is an array of 0.0 and 1.0 with the shape of t (for a single
    number, an array with no axes: it can be used as a number). The letter u is Lebl's: it is not the
    u_1, u_2 of variation of parameters. ValueError: a that is not one finite real number; t that is
    not made of real numbers (text, a boolean, a complex number, NaN, or a SymPy letter: in a formula
    typed with SymPy the step is sp.Heaviside, which notebook 20 calls u; a SymPy number such as sp.pi
    is a number).
    """
    start = _one_real_number("a", a)
    if type(t).__module__.startswith("sympy") and getattr(t, "free_symbols", True):
        raise ValueError("ol.step works with numbers: in a formula typed with SymPy write u(t - 2), the step "
                         f"of notebook 20 (it is sp.Heaviside), got {t!r}")
    return np.where(_real_array("t", t) >= start, 1.0, 0.0)


def convolution(f: Callable[[float], float], g: Callable[[float], float], t: ArrayLike, *,
                breaks: Sequence[float] = ()) -> np.ndarray:
    """The convolution (f * g)(t): the integral from 0 to t of f(tau) g(t - tau) d tau.

    Lebl, Section 6.3.1, formula (6.3.1), used in notebook 20. f and g are functions of one number
    (the time, 0 or more) that return one real number; t is a number, a list or an array of times, each
    0 or more; the result is an array with the shape of t (for a single number, an array with no axes),
    and (f * g)(0) = 0: at t = 0 the result is 0 and f and g are not evaluated. The star is not a
    product: for f = g = 1 it gives t, not 1. Units: with t in seconds, the result is in the units of f
    times the units of g times seconds. ``breaks`` are the times at which f or g jumps (a switch, the
    ends of a pulse): give them, and the integral is cut at tau = break (for f) and at tau = t - break
    (for g).

    How it is computed, for each t: by ``scipy.integrate.quad`` on equal pieces of [0, t], cut also at
    the breaks and at the jumps that a first look at f and g finds by itself (CONVOLUTION_SCAN equal
    steps of [0, t]; cuts nearer than CONVOLUTION_NARROW t are one cut). The sum is computed twice,
    with different pieces (CONVOLUTION_PIECES), and it is accepted when quad is in doubt on no piece and
    the two sums and quad's own error estimates agree to CONVOLUTION_RTOL max(1, |value|), the rule of
    ``check``. Otherwise a ValueError says so: that is the safe answer.

    What is promised, and what is not (measured; "about"):
      * The accuracy is absolute below 1 and relative above: about 1e-8 max(1, |(f * g)(t)|)
        (CONVOLUTION_RTOL). So it depends on the units chosen for f and g: a result near 0 that is
        made of pieces of size 1e6 (1e6 cos t against 1, at t = 2 pi) cannot reach it and is refused.
      * The search for switches is a help, not a guarantee. A switch or a pulse that is not named in
        ``breaks`` is looked for, and it is found by the first look when it lasts longer than one
        step of that look, t / 256 (CONVOLUTION_SCAN), AND its jump is more than 8 times
        (CONVOLUTION_JUMP_RATIO) what the function changes over a step around it, as it is for a pulse
        on 0 or on a constant. Otherwise the answer is a ValueError or, when both sums miss it, a
        wrong number with no error, as if the pulse were not there. When f or g switches, name the
        times in ``breaks``.
      * f and g must be finite numbers at every time from 0 to t, the ends included: 1 / sqrt(t) is
        refused, although its convolutions exist. (A function that is not finite only on an interval
        shorter than a step of the first look can pass unseen, as a short pulse can.)
      * Each t costs about 450 evaluations of f and of g for smooth functions and about 700 to 900
        with one or two jumps: use it on tens of times, not on thousands. It is a numerical
        integral, slower and less exact than a formula.

    What the search misses (measured on seeded pulses):
      * A shorter pulse (0.01 wide when t = 10). On 0: all right from t / 256 up, one in ten wrong at
        t / 300, about half at t / 500, seven in ten at t / 1000.
      * A low pulse, up to about t / 100 wide, on a function that is itself changing (10 sin t plus a
        pulse of height 1.5 and width 0.05, when t = 10; a pulse of height 0.5 and width 0.1 on
        sin 2t, when t = 20): it is not seen when its height is less than about 8 times the change of
        the function over one step of the first look. On a ramp, a sine or an exponential: about half
        wrong at t / 250, one in eight at t / 128, a few in a hundred at t / 100.
      * A switch that closes steeply without a jump (a ramp from 0 to 1 that lasts 0.0003, when
        t = 50): a few in a hundred wrong when the ramp lasts a hundredth of a step of the first
        look, about one in three at a thousandth.
    With the times named in ``breaks`` all of these are computed.

    ValueError, with one sentence: f or g that is not a function, that does not take one number, that
    returns something that is not one finite real number, or that stops with an error of its own (the
    sentence quotes it); t that is not made of finite real numbers, or a negative t; breaks that are
    not finite times >= 0; a product f(tau) g(t - tau) too large for the computer; an integral on
    which the two computations do not agree. SciPy is imported here, not at the top: the rest of odelib
    needs NumPy only.
    """
    import math

    from scipy.integrate import quad

    for name, function in (("f", f), ("g", g)):
        if not callable(function):
            raise ValueError(f"{name} must be a function of one number, for example lambda t: np.exp(-2 * t)")
    points = _real_array("t", t)
    if not bool(np.all(np.isfinite(points) & (points >= 0))):
        raise ValueError(f"t must be made of finite times, each 0 or more (f and g are defined for "
                         f"t >= 0 only), got {t!r}")
    try:
        jumps = [math.nan] if isinstance(breaks, (str, bytes)) else sorted(
            {_one_real_number("a break", point) for point in breaks})
    except (TypeError, ValueError, OverflowError):        # not a list, or a point that is not a number
        jumps = [math.nan]
    if not all(math.isfinite(point) and point >= 0 for point in jumps):
        raise ValueError(f"breaks must be a list of finite times, each 0 or more, got {breaks!r}")
    digits = round(-math.log10(CONVOLUTION_RTOL))
    in_doubt = (f"the convolution could not be computed to {digits} digits: if f or g has a switch or a "
                "short pulse, name the times at which it jumps in breaks=[...]; or f(tau) g(t - tau) is so "
                f"much larger than the result that {digits} digits cannot be reached")

    def value_of(function: Callable[[float], float], name: str, time: float) -> float:
        """function(time) as a finite float, or a ValueError with the sentence for what went wrong."""
        try:
            raw = function(time)
        except TypeError as problem:
            hint = " (a power is written t ** 2, not t ^ 2)" if "^" in str(problem) else ""
            raise ValueError(f"{name} must take one number and return one number ({problem}){hint}") from None
        except Exception as problem:                      # whatever else it does: said in one sentence
            raise ValueError(f"{name} stopped with an error at the time {time:.3g} "
                             f"({type(problem).__name__}: {problem})") from None
        try:
            refused = isinstance(raw, (str, bytes)) or np.iscomplexobj(raw) or np.ndim(raw) != 0
            value = math.nan if refused else float(raw)
        except Exception:                                 # whatever float() raises for such a value
            refused = True
        if refused:
            raise ValueError(f"{name} must return one real number, got {raw!r} at the time {time:.3g}")
        if not math.isfinite(value):
            raise ValueError(f"{name} is not a finite number at the time {time:.3g} (it gives {value!r}): "
                             "the convolution needs f and g at every time from 0 to t")
        return value

    def seen_jumps(function: Callable[[float], float], name: str, end: float) -> set[float]:
        """The times in (0, end) at which a look at the function on equal steps sees a jump."""
        times = [end * index / CONVOLUTION_SCAN for index in range(CONVOLUTION_SCAN + 1)]
        values = [value_of(function, name, time) for time in times]
        changes = [abs(after - before) for before, after in zip(values[:-1], values[1:], strict=True)]
        floor = CONVOLUTION_RTOL * max(abs(value) for value in values)
        marked = []
        for index, change in enumerate(changes):
            around = sorted(changes[max(index - CONVOLUTION_WINDOW, 0):index]
                            + changes[index + 1:index + 1 + CONVOLUTION_WINDOW])
            typical = around[len(around) // 2]            # the median of the changes on the steps around
            if change > CONVOLUTION_JUMP_RATIO * typical and change > floor:
                marked.append(index)
        found: set[float] = set()
        if len(marked) <= CONVOLUTION_MAX_JUMPS:
            for index in marked:                          # halve the step: where is the function half way
                low, high = times[index], times[index + 1]                # between its two sides?
                below, above = values[index], values[index + 1]
                middle = (low + high) / 2
                while low < middle < high:
                    between = value_of(function, name, middle)
                    if abs(between - below) <= abs(between - above):
                        low = middle
                    else:
                        high = middle
                    middle = (low + high) / 2
                # low and high are now neighboring numbers. The step is cut there and at its two ends, so
                # that a switch that closes steeply without a jump lies in two pieces of its own.
                found.update((times[index], high, times[index + 1]))
        return found

    def by_pieces(end: float, pieces: int, cuts: set[float]) -> tuple[float, float]:
        """The sum of the integrals over the pieces of [0, end] and of the errors that quad estimates."""
        edges = [0.0]
        for edge in sorted(cuts | {end * index / pieces for index in range(1, pieces)}):
            if min(edge - edges[-1], end - edge) > CONVOLUTION_NARROW * end:
                edges.append(edge)                        # (a cut next to another one is left out)
        edges.append(end)
        share = CONVOLUTION_RTOL / (len(edges) - 1)
        total = error = 0.0
        for left, right in zip(edges[:-1], edges[1:], strict=True):
            part, estimate, _, *doubt = quad(
                lambda tau: value_of(f, "f", tau) * value_of(g, "g", end - tau), left, right,
                epsabs=share, epsrel=share, limit=QUAD_LIMIT, full_output=True)
            if doubt:                                     # quad adds a message when it is in doubt
                raise ValueError(in_doubt)
            total += part
            error += estimate
        return total, error

    def at(end: float) -> float:
        """(f * g)(end) for one time end >= 0."""
        if end == 0:
            return 0.0
        of_f = set(jumps) | seen_jumps(f, "f", end)       # f jumps at tau = these times
        of_g = set(jumps) | seen_jumps(g, "g", end)       # g(end - tau) jumps where end - tau = these times
        cuts = {time for time in of_f | {end - time for time in of_g} if 0.0 < time < end}
        (value, estimate), (other, other_estimate) = (by_pieces(end, pieces, cuts)
                                                      for pieces in CONVOLUTION_PIECES)
        if not (math.isfinite(value) and math.isfinite(other)):
            raise ValueError(f"f(tau) g(t - tau) is too large for the computer between 0 and t = {end:.3g}")
        if not max(estimate, other_estimate, abs(value - other)) <= CONVOLUTION_RTOL * max(1.0, abs(value)):
            raise ValueError(in_doubt)
        return value

    with np.errstate(all="ignore"):
        values = np.array([at(float(end)) for end in points.ravel()], dtype=float)
    return values.reshape(points.shape)


def impulse_response(t: ArrayLike, a: float, b: float, c: float) -> np.ndarray:
    """The impulse response of a y'' + b y' + c y: the solution of a y'' + b y' + c y = delta(t) from rest.

    Lebl, Section 6.4.3 and Example 6.4.2 (x'' + omega_0^2 x = delta(t) gives sin(omega_0 t) / omega_0),
    used in notebook 20. The system is at rest before the blow, y = 0 for t < 0; the unit impulse at
    t = 0 changes the velocity at once from 0 to 1 / a and the position not at all, so for t >= 0 the
    result is the free motion with y(0) = 0 and y'(0) = 1 / a. That motion is computed by
    ``constant_coefficient_solution``: nothing is solved again here. Its transform is the transfer
    function H(s) = 1 / (a s^2 + b s + c) (Lebl, Section 6.2), and the solution from rest for any input
    f is the convolution of f with this function: ``convolution(f, lambda t: impulse_response(t, a, b,
    c), t)`` equals ``forced_response(t, a, b, c, f)``. Units, for a mass a in kg, a damper b in N s/m
    and a spring c in N/m, with t in seconds: meters for each newton second of impulse; a blow of
    impulse P gives P times this function. t is a number, a list or an array, and the result is an
    array with the shape of t (for a single number, an array with no axes). ValueError: a = 0 (the
    equation must be of second order); a, b or c that is not one finite real number; t that is not made
    of finite real numbers; times so large that the response is not a number of the computer.
    """
    a, b, c = _one_real_number("a", a), _one_real_number("b", b), _one_real_number("c", c)
    if a == 0:
        raise ValueError("a must be nonzero: the equation must be of second order")
    points = _real_array("t", t)
    if not bool(np.all(np.isfinite(points))):
        raise ValueError(f"t must be made of finite times, got {t!r}")
    with np.errstate(over="ignore", invalid="ignore"):
        after = constant_coefficient_solution(np.maximum(points, 0.0), a, b, c, 0.0, 1.0 / a)
    if not bool(np.all(np.isfinite(after))):
        raise ValueError("the impulse response is too large for the computer at these times, or a is too "
                         "near to 0: use times closer to 0")
    return np.where(points >= 0, after, 0.0)


# ---------------------------------------------------------------------------
# Series solutions (notebook 21; Lebl, Sections 7.1 to 7.3)
# ---------------------------------------------------------------------------
# Conventions: x is the independent variable, as in Lebl's Chapter 7; a_k are the coefficients and k
# their index; x0 is the center of the series; r is the exponent of the method of Frobenius,
# y = x^r (a_0 + a_1 x + a_2 x^2 + ...), and p0, q0 are the two limits of a regular singular point.
SERIES_CENTER = 0.0             # the default x0 of series_value: a series in powers of x
# The four cases of the indicial equation r(r - 1) + p0 r + q0 = 0 (Lebl, Section 7.3.2, items iii to vi
# of the list after Theorem 7.3.1), as indicial_roots names them:
INDICIAL_NOT_INTEGER = "two real roots, difference not a whole number"
INDICIAL_DOUBLE = "double root"
INDICIAL_INTEGER = "two real roots, difference a whole number"
INDICIAL_COMPLEX = "complex roots"
# The roots differ by a whole number m (m = 0: a double root) when their squared difference,
# (r1 - r2)^2 = (p0 - 1)^2 - 4 q0, is within INDICIAL_TOLERANCE max(1, (p0 - 1)^2 + 4 |q0|) of m^2: about
# four digits above the rounding of the computer. The squares are compared, not the roots: the square root
# would turn a rounding of 1e-16 in p0 or q0 into a gap of 1e-8 between the two roots of a double root.
INDICIAL_TOLERANCE = 1e-12
# The band INDICIAL_TOLERANCE max(1, ...) grows with the roots. When it reaches INDICIAL_AMBIGUOUS, a double
# root (squared difference 0) and two roots 1 apart (squared difference 1) can no longer be told apart, and
# indicial_roots refuses: this happens when (p0 - 1)^2 + 4|q0| reaches 2.5e11, for example for a double
# root of size about 1.8e5.
INDICIAL_AMBIGUOUS = 0.25


class _UnknownCoefficient(Exception):
    """Raised when a recurrence asks for a coefficient that series_coefficients does not know (yet)."""


class _EarlierCoefficients:
    """The coefficients a[0], ..., a[k - 1] as a recurrence sees them while a_k is computed.

    A plain list would read a[-1] as its last item: a recurrence that reaches back further than the
    initial values would then get a wrong number with no error. Here every index outside 0, ..., k - 1
    is refused with a sentence. The known coefficients can be looped over (for v in a, sum(a),
    np.array(a)) and sliced with ends 0 or more (a[:k], a[k - 2:k]).
    """

    def __init__(self, values: list[float]) -> None:
        self._values = values
        self.k = len(values)

    def __len__(self) -> int:
        return self.k

    def __iter__(self) -> _Iterator[float]:
        return iter(self._values[:self.k])                # a_0 to a_(k-1)

    def __getitem__(self, index: int | slice) -> float | list[float]:
        if isinstance(index, slice):                      # a[:k], a[k - 2:k]: a list of known coefficients
            if any(end is not None and end < 0 for end in (index.start, index.stop)):
                raise _UnknownCoefficient(f"the recurrence asked for a[{index.start}:{index.stop}] at k = "
                                          f"{self.k}: a slice of a must start and stop at 0 or above (a "
                                          "negative end is not counted from the end here)")
            return self._values[:self.k][index]
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise _UnknownCoefficient(f"the recurrence asked for a[{index!r}] at k = {self.k}: an index of a "
                                      "must be a whole number, such as k - 2")
        if index < 0:
            back = self.k - index                         # how many places back from a_k that is
            raise _UnknownCoefficient(f"the recurrence asked for a[{index}] at k = {self.k}: there is no "
                                      f"coefficient with a negative index (use a[k - {-index}], not "
                                      f"a[{index}], for the coefficient {-index} place(s) back); if your "
                                      f"rule reaches back {back} places, give at least {back} initial values")
        if index >= self.k:
            raise _UnknownCoefficient(f"the recurrence asked for a[{index}] at k = {self.k}: only a[0] to "
                                      f"a[{self.k - 1}] are known when a_{self.k} is computed")
        return self._values[index]


def _next_coefficient(recurrence: Callable[[int, Sequence[float]], float], k: int,
                      earlier: _EarlierCoefficients) -> float:
    """recurrence(k, earlier) as a finite float, or a ValueError with one sentence that names k."""
    try:
        raw = recurrence(k, earlier)
    except _UnknownCoefficient as problem:
        raise ValueError(str(problem)) from None
    except TypeError as problem:
        hint = " (a power is written k ** 2, not k ^ 2)" if "^" in str(problem) else ""
        if "positional argument" in str(problem):         # the function does not take (k, a)
            raise ValueError(f"the recurrence stopped at k = {k} with a TypeError ({problem}): it must take "
                             "two arguments, k and the list a of the earlier coefficients, and return one "
                             f"number{hint}") from None
        raise ValueError(f"the recurrence stopped with an error at k = {k} (TypeError: {problem}){hint}"
                         ) from None
    except ZeroDivisionError as problem:
        raise ValueError(f"the recurrence stopped with an error at k = {k} (ZeroDivisionError: {problem}): a "
                         "division by 0 (if you wrote a power with ^, write ** instead)") from None
    except Exception as problem:                          # whatever else it does: said in one sentence
        raise ValueError(f"the recurrence stopped with an error at k = {k} "
                         f"({type(problem).__name__}: {problem})") from None
    try:
        refused = (isinstance(raw, (bool, np.bool_, str, bytes)) or np.iscomplexobj(raw)
                   or np.ndim(raw) != 0)
        value = float("nan") if refused else float(raw)
    except Exception:                                     # whatever float() raises for such a value
        refused = True
    if refused:
        raise ValueError(f"the recurrence must return one real number, got {raw!r} at k = {k}")
    if not np.isfinite(value):
        raise ValueError(f"the recurrence gives {value!r} at k = {k}, not a finite number (a division by 0?)")
    return value


def series_coefficients(recurrence: Callable[[int, Sequence[float]], float], initial: Sequence[float],
                        n: int) -> np.ndarray:
    """The first n coefficients a_0, ..., a_(n-1) of a power series, from the first ones and a recurrence.

    Lebl, Section 7.2 (the recurrence relations of Examples 7.2.1 to 7.2.3) and Section 7.3 (Example 7.3.2,
    with the exponent r inside the recurrence), used in notebook 21. ``initial`` gives the first
    coefficients a_0, a_1, ...: the ones that the equation leaves free, and any that it fixes before the
    recurrence starts (Airy's a_2 = 0). For every later k, ``recurrence(k, a)`` must return a_k, where
    a[j] is the coefficient a_j found before, for 0 <= j < k; a can also be looped over (sum(a)) and sliced
    with ends 0 or more (a[k - 2:k]), and len(a) is k. For y'' - y = 0 (Lebl, Example 7.2.1) the
    rule is a_k = a_(k-2) / (k (k - 1)), and
    ``series_coefficients(lambda k, a: a[k - 2] / (k * (k - 1)), [1, 0], 10)`` gives the coefficients of
    cosh x. The result is an array of n floats; the coefficients that are 0 are counted too.

    What is promised, and what is not:
      * The numbers are those of the recurrence, in floating point. Nothing here checks that the recurrence
        belongs to the equation: check the first coefficients by hand, or put the series in the equation.
      * The coefficients say nothing about where the series converges, or how many terms are needed: see
        series_value.
      * a[j] for j outside 0, ..., k - 1 is refused, a negative j included (Python would read a[-1] as the
        last coefficient, a wrong number with no error; write a[k - 1]): a recurrence that reaches back
        3 places needs at least 3 initial values. A slice that runs past k is cut at k, as for a list.
      * Real coefficients only: complex roots of the indicial equation (Lebl, Section 7.3.2, item vi of
        the list after Theorem 7.3.1) are not handled.
      * The recurrence is called once for each k from len(initial) to n - 1; n is at most MAX_STEPS.

    ValueError, with one sentence: a recurrence that is not a function; initial that is not a list of
    finite real numbers, at least one; n that is not a whole number (20, not 20.0) from len(initial) to
    MAX_STEPS; a recurrence that asks for a coefficient that is not known, that stops with an error of its
    own, or that returns something that is not one finite real number (each sentence names k: a division
    by 0 at k = 2, for example).
    """
    if not callable(recurrence):
        raise ValueError("recurrence must be a function of k and a, for example "
                         "lambda k, a: a[k - 2] / (k * (k - 1))")
    advice = f"initial must be a list of finite real numbers, a_0 first (at least one), got {initial!r}"
    try:
        start = _real_array("initial", initial)
    except ValueError:
        raise ValueError(advice) from None
    if start.ndim != 1 or start.size == 0 or not bool(np.all(np.isfinite(start))):
        raise ValueError(advice)
    whole = isinstance(n, (int, np.integer)) and not isinstance(n, (bool, np.bool_))
    if not whole or not start.size <= n <= MAX_STEPS:
        raise ValueError(f"n, the number of coefficients, must be a whole number such as 20 (not 20.0), from "
                         f"{start.size} (the number of initial values) to {MAX_STEPS}, got {n!r}")
    values = [float(value) for value in start]
    earlier = _EarlierCoefficients(values)
    with np.errstate(all="ignore"):                       # a division by 0 is reported below, with its k
        for k in range(start.size, int(n)):
            earlier.k = k
            values.append(_next_coefficient(recurrence, k, earlier))
    return np.array(values, dtype=float)


def series_value(coefficients: ArrayLike, x: ArrayLike, x0: float = SERIES_CENTER) -> np.ndarray:
    """The partial sum a_0 + a_1 (x - x0) + a_2 (x - x0)^2 + ... + a_(n-1) (x - x0)^(n-1), at x.

    Lebl, Section 7.1.1 (the partial sum of a power series), used in notebook 21. ``coefficients`` are
    a_0, ..., a_(n-1), for example from ``series_coefficients``; x is a number, a list or an array, and x0
    is the center. The sum is computed by Horner's rule, a_0 + h (a_1 + h (a_2 + ...)) with h = x - x0:
    n - 1 multiplications and no powers. The result is an array with the shape of x (for a single number,
    an array with no axes: it can be used as a number).

    What is promised, and what is not:
      * This is the value of a polynomial, not of the function that the whole series stands for. The two
        agree only inside the radius of convergence, and only when enough terms are taken; outside the
        radius more terms make the partial sum worse, not better (the series x - x^3/3 + x^5/5 - ... of
        arctan x at x = 1.1: off by 0.17 with 10 terms that are not 0, by 13 with 40). Nothing here knows
        the radius.
      * Floating point: when the terms are much larger than their sum (an alternating series far from the
        center), digits are lost by cancellation, whatever the number of terms. The series of the Bessel
        function J_0 at x = 20 has terms as large as about 8e6 and the sum 0.167: no number of terms
        brings it closer than about 1e-9 (with 30 terms that are not 0 it is off by about 1e-5: too few
        terms). Compare with an independent value (scipy.special) when you can.

    ValueError: coefficients that are not a list of finite real numbers, at least one; x that is not made
    of finite real numbers; x0 that is not one finite real number; a partial sum that is too large for the
    computer at one of the x.
    """
    advice = ("coefficients must be a list of finite real numbers, a_0 first (at least one), "
              f"got {coefficients!r}")
    try:
        terms = _real_array("coefficients", coefficients)
    except ValueError:
        raise ValueError(advice) from None
    if terms.ndim != 1 or terms.size == 0 or not bool(np.all(np.isfinite(terms))):
        raise ValueError(advice)
    points = _real_array("x", x)
    if not bool(np.all(np.isfinite(points))):
        raise ValueError(f"x must be made of finite real numbers, got {x!r}")
    center = _one_real_number("x0", x0)
    with np.errstate(over="ignore", invalid="ignore"):   # x - x0 itself may be too large: said below
        h = points - center
        total = np.full(h.shape, terms[-1])
        for coefficient in terms[-2::-1]:                 # Horner: from the last coefficient to a_0
            total = total * h + coefficient
    if not bool(np.all(np.isfinite(total))):              # (an infinite h makes the sum infinite too)
        raise ValueError("the partial sum is too large for the computer at some of these x: use x nearer "
                         "to x0, or fewer coefficients")
    return np.asarray(total, dtype=float).reshape(points.shape)


def indicial_roots(p0: float, q0: float) -> tuple[float | complex, float | complex, str]:
    """The two roots of the indicial equation r(r - 1) + p0 r + q0 = 0, and which of four cases they are.

    Lebl, Section 7.3.2 (the method of Frobenius), used in notebook 21. For an equation
    P(x) y'' + Q(x) y' + R(x) y = 0 with a regular singular point at x = 0, p0 and q0 are the two limits
    of Lebl's definition, p0 = lim x Q(x) / P(x) and q0 = lim x^2 R(x) / P(x) as x -> 0; for the Euler
    equation x^2 y'' + p0 x y' + q0 y = 0 they are its two coefficients. The result is (r1, r2, case):
      * INDICIAL_NOT_INTEGER: two real roots whose difference is not a whole number, r1 > r2 (floats);
      * INDICIAL_DOUBLE: one real root, twice: r1 = r2 (floats);
      * INDICIAL_INTEGER: two real roots that differ by a whole number 1, 2, 3, ..., r1 > r2 (floats);
      * INDICIAL_COMPLEX: r1 = alpha + i beta and r2 = alpha - i beta with beta > 0 (complex numbers).
    These are the cases iii, iv, v and vi of the list after Lebl's Theorem 7.3.1: the case says what a
    second solution looks like.

    What is promised, and what is not:
      * "A whole number" and "double" are decided with INDICIAL_TOLERANCE, on the squared difference
        (r1 - r2)^2 = (p0 - 1)^2 - 4 q0, which the computer knows better than the difference itself. So p0
        and q0 typed with decimals are read as meant (0.7 and 0.0225 give the double root 0.15 only up to
        rounding). The price, for numbers p0 and q0 of size about 1: two roots closer than about 1e-6 are
        called a double root, and a difference within about 5e-13 of a whole number is called that whole
        number. These limits grow with the roots: two roots closer than about
        1e-6 sqrt((p0 - 1)^2 + 4|q0|) are called a double root; for example 1000 and 1000.002 are
        returned as the double root 1000.001. When (p0 - 1)^2 + 4|q0| reaches 2.5e11 (INDICIAL_AMBIGUOUS;
        for example a double root of size about 1.8e5) the function refuses, since it could no longer tell
        a double root from two roots 1 apart. Just below that limit the bands are wide: two roots up to
        0.5 apart are called a double root (170000.4 and 170000), and a difference between about 0.87
        and 1.12 is called 1.
      * Nothing is checked about the equation: not that x = 0 is a regular singular point, not that p0 and
        q0 are its limits.
      * The roots are not the solutions: with a double root, or two roots that differ by a whole number, the
        second solution may contain ln x (Lebl, items iv and v; in item v the term with ln x may also be
        absent, as for every Euler equation).

    ValueError: p0 or q0 that is not one finite real number, or so large that the roots are not numbers of
    the computer, or that the case cannot be decided ((p0 - 1)^2 + 4|q0| of 2.5e11 and more).
    """
    p0, q0 = _one_real_number("p0", p0), _one_real_number("q0", q0)
    middle = (1.0 - p0) / 2                               # r^2 - 2 middle r + q0 = 0: the mean of the roots
    with np.errstate(over="ignore", invalid="ignore"):
        squared = float(np.float64(1.0 - p0) ** 2 - 4 * np.float64(q0))          # (r1 - r2)^2
        size = max(1.0, float(np.float64(1.0 - p0) ** 2 + 4 * abs(np.float64(q0))))
    if not (np.isfinite(squared) and np.isfinite(size)):
        raise ValueError(f"p0 = {p0!r} and q0 = {q0!r} are too large for the computer: the roots cannot be "
                         "computed")
    band = INDICIAL_TOLERANCE * size                      # how far squared may be from the truth
    if band >= INDICIAL_AMBIGUOUS:
        largest = abs(middle) + float(np.sqrt(abs(squared))) / 2      # the size of the larger root
        raise ValueError(f"p0 = {p0!r} and q0 = {q0!r}: the roots, up to about {largest:.3g} in size, are "
                         "too large for the computer to tell a double root from two roots 1 apart")
    if abs(squared) <= band:
        return middle, middle, INDICIAL_DOUBLE
    half_gap = float(np.sqrt(abs(squared))) / 2           # |r1 - r2| / 2, or beta for complex roots
    if squared < 0:
        return complex(middle, half_gap), complex(middle, -half_gap), INDICIAL_COMPLEX
    larger = middle + (half_gap if middle >= 0 else -half_gap)     # the root farther from 0: no cancellation
    other = q0 / larger + 0.0                             # the product of the roots is q0 (+ 0.0: no -0.0)
    r1, r2 = max(larger, other), min(larger, other)
    whole = np.float64(round(2 * half_gap))               # the whole number nearest to r1 - r2
    with np.errstate(over="ignore", invalid="ignore"):
        integer = bool(abs(squared - whole ** 2) <= band)
    return r1, r2, INDICIAL_INTEGER if integer else INDICIAL_NOT_INTEGER


# ---------------------------------------------------------------------------
# Numerical methods, part 2 (notebook 22; Lebl, Section 1.7)
# ---------------------------------------------------------------------------
# Conventions: t is the independent variable (Lebl writes x in Section 1.7, as notebook 04 does), h the
# step, N the number of steps. A "method" is euler, heun or rk4 above, or a function of a student with the
# same call, method(f, t_span, y0, h) -> (t, y). The error of a system is its largest component (the
# largest of the absolute errors of the unknowns), unless one component is asked for.
# Backward Euler solves an equation at each step by Newton's method, which stops when a correction is
# below NEWTON_TOLERANCE max(1, |y|) (about six digits above the rounding of the computer), and gives up
# with a ValueError after NEWTON_MAX_ITERATIONS corrections (Newton's method usually needs two to five).
NEWTON_TOLERANCE = 1e-10
NEWTON_MAX_ITERATIONS = 50
# Each step of backward_euler costs several evaluations of f and a linear solve, about 7 times a step of
# euler: its own, smaller limit of steps keeps the browser from freezing (MAX_STEPS is sized for euler).
BACKWARD_MAX_STEPS = 100_000


def _state(name: str, value: ArrayLike) -> np.ndarray:
    """A number or a list of numbers as a flat array of finite floats (at least one), or a ValueError."""
    try:
        values = _real_array(name, value)
    except ValueError:
        values = None
    if values is None or values.ndim > 1 or values.size == 0 or not bool(np.all(np.isfinite(values))):
        raise ValueError(f"{name} must be a finite real number, or a list of them (one for each unknown), "
                         f"got {value!r}")
    return np.atleast_1d(values)


def _slope_at(f: RHS, t: float, y: np.ndarray) -> np.ndarray:
    """f(t, y) as a float array of y.size finite numbers, or a ValueError that names t."""
    with np.errstate(all="ignore"):                       # a value that is not finite is reported below
        result = f(t, y.copy())
    try:
        values = np.asarray(result, dtype=float)
    except (TypeError, ValueError):
        values = None
    if values is None or values.size != y.size:
        raise ValueError(f"f(t, y) must return {y.size} number(s), one for each unknown; got {result!r} at "
                         f"t = {t:.6g}")
    values = values.reshape(y.shape)
    if not bool(np.all(np.isfinite(values))):
        raise ValueError(f"f(t, y) is not a finite number at t = {t:.6g} (y = {y})")
    return values


def _exact_at(exact: Callable[[float], ArrayLike], t: float, size: int, component: int | None) -> np.ndarray:
    """The exact solution at t, as the components that are compared (all, or the one asked for)."""
    result = exact(t)                                     # (an error inside exact is its own message)
    try:
        values = np.atleast_1d(np.asarray(result, dtype=float))
    except (TypeError, ValueError):
        values = None
    if values is None or values.ndim != 1 or values.size != size or not bool(np.all(np.isfinite(values))):
        raise ValueError(f"exact(t) must return {size} finite number(s), one for each unknown, at "
                         f"t = {t:.6g}")
    return values if component is None else values[component:component + 1]


def _component(component: int | None, size: int) -> int | None:
    """The index of the component to compare (None: all of them), or a ValueError."""
    if component is None:
        return None
    whole = isinstance(component, (int, np.integer)) and not isinstance(component, (bool, np.bool_))
    if not whole or not 0 <= component < size:
        raise ValueError(f"component must be None or a whole number from 0 to {size - 1} (0 for the first "
                         f"unknown), got {component!r}")
    return int(component)


def _start_on_exact(y0: np.ndarray, exact: Callable[[float], ArrayLike], t0: float) -> None:
    """A ValueError when y0 is not the exact value at t0 (within CHECK_TOLERANCE, as check)."""
    at_start = _exact_at(exact, t0, y0.size, None)
    if not np.allclose(y0, at_start, rtol=CHECK_TOLERANCE, atol=CHECK_TOLERANCE):
        raise ValueError(f"y0 must be the exact value at t0 = {t0:.6g}, exact(t0) = {at_start}; got {y0}")


def _last_value(method: Callable, f: RHS, t_span: tuple[float, float], y0: np.ndarray,
                h: float) -> np.ndarray:
    """The last value of method(f, t_span, y0, h), as a flat float array of y0.size numbers.

    The method's own errors (its sentences on the grid, an overflow, an error in a student's function) are
    not caught: they say what went wrong. Only a result of the wrong form is reported here.
    """
    result = method(f, t_span, y0 if y0.size > 1 else float(y0[0]), h)
    try:
        values = np.asarray(result[1], dtype=float)
        last = np.atleast_1d(values[-1]).ravel() if values.ndim > 0 else None
    except (TypeError, IndexError, KeyError, ValueError):
        last = None
    if last is None or last.size != y0.size:
        raise ValueError("method must be called as method(f, t_span, y0, h) and return (t, y), as ol.euler, "
                         "ol.heun and ol.rk4 do")
    return last


def local_error(method: Callable, f: RHS, t0: float, y0: ArrayLike, h: float,
                exact: Callable[[float], ArrayLike], *, component: int | None = None) -> float:
    """The error of ONE step of ``method``, started on the exact solution: |y_1 - exact(t0 + h)|.

    Lebl, Section 1.7 (Euler's method and its error), used in notebook 22. ``method`` is ol.euler, ol.heun,
    ol.rk4, or a function with the same call, method(f, t_span, y0, h) -> (t, y). y0 must be the exact
    value exact(t0) (within CHECK_TOLERANCE), so that the step starts on the solution and only the error
    of this one step is measured: the **local error**. ``exact(t)`` returns the exact solution at t (a
    number, or one number for each unknown of a system). For a system the result is the largest of the
    errors of the unknowns; with ``component=j`` only the error of unknown j (0 for the first).

    What is promised, and what is not:
      * A method of order p has a local error of size about h^(p+1) for small h (Euler: (h^2/2)|y''|,
        from Taylor's formula): halving h divides it by about 2^(p+1). This is a statement for small h;
        for a large h, or near the rounding of the computer (errors near 1e-15), the ratios are not those.
      * The error is measured at t0 + h only, not along the step.

    ValueError: method, f or exact that is not a function; t0 that is not one finite real number; h that
    is not a positive number; y0 that is not made of finite real numbers, or is not exact(t0); exact(t)
    returning the wrong number of values; component out of range; a method that does not return (t, y).
    The method's own errors are not caught (its sentences, a FloatingPointError on overflow).
    """
    for name, function in (("method", method), ("f", f), ("exact", exact)):
        if not callable(function):
            raise ValueError(f"{name} must be a function, got {function!r}")
    t0 = _one_real_number("t0", t0)
    h = _require_positive("h", _one_real_number("h", h))
    start = _state("y0", y0)
    which = _component(component, start.size)
    _start_on_exact(start, exact, t0)
    last = _last_value(method, f, (t0, t0 + h), start, h)
    target = _exact_at(exact, t0 + h, start.size, which)
    compared = last if which is None else last[which:which + 1]
    return float(np.max(np.abs(compared - target)))


def global_error(method: Callable, f: RHS, t_span: tuple[float, float], y0: ArrayLike,
                 exact: Callable[[float], ArrayLike], steps: Sequence[int], *,
                 component: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The error at the end, |y_N - exact(t1)|, for each number of steps N in ``steps``: a convergence study.

    Lebl, Section 1.7 (Table 1.7.1: the error at x = 2 as h is halved), used in notebook 22. For each N,
    h = (t1 - t0) / N and ``method`` (ol.euler, ol.heun, ol.rk4 or a function with the same call) is run
    from t0 to t1; the error is the **global error** at t1, where the errors of all the steps have added
    up. y0 must be the exact value exact(t0) (within CHECK_TOLERANCE). For a system the error is the
    largest of the errors of the unknowns; with ``component=j`` only the error of unknown j (0 for the
    first). Returns (h values, errors), two float arrays in the order of ``steps``: ready for a log-log
    plot, and for ``observed_order(errors, ratio)`` when each N is ``ratio`` times the one before.

    What is promised, and what is not:
      * A method of order p has a global error of size about h^p for small h: halving h divides it by about
        2^p. The observed orders approach p only as h becomes small, and only while the step is stable and
        the error is far above the rounding of the computer (for RK4 on y' = y cos t the ratios are off
        below about 1e-13 and the errors stop falling near 1e-15: notebook 22, Exercise 4(i)).
      * An error that is exactly 0 (a method that is exact on this equation) cannot go into observed_order.

    ValueError: method, f or exact that is not a function; t_span that is not t0 < t1, finite; y0 that is
    not made of finite real numbers, or is not exact(t0); steps that are not whole numbers from 1 to
    MAX_STEPS (at least one); exact(t) returning the wrong number of values; component out of range; a
    method that does not return (t, y). The method's own errors are not caught (its sentences, a
    FloatingPointError on overflow).
    """
    for name, function in (("method", method), ("f", f), ("exact", exact)):
        if not callable(function):
            raise ValueError(f"{name} must be a function, got {function!r}")
    try:
        t0, t1 = (_one_real_number("t0", t_span[0]), _one_real_number("t1", t_span[1]))
        extra = len(t_span) != 2
    except (TypeError, IndexError, KeyError):
        t0 = t1 = 0.0
        extra = True
    if extra or t1 <= t0:
        raise ValueError(f"t_span must be two finite numbers (t0, t1) with t0 < t1, got {t_span!r}")
    start = _state("y0", y0)
    which = _component(component, start.size)
    advice = (f"steps must be a list of whole numbers of steps N from 1 to {MAX_STEPS}, such as "
              f"[10, 20, 40], got {steps!r}")
    try:
        counts = list(steps)
    except TypeError:
        raise ValueError(advice) from None
    whole = all(isinstance(n, (int, np.integer)) and not isinstance(n, (bool, np.bool_)) for n in counts)
    if not counts or not whole or not all(1 <= n <= MAX_STEPS for n in counts):
        raise ValueError(advice)
    _start_on_exact(start, exact, t0)
    target = _exact_at(exact, t1, start.size, which)
    sizes, errors = [], []
    for n in counts:
        h = (t1 - t0) / int(n)
        last = _last_value(method, f, (t0, t1), start, h)
        compared = last if which is None else last[which:which + 1]
        sizes.append(h)
        errors.append(float(np.max(np.abs(compared - target))))
    return np.array(sizes, dtype=float), np.array(errors, dtype=float)


def _newton_matrix(f: RHS, jacobian: Callable | None, t: float, y: np.ndarray, h: float) -> np.ndarray:
    """I - h J, with J the matrix of the derivatives of f(t, y) in y: given, or by central differences."""
    size = y.size
    if jacobian is not None:
        result = jacobian(t, y.copy())                # (an error inside jacobian is its own message)
        try:
            given = np.asarray(result, dtype=float)
        except (TypeError, ValueError):
            given = None
        if given is None or given.size != size * size or not bool(np.all(np.isfinite(given))):
            raise ValueError(f"jacobian(t, y) must return {size} x {size} finite number(s) (the derivatives "
                             f"of f in y) at t = {t:.6g}")
        derivatives = given.reshape(size, size)
    else:
        columns = []
        for j in range(size):
            step = JACOBIAN_STEP * max(1.0, abs(y[j]))
            plus, minus = y.copy(), y.copy()
            plus[j] += step
            minus[j] -= step
            columns.append((_slope_at(f, t, plus) - _slope_at(f, t, minus)) / (2 * step))
        derivatives = np.column_stack(columns)
    return np.eye(size) - h * derivatives


def _backward_euler_step(f: RHS, jacobian: Callable | None, t: float, y: np.ndarray, h: float) -> np.ndarray:
    """y_(n+1), the solution Y of Y = y_n + h f(t + h, Y), by Newton's method from Euler's guess."""
    t_next = t + h
    guess = y + h * _slope_at(f, t, y)                    # Euler's step: the first guess
    for _ in range(NEWTON_MAX_ITERATIONS):
        residual = guess - y - h * _slope_at(f, t_next, guess)
        try:
            correction = np.linalg.solve(_newton_matrix(f, jacobian, t_next, guess, h), residual)
        except np.linalg.LinAlgError:
            raise ValueError(f"backward Euler: the equation of the step to t = {t_next:.6g} cannot be solved "
                             "(the matrix I - h J is singular): try another h") from None
        with np.errstate(over="ignore", invalid="ignore"):
            guess = guess - correction
        if not bool(np.all(np.isfinite(guess))):
            raise ValueError(f"backward Euler: the values became too large for the computer in the step to "
                             f"t = {t_next:.6g} (blow-up of the solution, or a step too large for Newton's "
                             "method): try a smaller h")
        if np.max(np.abs(correction)) <= NEWTON_TOLERANCE * max(1.0, float(np.max(np.abs(guess)))):
            return guess
    raise ValueError(f"backward Euler: Newton's method did not converge in the step to t = {t_next:.6g} "
                     f"({NEWTON_MAX_ITERATIONS} corrections): try a smaller h")


def backward_euler(f: RHS, t_span: tuple[float, float], y0: ArrayLike, h: float,
                   jacobian: Callable | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Backward (implicit) Euler y_(n+1) = y_n + h f(t_(n+1), y_(n+1)): the slope at the END of the step.

    Notebook 22 (Lebl, Section 1.7, names stiff problems; A. Meseguer, Fundamentals of Numerical
    Mathematics for Physicists and Engineers, Section 8.2). y_(n+1) appears on both sides, so each step
    solves an equation for it: by Newton's method, started from Euler's step, with the matrix of the
    derivatives of f in y given by ``jacobian(t, y)`` (an n x n array; a number for one equation) or, when
    it is None, computed by central differences with the step JACOBIAN_STEP max(1, |y_j|). Newton stops
    when a correction is below NEWTON_TOLERANCE max(1, |y|). Same grid and same result as ol.euler: the
    times t0, t0 + h, ..., t1 (a whole number of steps), and (t, y), y a 1-D array for one equation and
    an array of shape (N + 1, n) for a system of n equations.

    What is promised, and what is not:
      * Order 1, like Euler: halving h halves the error. It is not more accurate than Euler; it is more
        stable. On the test equation y' = lambda y it multiplies by 1 / (1 - h lambda) at each step, which
        is less than 1 in size for every h > 0 when lambda < 0: the values decay, as the solution does,
        whatever the step (Euler multiplies by 1 + h lambda and grows when h > 2 / |lambda|).
      * Forward in time only (t0 < t1), as euler.
      * For a growing solution (lambda > 0) it is not stable in this sense: with h lambda = 1 the equation
        of the step cannot be solved (reported as singular, or as Newton not converging when the
        derivatives are computed by differences), and with h lambda > 1 the values change sign at every
        step.
      * Newton's method can fail for a strongly nonlinear f and a large h: then a ValueError names the
        time; a smaller h helps.
      * At most BACKWARD_MAX_STEPS steps (each costs about 7 steps of euler).

    ValueError: f (or jacobian) that is not a function; the grid rules of euler (t0 < t1, finite, h > 0,
    a whole number of steps), and at most BACKWARD_MAX_STEPS steps; y0 that is not made of finite real
    numbers; f(t, y) or jacobian(t, y) returning the wrong number of values, or values that are not
    finite (the sentence names t); Newton's method not converging, or values too large for the computer
    (the sentence names t). An error raised inside f or jacobian is not caught: it is its own message.
    """
    if not callable(f):
        raise ValueError(f"f must be a function of t and y, got {f!r}")
    if jacobian is not None and not callable(jacobian):
        raise ValueError(f"jacobian must be None or a function of t and y, got {jacobian!r}")
    t = _grid(t_span, h)
    if t.size - 1 > BACKWARD_MAX_STEPS:
        raise ValueError(f"too many steps ({t.size - 1}, the limit of backward_euler is "
                         f"{BACKWARD_MAX_STEPS}): use a larger step h")
    start = _state("y0", y0)
    y = np.empty((t.size, start.size))
    y[0] = start
    step = float(h)
    for i in range(t.size - 1):
        y[i + 1] = _backward_euler_step(f, jacobian, t[i], y[i], step)
    return t, (y[:, 0] if y.shape[1] == 1 else y)


def embedded_step(f: RHS, t: float, y: ArrayLike, h: float) -> tuple[float | np.ndarray, float]:
    """One step of Heun's method and an estimate of the error of Euler's step: (Heun's value, |Heun - Euler|).

    Notebook 22: the idea of the step control of scipy's solve_ivp, shown with the simplest pair. From the
    point (t, y) both methods use the slope k_1 = f(t, y); Euler's step is y + h k_1, and Heun's adds the
    slope k_2 = f(t + h, y + h k_1) at the end, y + h (k_1 + k_2) / 2 (Lebl, the improved Euler of the
    exercises of Section 1.7). Heun's value is far more accurate for small h, so the difference of the
    two, (h / 2) |k_2 - k_1|, estimates the local error of Euler's step. Returns (Heun's value, the
    estimate): the value is a float for one equation and an array for a system; the estimate is the
    largest of its components.

    What is promised, and what is not:
      * The estimate is of Euler's error, not of Heun's: for small h a solver that keeps Heun's value with
        Euler's estimate is on the safe side (Heun's error is of size h^3, the estimate of size h^2); for a
        large h it is not (the estimate can even be 0).
      * It is an estimate, close to the true local error of Euler only for small h (notebook 22 checks it
        on y' = y cos t); for a large h it can be far off in either direction.

    ValueError: f that is not a function; t that is not one finite real number; h that is not a positive
    number; y that is not made of finite real numbers; f(t, y) returning the wrong number of values, or
    values that are not finite (the sentence names t).
    """
    if not callable(f):
        raise ValueError(f"f must be a function of t and y, got {f!r}")
    t = _one_real_number("t", t)
    h = _require_positive("h", _one_real_number("h", h))
    start = _state("y", y)
    first = _slope_at(f, t, start)
    euler_value = start + h * first
    second = _slope_at(f, t + h, euler_value)
    heun_value = start + h * (first + second) / 2
    estimate = float(np.max(np.abs(heun_value - euler_value)))
    return (float(heun_value[0]) if heun_value.size == 1 and np.ndim(y) == 0 else heun_value), estimate


# ---------------------------------------------------------------------------
# Checking an answer found by hand (exercises "by hand, then check")
# ---------------------------------------------------------------------------
CHECK_TOLERANCE = 1e-6          # relative and absolute tolerance of check()
ROUNDED = 2e-2                  # a wrong value this close to the right one is probably a rounded one


def _decimal_comma(value, expected):
    """The pair (3, 5) when 3,5 was typed where one number is expected (also inside a list), or None."""
    def pair(item):
        return (isinstance(item, tuple) and len(item) == 2
                and all(isinstance(part, int) and not isinstance(part, bool) for part in item) and item[1] >= 0)

    if pair(value) and np.ndim(expected) == 0:
        return value
    if isinstance(value, (list, tuple)) and np.ndim(expected) == 1:
        return next((item for item in value if pair(item)), None)
    return None


def check(label: str, value, expected, tol: float = CHECK_TOLERANCE) -> bool:
    """Compare an answer found by hand with the expected one, and say so in one line.

    Prints ``label: correct`` or ``label: not yet ...`` and returns True or False. It never
    raises for a wrong, missing (``None``) or non-numeric answer, so a notebook keeps running;
    the expected value is not printed. Numbers, complex numbers and arrays are accepted
    (text is not, even "3"). Every entry must be within ``tol`` of the expected one,
    relative to its size: |value - expected| <= tol * max(1, |expected|). So ``tol=1e-2``
    means one per cent for large values and 0.01 for values smaller than 1.
    """
    tol = _require_positive("tol", tol)
    if isinstance(value, (str, bytes)):
        print(f"{label}: not yet (type your answer without quotes; got {value!r})")
        return False
    if value is None:
        print(f"{label}: no answer yet (replace None with your value)")
        return False
    comma = _decimal_comma(value, expected)
    if comma is not None:
        whole, decimals = comma
        print(f"{label}: not yet ({whole},{decimals} is read by Python as the two numbers {whole} and {decimals}: "
              f"write {whole}.{decimals}, with a point)")
        return False
    try:
        got = np.asarray(value, dtype=complex)
        want = np.asarray(expected, dtype=complex)
    except (TypeError, ValueError):
        print(f"{label}: not yet (a number is expected, got {value!r})")
        return False
    if got.shape != want.shape:
        advice = "; a comma separates two numbers, and decimals take a point: 0.5" if isinstance(value, tuple) else ""
        print(f"{label}: not yet ({want.size} value(s) expected, got {got.size}{advice})")
        return False
    gap = np.abs(got - want)
    if np.all(gap <= tol * np.maximum(1.0, np.abs(want))):
        print(f"{label}: correct")
        return True
    if got.ndim == 0 and np.isfinite(want) and gap <= ROUNDED * max(1.0, abs(want)):
        print(f"{label}: not yet (your value {value!r} is close: if you rounded it, type the exact expression, "
              "for example 1 / 3 or np.sqrt(2))")
        return False
    print(f"{label}: not yet (your value {value!r}, try again)")
    return False


FORMULA_SAMPLES = 7             # points at which check_formula compares two formulas


def check_formula(label: str, expression, expected, symbol, interval: tuple[float, float],
                  tol: float = CHECK_TOLERANCE) -> bool:
    """Compare a formula typed by the student (a SymPy expression) with the expected one.

    Two formulas are taken as equal when they have the same values at ``FORMULA_SAMPLES``
    points of ``interval``, so any equivalent way of writing the answer is accepted. Like
    ``check`` it prints one line, returns True or False, never shows the expected formula
    and never raises for a wrong answer: a formula written with another letter, or one that
    cannot be evaluated, gets a hint instead of a traceback. SymPy is imported here, not at
    the top: the rest of odelib needs NumPy only.
    """
    import sympy as sp

    tol = _require_positive("tol", tol)
    if expression is None:
        print(f"{label}: no answer yet (replace None with your formula)")
        return False
    if isinstance(expression, (str, bytes)):
        print(f"{label}: not yet (type the formula without quotes)")
        return False
    points = np.linspace(*interval, FORMULA_SAMPLES)

    def values(formula):
        return np.asarray(sp.lambdify(symbol, formula, "numpy")(points), dtype=complex) + 0 * points

    try:
        formula = sp.sympify(expression)
        others = sorted(str(s) for s in formula.free_symbols - {symbol})
        if others:
            print(f"{label}: not yet (write the formula with the letter {symbol} only; "
                  f"it also contains {', '.join(others)})")
            return False
        with np.errstate(all="ignore"):
            got = values(formula)
    except Exception:                                    # anything SymPy or NumPy cannot digest
        print(f"{label}: not yet (the formula could not be evaluated: use the letter {symbol} "
              "and SymPy functions such as sp.exp, sp.log, sp.sin)")
        return False
    want = values(sp.sympify(expected))
    if np.all(np.isfinite(got)) and np.all(np.abs(got - want) <= tol * np.maximum(1.0, np.abs(want))):
        print(f"{label}: correct")
        return True
    print(f"{label}: not yet (your formula does not give the right values, try again)")
    return False


RESIDUAL_ROUNDING = 1e-10       # rounding allowed in a residual, relative to the size of the answer


def check_solution(label: str, expression, residual, symbol, interval: tuple[float, float],
                   conditions: Sequence[tuple] = (), tol: float = CHECK_TOLERANCE) -> bool:
    """Check a formula by substituting it in the equation: for answers that are not unique.

    A particular solution of a non-homogeneous equation can be written in many ways that are
    not equal as functions (any solution of the homogeneous equation may be added), so it
    cannot be compared with one expected formula. ``residual`` is a function that takes a SymPy
    expression y and returns "left-hand side minus right-hand side", for example
    ``lambda y: sp.diff(y, x, 2) - 9 * y - sp.exp(3 * x)``. The answer is right when the residual
    is zero at ``FORMULA_SAMPLES`` points of ``interval`` (within ``tol`` times the size of the
    right-hand side there, so that a large solution of the homogeneous equation is not taken
    for an answer) and when every condition is met. A condition is ``(point, value)`` for y(point) =
    value, or ``(point, value, order)`` for the derivative of that order. Same manners as
    ``check_formula``: one line, True or False, nothing is revealed, a wrong answer never raises.
    """
    import sympy as sp

    tol = _require_positive("tol", tol)
    if expression is None:
        print(f"{label}: no answer yet (replace None with your formula)")
        return False
    if isinstance(expression, (str, bytes)):
        print(f"{label}: not yet (type the formula without quotes)")
        return False
    points = np.linspace(*interval, FORMULA_SAMPLES).astype(complex)     # complex: the logarithm of a
    #                               negative number, which SymPy's integrals may contain, is then a number

    def values(formula):
        return np.asarray(sp.lambdify(symbol, formula, "numpy")(points), dtype=complex) + 0 * points

    try:
        formula = sp.sympify(expression)
        others = sorted(str(s) for s in formula.free_symbols - {symbol})
        if others:
            print(f"{label}: not yet (write the formula with the letter {symbol} only; "
                  f"it also contains {', '.join(others)})")
            return False
        formula = formula.replace(sp.Abs, lambda inside: sp.sqrt(inside ** 2))    # same values, and it can be differentiated
        with np.errstate(all="ignore"):
            size = values(formula)
            gap = values(residual(formula))
            forcing = values(residual(sp.Integer(0)))        # minus the right-hand side of the equation
            misses = []
            for condition in conditions:
                point, wanted = condition[0], condition[1]
                order = condition[2] if len(condition) > 2 else 0
                found = complex(sp.diff(formula, symbol, order).subs(symbol, point).evalf())
                if not abs(found - wanted) <= tol * max(1.0, abs(wanted)):
                    misses.append("y" + "'" * order + f"({float(point):g})")      # the point may be sp.pi
    except Exception:                                    # anything SymPy or NumPy cannot digest
        print(f"{label}: not yet (the formula could not be evaluated: use the letter {symbol} "
              "and SymPy functions such as sp.exp, sp.log, sp.sin)")
        return False
    if not (np.all(np.isfinite(size)) and np.all(np.isfinite(gap))
            and np.all(np.abs(gap) <= tol * np.maximum(1.0, np.abs(forcing))
                       + RESIDUAL_ROUNDING * np.maximum(1.0, np.abs(size)))):
        print(f"{label}: not yet (your formula does not satisfy the equation, try again)")
        return False
    if misses:
        print(f"{label}: not yet (the equation is satisfied, but not the condition on {', '.join(misses)})")
        return False
    print(f"{label}: correct")
    return True


def check_vector(label: str, value, expected, tol: float = CHECK_TOLERANCE, *,
                 up_to_scale: bool = False) -> bool:
    """Check an answer that is a vector or a matrix, and say so in one line (notebook 17).

    ``tol`` comes fourth, as in ``check`` and ``check_formula``. With ``up_to_scale=False`` this is
    ``check`` itself: the same rule and the same messages (a matrix is typed as a list of rows,
    [[a11, a12], [a21, a22]]). With ``up_to_scale=True`` the answer is accepted
    when it is a nonzero multiple, real or complex, of the vector ``expected``: eigenvectors are never
    unique (Lebl asks the question after Example 3.4.3). The best multiple
    k = (conj(e) . value) / (conj(e) . e) is computed, and the answer is right when
    |value - k e| <= tol * |value|: the direction is compared, whatever the length of the vector.
    Like ``check`` it prints one line, returns True or False, never shows the expected vector and
    never raises for a wrong answer; the zero vector gets its own message, and a column (as SymPy's
    ``eigenvects`` gives it) is read as a list. It raises ValueError only
    for a mistake of the notebook: with ``up_to_scale=True``, an expected vector that is not a flat
    list of finite numbers, or that is zero.
    """
    tol = _require_positive("tol", tol)
    if not up_to_scale:
        return check(label, value, expected, tol)
    try:
        direction = np.asarray(expected, dtype=complex)
    except (TypeError, ValueError):
        direction = None
    if (direction is None or direction.ndim != 1 or not np.all(np.isfinite(direction))
            or not np.any(direction != 0)):
        raise ValueError("expected must be a nonzero vector of finite numbers when up_to_scale=True")
    if isinstance(value, (str, bytes)):
        print(f"{label}: not yet (type your answer without quotes; got {value!r})")
        return False
    if value is None:
        print(f"{label}: no answer yet (replace None with your value)")
        return False
    comma = _decimal_comma(value, expected)
    if comma is not None:
        whole, decimals = comma
        print(f"{label}: not yet ({whole},{decimals} is read by Python as the two numbers "
              f"{whole} and {decimals}: write {whole}.{decimals}, with a point)")
        return False
    try:
        got = np.asarray(value, dtype=complex)
    except (TypeError, ValueError):
        print(f"{label}: not yet (a list of {direction.size} numbers is expected, got {value!r})")
        return False
    if got.size != direction.size:
        print(f"{label}: not yet ({direction.size} value(s) expected, got {got.size})")
        return False
    if got.ndim == 2 and got.shape[1] == 1:               # a column, as SymPy gives it
        got = got[:, 0]
    if got.ndim != 1:
        print(f"{label}: not yet (one list of {direction.size} numbers is expected, "
              "without brackets inside it)")
        return False
    if np.all(got == 0):
        print(f"{label}: not yet (the zero vector is never an eigenvector)")
        return False
    with np.errstate(all="ignore"):
        multiple = np.vdot(direction, got) / np.vdot(direction, direction)
        gap = np.linalg.norm(got - multiple * direction)
        right = bool(np.isfinite(gap) and gap <= tol * np.linalg.norm(got))
    if right:
        print(f"{label}: correct")
        return True
    print(f"{label}: not yet (your vector does not point along an eigenvector, try again)")
    return False


# The words of the "classify" exercises of notebook 18 (Lebl, Table 3.5.1 and Section 8.2.2)
PORTRAIT_WORDS = (KIND_SOURCE, KIND_SINK, KIND_SADDLE, KIND_CENTER, KIND_SPIRAL_SOURCE, KIND_SPIRAL_SINK)
STABILITY_WORDS = (STABILITY_ASYMPTOTIC, STABILITY_STABLE, STABILITY_UNSTABLE)
# read-only (a notebook cell cannot change it by accident): Lebl's other names, then other books'
KIND_SYNONYMS = MappingProxyType({
    "unstable node": KIND_SOURCE, "stable node": KIND_SINK, "centre": KIND_CENTER,
    "center point": KIND_CENTER, "saddle point": KIND_SADDLE,
    "unstable spiral": KIND_SPIRAL_SOURCE, "stable spiral": KIND_SPIRAL_SINK,
    "unstable focus": KIND_SPIRAL_SOURCE, "stable focus": KIND_SPIRAL_SINK})


def _word(text: str) -> str:
    """A word as it is compared: lower case, no spaces around, "-" and "_" read as spaces, single spaces."""
    return " ".join(text.lower().replace("-", " ").replace("_", " ").split())


def check_choice(label: str, value, expected: str, choices: Sequence[str],
                 synonyms: Mapping[str, str] | None = None) -> bool:
    """Check an answer that is a word, such as "saddle", and say so in one line (notebook 18).

    The same manners as ``check``: one line, True or False, the expected word is never shown and an
    answer never raises. The answer is normalised (capitals, spaces around it, "-" and "_" and repeated
    spaces do not matter: " Spiral-sink " is "spiral sink") and then mapped through ``synonyms`` (for
    example KIND_SYNONYMS, Lebl's other names: "stable node" is "sink"). Messages: None -> no answer
    yet; not a string -> type the word in quotes; a word that is not among ``choices`` -> the list of
    the choices (the menu of the exercise, not its answer); otherwise "correct" or "not yet, try
    again". ValueError only for a mistake of the notebook: ``expected`` not among ``choices``, a choice
    that is not text, or ``synonyms`` that is not a mapping of words to words. Choices used by notebook
    18: PORTRAIT_WORDS (the six words of Lebl's Table 3.5.1) and STABILITY_WORDS.
    """
    if isinstance(choices, str) or not all(isinstance(choice, str) for choice in choices):
        raise ValueError(f"choices must be a list of words, got {choices!r}")
    words_only = isinstance(synonyms, Mapping) and all(
        isinstance(key, str) and isinstance(word, str) for key, word in synonyms.items())
    if synonyms is not None and not words_only:
        raise ValueError(f"synonyms must map words to words, got {synonyms!r}")
    table = {_word(key): _word(word) for key, word in (synonyms or {}).items()}
    words = [_word(choice) for choice in choices]
    if not isinstance(expected, str) or table.get(_word(expected), _word(expected)) not in words:
        raise ValueError(f"expected must be one of the choices {list(choices)!r}, got {expected!r}")
    if value is None:
        print(f"{label}: no answer yet (replace None with your answer, a word in quotes)")
        return False
    if not isinstance(value, str):
        print(f'{label}: not yet (type the word in quotes: "...", with your word in place of the dots)')
        return False
    answer = table.get(_word(value), _word(value))
    if answer not in words:
        print(f"{label}: not yet (not one of the words: {', '.join(choices)})")
        return False
    if answer == table.get(_word(expected), _word(expected)):
        print(f"{label}: correct")
        return True
    print(f"{label}: not yet, try again")
    return False


# ---------------------------------------------------------------------------
# The student guide: one text, shown by ol.guide(), by notebook 00 and by the home page of the site
# ---------------------------------------------------------------------------
GUIDE_START = (
    ("Open a notebook",
     "Click its name on the home page. The first time, the browser prepares Python: wait about a minute. "
     "What you see at first (numbers, plots) was stored when the notebook was made: nothing has run yet."),
    ("Run everything, first",
     "Click the button with two triangles (>>) in the bar at the top; a window asks about restarting "
     "the kernel (the Python behind the page): click Confirm Kernel Restart. Or use the menu Run > Run All "
     "Cells. Do this every time you open a notebook: each cell needs the cells above it. The run stops at "
     "the first red message."),
    ("Run one cell",
     "Click inside a grey box (a code cell) and press Shift+Enter, or click the single triangle in the bar "
     "at the top. Enter alone only starts a new line."),
    ("Answer an exercise",
     "Replace None with your answer and press Shift+Enter. The lines to change start with my_ and are above "
     "the line that starts with # ----; in a 'Fill in' exercise it is the line return None, and the cell "
     "below it does the check: run that one too. The reply is 'correct', 'not yet' (with the reason) or "
     "'no answer yet'. Try as many times as you like."),
    ("Keep your work",
     "Your changes stay in this browser on this computer only. To keep a notebook or hand it in: "
     "menu File > Download; the file to hand in is the one that ends with .ipynb."),
)

# on paper, in Python, the usual mistake
GUIDE_TYPING = (
    ("3.5", "3.5", "3,5 (a comma: Python reads the two numbers 3 and 5)"),
    ("2 to the power 10", "2 ** 10", "2^10 (no red message with whole numbers: Python computes something else, here 8)"),
    ("3x,  2(x + 1)", "3 * x,  2 * (x + 1)", "3x,  2(x + 1) (the product is always written)"),
    ("3/4", "3 / 4", ""),
    ("1 over 2x,  x to the power 1/2", "1 / (2 * x),  x ** (1 / 2)", "1 / 2 * x,  x ** 1 / 2 (brackets around a denominator and an exponent)"),
    ("one third, the square root of 2", "1 / 3,  np.sqrt(2)", "0.33,  1.41 (type the exact expression, not a rounded number)"),
    ("square root of 2, in a formula", "sp.sqrt(2)", "sqrt(2)"),
    ("e to the power -2x", "np.exp(-2 * x)   or, in a formula, sp.exp(-2 * x)", "e**(-2x),  exp(-2 * x)"),
    ("ln x", "np.log(x)   or, in a formula, sp.log(x)", "ln(x)"),
    ("sin x, cos x", "np.sin(x), np.cos(x)   or sp.sin(x), sp.cos(x)", "sin x,  sin(x),  sen(x)"),
    ("pi", "np.pi   or, in a formula, sp.pi", "pi"),
    ("a list of numbers", "[0, 2, 1]", "0 2 1"),
    ("yes, no", "True, False", "true, yes, \"True\" (no quotes)"),
    ("infinitely many", "np.inf", "infinity"),
)
GUIDE_TYPING_RULE = ("np. is for numbers, sp. is for formulas that contain a letter such as x or t: an exercise that "
                     "asks for a formula says so, and names the letter. Inside a function (a line that starts with "
                     "return) and in plots use np.: there the letter stands for numbers.")

# what the last line of a red message says, what it means, what to do
GUIDE_ERRORS = (
    ("NameError: name 'np' is not defined   (or 'ol', 'sp', 'plt', 'x', ...)",
     "The cell was run before the cells above it.",
     "Run the whole notebook: the >> button, or Run > Run All Cells."),
    ("NameError: name 'exp' is not defined   (or 'sqrt', 'sin', 'pi', 'e', ...)",
     "A function or a constant was typed without np. or sp. in front.",
     "Write np.exp(2) with numbers, sp.exp(x) in a formula. See 'How to type mathematics'."),
    ("NameError: name 'my_x' is not defined",
     "The line my_x = ... was deleted, or was typed with two = signs.",
     "Type it again above the line # ----, with one = sign: my_x = None."),
    ("NameError: name '...' is not defined   (any other name)",
     "A name is used that Python does not know: often a typing mistake.",
     "Check the spelling, capital letters included."),
    ("SyntaxError: invalid decimal literal",
     "A product without its sign (3x, 2x**2), or a unit after a number (3.5m).",
     "Write 3 * x; type the number without its unit."),
    ("SyntaxError: invalid syntax   or   '(' was never closed",
     "A bracket is missing, nothing is typed after =, or something that is not Python was typed (a word, %, :).",
     "Look at the place marked with ^ under the line, and at the line before it."),
    ("SyntaxError: invalid character ...",
     "A sign was pasted from a document: a long minus, a multiplication cross, a curly quote.",
     "Type the signs - * / and \" on the keyboard."),
    ("IndentationError: unexpected indent",
     "A line starts with a space.",
     "Lines start at the left edge; only the lines inside def or for are moved right, by four spaces."),
    ("TypeError: 'int' object is not callable   (or 'float', 'Symbol', ...)",
     "A product without its sign before a bracket: 2(3 + 1). (Ignore the words about a comma that Python adds.)",
     "Write 2 * (3 + 1)."),
    ("TypeError: unsupported operand type(s) for ^",
     "A power was typed with ^. (With whole numbers there is no message, only a wrong value.)",
     "Write two stars for a power: x ** 3."),
    ("TypeError: loop of ufunc does not support argument ...",
     "A np. function was used in a formula with a letter.",
     "In a formula use sp.exp, sp.sin, sp.sqrt, ..."),
    ("AssertionError   (nothing after it)",
     "You changed a number in an example. The cell compares its result with a value of the book, "
     "and they no longer agree. Nothing is broken.",
     "Put the old number back, or delete the line that starts with assert and go on."),
    ("AssertionError: some words",
     "In a 'Fill in' exercise: the function you completed does not give the right values yet; the words say what failed.",
     "Correct the line return ... in the cell above, run that cell, then this one again."),
    ("... : not yet (...)",
     "This is not an error: the cell ran, and your answer is not the expected one.",
     "Read the words in brackets and try again."),
    ("A cell shows [*] and never finishes",
     "Python is busy, or stuck. (The first time you open the site, a minute or two of [*] is normal: wait.)",
     "Menu Kernel > Restart Kernel, then run everything again. The square button may not stop it in the browser."),
)

# how do I ...
GUIDE_HOW = (
    ("get an empty cell (for ol.guide(), or for an 'Explore' exercise)",
     "click the + in the bar at the top: a new grey box appears under the cell you were in. Type in it, press Shift+Enter."),
    ("type in a cell that says # your code here",
     "a line that starts with # is a comment. Click at its end, press Enter, and type on the new line."),
    ("undo a change",
     "Ctrl+Z (Cmd+Z on a Mac) inside the cell. To clear an answer, type None again."),
    ("repair a text cell that turned into strange symbols ($, **)",
     "you double-clicked it. Press Shift+Enter."),
    ("see what a cell is doing",
     "look at its left: [ ] not run yet, [*] running, a number such as [5] finished."),
    ("stop a cell that never finishes",
     "menu Kernel > Restart Kernel, then run everything again."),
    ("print a notebook",
     "menu File > Print."),
    ("get the original notebook back, or a version corrected by my instructor",
     "the browser keeps showing your copy. Menu File > Open..., right-click the notebook, Delete; "
     "then open it again from the home page."),
)

GUIDE_WORDS = (
    ("cell", "one box of the notebook: text, or code (a grey box)"),
    ("to run a cell", "to make Python execute its code: Shift+Enter"),
    ("output", "what appears under a code cell after it has run: numbers, a plot, a red message"),
    ("kernel", "the Python that runs behind the notebook; 'restart the kernel' means: start it afresh "
               "(what you typed in the cells stays, what Python had computed is forgotten)"),
    ("# comment", "everything after # on a line is for the reader; Python ignores it"),
    ("variable", "a name for a value: minutes = 5"),
    ("None", "'nothing yet': the place where your answer goes"),
    ("np, sp, plt, ol", "the tools every notebook loads: NumPy (numbers), SymPy (formulas), plots, and the course library"),
    ("assert", "a line that stops the cell if a check fails: this is how the notebooks check themselves"),
    ("Trusted, Last Checkpoint", "words of the program at the top of the page: you can ignore them"),
)

GUIDE_TOPICS = {"start": "Five steps", "typing": "How to type mathematics", "errors": "When you see red",
                "how": "How do I ...", "words": "Words"}
GUIDE_WIDTH = 100               # characters per line of the printed guide


def guide(topic: str | None = None) -> None:
    """Print the student guide, or one part: "start", "typing", "errors", "how" or "words".

    The same text is on the home page of the course site and in notebook 00. Type ``ol.guide()``
    in any code cell, at any time; ``ol.guide("errors")`` shows only what the red messages mean.
    """
    import textwrap

    if topic is not None and topic not in GUIDE_TOPICS:
        print(f"ol.guide(): the parts are {', '.join(repr(name) for name in GUIDE_TOPICS)}; "
              "with nothing in the brackets the whole guide is printed")
        return
    wanted = GUIDE_TOPICS if topic is None else {topic: GUIDE_TOPICS[topic]}

    def wrapped(text: str, indent: str) -> str:
        return textwrap.fill(text, GUIDE_WIDTH, initial_indent=indent, subsequent_indent=indent)

    for name, title in wanted.items():
        print(title.upper())
        if name == "start":
            for number, (step, text) in enumerate(GUIDE_START, start=1):
                print(wrapped(f"{number}. {step}. {text}", "  "))
        elif name == "typing":
            for paper, python, mistake in GUIDE_TYPING:
                print(f"  {paper}")
                print(f"      write: {python}")
                if mistake:
                    print(wrapped("not:   " + mistake, "      "))
            print(wrapped(GUIDE_TYPING_RULE, "  "))
        elif name == "errors":
            print("  Read the LAST line of the red message, and the line that starts with 'Hint' next to it.")
            for message, meaning, remedy in GUIDE_ERRORS:
                print(f"  {message}")
                print(wrapped(meaning + " " + remedy, "      "))
        elif name == "how":
            for question, answer in GUIDE_HOW:
                print(wrapped(f"... {question}?", "  "))
                print(wrapped(answer[:1].upper() + answer[1:], "      "))
        else:
            for word, meaning in GUIDE_WORDS:
                print(wrapped(f"{word}: {meaning}", "  "))
        print()


# ---------------------------------------------------------------------------
# A hint next to the usual error messages of a beginner
# ---------------------------------------------------------------------------
HINT_LABEL = "Hint (MATH 331): "
# what a beginner types (in small letters) -> the name in NumPy and SymPy
FUNCTION_NAMES = {"exp": "exp", "sqrt": "sqrt", "sin": "sin", "cos": "cos", "tan": "tan", "log": "log", "ln": "log",
                  "arctan": "arctan", "atan": "arctan", "sinh": "sinh", "cosh": "cosh", "abs": "abs",
                  "sen": "sin", "tg": "tan", "tang": "tan", "arctg": "arctan", "root": "sqrt", "radq": "sqrt"}
CONSTANT_NAMES = {"pi": ("np.pi", "sp.pi"), "e": ("np.e", "sp.E"), "inf": ("np.inf", "sp.oo"),
                  "infinity": ("np.inf", "sp.oo"), "oo": ("np.inf", "sp.oo"), "i": ("1j", "sp.I")}
WORD_NAMES = {"true": "True", "false": "False", "none": "None", "yes": "True", "no": "False",
              "si": "True", "vero": "True", "falso": "False", "t": "True", "f": "False"}
SETUP_NAMES = ("np", "sp", "plt", "ol", "solve_ivp", "interact", "FloatSlider", "IntSlider")
GUIDE_PARTS_TYPED = ("guide", "errors", "typing", "start", "how", "words")
RUN_ALL = "run the whole notebook first (the >> button, or the menu Run > Run All Cells)"
SAMPLE = {"exp": "2", "log": "2", "sqrt": "2", "abs": "-2"}      # a number to show in "np.exp(2)"


def _syntax_hint(message: str) -> str | None:
    if "leading zeros" in message:
        return "a number cannot start with 0. If this is a decimal comma, write 3.05 with a point, not 3,05."
    if "decimal literal" in message or "imaginary literal" in message:
        return ("a product must be written with its sign: 3 * x, not 3x. If it is a unit after a number "
                "(3.5m), type the number only.")
    if any(words in message for words in ("never closed", "unmatched", "does not match", "incomplete input")):
        return "count the brackets near the place marked with ^: every ( needs its )."
    if "unterminated string" in message:
        return ("a quote or an apostrophe has no partner. A derivative is not written y': remove the apostrophe. "
                "Text is written between two quotes, \"like this\".")
    if "invalid character" in message or "non-printable" in message:
        return ("a sign pasted from a document is not a keyboard sign. Type - * / ** and \" on the keyboard, "
                "and delete the spaces that came with the paste.")
    if "cannot assign" in message or "Maybe you meant '=='" in message:
        return "one = sign only, with the name (my_...) on its left: my_x = 3.5, not 3.5 = my_x and not y(x) = ..."
    if "expected ':'" in message:
        return "a line that starts with def or for ends with a colon."
    if "indent" in message:
        return ("lines start at the left edge; only the lines inside def or for are moved right, "
                "by four spaces. Delete the spaces at the start of the line.")
    if message.startswith("invalid syntax"):
        return ("Python cannot read the place marked with ^. The usual causes: nothing after =, a product "
                "without its sign (2 x), a word or a sign Python does not have (%, :, |) after a number, "
                "a missing bracket.")
    return None


def hint_for(error: BaseException) -> str | None:
    """One sentence of advice for the errors a beginner meets most often, or None.

    Used by the notebooks to add a line next to the red message (see ``ol.guide("errors")``);
    it looks only at the kind of error and at its text.
    """
    text = str(error)
    if isinstance(error, SyntaxError):                    # IndentationError is one of them
        return _syntax_hint(error.msg or "")
    if isinstance(error, AssertionError):
        if text:
            return None                                   # the check says itself what failed
        return ("this cell compares its result with a value of the book (the line that starts with "
                "assert). You changed something, so they no longer agree: nothing is broken. Put the old "
                "value back, or delete that line and go on.")
    if isinstance(error, NameError):
        name = text.split("'")[1] if text.count("'") >= 2 else ""
        small = name.lower()
        if name in SETUP_NAMES:
            return f"'{name}' is loaded by the first code cell of the notebook: {RUN_ALL}."
        if name.startswith("my_"):
            return (f"the line {name} = ... is missing, or was typed with two = signs. Type it again above "
                    f"the line # ----, with one = sign, for example {name} = None.")
        if small in FUNCTION_NAMES:
            proper = FUNCTION_NAMES[small]
            sample = SAMPLE.get(proper, "1")
            return (f"write np.{proper}({sample}) with numbers, or sp.{proper}(x) in a formula with a letter"
                    + ("" if small == proper and name == small else f" ({name} is called {proper} here, in small letters)")
                    + ".")
        if small in CONSTANT_NAMES:
            number, symbol = CONSTANT_NAMES[small]
            return f"write {number} with numbers, or {symbol} in a formula."
        if small in WORD_NAMES and name != WORD_NAMES[small]:
            return f"write {WORD_NAMES[small]} (capital letter, no quotes)."
        if name in GUIDE_PARTS_TYPED:
            return "the guide is called with ol. and quotes: ol.guide() or ol.guide(\"errors\")."
        if name == "math":
            return "this course uses np for numbers: np.sqrt(2), np.pi, np.exp(1)."
        if len(name) == 1 and name.isalpha():
            return (f"the letter {name} has no value here. A constant such as C must be found from the "
                    f"initial condition, or left out; in a formula use the letter the exercise names. "
                    f"If the notebook defines {name} above, {RUN_ALL}.")
        return (f"Python does not know the name '{name}'. Either it is a typing mistake (capital letters "
                f"count), or the cell that defines it has not run: {RUN_ALL}.")
    if isinstance(error, TypeError):
        if "'NoneType' object is not callable" in text:
            return "the word None is still there: delete it, and type your answer in its place."
        if "is not callable" in text:
            return "a product must be written with its sign: 2 * (3 + 1), not 2(3 + 1); x * (x + 1), not x(x + 1)."
        if "unsupported operand type(s) for ^" in text:
            return "a power is written with two stars: x ** 3, not x^3."
        if "ufunc" in text and "ellipsis" in text:
            return "replace the three dots with your number: np.sqrt(16)."
        if "ufunc" in text:
            return ("np.exp, np.sin, np.sqrt, ... work with numbers. In a formula that contains a letter "
                    "use sp.exp, sp.sin, sp.sqrt, ...")
        if "truth value of Relational" in text:
            return ("a function that is used for numbers and plots (a line that starts with return) needs "
                    "np.exp, np.sin, ..., not sp.exp, sp.sin, ...")
    if isinstance(error, AttributeError) and ("module 'sympy'" in text or "module 'numpy'" in text):
        name = text.split("'")[3] if text.count("'") >= 4 else ""
        if name.lower() in CONSTANT_NAMES:
            number, symbol = CONSTANT_NAMES[name.lower()]
            return f"the constant is written {number} with numbers, or {symbol} in a formula."
        if name.lower() in FUNCTION_NAMES:
            return f"the function is called {FUNCTION_NAMES[name.lower()]}, in small letters."
        return f"np and sp have nothing called '{name}': check the spelling (see ol.guide(\"typing\"))."
    return None


def _install_hints() -> bool:
    """In a notebook, print ``hint_for`` next to the red message of an error. Returns True when installed.

    A function is registered for IPython's event "post_run_cell", which is called after every
    cell with the result of the run. Nothing about the error itself is changed: the red message
    is shown and the notebook stops exactly as without this library; one line is added.
    """
    try:
        shell = get_ipython()                             # defined by IPython and by Jupyter, not by plain Python
    except NameError:
        return False
    if shell is None or not hasattr(shell, "events"):
        return False
    if getattr(shell, "_math331_hints", False):           # the library was imported twice: once is enough
        return True

    def after_cell(result):
        try:
            error = getattr(result, "error_in_exec", None) or getattr(result, "error_before_exec", None)
            hint = hint_for(error) if error is not None else None
            if hint:
                import sys
                print(HINT_LABEL + hint, file=sys.stderr)
        except Exception:                                 # a hint must never add an error of its own
            pass

    shell.events.register("post_run_cell", after_cell)
    shell._math331_hints = True
    return True


HINTS_INSTALLED = _install_hints()
