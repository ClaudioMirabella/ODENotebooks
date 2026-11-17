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
Plotting
    slope_field                   short segments of slope f(t, y)             (Lebl 1.2)
    phase_line                    critical points and arrows of x' = f(x)     (Lebl 1.6)
Second-order linear equations
    characteristic_roots          roots of a r^2 + b r + c = 0                (Lebl 2.2)
    constant_coefficient_solution exact y(t) of a y'' + b y' + c y = 0, any a, b, c (Lebl 2.2)
    damping_regime                over-, critically, under-damped, undamped   (Lebl 2.4)
    free_response                 exact x(t) of m x'' + c x' + k x = 0         (Lebl 2.2, 2.4)
    wronskian_values              W = y1 y2' - y1' y2 from sampled values     (notebook 05)
Exercises
    check                         compare an answer found by hand with the expected one
    check_formula                 the same for a formula typed as a SymPy expression

Error handling: invalid arguments raise ``ValueError`` (never a silent NaN), with one
documented exception: ``logistic_solution`` returns NaN for times at or beyond the
blow-up time, where the solution does not exist.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

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
