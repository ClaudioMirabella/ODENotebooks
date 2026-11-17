"""Independent cross-validation of odelib (review of 2026-11-17).

Every analytical function of odelib is compared here with two things it does not use:

* the closed-form solution that SymPy derives from the differential equation itself
  (``sympy.dsolve``), evaluated at parameters drawn at random (fixed seed);
* an independent numerical integration with SciPy (``solve_ivp``, tight tolerances).

The numerical methods of odelib (Euler, Heun, RK4, solution_curve) are checked against the
classical results: orders of convergence 1, 2 and 4, and exactness on polynomials of the
right degree. Values printed in the textbooks are in test_odelib.py; this file does not
repeat them. Skipped where SymPy or SciPy are not installed.

Run from the ode/notebooks folder:  python -m pytest tests/test_cross_validation.py
"""

import numpy as np
import pytest

import odelib as ol

sp = pytest.importorskip("sympy")
solve_ivp = pytest.importorskip("scipy.integrate").solve_ivp

SEED = 331
CASES = 5                                   # random parameter sets per test
TIGHT = {"rtol": 1e-11, "atol": 1e-13}
t = sp.symbols("t")
y = sp.Function("y")


def draws(low, high, size=None):
    """Random numbers in [low, high), the same at every run."""
    return np.random.default_rng(SEED).uniform(low, high, size)


def closed_form(equation, initial_conditions):
    """Right-hand side of the solution SymPy finds for an initial value problem, as a function."""
    solution = sp.dsolve(equation, y(t), ics=initial_conditions)
    return sp.lambdify(t, solution.rhs, "numpy")


def integrate(rhs, t_end, start, points):
    result = solve_ivp(rhs, (0.0, t_end), np.atleast_1d(start), t_eval=points, **TIGHT)
    assert result.success
    return result.y


# --- first-order analytical solutions against dsolve and solve_ivp -----------------------------
@pytest.mark.parametrize("P0, k", list(zip(draws(0.5, 50, CASES), draws(-2, 2, CASES))))
def test_exponential_solution(P0, k):
    points = np.linspace(0.0, 2.0, 9)
    exact = closed_form(sp.Eq(y(t).diff(t), sp.Float(k) * y(t)), {y(0): sp.Float(P0)})
    assert ol.exponential_solution(points, P0, k) == pytest.approx(exact(points), rel=1e-10)
    assert ol.exponential_solution(points, P0, k) == pytest.approx(
        integrate(lambda s, v: k * v, 2.0, P0, points)[0], rel=1e-8)


@pytest.mark.parametrize("x0, k, M", list(zip(draws(0.1, 9, CASES), draws(0.05, 0.5, CASES), draws(2, 12, CASES))))
def test_logistic_solution(x0, k, M):
    points = np.linspace(0.0, 5.0, 11)
    numerical = integrate(lambda s, v: k * v * (M - v), 5.0, x0, points)[0]
    assert ol.logistic_solution(points, x0, k, M) == pytest.approx(numerical, rel=1e-8)
    # the equation itself, by central differences on the formula
    dt = 1e-5
    slope = (ol.logistic_solution(points + dt, x0, k, M) - ol.logistic_solution(points - dt, x0, k, M)) / (2 * dt)
    value = ol.logistic_solution(points, x0, k, M)
    assert slope == pytest.approx(k * value * (M - value), rel=1e-6, abs=1e-9)


def test_logistic_blow_up_time_is_where_the_numerical_solution_escapes():
    x0, k, M = -1.0, 0.3, 4.0                         # negative start: finite-time blow-up
    blow_up = ol.logistic_blowup_time(x0, k, M)
    assert blow_up == pytest.approx(np.log(1 - M / x0) / (k * M))        # from the closed form
    before = integrate(lambda s, v: k * v * (M - v), 0.99 * blow_up, x0, [0.99 * blow_up])[0, 0]
    assert before < -50 * abs(x0)                     # already very large just before it


@pytest.mark.parametrize("x0, x_inf, tau", list(zip(draws(-5, 5, CASES), draws(-5, 5, CASES)[::-1], draws(0.2, 4, CASES))))
def test_relaxation(x0, x_inf, tau):
    points = np.linspace(0.0, 6.0, 13)
    exact = closed_form(sp.Eq(y(t).diff(t), (sp.Float(x_inf) - y(t)) / sp.Float(tau)), {y(0): sp.Float(x0)})
    assert ol.relaxation(points, x0, x_inf, tau) == pytest.approx(exact(points), rel=1e-10, abs=1e-12)
    assert ol.relaxation(points, x0, x_inf, tau) == pytest.approx(
        integrate(lambda s, v: (x_inf - v) / tau, 6.0, x0, points)[0], rel=1e-8, abs=1e-10)


@pytest.mark.parametrize("h0, ratio", list(zip(draws(0.2, 3, CASES), draws(1e-4, 1e-2, CASES))))
def test_torricelli(h0, ratio):
    g = ol.GRAVITY
    drain = ol.torricelli_drain_time(h0, ratio)
    points = np.linspace(0.0, 0.9 * drain, 10)       # before the tank is empty the problem is regular
    numerical = integrate(lambda s, v: -ratio * np.sqrt(2 * g * np.maximum(v, 0.0)), 0.9 * drain, h0, points)[0]
    assert ol.torricelli_height(points, h0, ratio) == pytest.approx(numerical, rel=1e-7)
    # the draining time from the separated equation: integral of dh / sqrt(h) from 0 to h0
    assert drain == pytest.approx(2 * np.sqrt(h0) / (ratio * np.sqrt(2 * g)))


@pytest.mark.parametrize("v_t", draws(3, 80, CASES))
def test_quadratic_drag(v_t):
    g = ol.GRAVITY
    points = np.linspace(0.0, 4 * v_t / g, 9)
    numerical = integrate(lambda s, v: g * (1 - (v / v_t) ** 2), points[-1], 0.0, points)[0]
    assert ol.quadratic_drag_velocity(points, v_t) == pytest.approx(numerical, rel=1e-8, abs=1e-9)
    symbolic = sp.Float(v_t) * sp.tanh(sp.Float(g) * t / sp.Float(v_t))
    residual = sp.simplify(symbolic.diff(t) - sp.Float(g) * (1 - symbolic ** 2 / sp.Float(v_t) ** 2))
    assert abs(float(residual.subs(t, 1.234))) < 1e-9


LINEAR_PROBLEMS = (                                   # p(x), f(x), x0, y0: y' + p y = f
    (lambda s: 2 * s, lambda s: s, 0.0, 3.0, 2 * t, t),
    (lambda s: 1 / (1 + s), lambda s: np.cos(s), 0.0, -1.0, 1 / (1 + t), sp.cos(t)),
    (lambda s: -0.5, lambda s: np.exp(-s), 1.0, 2.0, sp.Float(-0.5), sp.exp(-t)),
    (lambda s: np.sin(s), lambda s: np.sin(s), 0.0, 0.0, sp.sin(t), sp.sin(t)),
)


@pytest.mark.parametrize("p, f, x0, y0, p_sym, f_sym", LINEAR_PROBLEMS)
def test_linear_first_order(p, f, x0, y0, p_sym, f_sym):
    points = np.linspace(x0, x0 + 2.0, 9)
    exact = closed_form(sp.Eq(y(t).diff(t) + p_sym * y(t), f_sym), {y(x0): y0})
    assert ol.linear_first_order(p, f, x0, y0, points) == pytest.approx(exact(points), rel=1e-8, abs=1e-10)
    numerical = solve_ivp(lambda s, v: f(s) - p(s) * v, (x0, points[-1]), [y0], t_eval=points, **TIGHT).y[0]
    assert ol.linear_first_order(p, f, x0, y0, points) == pytest.approx(numerical, rel=1e-8, abs=1e-10)


# --- second-order equations with constant coefficients ------------------------------------------
SECOND_ORDER = (                                      # a, b, c: real distinct, repeated, complex roots
    (1.0, 3.0, 2.0), (2.0, -1.0, -3.0), (1.0, 4.0, 4.0), (1.0, -6.0, 9.0),
    (1.0, 2.0, 5.0), (3.0, 0.0, 12.0), (1.0, 0.4, 9.0),
)


@pytest.mark.parametrize("a, b, c", SECOND_ORDER)
def test_constant_coefficient_solution(a, b, c):
    y0, v0 = 1.5, -0.7
    points = np.linspace(0.0, 2.0, 11)
    a_s, b_s, c_s = (sp.Rational(str(value)) for value in (a, b, c))      # exact fractions: 0.4 -> 2/5
    equation = sp.Eq(a_s * y(t).diff(t, 2) + b_s * y(t).diff(t) + c_s * y(t), 0)
    exact = closed_form(equation, {y(0): sp.Rational(str(y0)), y(t).diff(t).subs(t, 0): sp.Rational(str(v0))})
    ours = ol.constant_coefficient_solution(points, a, b, c, y0, v0)
    assert ours == pytest.approx(np.real(exact(points)), rel=1e-9, abs=1e-10)
    numerical = integrate(lambda s, u: [u[1], -(b * u[1] + c * u[0]) / a], 2.0, [y0, v0], points)[0]
    assert ours == pytest.approx(numerical, rel=1e-7, abs=1e-9)


@pytest.mark.parametrize("a, b, c", SECOND_ORDER)
def test_characteristic_roots(a, b, c):
    ours = sorted(ol.characteristic_roots(a, b, c), key=lambda z: (z.real, z.imag))
    reference = sorted(np.roots([a, b, c]), key=lambda z: (z.real, z.imag))
    assert np.allclose(ours, reference, atol=1e-6)    # a double root is found by np.roots only to ~1e-8
    for root in ours:                                 # each one satisfies a r^2 + b r + c = 0
        assert abs(a * root ** 2 + b * root + c) < 1e-9


@pytest.mark.parametrize("m, c, k, regime", [(1.0, 0.0, 4.0, "undamped"), (1.0, 1.0, 4.0, "underdamped"),
                                             (1.0, 4.0, 4.0, "critically damped"), (2.0, 9.0, 4.0, "overdamped")])
def test_free_response_and_regime(m, c, k, regime):
    assert ol.damping_regime(m, c, k) == regime
    x0, v0 = 0.8, 1.1
    points = np.linspace(0.0, 6.0, 25)
    numerical = integrate(lambda s, u: [u[1], -(c * u[1] + k * u[0]) / m], 6.0, [x0, v0], points)[0]
    assert ol.free_response(points, m, c, k, x0, v0) == pytest.approx(numerical, rel=1e-7, abs=1e-9)


def test_wronskian_values_against_sympy_and_abel():
    # two solutions of y'' + 2y' + 5y = 0: W = y1 y2' - y1' y2 = W(0) exp(-2t)  (Abel)
    y1, y2 = sp.exp(-t) * sp.cos(2 * t), sp.exp(-t) * sp.sin(2 * t)
    points = np.linspace(0.0, 3.0, 13)
    values = [sp.lambdify(t, expr, "numpy")(points) for expr in (y1, y1.diff(t), y2, y2.diff(t))]
    symbolic = sp.lambdify(t, sp.simplify(y1 * y2.diff(t) - y1.diff(t) * y2), "numpy")(points)
    assert ol.wronskian_values(*values) == pytest.approx(symbolic, rel=1e-12)
    assert ol.wronskian_values(*values) == pytest.approx(2 * np.exp(-2 * points), rel=1e-12)


# --- numerical methods: the classical orders -----------------------------------------------------
@pytest.mark.parametrize("method, order", [(ol.euler, 1), (ol.heun, 2), (ol.rk4, 4)])
def test_order_of_convergence(method, order):
    # y' = y cos t, y(0) = 1  ->  y = exp(sin t): smooth, not a polynomial
    def rhs(s, v):
        return v * np.cos(s)
    steps = (0.1, 0.05, 0.025, 0.0125)
    errors = [abs(method(rhs, (0.0, 2.0), [1.0], h)[1][-1] - np.exp(np.sin(2.0))) for h in steps]
    assert ol.observed_order(errors) == pytest.approx(order, abs=0.15)


@pytest.mark.parametrize("method, degree", [(ol.euler, 1), (ol.heun, 2), (ol.rk4, 4)])
def test_methods_are_exact_on_polynomials_of_their_degree(method, degree):
    # y' = d t^(d-1), y(0) = 0  ->  y = t^d: a method of order d has no error here
    grid, values = method(lambda s, v: degree * s ** (degree - 1), (0.0, 2.0), [0.0], 0.25)
    assert values == pytest.approx(grid ** degree, abs=1e-12)
    grid, values = method(lambda s, v: (degree + 1) * s ** degree, (0.0, 2.0), [0.0], 0.25)
    assert abs(values[-1] - 2.0 ** (degree + 1)) > 1e-6          # and it does one degree higher


def test_euler_stability_bound():
    # y' = -y / tau: Euler's iterates are (1 - h/tau)^n; they decay only for h < 2 tau (notebook 13)
    tau = 2.0
    for h, decays in ((1.0, True), (3.0, True), (5.0, False)):
        grid, values = ol.euler(lambda s, v: -v / tau, (0.0, 15.0), [1.0], h)
        assert values == pytest.approx((1 - h / tau) ** np.arange(grid.size), abs=1e-12)
        assert (abs(values[-1]) < 1.0) == decays


@pytest.mark.parametrize("rhs, start", [(lambda s, v: s * v, (0.0, 0.2)), (lambda s, v: s - v, (0.5, -1.0)),
                                        (lambda s, v: np.cos(s) - v ** 3, (-1.0, 0.3))])
def test_solution_curve_against_solve_ivp(rhs, start):
    window = (-2.0, 2.0)
    grid, values = ol.solution_curve(rhs, start, window, (-50.0, 50.0), h=0.01)
    assert grid[0] == pytest.approx(window[0]) and grid[-1] == pytest.approx(window[1])
    for end in window:                                # one integration to the left, one to the right
        reference = solve_ivp(rhs, (start[0], end), [start[1]], **TIGHT).y[0, -1]
        assert np.interp(end, grid, values) == pytest.approx(reference, rel=1e-6, abs=1e-8)


# --- robustness in a classroom: a silly argument must fail quickly, not freeze the browser ------
def test_a_tiny_step_is_refused_instead_of_running_for_ever():
    with pytest.raises(ValueError, match="too many steps"):
        ol.solution_curve(lambda s, v: v, (0.0, 1.0), (-3.0, 3.0), (-3.0, 3.0), h=1e-12)
    with pytest.raises(ValueError, match="too many steps"):
        ol.euler(lambda s, v: v, (0.0, 1.0), [1.0], 1e-9)


# --- forced_response (notebook 14) against an independent numerical integration ----------
FORCINGS = {"tan": (np.tan, 1.2), "sin of t squared": (lambda s: np.sin(s * s), 6.0),
            "a kink": (abs, 3.0), "one over cosh": (lambda s: 1 / np.cosh(s), 3.0)}


@pytest.mark.parametrize("name", sorted(FORCINGS))
@pytest.mark.parametrize("roots", ["real", "repeated", "complex"])
def test_forced_response_agrees_with_solve_ivp(name, roots):
    forcing, t_end = FORCINGS[name]
    a, shift, y0, v0 = draws(0.5, 2.0, 4)
    if roots == "real":
        b, c = a * (2 * shift + 1), a * shift * (shift + 1)          # roots -shift and -shift - 1
    elif roots == "repeated":
        b, c = 2 * a * shift, a * shift * shift                      # the root -shift, twice
    else:
        b, c = 2 * a * shift, a * (shift * shift + 4)                # roots -shift +- 2i
    points = np.linspace(0.0, t_end, 25)
    reference = integrate(lambda s, u: [u[1], (forcing(s) - b * u[1] - c * u[0]) / a], t_end, [y0, v0], points)
    assert ol.forced_response(points, a, b, c, forcing, y0, v0) == pytest.approx(reference[0], rel=1e-7, abs=1e-7)


# --- cosine_forced_response (notebook 15) --------------------------------------------------
@pytest.mark.parametrize("damping", ["none", "light", "critical", "heavy"])
@pytest.mark.parametrize("detuning", [0.0, 1e-6, -0.2, 1.5])
def test_cosine_forced_response_agrees_with_forced_response_and_solve_ivp(damping, detuning):
    m, k, F0, x0, v0 = draws(0.5, 3.0, 5)
    critical = 2 * np.sqrt(m * k)
    c = {"none": 0.0, "light": 0.1 * critical, "critical": critical, "heavy": 2.5 * critical}[damping]
    omega = np.sqrt(k / m) * (1 + detuning)
    points = np.linspace(0.0, 12.0, 25)

    def force(s):
        return F0 * np.cos(omega * s)

    exact = ol.cosine_forced_response(points, m, c, k, F0, omega, x0, v0)
    reference = integrate(lambda s, u: [u[1], (force(s) - c * u[1] - k * u[0]) / m], 12.0, [x0, v0], points)
    assert exact == pytest.approx(reference[0], rel=1e-7, abs=1e-7)
    assert exact == pytest.approx(ol.forced_response(points, m, c, k, force, x0, v0), rel=1e-7, abs=1e-7)


def test_steady_state_is_the_particular_solution_sympy_finds():
    m, c, k, F0, omega = (sp.Rational(str(round(value, 2))) for value in draws(0.5, 3.0, 5))
    A, B = sp.symbols("A B")
    guess = A * sp.cos(omega * t) + B * sp.sin(omega * t)
    residual = sp.expand(m * sp.diff(guess, t, 2) + c * sp.diff(guess, t) + k * guess - F0 * sp.cos(omega * t))
    found = sp.solve([residual.coeff(sp.cos(omega * t)), residual.coeff(sp.sin(omega * t))], [A, B])
    amplitude, phase = ol.steady_state(*(float(value) for value in (m, c, k, F0, omega)))
    assert amplitude == pytest.approx(float(sp.sqrt(found[A] ** 2 + found[B] ** 2)))
    assert phase == pytest.approx(float(sp.atan2(found[B], found[A])))


# --- higher_order_solution and beam_deflection (notebook 16) -------------------------------
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_higher_order_solution_agrees_with_solve_ivp(order):
    numbers = draws(-2.0, 2.0, 2 * order + 1)
    coefficients = np.concatenate([[1.5], numbers[:order]])
    initial = numbers[order + 1:]
    points = np.linspace(0.0, 2.0, 21)

    def rhs(s, u):
        return np.concatenate([u[1:], [-(coefficients[:0:-1] @ u) / coefficients[0]]])

    reference = integrate(rhs, 2.0, initial, points)
    assert ol.higher_order_solution(points, coefficients, initial) == pytest.approx(reference[0], rel=1e-7, abs=1e-7)


def test_higher_order_solution_agrees_with_dsolve_for_order_three():
    a2, a1, a0, y0, v0, w0 = (sp.Rational(str(round(value, 1))) for value in draws(-2.0, 2.0, 6))
    equation = y(t).diff(t, 3) + a2 * y(t).diff(t, 2) + a1 * y(t).diff(t) + a0 * y(t)
    exact = closed_form(equation, {y(0): y0, y(t).diff(t).subs(t, 0): v0, y(t).diff(t, 2).subs(t, 0): w0})
    points = np.linspace(0.0, 1.5, 16)
    computed = ol.higher_order_solution(points, [1, float(a2), float(a1), float(a0)], [float(y0), float(v0), float(w0)])
    assert computed == pytest.approx(np.real(np.asarray(exact(points), dtype=complex)), rel=1e-8, abs=1e-8)


@pytest.mark.parametrize("left, right", [("supported", "supported"), ("fixed", "free"), ("fixed", "fixed"),
                                         ("fixed", "supported"), ("free", "fixed")])
def test_beam_deflection_agrees_with_solve_bvp(left, right):
    solve_bvp = pytest.importorskip("scipy.integrate").solve_bvp
    load = draws(-1.0, 2.0, 3)                               # a quadratic load
    length, rigidity = draws(0.5, 3.0, 2)
    ends = {"supported": (0, 2), "fixed": (0, 1), "free": (2, 3)}

    def rhs(s, u):
        return np.vstack([u[1], u[2], u[3], np.polyval(load, s) / rigidity])

    def boundary(ua, ub):
        return np.array([ua[i] for i in ends[left]] + [ub[i] for i in ends[right]])

    mesh = np.linspace(0.0, length, 41)
    result = solve_bvp(rhs, boundary, mesh, np.zeros((4, mesh.size)), tol=1e-10)
    assert result.success
    assert ol.beam_deflection(mesh, load, left, right, length, rigidity) == pytest.approx(result.sol(mesh)[0], rel=1e-6, abs=1e-8)
