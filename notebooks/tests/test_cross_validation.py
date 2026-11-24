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


# --- linear systems x' = A x (notebook 17) ------------------------------------------------------
SYSTEM_CASES = 8                            # random matrices per case
ENTRY_RANGE = (-3, 4)                       # integers from -3 to 3 for the SymPy comparisons
NUMERATOR_RANGE = (-9, 10)                  # rational coefficients k / DENOMINATOR, k from -9 to 9
DENOMINATOR = 4
SYSTEM_END = 2.0


def matrices_of_case(case):
    """Seeded random 2 x 2 matrices of one case (a defective one: lambda I + u w^T with w.u = 0)."""
    rng = np.random.default_rng(SEED)
    found = []
    while len(found) < SYSTEM_CASES:
        if case == ol.CASE_REPEATED_ONE:
            p, q, size, value = rng.integers(*ENTRY_RANGE, 4)
            if (p, q) != (0, 0) and size != 0:
                found.append(value * np.eye(2) + size * np.outer([p, q], [-q, p]))
            continue
        A = rng.integers(*ENTRY_RANGE, (2, 2)).astype(float)
        discriminant = np.trace(A) ** 2 - 4 * np.linalg.det(A)
        if (case == ol.CASE_REAL and discriminant > 0.5) or (case == ol.CASE_COMPLEX and discriminant < -0.5):
            found.append(A)
    return found


@pytest.mark.parametrize("case", [ol.CASE_REAL, ol.CASE_COMPLEX, ol.CASE_REPEATED_ONE])
def test_eigen_solution_agrees_with_solve_ivp(case):
    points = np.linspace(0.0, SYSTEM_END, 21)
    for A in matrices_of_case(case):
        assert ol.two_by_two_case(A).case == case
        start = draws(-1.0, 1.0, 2)
        reference = integrate(lambda s, v, A=A: A @ v, SYSTEM_END, start, points)
        assert ol.eigen_solution(points, A, start) == pytest.approx(reference.T, rel=1e-8, abs=1e-9)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_companion_matrix_has_the_characteristic_polynomial_sympy_finds(order):
    lam = sp.symbols("lambda")
    numerators = np.random.default_rng(SEED + order).integers(*NUMERATOR_RANGE, order + 1)
    coefficients = [sp.Rational(int(value), DENOMINATOR) for value in numerators]
    if coefficients[0] == 0:
        coefficients[0] = sp.Integer(1)
    matrix = sp.Matrix(ol.companion_matrix([float(value) for value in coefficients])).applyfunc(sp.nsimplify)
    polynomial = sum(value * lam ** (order - index) for index, value in enumerate(coefficients))
    expected = polynomial / coefficients[0]
    assert sp.simplify(matrix.charpoly(lam).as_expr() - expected) == 0


def _order(z):
    return (z.real, -z.imag)


@pytest.mark.parametrize("case", [ol.CASE_REAL, ol.CASE_COMPLEX, ol.CASE_REPEATED_ONE])
def test_two_by_two_case_agrees_with_sympy_eigenvects(case):
    for A in matrices_of_case(case):
        found = ol.two_by_two_case(A)
        exact = sp.Matrix(A.astype(int).tolist()).eigenvects()
        values = sorted((complex(value) for value, times, _ in exact for _ in range(times)), key=_order)
        assert sorted(np.array(found.eigenvalues, dtype=complex), key=_order) == pytest.approx(values)
        geometric = sum(len(vectors) for _, _, vectors in exact)
        assert len(found.eigenvectors) == geometric
        for value, vector in zip(found.eigenvalues, found.eigenvectors, strict=False):
            sympy_vector = next(np.array(vectors[0], dtype=complex).ravel()
                                for exact_value, _, vectors in exact
                                if abs(complex(exact_value) - value) < 1e-9)
            k = np.vdot(sympy_vector, vector) / np.vdot(sympy_vector, sympy_vector)
            assert abs(k) > 0 and np.allclose(vector, k * sympy_vector, atol=1e-12)    # parallel to SymPy's


# --- the phase plane and nonlinear systems (notebook 18) ----------------------------------------
RANDOM_MATRICES = 500                       # seeded random integer matrices
PHASE_ENTRY_RANGE = (-5, 6)                 # integers from -5 to 5: singular and repeated ones included
JACOBIAN_POINTS = 5                         # seeded random points per system
POINT_RANGE = (-2.0, 2.0)
JACOBIAN_AGREEMENT = 1e-8                   # relative, as in test_odelib.py
X, Y = sp.symbols("x y", real=True)
# Lebl's systems: Examples 8.1.1, 8.2.1, 8.2.2 and Exercises 8.1.3 (a)-(c), 8.2.1 (a)-(c), 8.2.4
SYSTEMS = {
    "Example 8.1.1": (Y, -X + X ** 2),
    "Example 8.2.1": (-Y - X ** 2, -X + Y ** 2),
    "Example 8.2.2": (Y + Y ** 2 * sp.exp(X), X),
    "Exercise 8.1.3 (a)": (X ** 2 - Y ** 2, X ** 2 + Y ** 2 - 1),
    "Exercise 8.1.3 (b)": (-Y, 3 * X + Y * X ** 2),
    "Exercise 8.1.3 (c)": (X ** 2 + Y, Y ** 2 + X),
    "Exercise 8.2.1 (a)": (-X + 3 * X ** 2, -Y),
    "Exercise 8.2.1 (b)": (X ** 2 + Y ** 2 - 1, X),
    "Exercise 8.2.1 (c)": (Y * sp.exp(X), Y - X + Y ** 2),
    "Exercise 8.2.4": (-X * Y, X ** 2 - 1 - Y),
}


def exact_kind(matrix):
    """The picture of x' = A x from SymPy's exact eigenvalues (Lebl, Table 3.5.1 and the borderlines)."""
    if matrix.is_zero_matrix:
        return ol.KIND_ALL_EQUILIBRIA
    values = [value for value, times in matrix.eigenvals().items() for _ in range(times)]
    real = [sp.re(sp.simplify(value)) for value in values]
    imaginary = [sp.im(sp.simplify(value)) for value in values]
    zeros = [sp.simplify(value) == 0 for value in values]
    if any(zeros):
        return ol.KIND_PARALLEL_LINES if all(zeros) else ol.KIND_LINE_OF_EQUILIBRIA
    if imaginary[0] != 0:
        return (ol.KIND_CENTER if real[0] == 0 else ol.KIND_SPIRAL_SOURCE if real[0] > 0
                else ol.KIND_SPIRAL_SINK)
    if sp.simplify(values[0] - values[1]) == 0:
        star = matrix == values[0] * sp.eye(2)
        if values[0] > 0:
            return ol.KIND_STAR_SOURCE if star else ol.KIND_DEGENERATE_SOURCE
        return ol.KIND_STAR_SINK if star else ol.KIND_DEGENERATE_SINK
    if real[0] * real[1] < 0:
        return ol.KIND_SADDLE
    return ol.KIND_SOURCE if real[0] > 0 else ol.KIND_SINK


def test_classify_linear_2d_agrees_with_sympy_on_random_integer_matrices():
    matrices = np.random.default_rng(SEED).integers(*PHASE_ENTRY_RANGE, (RANDOM_MATRICES, 2, 2))
    kinds = set()
    for A in matrices:
        expected = exact_kind(sp.Matrix(A.tolist()))
        assert ol.classify_linear_2d(A).kind == expected, A
        kinds.add(expected)
    assert {ol.KIND_SADDLE, ol.KIND_SOURCE, ol.KIND_SINK, ol.KIND_SPIRAL_SOURCE, ol.KIND_SPIRAL_SINK,
            ol.KIND_CENTER, ol.KIND_LINE_OF_EQUILIBRIA} <= kinds          # the draw reaches the main kinds


@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_jacobian_agrees_with_sympy(name):
    f, g = SYSTEMS[name]
    exact = sp.lambdify((X, Y), sp.Matrix([f, g]).jacobian([X, Y]), "numpy")
    field = sp.lambdify((X, Y), (f, g), "numpy")
    for x, y in draws(*POINT_RANGE, (JACOBIAN_POINTS, 2)):
        expected = np.array(exact(x, y), dtype=float)
        assert ol.jacobian(field, (x, y)) == pytest.approx(expected, rel=JACOBIAN_AGREEMENT,
                                                           abs=JACOBIAN_AGREEMENT)


@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_classify_equilibrium_2d_agrees_with_sympy(name):
    f, g = SYSTEMS[name]
    field = sp.lambdify((X, Y), (f, g), "numpy")
    jacobian = sp.Matrix([f, g]).jacobian([X, Y])
    points = [point for point in sp.solve([f, g], [X, Y], dict=True)
              if point[X].is_real and point[Y].is_real]
    assert points, name
    for point in points:
        exact_j = sp.simplify(jacobian.subs(point))
        kind = exact_kind(exact_j)
        found = ol.classify_equilibrium_2d(field, (float(point[X]), float(point[Y])))
        assert found.linear.kind == kind, (name, point)
        repelling_line = kind == ol.KIND_LINE_OF_EQUILIBRIA and exact_j.trace() > 0     # unstable (Lyapunov)
        inconclusive = not repelling_line and kind in (ol.KIND_CENTER, ol.KIND_LINE_OF_EQUILIBRIA,
                                                       ol.KIND_PARALLEL_LINES, ol.KIND_ALL_EQUILIBRIA)
        assert (found.stability == ol.STABILITY_INCONCLUSIVE) == inconclusive


# --- the Laplace transform (notebook 19; Lebl, Sections 6.1 and 6.2) --------------------------------
ROW_AGREEMENT = 1e-12                       # two ways of computing one formula of the table (relative)
LAPLACE_SETS = 4                            # seeded random sets of constants (a, omega, n)
LAPLACE_OFFSETS = (0.5, 1.5, 4.0)           # each transform is compared at s = s_min + offset
LAPLACE_QUARTERS = (-8, 9)                  # a is drawn among -8/4, ..., 8/4 and omega among 1/4, ..., 8/4
LAPLACE_POWERS = (0, 7)                     # n is drawn among 0, ..., 6
LAPLACE_TIMES = np.linspace(0.0, 3.0, 13)   # where two functions of t are compared


def promised(found, exact, s):
    """True when found is as near to exact as laplace_numeric promises: rtol max(1 / max(|s|, 1), |F(s)|)."""
    return abs(found - exact) <= ol.LAPLACE_RTOL * max(1 / max(abs(s), ol.LAPLACE_UNIT_RATE), abs(exact))
S = sp.symbols("s")
T_POSITIVE = sp.symbols("t", positive=True)     # for inverse transforms: SymPy then drops its step function


def laplace_constants():
    """The constants of the notebook's table, then seeded random ones (exact fractions for SymPy)."""
    rng = np.random.default_rng(SEED)
    sets = [(sp.Rational(ol.LAPLACE_DEMO_A), sp.Rational(ol.LAPLACE_DEMO_OMEGA), ol.LAPLACE_DEMO_N)]
    for _ in range(LAPLACE_SETS):
        a = sp.Rational(int(rng.integers(*LAPLACE_QUARTERS)), 4)
        omega = sp.Rational(int(rng.integers(1, LAPLACE_QUARTERS[1])), 4)
        sets.append((a, omega, int(rng.integers(*LAPLACE_POWERS))))
    return sets


def table_in_sympy(a, omega, n):
    """The functions f(t) of Lebl's Table 6.1.1 as SymPy expressions, by the names of ol.laplace_table."""
    return {"1": sp.Integer(1), "t": t, "t^2": t ** 2, "t^n": t ** n, "e^(-at)": sp.exp(-a * t),
            "sin": sp.sin(omega * t), "cos": sp.cos(omega * t), "sinh": sp.sinh(omega * t),
            "cosh": sp.cosh(omega * t)}


@pytest.mark.parametrize("a, omega, n", laplace_constants())
def test_laplace_table_agrees_with_sympy_laplace_transform(a, omega, n):
    functions = table_in_sympy(a, omega, n)
    table = ol.laplace_table(a=float(a), omega=float(omega), n=n)
    assert [row.name for row in table] == list(functions)
    for row in table:
        transform, edge, _ = sp.laplace_transform(functions[row.name], t, S)
        assert float(edge) == pytest.approx(row.s_min)              # SymPy: the transform holds for s > edge
        exact = sp.lambdify(S, transform, "numpy")
        for offset in LAPLACE_OFFSETS:
            s = row.s_min + offset
            assert row.transform(s) == pytest.approx(float(exact(s)), rel=ROW_AGREEMENT)
            assert promised(ol.laplace_numeric(row.time, s), float(exact(s)), s)
        assert row.time(LAPLACE_TIMES) == pytest.approx(
            sp.lambdify(t, functions[row.name], "numpy")(LAPLACE_TIMES) + 0 * LAPLACE_TIMES)


# F(s), its partial fractions as printed, and its inverse transform f(t) as printed
INVERSES = {
    "Example 6.1.6": ((S ** 2 + S + 1) / (S ** 3 + S), 1 / S + 1 / (S ** 2 + 1), 1 + sp.sin(T_POSITIVE)),
    "Example 6.1.7": (1 / (S ** 2 + 4 * S + 8), 1 / ((S + 2) ** 2 + 4),
                      sp.exp(-2 * T_POSITIVE) * sp.sin(2 * T_POSITIVE) / 2),
    "Example 6.2.1": (S / ((S ** 2 + 1) * (S ** 2 + 4)) + 1 / (S ** 2 + 1),
                      S / (3 * (S ** 2 + 1)) - S / (3 * (S ** 2 + 4)) + 1 / (S ** 2 + 1),
                      sp.cos(T_POSITIVE) / 3 - sp.cos(2 * T_POSITIVE) / 3 + sp.sin(T_POSITIVE)),
    "Example 6.2.4": (1 / (S * (S ** 2 + 1)), 1 / S - S / (S ** 2 + 1), 1 - sp.cos(T_POSITIVE)),
    "Exercise 6.1.13": (8 / (S ** 3 * (S + 2)), 4 / S ** 3 - 2 / S ** 2 + 1 / S - 1 / (S + 2),
                        2 * T_POSITIVE ** 2 - 2 * T_POSITIVE + 1 - sp.exp(-2 * T_POSITIVE)),
    "Exercise 6.1.7": (1 / ((S - 1) ** 2 * (S + 1)),
                       -1 / (4 * (S - 1)) + 1 / (2 * (S - 1) ** 2) + 1 / (4 * (S + 1)),
                       (2 * T_POSITIVE - 1) * sp.exp(T_POSITIVE) / 4 + sp.exp(-T_POSITIVE) / 4),
    # the circuit of notebook 19: q'' + 2 q' + 5 q = 8 e^(-t), q(0) = 1, q'(0) = 0
    "notebook 19, circuit": ((8 / (S + 1) + S + 2) / (S ** 2 + 2 * S + 5),
                             2 / (S + 1) - S / (S ** 2 + 2 * S + 5),
                             sp.exp(-T_POSITIVE) * (2 - sp.cos(2 * T_POSITIVE) + sp.sin(2 * T_POSITIVE) / 2)),
    # the same circuit with R = 6 ohm: the root -1 of the source is also a root of the circuit
    "notebook 19, R = 6": ((8 / (S + 1) + S + 6) / (S ** 2 + 6 * S + 5),
                           3 / (4 * (S + 1)) + 2 / (S + 1) ** 2 + 1 / (4 * (S + 5)),
                           (sp.Rational(3, 4) + 2 * T_POSITIVE) * sp.exp(-T_POSITIVE)
                           + sp.exp(-5 * T_POSITIVE) / 4),
    # and with a battery of 8 V in place of the fading source (Exercise 3(v))
    "notebook 19, battery": ((8 / S + S + 2) / (S ** 2 + 2 * S + 5),
                             8 / (5 * S) - 3 * (S + 2) / (5 * (S ** 2 + 2 * S + 5)),
                             sp.Rational(8, 5) - sp.Rational(3, 5) * sp.exp(-T_POSITIVE) * (
                                 sp.cos(2 * T_POSITIVE) + sp.sin(2 * T_POSITIVE) / 2)),
    # Exercise 2(d) of notebook 19 (Lebl, Exercise 6.2.4 with numbers):
    # x'' + 2 x' + 10 x = 0, x(0) = 2, x'(0) = 1
    "notebook 19, spring": ((2 * S + 5) / (S ** 2 + 2 * S + 10),
                            2 * (S + 1) / ((S + 1) ** 2 + 9) + 3 / ((S + 1) ** 2 + 9),
                            sp.exp(-T_POSITIVE) * (2 * sp.cos(3 * T_POSITIVE) + sp.sin(3 * T_POSITIVE))),
}


@pytest.mark.parametrize("name", sorted(INVERSES))
def test_the_inverse_transforms_of_notebook_19_agree_with_sympy(name):
    transform, fractions, function = INVERSES[name]
    assert sp.simplify(sp.apart(transform, S) - fractions) == 0          # the partial fractions as printed
    inverse = sp.inverse_laplace_transform(transform, S, T_POSITIVE)
    points = LAPLACE_TIMES[1:]                                           # t > 0
    assert sp.lambdify(T_POSITIVE, inverse, "numpy")(points) == pytest.approx(
        sp.lambdify(T_POSITIVE, function, "numpy")(points))
    forward = sp.lambdify(T_POSITIVE, function, "numpy")                 # and the answer transformed forward
    edge = max([float(sp.re(root)) for root in sp.roots(sp.denom(sp.together(transform)), S)])
    for offset in LAPLACE_OFFSETS:
        s = edge + offset
        assert promised(ol.laplace_numeric(forward, s), float(transform.subs(S, s)), s)


@pytest.mark.parametrize("coefficients, right, start, name", [
    ((1, 0, 1), sp.cos(2 * t), (0, 1), "Example 6.2.1"),             # x'' + x = cos 2t, x(0) = 0, x'(0) = 1
    ((1, 2, 5), 8 * sp.exp(-t), (1, 0), "notebook 19, circuit"),
    ((1, 6, 5), 8 * sp.exp(-t), (1, 0), "notebook 19, R = 6"),
    ((1, 2, 5), 8 + 0 * t, (1, 0), "notebook 19, battery"),
    ((1, 2, 10), 0 * t, (2, 1), "notebook 19, spring"),
])
def test_the_initial_value_problems_of_notebook_19_against_dsolve_and_forced_response(coefficients, right,
                                                                                      start, name):
    a, b, c = coefficients
    equation = sp.Eq(a * y(t).diff(t, 2) + b * y(t).diff(t) + c * y(t), right)
    exact = closed_form(equation, {y(0): start[0], y(t).diff(t).subs(t, 0): start[1]})
    printed = sp.lambdify(T_POSITIVE, INVERSES[name][2], "numpy")
    assert printed(LAPLACE_TIMES) == pytest.approx(exact(LAPLACE_TIMES))
    # the third, independent answer of the notebook: variation of parameters by quadrature
    forcing = sp.lambdify(t, right, "numpy")
    assert ol.forced_response(LAPLACE_TIMES, a, b, c, forcing, *start) == pytest.approx(exact(LAPLACE_TIMES))


# --- steps, impulses and convolution (notebook 20; Lebl, Sections 6.2 to 6.4) -----------------------
TAU = sp.symbols("tau", positive=True)      # the variable inside a convolution
CONVOLUTION_TIMES = np.linspace(0.0, 4.0, 9)
STEP_S = (0.5, 1.0, 4.0)                    # the values of s at which two transforms are compared
STEP_AGREEMENT = 1e-12                      # two formulas for one number (relative)
QUADRATURE_AGREEMENT = 1e-7                 # forced_response: one quad at its default tolerance
DELAYS = (sp.Rational(1, 2), 1, 2)          # the delays a of the second shifting property
FIRING_TIMES = (0.5, 2.0, 4.0, 2 * np.pi, 8.0)      # seconds for which the rocket of Example 6.2.2 fires
PULSE_WIDTHS = (2.0, 1.0, 0.5, 0.05)        # the widths b of the unit pulse of notebook 20
EQUATION_RANGES = ((1, 5), (0, 5), (1, 10))         # whole a, b, c of a y'' + b y' + c y (high excluded)
POWER_RANGE = (0, 3)                        # f = t^n e^(-p t) cos(w t): n among 0, 1, 2
RATE_RANGE = (-1, 3)                        # p among -1, 0, 1, 2
FREQUENCY_RANGE = (0, 4)                    # w among 0, 1, 2, 3
SEEDED_CONVOLUTIONS = 3                     # seeded pairs that SymPy integrates (each costs about a second)


def unit_step(argument):
    """Lebl's unit step in SymPy, with the value 1 at the jump (the u of notebook 20)."""
    return sp.Heaviside(argument, 1)


def as_function(expression):
    """A SymPy formula in T_POSITIVE as a function of a number or an array."""
    return sp.lambdify(T_POSITIVE, expression, "numpy")


# f, g and the convolution f * g as printed in the book (or in notebook 20)
_T = T_POSITIVE
CONVOLUTIONS = {
    "Example 6.3.1": (sp.exp(_T), _T, sp.exp(_T) - _T - 1),
    "Example 6.3.2, omega = 3": (sp.sin(3 * _T), sp.cos(3 * _T), _T * sp.sin(3 * _T) / 2),
    "Example 6.3.3": (_T, sp.exp(-_T), sp.exp(-_T) + _T - 1),
    "Exercise 6.3.10": (sp.cos(_T), sp.exp(-_T), (sp.cos(_T) + sp.sin(_T) - sp.exp(-_T)) / 2),
    "Exercise 6.3.11": (5 * _T, sp.sin(_T), 5 * _T - 5 * sp.sin(_T)),
    "Exercise 6.3.12": (sp.sin(_T), sp.sin(_T), (sp.sin(_T) - _T * sp.cos(_T)) / 2),
    "Exercise 6.4.11": (sp.cos(_T), sp.sin(_T), _T * sp.sin(_T) / 2),
    # notebook 20: the force 8 e^(-t) on the hammer's system, whose impulse response is e^(-t) sin(2t) / 2
    "notebook 20, fading force": (8 * sp.exp(-_T), sp.exp(-_T) * sp.sin(2 * _T) / 2,
                                  2 * sp.exp(-_T) * (1 - sp.cos(2 * _T))),
    "notebook 20, 1 * 1": (sp.Integer(1), sp.Integer(1), _T),
    "notebook 20, resonance": (sp.cos(2 * _T), sp.sin(2 * _T) / 2, _T * sp.sin(2 * _T) / 4),
}


def by_the_definition(f, g):
    """(f * g)(t) from formula (6.3.1), integrated by SymPy."""
    return sp.integrate(f.subs(_T, TAU) * g.subs(_T, _T - TAU), (TAU, 0, _T))


@pytest.mark.parametrize("name", sorted(CONVOLUTIONS))
def test_the_convolutions_of_notebook_20_agree_with_sympy(name):
    f, g, printed = CONVOLUTIONS[name]
    assert sp.simplify(by_the_definition(f, g) - printed) == 0
    points = CONVOLUTION_TIMES
    expected = as_function(printed)(points) + 0 * points
    found = ol.convolution(as_function(f), as_function(g), points)
    assert np.all(np.abs(found - expected) <= ol.CONVOLUTION_RTOL * np.maximum(1, np.abs(expected)))
    # Theorem 6.3.1: the transform of the convolution is the product of the transforms
    transforms = [sp.laplace_transform(part, _T, S, noconds=True) for part in (f, g, printed)]
    assert sp.simplify(transforms[2] - transforms[0] * transforms[1]) == 0


def test_convolution_agrees_with_sympy_on_seeded_functions():
    rng = np.random.default_rng(SEED)
    for _ in range(SEEDED_CONVOLUTIONS):
        n, p, w = (int(rng.integers(*limits)) for limits in (POWER_RANGE, RATE_RANGE, FREQUENCY_RANGE))
        q, v = (int(rng.integers(*limits)) for limits in (RATE_RANGE, FREQUENCY_RANGE))
        f = _T ** n * sp.exp(-p * _T) * sp.cos(w * _T)
        g = sp.exp(-q * _T) * sp.sin(v * _T) + 1
        expected = as_function(by_the_definition(f, g))(CONVOLUTION_TIMES[1:])
        found = ol.convolution(as_function(f), as_function(g), CONVOLUTION_TIMES[1:])
        assert np.all(np.abs(found - expected) <= ol.CONVOLUTION_RTOL * np.maximum(1, np.abs(expected)))


@pytest.mark.parametrize("delay", DELAYS)
def test_the_second_shifting_property_agrees_with_sympy(delay):
    # (6.2.2): L{f(t - a) u(t - a)} = e^(-as) L{f(t)}, for f = t^2, e^(-2t), sin 3t, 1
    for f in (_T ** 2, sp.exp(-2 * _T), sp.sin(3 * _T), sp.Integer(1)):
        delayed = f.subs(_T, _T - delay) * unit_step(_T - delay)
        by_sympy = sp.laplace_transform(delayed, _T, S, noconds=True)
        by_rule = sp.exp(-delay * S) * sp.laplace_transform(f, _T, S, noconds=True)
        for s in STEP_S[:2]:                                # two values of s are enough here
            exact = complex(by_rule.subs(S, s))
            assert complex(by_sympy.subs(S, s)) == pytest.approx(exact, rel=STEP_AGREEMENT)
            found = ol.laplace_numeric(as_function(delayed), s, breaks=[float(delay)])
            assert promised(found, exact.real, s)


def test_t_times_a_step_against_sympy():
    # notebook 20, Section 3: L{t u(t - 1)} = e^(-s) (1/s^2 + 1/s), not e^(-s)/s^2
    by_sympy = sp.laplace_transform(_T * unit_step(_T - 1), _T, S, noconds=True)
    assert sp.simplify(by_sympy - sp.exp(-S) * (1 / S ** 2 + 1 / S)) == 0
    assert sp.simplify(by_sympy - sp.exp(-S) / S ** 2) != 0
    with pytest.raises(ValueError, match="in a formula typed with SymPy write u"):
        ol.step(_T, 1)                                      # ol.step is for numbers, not for SymPy's letters
    with pytest.raises(ValueError, match="in a formula typed with SymPy write u"):
        ol.step(_T - 1)
    # a SymPy number is a number
    assert float(ol.step(sp.pi, 3)) == 1.0 and float(ol.step(sp.Integer(2), 3)) == 0.0
    assert float(ol.step(sp.Rational(7, 2), sp.Rational(7, 2))) == 1.0
    # Exercise 1(b) of the notebook: L{t^2 u(t - 1)} = e^(-s) (2/s^3 + 2/s^2 + 1/s)
    squared = sp.laplace_transform(_T ** 2 * unit_step(_T - 1), _T, S, noconds=True)
    assert sp.simplify(squared - sp.exp(-S) * (2 / S ** 3 + 2 / S ** 2 + 1 / S)) == 0


def test_the_rocket_of_example_6_2_2_against_sympy():
    assert sp.simplify(sp.apart(1 / (S * (S ** 2 + 1)), S) - (1 / S - S / (S ** 2 + 1))) == 0
    assert sp.simplify(sp.inverse_laplace_transform(1 / (S * (S ** 2 + 1)), S, _T) - (1 - sp.cos(_T))) == 0
    transform = (sp.exp(-S) - sp.exp(-5 * S)) / (S * (S ** 2 + 1))
    printed = (1 - sp.cos(_T - 1)) * unit_step(_T - 1) - (1 - sp.cos(_T - 5)) * unit_step(_T - 5)
    points = np.linspace(0.0, 12.0, 49)
    by_sympy = as_function(sp.inverse_laplace_transform(transform, S, _T))(points)
    assert by_sympy == pytest.approx(as_function(printed)(points))
    force = as_function(unit_step(_T - 1) - unit_step(_T - 5))
    by_parameters = ol.forced_response(points[::4], 1, 0, 1, lambda time: float(force(time)))
    assert by_parameters == pytest.approx(as_function(printed)(points[::4]), abs=QUADRATURE_AGREEMENT)
    # after the stop, for t >= 5: cos(t - 5) - cos(t - 1) = 2 sin 2 sin(t - 3)
    after = sp.cos(_T - 5) - sp.cos(_T - 1)
    assert sp.simplify(sp.expand_trig(after - 2 * sp.sin(2) * sp.sin(_T - 3))) == 0


@pytest.mark.parametrize("duration", FIRING_TIMES)
def test_the_amplitude_after_a_firing_of_any_duration(duration):
    # Lebl, Exercise 6.2.10: fired from t = 1 for d seconds, then x = 2 sin(d/2) sin(t - 1 - d/2)
    off = 1 + duration
    equation = sp.Eq(y(t).diff(t, 2) + y(t), 1)             # while the rocket fires, from rest at t = 1
    firing = sp.dsolve(equation, y(t), ics={y(1): 0, y(t).diff(t).subs(t, 1): 0}).rhs
    at_stop = [float(firing.subs(t, off)), float(firing.diff(t).subs(t, off))]
    assert float(np.hypot(*at_stop)) == pytest.approx(2 * abs(np.sin(duration / 2)), abs=STEP_AGREEMENT)
    points = off + np.linspace(0.0, 7.0, 15)
    free = at_stop[0] * np.cos(points - off) + at_stop[1] * np.sin(points - off)
    as_one_sine = 2 * np.sin(duration / 2) * np.sin(points - 1 - duration / 2)
    assert free == pytest.approx(as_one_sine, abs=STEP_AGREEMENT)
    assert free == pytest.approx(np.cos(points - off) - np.cos(points - 1), abs=STEP_AGREEMENT)


def test_the_unit_pulse_of_notebook_20_and_its_limits():
    width = sp.symbols("b", positive=True)
    # its transform (1 - e^(-bs))/(bs) tends to 1, and the amplitude (2/b) sin(b/2) after it tends to 1
    transform = sp.laplace_transform((1 - unit_step(_T - width)) / width, _T, S, noconds=True)
    assert sp.simplify(transform - (1 - sp.exp(-width * S)) / (width * S)) == 0
    assert sp.limit((1 - sp.exp(-width * S)) / (width * S), width, 0) == 1
    assert sp.limit(2 / width * sp.sin(width / 2), width, 0) == 1
    for b in PULSE_WIDTHS:                                  # x'' + x = (u(t) - u(t - b))/b from rest
        points = b + np.linspace(0.0, 7.0, 15)
        after = (np.cos(points - b) - np.cos(points)) / b   # the two delayed pieces, for t >= b
        assert after == pytest.approx(2 / b * np.sin(b / 2) * np.sin(points - b / 2), abs=STEP_AGREEMENT)
        by_convolution = ol.convolution(lambda time, b=b: (1.0 - ol.step(time, b)) / b, np.sin, points,
                                        breaks=[b])
        assert by_convolution == pytest.approx(after, abs=ol.CONVOLUTION_RTOL)


def test_the_hammer_of_notebook_20_against_sympy():
    # x'' + 2x' + 5x = 3 delta(t) from rest: X(s) = 3/(s^2 + 2s + 5), x = (3/2) e^(-t) sin 2t
    printed = sp.Rational(3, 2) * sp.exp(-_T) * sp.sin(2 * _T)
    assert sp.simplify(sp.inverse_laplace_transform(3 / (S ** 2 + 2 * S + 5), S, _T) - printed) == 0
    assert sp.laplace_transform(sp.DiracDelta(_T - 2), _T, S, noconds=True) == sp.exp(-2 * S)
    # the same function is the free motion with x(0) = 0 and x'(0) = P/m = 3
    equation = sp.Eq(y(t).diff(t, 2) + 2 * y(t).diff(t) + 5 * y(t), 0)
    free = closed_form(equation, {y(0): 0, y(t).diff(t).subs(t, 0): 3})
    assert free(LAPLACE_TIMES) == pytest.approx(as_function(printed)(LAPLACE_TIMES))
    assert 3 * ol.impulse_response(LAPLACE_TIMES, 1, 2, 5) == pytest.approx(free(LAPLACE_TIMES))
    assert printed.subs(_T, 0) == 0 and printed.diff(_T).subs(_T, 0) == 3
    # the largest displacement: where tan 2t = 2
    peak = sp.atan(2) / 2
    assert sp.simplify(printed.diff(_T).subs(_T, peak)) == 0
    assert float(printed.subs(_T, peak)) == pytest.approx(0.7713, abs=1e-4)


def test_impulse_response_agrees_with_dsolve_and_with_the_transfer_function():
    rng = np.random.default_rng(SEED)
    for _ in range(CASES):
        a, b, c = (int(rng.integers(*limits)) for limits in EQUATION_RANGES)
        equation = sp.Eq(a * y(t).diff(t, 2) + b * y(t).diff(t) + c * y(t), 0)
        exact = closed_form(equation, {y(0): 0, y(t).diff(t).subs(t, 0): sp.Rational(1, a)})
        points = LAPLACE_TIMES
        assert ol.impulse_response(points, a, b, c) == pytest.approx(exact(points) + 0 * points)
        inverse = sp.inverse_laplace_transform(1 / (a * S ** 2 + b * S + c), S, _T)       # of H(s)
        assert as_function(inverse)(points[1:]) == pytest.approx(exact(points[1:]))


def test_the_exercises_of_notebook_20_against_sympy():
    points = np.linspace(0.0, 6.0, 25)
    # Lebl, Exercise 6.2.6: x'' + x = u(t - 1) from rest gives (1 - cos(t - 1)) u(t - 1)
    answer_626 = (1 - sp.cos(_T - 1)) * unit_step(_T - 1)
    inverse = sp.inverse_laplace_transform(sp.exp(-S) / (S * (S ** 2 + 1)), S, _T)
    assert as_function(inverse)(points) == pytest.approx(as_function(answer_626)(points))
    found = ol.convolution(lambda time: ol.step(time, 1), np.sin, points)
    assert found == pytest.approx(as_function(answer_626)(points), abs=ol.CONVOLUTION_RTOL)
    # Lebl, Exercise 6.4.3: x'' + 4x = 4 delta(t - 1) from rest gives 2 sin(2 (t - 1)) u(t - 1)
    answer_643 = 2 * sp.sin(2 * (_T - 1)) * unit_step(_T - 1)
    inverse = sp.inverse_laplace_transform(4 * sp.exp(-S) / (S ** 2 + 4), S, _T)
    assert as_function(inverse)(points) == pytest.approx(as_function(answer_643)(points))
    assert 4 * ol.impulse_response(points - 1, 1, 0, 4) == pytest.approx(as_function(answer_643)(points))
    # Lebl, Exercise 6.2.14 (answer printed): x'' - x = (t^2 - 1) u(t - 1), x(0) = 1, x'(0) = 2
    before = -sp.exp(-_T) / 2 + 3 * sp.exp(_T) / 2
    after = 2 * sp.exp(_T - 1) - _T ** 2 - 1 + before
    assert sp.simplify(before.diff(_T, 2) - before) == 0
    assert sp.simplify(after.diff(_T, 2) - after - (_T ** 2 - 1)) == 0
    assert before.subs(_T, 0) == 1 and before.diff(_T).subs(_T, 0) == 2
    assert sp.simplify((after - before).subs(_T, 1)) == 0                   # no jump of x at t = 1
    assert sp.simplify((after - before).diff(_T).subs(_T, 1)) == 0          # and none of x'
    # Exercise 3(v) of the notebook: a second blow at t = pi stops the mass of x'' + x
    later = np.pi + np.linspace(0.0, 6.0, 13)
    both = ol.impulse_response(later, 1, 0, 1) + ol.impulse_response(later - np.pi, 1, 0, 1)
    assert both == pytest.approx(0.0, abs=STEP_AGREEMENT)
    # Exercise 3(ii): two firings of 4 seconds, the second from 1 + 3 pi: the mass ends at rest
    second = 1 + 3 * np.pi
    end = second + 4 + np.linspace(0.0, 6.0, 13)
    switches = ((1, 1), (-1, 5), (1, second), (-1, second + 4))             # on, off, on, off
    twice = sum(sign * (1 - np.cos(end - time)) for sign, time in switches)
    assert twice == pytest.approx(0.0, abs=STEP_AGREEMENT)
