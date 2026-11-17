"""Tests of odelib against values printed in the reference textbooks.

Lebl = J. Lebl, Notes on Diffy Qs: Differential Equations for Engineers (free on LibreTexts)
Adkins-Davidson = W.A. Adkins, M.G. Davidson, Ordinary Differential Equations, Springer 2012
Run from the ode/notebooks folder:  python -m pytest
"""

import numpy as np
import pytest

import odelib as ol


# --- exponential growth and decay --------------------------------------------------------
def test_lebl_example_0_2_1_bacteria():
    # 100 bacteria at t = 0, 200 at t = 10 s  ->  k = ln 2 / 10, P(60) = 6400, P(61) ~ 6859.35
    k = ol.growth_rate_from_two_values(0, 100, 10, 200)
    assert k == pytest.approx(np.log(2) / 10)
    assert ol.exponential_solution(60, 100, k) == pytest.approx(6400)
    assert ol.exponential_solution(61, 100, k) == pytest.approx(6859.35, abs=0.01)


def test_half_life_and_decay_constant_are_inverse():
    assert ol.half_life(ol.decay_constant(5730.0)) == pytest.approx(5730.0)
    # the amount halves after each half-life, whatever N0
    lam = ol.decay_constant(5730.0)
    for N0 in (1.0, 2.0, 37.5):
        assert ol.exponential_solution(5730.0, N0, -lam) == pytest.approx(N0 / 2)


def test_adkins_davidson_example_8_radiocarbon():
    # 75% of carbon-14 left, half-life 5730 years  ->  about 2378 years
    assert ol.age_from_fraction(0.75, 5730) == pytest.approx(2378, abs=0.5)


@pytest.mark.parametrize("bad", [0.0, -1.0, 1.5])
def test_age_rejects_impossible_fractions(bad):
    with pytest.raises(ValueError):
        ol.age_from_fraction(bad, 5730)


# --- logistic equation x' = k x (M - x) -----------------------------------------------------
@pytest.mark.parametrize("x0", [0.5, 2.5, 5.0, 7.0])
def test_logistic_solution_satisfies_the_equation(x0):
    k, M = 0.1, 5.0                                   # Lebl Figure 1.6.2
    t = np.linspace(0, 20, 2001)
    x = ol.logistic_solution(t, x0, k, M)
    dxdt = np.gradient(x, t)
    residual = dxdt - k * x * (M - x)
    assert np.max(np.abs(residual[2:-2])) < 1e-4
    assert x[0] == pytest.approx(x0)
    assert ol.logistic_solution(1e3, x0, k, M) == pytest.approx(M)   # stable equilibrium M


def test_logistic_negative_initial_value_blows_up_in_finite_time():
    k, M, x0 = 0.1, 5.0, -0.5
    t_star = ol.logistic_blowup_time(x0, k, M)
    assert t_star == pytest.approx(np.log((M - x0) / -x0) / (k * M))
    assert ol.logistic_solution(0.999 * t_star, x0, k, M) < -100
    assert np.isnan(ol.logistic_solution(1.01 * t_star, x0, k, M))
    assert ol.logistic_blowup_time(1.0, k, M) == np.inf


def test_lebl_harvesting_equilibria():
    # Lebl Section 1.6: M = 8, k = 0.1;  h = 1 -> B ~ 1.55, A ~ 6.45;  h = 1.6 -> A = B = 4;  h = 2 -> none
    B, A = ol.harvesting_equilibria(0.1, 8, 1.0)
    assert (B, A) == (pytest.approx(1.55, abs=0.01), pytest.approx(6.45, abs=0.01))
    assert ol.harvesting_equilibria(0.1, 8, 1.6) == (pytest.approx(4.0),)
    assert ol.harvesting_equilibria(0.1, 8, 2.0) == ()


# --- numerical methods ------------------------------------------------------------------------
LEBL_TABLE_1_7_1 = {  # Euler for y' = y^2/3, y(0) = 1, approximations of y(2) (exact value 3)
    1: 1.92593, 0.5: 2.20861, 0.25: 2.47250, 0.125: 2.68034,
    0.0625: 2.82040, 0.03125: 2.90412, 0.015625: 2.95035, 0.0078125: 2.97472,
}


@pytest.mark.parametrize(("h", "value"), LEBL_TABLE_1_7_1.items())
def test_euler_reproduces_lebl_table_1_7_1(h, value):
    # Tolerance: one unit in the 5th decimal. For h = 0.25 Euler gives 2.472494, which rounds
    # to 2.47249; the book prints 2.47250 (its own error column, 0.52751, agrees with 2.472494).
    _, y = ol.euler(lambda x, y: y**2 / 3, (0.0, 2.0), 1.0, h)
    assert y[-1] == pytest.approx(value, abs=1e-5)


def test_lebl_exercise_1_7_9_improved_euler():
    # y' = x + y, y(0) = 1, exact y = 2e^x - x - 1, y(1) = 2e - 2 ~ 3.4366
    f = lambda x, y: x + y                                            # noqa: E731
    exact = 2 * np.e - 2
    assert ol.heun(f, (0, 1), 1.0, 0.25)[1][-1] == pytest.approx(3.3897, abs=1e-4)
    assert ol.heun(f, (0, 1), 1.0, 0.125)[1][-1] == pytest.approx(3.4237, abs=1e-4)
    assert ol.euler(f, (0, 1), 1.0, 0.25)[1][-1] == pytest.approx(2.8828, abs=1e-4)
    assert ol.euler(f, (0, 1), 1.0, 0.125)[1][-1] == pytest.approx(3.1316, abs=1e-4)
    assert exact - ol.heun(f, (0, 1), 1.0, 0.25)[1][-1] == pytest.approx(0.046852, abs=1e-6)


@pytest.mark.parametrize(("method", "order"), [(ol.euler, 1), (ol.heun, 2), (ol.rk4, 4)])
def test_observed_orders_of_accuracy(method, order):
    steps = [0.1 / 2**i for i in range(5)]
    errors = [abs(method(lambda t, y: y, (0, 1), 1.0, h)[1][-1] - np.e) for h in steps]
    assert ol.observed_order(errors)[-1] == pytest.approx(order, abs=0.1)


def test_euler_detects_blow_up_instead_of_returning_garbage():
    # Lebl Table 1.7.2: y' = y^2/3 has no solution at x = 3 (it blows up there)
    with pytest.raises(FloatingPointError):
        ol.euler(lambda x, y: y**2 / 3, (0, 30), 1.0, 0.5)


def test_grid_must_contain_whole_number_of_steps():
    with pytest.raises(ValueError):
        ol.euler(lambda t, y: y, (0, 1), 1.0, 0.3)


def test_systems_are_supported():
    # x'' = -x as a system: (x, v)' = (v, -x); after one period the state returns to the start
    t, y = ol.rk4(lambda t, u: np.array([u[1], -u[0]]), (0, 2 * np.pi), [1.0, 0.0], 2 * np.pi / 1000)
    assert y[-1] == pytest.approx([1.0, 0.0], abs=1e-9)


def test_classify_equilibrium():
    assert ol.classify_equilibrium(-0.5) == "stable"
    assert ol.classify_equilibrium(0.5) == "unstable"
    assert ol.classify_equilibrium(0.0) == "inconclusive"


# --- second-order linear equations (notebooks 05-07) -------------------------------------
def test_characteristic_roots_ordering():
    assert ol.characteristic_roots(1, -6, 8) == (4, 2)
    r1, r2 = ol.characteristic_roots(1, -6, 13)                       # 3 +- 2i
    assert r1 == pytest.approx(3 + 2j) and r2 == pytest.approx(3 - 2j)
    with pytest.raises(ValueError):
        ol.characteristic_roots(0, 1, 1)


@pytest.mark.parametrize("coeffs, ic, exact", [
    # Lebl Section 2.2 (opening example): y'' - 6y' + 8y = 0, y(0) = -2, y'(0) = 6  ->  -7 e^{2x} + 5 e^{4x}
    ((1, -6, 8), (-2, 6), lambda x: -7 * np.exp(2 * x) + 5 * np.exp(4 * x)),
    # Lebl Example 2.2.4: y'' - 6y' + 13y = 0, y(0) = 0, y'(0) = 10  ->  5 e^{3x} sin 2x
    ((1, -6, 13), (0, 10), lambda x: 5 * np.exp(3 * x) * np.sin(2 * x)),
    # Adkins-Davidson Sect. 3.3 Example 7: y'' + 2y' + y = 0, y(0) = 2, y'(0) = -3  ->  2 e^{-t} - t e^{-t}
    ((1, 2, 1), (2, -3), lambda x: (2 - x) * np.exp(-x)),
])
def test_constant_coefficient_solution(coeffs, ic, exact):
    x = np.linspace(0, 1.5, 31)
    assert ol.constant_coefficient_solution(x, *coeffs, *ic) == pytest.approx(exact(x), rel=1e-12, abs=1e-12)


def test_lebl_example_2_4_1_amplitude():
    # 2x'' + 8x = 0, x(0) = 0.5, x'(0) = 1: w0 = 2, amplitude sqrt(0.5^2 + (1/2)^2) = sqrt(0.5)
    t = np.linspace(0, np.pi, 20001)                                  # one full period
    x = ol.free_response(t, 2, 0, 8, 0.5, 1)
    assert np.max(np.abs(x)) == pytest.approx(np.sqrt(0.5), rel=1e-6)
    assert ol.damping_regime(2, 0, 8) == "undamped"


@pytest.mark.parametrize("c, regime",
                         [(1.0, "underdamped"), (4.0, "critically damped"), (10.0, "overdamped")])
def test_free_response_matches_rk4_in_every_regime(c, regime):
    # m = 1, k = 4: critical damping at c = 2 sqrt(mk) = 4
    m, k, x0, v0 = 1.0, 4.0, 1.0, -2.0
    assert ol.damping_regime(m, c, k) == regime
    t, u = ol.rk4(lambda t, u: np.array([u[1], -(c * u[1] + k * u[0]) / m]), (0, 5), [x0, v0], 0.001)
    assert ol.free_response(t, m, c, k, x0, v0) == pytest.approx(u[:, 0], abs=1e-10)


def test_free_response_is_continuous_across_critical_damping():
    t = np.linspace(0, 5, 101)
    near = [ol.free_response(t, 1, 4 * (1 + s), 4, 1, 0) for s in (-1e-6, 0, 1e-6)]
    assert near[0] == pytest.approx(near[1], abs=1e-5) and near[2] == pytest.approx(near[1], abs=1e-5)


def test_airy_wronskian_dlmf_9_2_7():
    # W{Ai, Bi} = 1/pi for every x (DLMF 9.2.7); y'' - x y = 0 has p = 0, so Abel says W is constant
    from scipy.special import airy
    x = np.linspace(-8, 3, 50)
    ai, aip, bi, bip = airy(x)
    assert ol.wronskian_values(ai, aip, bi, bip) == pytest.approx(np.full_like(x, 1 / np.pi), rel=1e-10)


def test_bessel_wronskian_dlmf_10_5_2():
    # W{J_nu, Y_nu}(x) = 2/(pi x) (DLMF 10.5.2): Abel with p = 1/x gives W = C/x
    from scipy.special import jv, jvp, yv, yvp
    x = np.linspace(0.5, 30, 60)
    for nu in (0, 1, 2.5):
        w = ol.wronskian_values(jv(nu, x), jvp(nu, x), yv(nu, x), yvp(nu, x))
        assert w == pytest.approx(2 / (np.pi * x), rel=1e-9)


def test_abel_formula_numerically():
    # y'' + (1/(1+t)) y' + y = 0 (variable p): integrate two solutions and compare W with Abel
    # W(t) = W(0) exp(-int_0^t p) = W(0)/(1 + t)
    def f(t, u):
        return np.array([u[1], -u[1] / (1 + t) - u[0], u[3], -u[3] / (1 + t) - u[2]])
    t, u = ol.rk4(f, (0, 10), [1.0, 0.0, 0.0, 1.0], 0.001)
    w = ol.wronskian_values(u[:, 0], u[:, 1], u[:, 2], u[:, 3])
    assert w == pytest.approx(1 / (1 + t), rel=1e-9)


# --- check(): answers found by hand ------------------------------------------------------
def test_check_accepts_a_correct_number(capsys):
    assert ol.check("x", 2.0, 2) is True
    assert capsys.readouterr().out.strip() == "x: correct"


def test_check_rejects_a_wrong_number_without_revealing_the_answer(capsys):
    expected = 0.36787944
    assert ol.check("maximum", 0.5, expected) is False
    out = capsys.readouterr().out
    assert "not yet" in out and "0.5" in out and str(expected) not in out


def test_check_missing_answer(capsys):
    assert ol.check("x", None, 2.0) is False
    assert "no answer yet" in capsys.readouterr().out


def test_check_tolerance():
    exact = 1 / np.e
    rounded = round(exact, 4)                       # a student's four-decimal answer
    assert ol.check("value", rounded, exact) is False
    assert ol.check("value", rounded, exact, tol=1e-3) is True


def test_check_arrays_and_complex_values():
    assert ol.check("system", [2, 1], np.array([2.0, 1.0])) is True
    assert ol.check("system", [1, 2], np.array([2.0, 1.0])) is False
    assert ol.check("system", [2, 1, 0], np.array([2.0, 1.0])) is False       # wrong size
    assert ol.check("product", 11 - 2j, (3 + 4j) * (1 - 2j)) is True
    assert ol.check("product", 11 + 2j, (3 + 4j) * (1 - 2j)) is False


def test_check_never_raises_on_a_non_numeric_answer(capsys):
    assert ol.check("x", "two", 2.0) is False
    assert "without quotes" in capsys.readouterr().out


@pytest.mark.parametrize("bad", [0.0, -1e-6, float("nan")])
def test_check_rejects_a_bad_tolerance(bad):
    with pytest.raises(ValueError):
        ol.check("x", 1.0, 1.0, tol=bad)


# --- solution_curve(): curves on a slope field ---------------------------------------------
WINDOW = (-3.0, 3.0)


def test_solution_curve_lebl_figure_1_2_3():
    # y' = x y through (0, 0.2): y = 0.2 exp(x^2 / 2), followed to the left and to the right
    y0 = 0.2
    t, y = ol.solution_curve(lambda x, y: x * y, (0.0, y0), WINDOW, WINDOW)
    assert np.all(np.diff(t) > 0) and t[0] < 0 < t[-1]
    assert y == pytest.approx(y0 * np.exp(t**2 / 2), rel=1e-8)
    assert np.interp(2.0, t, y) == pytest.approx(y0 * np.exp(2.0), rel=1e-6)
    assert y.max() <= WINDOW[1]                       # it stops where it leaves the window


def test_solution_curve_reaches_both_ends_when_it_stays_inside():
    # y' = -y through (0, 1) in a tall window: y = exp(-x) on the whole interval
    t_range, y_range = (-1.0, 2.0), (0.0, 5.0)
    t, y = ol.solution_curve(lambda x, y: -y, (0.0, 1.0), t_range, y_range)
    assert t[0] == pytest.approx(t_range[0]) and t[-1] == pytest.approx(t_range[1])
    assert y == pytest.approx(np.exp(-t), rel=1e-9)


def test_solution_curve_stops_at_a_blow_up_lebl_example_1_2_3():
    # y' = y^2, y(0) = 1: y = 1/(1 - x) blows up at x = 1; the curve ends inside the window
    with np.errstate(all="raise"):                    # and without floating-point warnings
        t, y = ol.solution_curve(lambda x, y: y * y, (0.0, 1.0), WINDOW, (-5.0, 5.0))
    assert t[-1] < 1.0 and np.all(np.isfinite(y)) and y.max() <= 5.0
    assert y == pytest.approx(1 / (1 - t), rel=1e-6)


def test_solution_curve_equilibrium_is_a_straight_line():
    t, y = ol.solution_curve(lambda x, y: x * y, (0.0, 0.0), WINDOW, WINDOW)
    assert np.all(y == 0.0) and t[0] == pytest.approx(WINDOW[0]) and t[-1] == pytest.approx(WINDOW[1])


@pytest.mark.parametrize("point, h", [((5.0, 0.0), 0.01), ((0.0, 5.0), 0.01), ((0.0, 0.0), 0.0),
                                      ((0.0, 0.0), -0.1), ((float("nan"), 0.0), 0.01)])
def test_solution_curve_rejects_bad_arguments(point, h):
    with pytest.raises(ValueError):
        ol.solution_curve(lambda x, y: y, point, WINDOW, WINDOW, h)


# --- Torricelli's law: draining a tank (notebook 10) -----------------------------------------
TANK = {"h0": 1.0, "area_ratio": (0.01 / 0.5) ** 2, "g": 9.81}     # 1 m of water, 1 cm hole, 0.5 m tank


def test_torricelli_drain_time_of_notebook_10():
    assert ol.torricelli_drain_time(**TANK) == pytest.approx(1128.8, abs=0.05)       # 18.8 minutes
    assert ol.torricelli_drain_time(TANK["h0"], TANK["area_ratio"]) == pytest.approx(
        np.sqrt(2 * TANK["h0"] / ol.GRAVITY) / TANK["area_ratio"])


def test_torricelli_scaling_laws():
    T = ol.torricelli_drain_time(**TANK)
    wider = dict(TANK, area_ratio=4 * TANK["area_ratio"])          # hole twice as wide
    higher = dict(TANK, h0=2 * TANK["h0"])
    assert ol.torricelli_drain_time(**wider) == pytest.approx(T / 4)
    assert ol.torricelli_drain_time(**higher) == pytest.approx(T * np.sqrt(2))


def test_torricelli_height_ends():
    T = ol.torricelli_drain_time(**TANK)
    assert ol.torricelli_height(0.0, **TANK) == pytest.approx(TANK["h0"])
    assert ol.torricelli_height(T, **TANK) == pytest.approx(0.0, abs=1e-12)
    assert np.all(ol.torricelli_height(np.array([T, 1.5 * T, 10 * T]), **TANK) == pytest.approx(0.0, abs=1e-12))
    # half of the height is lost in the first (1 - 1/sqrt 2) of the time, about 29 %
    assert ol.torricelli_height(T * (1 - 1 / np.sqrt(2)), **TANK) == pytest.approx(TANK["h0"] / 2)


def test_torricelli_height_satisfies_the_equation():
    T = ol.torricelli_drain_time(**TANK)
    t = np.linspace(0.05 * T, 0.95 * T, 19)
    dt = 1e-3
    slope = (ol.torricelli_height(t + dt, **TANK) - ol.torricelli_height(t - dt, **TANK)) / (2 * dt)
    law = -TANK["area_ratio"] * np.sqrt(2 * TANK["g"] * ol.torricelli_height(t, **TANK))
    assert slope == pytest.approx(law, rel=1e-6)


def test_torricelli_height_agrees_with_rk4():
    T = ol.torricelli_drain_time(**TANK)

    def draining(t, h):
        return -TANK["area_ratio"] * np.sqrt(2 * TANK["g"] * np.maximum(h, 0.0))

    t, h = ol.rk4(draining, (0.0, 1000.0), [TANK["h0"]], 1.0)      # stops before T = 1128.8 s
    assert t[-1] < T
    assert h == pytest.approx(ol.torricelli_height(t, **TANK), abs=1e-9)


@pytest.mark.parametrize("bad", [dict(TANK, h0=0.0), dict(TANK, area_ratio=-1.0), dict(TANK, g=0.0),
                                 dict(TANK, h0=float("nan"))])
def test_torricelli_rejects_bad_arguments(bad):
    with pytest.raises(ValueError):
        ol.torricelli_drain_time(**bad)
    with pytest.raises(ValueError):
        ol.torricelli_height(1.0, **bad)


def test_torricelli_height_rejects_negative_times():
    with pytest.raises(ValueError):
        ol.torricelli_height(-1.0, **TANK)


# --- linear_first_order(): the integrating factor with numerical integrals (notebook 11) -----
def test_linear_first_order_lebl_example_1_4_1():
    # y' + 2xy = exp(x - x^2), y(0) = -1  ->  y = exp(x - x^2) - 2 exp(-x^2)
    x = np.linspace(-1.0, 2.0, 13)                    # to the left and to the right of x0
    y = ol.linear_first_order(lambda s: 2 * s, lambda s: np.exp(s - s * s), 0.0, -1.0, x)
    assert y == pytest.approx(np.exp(x - x**2) - 2 * np.exp(-x**2), rel=1e-9, abs=1e-12)


def test_linear_first_order_lebl_example_1_4_2_salt_tank():
    # x' + 3x/(60 + 2t) = 0.5, x(0) = 10: 11.86 kg of salt when the tank is full (t = 20)
    def salt(t):
        return ol.linear_first_order(lambda s: 3 / (60 + 2 * s), lambda s: 0.5, 0.0, 10.0, t)
    assert salt(20.0) == pytest.approx(11.86, abs=0.005)
    t = np.linspace(0.0, 20.0, 5)
    constant = 4 * 60**1.5                            # Lebl: C = 4 (60^(3/2)), about 1859.03
    assert salt(t) == pytest.approx((60 + 2 * t) / 10 + constant * (60 + 2 * t) ** -1.5, rel=1e-9)


def test_linear_first_order_lebl_exercise_1_4_13():
    # P' = (2 - 0.1 t) P, P(0) = 1000  ->  P(5) = 1000 exp(8.75), about 6.31e6
    P5 = ol.linear_first_order(lambda t: -(2 - 0.1 * t), lambda t: 0.0, 0.0, 1000.0, 5.0)
    assert P5 == pytest.approx(1000 * np.exp(8.75), rel=1e-9)
    assert P5 == pytest.approx(6.31e6, rel=1e-3)


def test_linear_first_order_constant_coefficients_and_initial_value():
    # y' + 2y = 4, y(0) = 0  ->  y = 2 (1 - exp(-2x)); and y(x0) = y0 whatever the problem
    x = np.linspace(0.0, 3.0, 7)
    y = ol.linear_first_order(lambda s: 2.0, lambda s: 4.0, 0.0, 0.0, x)
    assert y == pytest.approx(2 * (1 - np.exp(-2 * x)), abs=1e-10)
    assert ol.linear_first_order(np.sin, np.cos, 1.5, -7.0, 1.5) == pytest.approx(-7.0)


def test_linear_first_order_shape_follows_x():
    assert np.shape(ol.linear_first_order(lambda s: 1.0, lambda s: 0.0, 0.0, 1.0, 2.0)) == ()
    assert np.shape(ol.linear_first_order(lambda s: 1.0, lambda s: 0.0, 0.0, 1.0, [[0.0, 1.0], [2.0, 3.0]])) == (2, 2)


@pytest.mark.parametrize("x0, y0, x", [(float("nan"), 1.0, 1.0), (0.0, float("inf"), 1.0), (0.0, 1.0, float("nan"))])
def test_linear_first_order_rejects_bad_arguments(x0, y0, x):
    with pytest.raises(ValueError):
        ol.linear_first_order(lambda s: 1.0, lambda s: 0.0, x0, y0, x)


# --- relaxation() and quadratic_drag_velocity(): first-order models (notebook 13) ------------
def test_relaxation_ends_and_time_constant():
    x0, x_inf, tau = 90.0, 20.0, 4.0
    assert ol.relaxation(0.0, x0, x_inf, tau) == pytest.approx(x0)
    assert ol.relaxation(1e3 * tau, x0, x_inf, tau) == pytest.approx(x_inf)
    closed = (ol.relaxation(tau, x0, x_inf, tau) - x0) / (x_inf - x0)
    assert closed == pytest.approx(1 - np.exp(-1))            # 63.2 % of the gap after one tau
    assert closed == pytest.approx(0.632, abs=5e-4)
    assert ol.relaxation(123.0, x_inf, x_inf, tau) == pytest.approx(x_inf)     # the equilibrium stays


def test_relaxation_agrees_with_rk4():
    x0, x_inf, tau = 0.0, 5.0, 2.0
    t, x = ol.rk4(lambda t, x: (x_inf - x) / tau, (0.0, 10.0), [x0], 0.01)
    assert x == pytest.approx(ol.relaxation(t, x0, x_inf, tau), abs=1e-9)


def test_relaxation_adkins_davidson_example_11_mixing():
    # 10 L, 1000 g of salt at t = 0; 400 g/L in at 3 L/min, out at 3 L/min:
    # y' = 1200 - 3y/10  ->  y = 4000 - 3000 exp(-3t/10): tau = 10/3 min, y_inf = 4000 g
    t = np.linspace(0.0, 15.0, 7)
    assert ol.relaxation(t, 1000.0, 4000.0, 10 / 3) == pytest.approx(4000 - 3000 * np.exp(-3 * t / 10))


def test_relaxation_lebl_example_1_3_3_coffee():
    # 89 degrees at t = 0, 85 after one minute, room at 22: 60 degrees after 9.21 minutes
    k = -np.log((85 - 22) / (89 - 22))
    assert k == pytest.approx(0.0616, abs=5e-5)
    assert ol.relaxation(9.21, 89.0, 22.0, 1 / k) == pytest.approx(60.0, abs=0.01)


def test_quadratic_drag_skydiver():
    m, c, g = 80.0, 0.25, 9.81
    v_t = np.sqrt(m * g / c)
    assert v_t == pytest.approx(56.0, abs=0.05)
    assert ol.quadratic_drag_velocity(0.0, v_t, g) == 0.0
    assert ol.quadratic_drag_velocity(1e3, v_t, g) == pytest.approx(v_t)
    assert ol.quadratic_drag_velocity(10.46, v_t, g) == pytest.approx(0.95 * v_t, rel=1e-3)
    t, v = ol.rk4(lambda t, v: g - c / m * v * v, (0.0, 20.0), [0.0], 0.01)
    assert v == pytest.approx(ol.quadratic_drag_velocity(t, v_t, g), abs=1e-8)


@pytest.mark.parametrize("tau", [0.0, -1.0, float("nan")])
def test_relaxation_rejects_a_bad_time_constant(tau):
    with pytest.raises(ValueError):
        ol.relaxation(1.0, 0.0, 1.0, tau)


@pytest.mark.parametrize("v_t, g", [(0.0, 9.81), (-5.0, 9.81), (50.0, 0.0)])
def test_quadratic_drag_rejects_bad_arguments(v_t, g):
    with pytest.raises(ValueError):
        ol.quadratic_drag_velocity(1.0, v_t, g)


# --- check(): corrections after the review of 2026-11-17 -----------------------------------
def test_check_tolerance_is_relative_to_the_size_of_the_answer():
    big = 1000 * np.exp(8.75)                         # Lebl, Exercise 1.4.13: about 6.31e6
    assert ol.check("P(5)", 6.31e6, big, tol=1e-2) is True
    assert ol.check("P(5)", 6.19e6, big, tol=1e-2) is False        # 1.9 % low: was accepted before
    assert ol.check("P(5)", 6.40e6, big, tol=1e-2) is False        # 1.4 % high: was accepted before
    small = np.exp(-3.0)                              # 0.0498: for small values tol is absolute
    assert ol.check("gap", 0.05, small, tol=1e-2) is True
    assert ol.check("gap", 0.06, small, tol=1e-2) is False


def test_check_rejects_text_even_when_it_looks_like_a_number(capsys):
    assert ol.check("x", "3", 3) is False
    assert "without quotes" in capsys.readouterr().out


# --- Stage 1 hardening (2026-11-17) -----------------------------------------------------------
def test_solution_curve_stops_at_a_vertical_tangent():
    # the circle x^2 + y^2 = 1 through (0, 1): y' = -x/y, the solution ends at x = +-1
    window = (-2.0, 2.0)
    t, y = ol.solution_curve(lambda x, y: -x / y, (0.0, 1.0), window, window)
    assert t[0] == pytest.approx(-1.0) and t[-1] == pytest.approx(1.0)     # it reaches the ends, and stops
    on_circle = np.sqrt(np.clip(1 - t**2, 0.0, None))
    assert y == pytest.approx(on_circle, abs=0.02)             # and it is the circle, nothing else
    assert y[1:-1] == pytest.approx(on_circle[1:-1], abs=1e-4)
    assert np.all(y > 0)


@pytest.mark.parametrize("x", [-1.0, 0.0])
def test_linear_first_order_refuses_to_cross_a_singularity(x):
    with pytest.raises(ValueError, match="continuous"):
        ol.linear_first_order(lambda s: 1 / s, lambda s: 1.0, 1.0, 1.0, x)


def test_linear_first_order_reports_an_overflow_instead_of_nan():
    with pytest.raises(ValueError, match="too large"):
        ol.linear_first_order(lambda s: 1.0, lambda s: 1.0, 0.0, 0.0, 800.0)


def test_one_value_of_g():
    assert ol.GRAVITY == 9.81
    assert ol.torricelli_drain_time(1.0, (0.01 / 0.5) ** 2) == pytest.approx(1128.8, abs=0.05)


# --- non-homogeneous equations (notebook 14) -----------------------------------------------
GRID = np.linspace(0.0, 2.0, 21)
# a, b, c, right-hand side, a particular solution printed in the book (rewritten in t)
BOOK_PARTICULAR = {
    "Lebl 2.5, opening example": (1, 5, 6, lambda s: 2 * s + 1, lambda s: (3 * s - 1) / 9),
    "Lebl 2.5, cosine": (1, 2, 2, lambda s: np.cos(2 * s), lambda s: (-np.cos(2 * s) + 2 * np.sin(2 * s)) / 10),
    "Lebl 2.5, the hiccup": (1, 0, -9, lambda s: np.exp(3 * s), lambda s: s * np.exp(3 * s) / 6),
    "Lebl 2.5, double root": (1, -6, 9, lambda s: np.exp(3 * s), lambda s: s ** 2 * np.exp(3 * s) / 2),
    "Adkins-Davidson 3.4, Example 1": (1, -1, -6, lambda s: np.exp(-s), lambda s: -np.exp(-s) / 4),
    "Adkins-Davidson 3.4, Example 5": (1, -5, 6, lambda s: 4 * np.exp(2 * s), lambda s: -4 * s * np.exp(2 * s)),
    "Adkins-Davidson 3.4, Example 7": (1, -3, 2, lambda s: 2 * s * np.exp(s), lambda s: -(2 * s + s ** 2) * np.exp(s)),
    "Adkins-Davidson 3.4, Example 9": (1, -5, 6, lambda s: 12 + 4 * np.exp(2 * s), lambda s: 2 - 4 * s * np.exp(2 * s)),
}
STEP = 1e-5                                 # for the derivative of a book solution at 0


@pytest.mark.parametrize("name", sorted(BOOK_PARTICULAR))
def test_forced_response_reproduces_the_particular_solutions_of_the_books(name):
    # with the initial data of the book's particular solution, the solution is that function
    a, b, c, forcing, book = BOOK_PARTICULAR[name]
    slope = (book(STEP) - book(-STEP)) / (2 * STEP)
    computed = ol.forced_response(GRID, a, b, c, forcing, y0=book(0.0), v0=slope)
    assert computed == pytest.approx(book(GRID), rel=1e-6, abs=1e-6)


def test_forced_response_lebl_opening_example_with_its_initial_data():
    # y'' + 5y' + 6y = 2x + 1, y(0) = 0, y'(0) = 1/3:  y = e^(-2x)/3 - 2e^(-3x)/9 + (3x - 1)/9
    exact = np.exp(-2 * GRID) / 3 - 2 * np.exp(-3 * GRID) / 9 + (3 * GRID - 1) / 9
    assert ol.forced_response(GRID, 1, 5, 6, lambda s: 2 * s + 1, 0.0, 1 / 3) == pytest.approx(exact, abs=1e-9)


def test_forced_response_without_forcing_is_the_homogeneous_solution_and_it_is_linear():
    a, b, c = 2.0, 3.0, 5.0
    free = ol.forced_response(GRID, a, b, c, lambda s: 0.0, 1.0, -2.0)
    assert free == pytest.approx(ol.constant_coefficient_solution(GRID, a, b, c, 1.0, -2.0))
    first = ol.forced_response(GRID, a, b, c, np.sin)
    second = ol.forced_response(GRID, a, b, c, np.exp)
    both = ol.forced_response(GRID, a, b, c, lambda s: 3 * np.sin(s) - 2 * np.exp(s))
    assert both == pytest.approx(3 * first - 2 * second, abs=1e-9)
    assert ol.forced_response(0.0, a, b, c, np.exp, 1.5, -0.5) == pytest.approx(1.5)      # a single number is accepted
    left = ol.forced_response(-GRID, 1, 0, -9, lambda s: np.exp(3 * s), 0.0, 1 / 6)         # to the left of 0
    assert left == pytest.approx(-GRID * np.exp(-3 * GRID) / 6, abs=1e-9)


def test_forced_response_refuses_what_it_cannot_compute():
    with pytest.raises(ValueError):
        ol.forced_response(GRID, 0, 1, 1, np.sin)                    # not of second order
    with pytest.raises(ValueError):
        ol.forced_response(GRID, 1, 1, 1, "sin")                     # not a function
    with pytest.raises(ValueError):
        ol.forced_response([0.0, np.inf], 1, 1, 1, np.sin)
    with pytest.raises(ValueError):
        ol.forced_response(GRID, 1, 1, 1, np.sin, y0=np.nan)
    with pytest.raises(ValueError):
        ol.forced_response(GRID, 1, 0, 1, np.tan)                    # tan is not continuous up to 2
    with pytest.raises(ValueError):
        ol.forced_response(GRID, 1, 1, 1, lambda s: None)            # f does not return a number
    with pytest.raises(ValueError):
        ol.forced_response(GRID, 1, 1, 1, lambda s: np.array([s, s]))


@pytest.mark.parametrize("coefficients, number, expected", [
    ((1, -1, -6), (2, 0), 0),               # Lebl, Exercise 2.5.1
    ((1, -4, 4), (2, 0), 2),                # Lebl, Exercise 2.5.2
    ((1, 0, 9), (0, 3), 1),                 # Lebl, Exercise 2.5.3
    ((1, -5, 6), (2, 0), 1),                # Adkins-Davidson 3.4, Example 5
    ((1, 2, 2), (-1, 1), 1),
    ((1, 2, 2), (0, 1), 0),                 # the trap: the roots are -1 +- i, not +- i
    ((1, 0, -9), (3, 0), 1),
    ((1, -6, 9), (3, 0), 2),
    ((1, 0, 0), (0, 0), 2),                 # y'' = polynomial
    ((1, 1, 0), (0, 0), 1),
    ((2, 0, 8), (0, -2), 1),                # the conjugate root counts too
])
def test_root_multiplicity(coefficients, number, expected):
    assert ol.root_multiplicity(*coefficients, *number) == expected


def test_root_multiplicity_near_a_root_and_invalid_arguments():
    assert ol.root_multiplicity(1, -6, 9, 3.001) == 0                # close to the double root, but not a root
    assert ol.root_multiplicity(1, -6, 9, 3.0001) == 0
    assert ol.root_multiplicity(1, -2, 1, 1.00002) == 0
    assert ol.root_multiplicity(1, -2.0001, 1.0001, 1.0) == 1        # roots 1 and 1.0001: distinct, as in
    grid = np.linspace(0.0, 1.0, 5)                                  # constant_coefficient_solution
    assert ol.constant_coefficient_solution(grid, 1, -2.0001, 1.0001, 0.0, 1.0) == pytest.approx(
        (np.exp(1.0001 * grid) - np.exp(grid)) / 0.0001, rel=1e-6)
    for coefficients, number in (((0.1, 0.3, 0.2), (-1, 0)), ((1e3, 0, 1e3), (0, 1)), ((1e-3, 2e-3, 1e-3), (-1, 0))):
        assert ol.root_multiplicity(*coefficients, *number) >= 1
    assert ol.root_multiplicity(0.1, -0.6, 0.9, 3.0) == 2            # coefficients that are not exact in binary
    assert ol.root_multiplicity(1, -(3 + 3.001), 3 * 3.001, 3.0) == 1          # two roots close to each other
    with pytest.raises(ValueError):
        ol.root_multiplicity(0, 1, 1, 1.0)
    with pytest.raises(ValueError):
        ol.root_multiplicity(1, 1, 1, np.nan)


# --- a spring driven by F0 cos(omega t) (notebook 15; Lebl, Section 2.6) ---------------------
LONG = np.linspace(0.0, 20.0, 401)


def test_lebl_example_2_6_1_beats():
    # 0.5 x'' + 8x = 10 cos(pi t) from rest:  x = 20/(16 - pi^2) (cos(pi t) - cos 4t)
    exact = 20 / (16 - np.pi ** 2) * (np.cos(np.pi * LONG) - np.cos(4 * LONG))
    assert 20 / (16 - np.pi ** 2) == pytest.approx(3.262, abs=5e-4)
    assert ol.cosine_forced_response(LONG, 0.5, 0.0, 8.0, 10.0, np.pi) == pytest.approx(exact, abs=1e-10)


def test_lebl_figure_2_6_3_and_exercise_2_6_8_pure_resonance():
    # F0 = 2, m = 1, omega = pi: x = (1/pi) t sin(pi t)
    assert ol.cosine_forced_response(LONG, 1.0, 0.0, np.pi ** 2, 2.0, np.pi) == pytest.approx(
        LONG * np.sin(np.pi * LONG) / np.pi, abs=1e-10)
    # Exercise 2.6.8: m = 5, k = 20, F0 = 5: resonance at omega = 2, amplitude 25 at t = 100
    assert np.sqrt(20 / 5) == 2.0
    assert 5.0 / (2 * 5 * 2.0) * 100.0 == pytest.approx(25.0)   # F0 t / (2 m omega) at t = 100
    times = np.linspace(95.0, 100.0, 2001)                    # the largest displacement near t = 100
    assert np.max(np.abs(ol.cosine_forced_response(times, 5.0, 0.0, 20.0, 5.0, 2.0))) == pytest.approx(25.0, rel=0.03)


def test_adkins_davidson_3_6_formula_18_both_cases_and_continuity():
    m, k, F0 = 2.0, 18.0, 3.0                                 # natural frequency 3
    for omega in (1.0, 2.9, 3.5):
        exact = F0 / (m * (9.0 - omega ** 2)) * (np.cos(omega * LONG) - np.cos(3.0 * LONG))
        assert ol.cosine_forced_response(LONG, m, 0.0, k, F0, omega) == pytest.approx(exact, abs=1e-9)
    at_resonance = F0 / (2 * m * 3.0) * LONG * np.sin(3.0 * LONG)
    assert ol.cosine_forced_response(LONG, m, 0.0, k, F0, 3.0) == pytest.approx(at_resonance, abs=1e-10)
    assert ol.cosine_forced_response(LONG, m, 0.0, k, F0, 3.0 + 1e-9) == pytest.approx(at_resonance, abs=1e-6)


def test_lebl_figures_2_6_4_and_2_6_5_and_exercise_2_6_6():
    amplitude, phase = ol.steady_state(1.0, 0.7, 1.0, 1.0, 1.1)                 # Figure 2.6.4
    assert amplitude == pytest.approx(1.253, abs=5e-4) and phase == pytest.approx(1.837, abs=5e-4)
    for damping, frequency, height in ((0.4, 0.959, 2.55), (0.8, 0.825, 1.36)):  # Figure 2.6.5
        best = ol.practical_resonance(1.0, damping, 1.0)
        assert best == pytest.approx(frequency, abs=5e-4)
        assert ol.steady_state(1.0, damping, 1.0, 1.0, best)[0] == pytest.approx(height, abs=5e-3)
    assert ol.practical_resonance(1.0, 1.6, 1.0) == 0.0
    for mass, stiffness in ((1.0, 1.0), (3.0, 1.0), (0.7, 11.0), (1e3, 1e-3)):              # the limit case c^2 = 2mk
        assert ol.practical_resonance(mass, np.sqrt(2 * mass * stiffness), stiffness) == 0.0
    best = ol.practical_resonance(4.0, 1.0, 4.0)                                # Exercise 2.6.6
    assert best == pytest.approx(np.sqrt(31) / (4 * np.sqrt(2))) and best == pytest.approx(0.984, abs=5e-4)
    assert ol.steady_state(4.0, 1.0, 4.0, 2.0, best)[0] == pytest.approx(16 / (3 * np.sqrt(7)))


@pytest.mark.parametrize("m, c, k", [(1.0, 0.4, 1.0), (4.0, 1.0, 4.0), (2.0, 0.1, 50.0), (1.0, 1.6, 1.0), (0.5, 3.0, 2.0)])
def test_the_response_curve_has_its_maximum_at_practical_resonance(m, c, k):
    frequencies = np.linspace(1e-3, 4 * np.sqrt(k / m), 40001)
    heights = np.array([ol.steady_state(m, c, k, 1.0, omega)[0] for omega in frequencies])
    best = ol.practical_resonance(m, c, k)
    if best == 0.0:
        assert np.all(np.diff(heights) < 0)                                    # decreasing: no practical resonance
    else:
        assert frequencies[np.argmax(heights)] == pytest.approx(best, abs=2 * frequencies[1])
        assert best < np.sqrt(k / m)


def test_steady_state_limits_phase_and_invalid_arguments():
    assert ol.steady_state(2.0, 0.3, 8.0, 5.0, 1e-9)[0] == pytest.approx(5.0 / 8.0)        # a very slow force: F0 / k
    assert ol.steady_state(2.0, 0.3, 8.0, 5.0, 1e4)[0] < 1e-7                              # a very fast force
    assert ol.steady_state(2.0, 0.3, 8.0, 5.0, 2.0)[1] == pytest.approx(np.pi / 2)         # at the natural frequency
    assert 0 < ol.steady_state(2.0, 0.3, 8.0, 5.0, 1.0)[1] < np.pi / 2 < ol.steady_state(2.0, 0.3, 8.0, 5.0, 3.0)[1] < np.pi
    assert ol.steady_state(2.0, 0.0, 8.0, 5.0, 1.0) == pytest.approx((5.0 / 6.0, 0.0))     # undamped, below
    assert ol.steady_state(2.0, 0.0, 8.0, 5.0, 3.0) == pytest.approx((0.5, np.pi))         # undamped, above
    with pytest.raises(ValueError, match="pure resonance"):
        ol.steady_state(2.0, 0.0, 8.0, 5.0, 2.0)
    for bad in ((0.0, 1.0, 1.0, 1.0, 1.0), (1.0, -1.0, 1.0, 1.0, 1.0), (1.0, 1.0, -1.0, 1.0, 1.0),
                (1.0, 1.0, 1.0, np.nan, 1.0), (1.0, 1.0, 1.0, 1.0, 0.0), (1.0, 1.0, 1.0, 1.0, np.inf)):
        with pytest.raises(ValueError):
            ol.steady_state(*bad)
        with pytest.raises(ValueError):
            ol.cosine_forced_response(LONG, *bad)
    with pytest.raises(ValueError):
        ol.practical_resonance(1.0, -0.1, 1.0)
    with pytest.raises(ValueError):
        ol.cosine_forced_response(LONG, 1.0, 0.0, 1.0, 1.0, 1.0, x0=np.nan)
    with pytest.raises(ValueError):
        ol.cosine_forced_response([0.0, np.nan], 1.0, 0.5, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        ol.cosine_forced_response(-1e4, 1.0, 0.7, 1.0, 1.0, 1.1)                # far in the past: overflow
    with pytest.raises(ValueError):
        ol.steady_state(1.0, 1e-300, 1.0, 1.0, 1.0)                             # (c omega)^2 underflows
    assert ol.steady_state(2.0, 0.3, 8.0, 0.0, 1.0) == (0.0, 0.0)               # no force


def test_cosine_forced_response_with_a_negligible_damping_is_the_undamped_one():
    # found by the review of notebook 15: c = 1e-17 gave 64.0 instead of 43.5 at t = 30
    undamped = ol.cosine_forced_response(30.0, 0.5, 0.0, 8.0, 10.0, 4.0)
    assert undamped == pytest.approx(43.5458, abs=1e-4)
    for tiny in (1e-200, 1e-17, 1e-14, 1e-12, 1e-10):
        assert ol.cosine_forced_response(30.0, 0.5, tiny, 8.0, 10.0, 4.0) == pytest.approx(undamped, rel=1e-6)
    assert ol.cosine_forced_response(30.0, 0.5, 1e-6, 8.0, 10.0, 4.0) == pytest.approx(undamped, rel=1e-4)


def test_cosine_forced_response_becomes_the_steady_solution():
    m, c, k, F0, omega = 1.0, 0.7, 1.0, 1.0, 1.1                                # Lebl, Figure 2.6.4
    amplitude, phase = ol.steady_state(m, c, k, F0, omega)
    late = np.linspace(40.0, 60.0, 201)
    for start in ((0.0, 0.0), (2.0, 0.0), (-1.0, 3.0)):
        assert ol.cosine_forced_response(late, m, c, k, F0, omega, *start) == pytest.approx(
            amplitude * np.cos(omega * late - phase), abs=1e-5)
    assert ol.cosine_forced_response(0.0, m, c, k, F0, omega, 2.0, -1.0) == pytest.approx(2.0)


# --- equations of order n and boundary value problems (notebook 16) -------------------------
SHORT = np.linspace(0.0, 1.5, 16)


def test_lebl_example_2_3_3_with_its_initial_conditions():
    # y^(3) - 3y'' - y' + 3y = 0, y(0) = 1, y'(0) = 2, y''(0) = 3:  y = -e^(-x)/4 + e^x + e^(3x)/4
    exact = -np.exp(-SHORT) / 4 + np.exp(SHORT) + np.exp(3 * SHORT) / 4
    assert ol.higher_order_solution(SHORT, [1, -3, -1, 3], [1, 2, 3]) == pytest.approx(exact, rel=1e-10)


def test_lebl_examples_2_3_4_and_2_3_5_repeated_roots():
    # 2.3.4: y^(4) - 3y^(3) + 3y'' - y' = 0, general solution (C1 + C2 x + C3 x^2) e^x + C4
    def first(s):
        return (1 - 2 * s + 3 * s ** 2) * np.exp(s) + 5

    # y, y', y'', y^(3) at 0 of (1 - 2x + 3x^2) e^x + 5:  6, -1, 3, 13
    assert ol.higher_order_solution(SHORT, [1, -3, 3, -1, 0], [6, -1, 3, 13]) == pytest.approx(first(SHORT), rel=1e-9)

    # 2.3.5: y^(4) - 4y^(3) + 8y'' - 8y' + 4y = 0, roots 1 +- i twice: x e^x cos x is a solution
    def second(s):
        return s * np.exp(s) * np.cos(s)

    # y, y', y'', y^(3) at 0 of x e^x cos x:  0, 1, 2, 0
    assert ol.higher_order_solution(SHORT, [1, -4, 8, -8, 4], [0, 1, 2, 0]) == pytest.approx(second(SHORT), abs=1e-9)


def test_higher_order_solution_small_orders_uniqueness_and_invalid_arguments():
    assert ol.higher_order_solution(SHORT, [2, 3, 5], [1.0, -2.0]) == pytest.approx(
        ol.constant_coefficient_solution(SHORT, 2, 3, 5, 1.0, -2.0))
    assert ol.higher_order_solution(SHORT, [1, 0.3], [4.0]) == pytest.approx(ol.exponential_solution(SHORT, 4.0, -0.3))
    # Lebl, Exercise 2.3.14: zero initial data give the zero solution, whatever the coefficients
    assert np.all(ol.higher_order_solution(SHORT, [1001, 3.2, np.pi, -np.sqrt(4)], [0, 0, 0]) == 0)
    assert ol.higher_order_solution(0.0, [1, -3, -1, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert ol.higher_order_solution(-SHORT, [1, 0, 1], [0.0, 1.0]) == pytest.approx(-np.sin(SHORT))   # to the left of 0
    for coefficients, initial in (([0, 1, 1], [1.0, 0.0]), ([1], []), ([1, 1, 1], [1.0]), ([1, np.nan, 1], [1.0, 0.0]),
                                  ([1, 1, 1], [1.0, np.inf])):
        with pytest.raises(ValueError):
            ol.higher_order_solution(SHORT, coefficients, initial)
    with pytest.raises(ValueError):
        ol.higher_order_solution(1e6, [1, -3, -1, 3], [1, 2, 3])               # e^(3 000 000)


def test_classify_linear_system_on_lebl_boundary_value_problems():
    # x'' + lambda x = 0 on [0, pi], x = A cos(s t) + B sin(s t), s = sqrt(lambda); conditions x(0) and x(pi)
    def conditions(lam):
        s = np.sqrt(lam)
        return [[1.0, 0.0], [np.cos(s * np.pi), np.sin(s * np.pi)]]

    kind, constants = ol.classify_linear_system(conditions(2.0), [0.0, 0.0])    # Example 4.1.2
    assert kind == "one solution" and constants == pytest.approx([0.0, 0.0])
    kind, constants = ol.classify_linear_system(conditions(1.0), [0.0, 0.0])    # Example 4.1.1
    assert kind == "infinitely many solutions" and np.allclose(np.array(conditions(1.0)) @ constants, 0.0)
    assert ol.classify_linear_system(conditions(1.0), [0.0, 1.0]) == ("no solution", None)
    kind, constants = ol.classify_linear_system(conditions(2.0), [0.0, 1.0])
    assert kind == "one solution" and constants == pytest.approx([0.0, 1 / np.sin(np.sqrt(2) * np.pi)])
    for lam in (4.0, 9.0, 16.0):                                                # the other eigenvalues k^2
        assert ol.classify_linear_system(conditions(lam), [0.0, 1.0])[0] == "no solution"
        assert ol.classify_linear_system(conditions(lam), [0.0, 0.0])[0] == "infinitely many solutions"


def test_classify_linear_system_general_cases_and_invalid_arguments():
    assert ol.classify_linear_system([[2.0]], [4.0])[1] == pytest.approx([2.0])
    assert ol.classify_linear_system([[0.0]], [0.0])[0] == "infinitely many solutions"
    assert ol.classify_linear_system([[0.0]], [1.0])[0] == "no solution"
    assert ol.classify_linear_system([[1.0, 2.0], [2.0, 4.0]], [3.0, 6.0])[0] == "infinitely many solutions"
    assert ol.classify_linear_system([[1.0, 2.0], [2.0, 4.0]], [3.0, 7.0])[0] == "no solution"
    assert ol.classify_linear_system([[1.0, 2.0], [2.0, 4.0 + 1e-3]], [3.0, 7.0])[0] == "one solution"   # close, not singular
    # found by the review of notebook 16: multiplying an equation by a number must not change the answer
    kind, solution = ol.classify_linear_system([[1e9, 2e9], [3.0, 4.0]], [1e9, 1.0])
    assert kind == "one solution" and solution == pytest.approx([-1.0, 1.0])
    assert ol.classify_linear_system([[1e10, 0.0], [0.0, 1.0]], [1e10, 1.0])[0] == "one solution"
    assert ol.classify_linear_system([[1e-12, 0.0], [0.0, 1e-12]], [1.0, 1.0])[0] == "one solution"
    # ... and a small right-hand side that cannot be matched is still "no solution"
    assert ol.classify_linear_system([[1.0, 0.0], [1.0, 0.0]], [1e-12, 2e-12])[0] == "no solution"
    assert ol.classify_linear_system([[1.0, 0.0], [0.0, 0.0]], [0.0, 1e-10])[0] == "no solution"
    assert ol.classify_linear_system([[1.0, 0.0], [0.0, 0.0]], [1e12, 1.0])[0] == "no solution"
    assert ol.classify_linear_system([[1.0, 0.0], [0.0, 0.0]], [1e12, 0.0])[0] == "infinitely many solutions"
    assert ol.classify_linear_system([[1.0, 0.0], [1.0, 0.0]], [3e-12, 3e-12])[0] == "infinitely many solutions"
    for matrix, right_side in (([[1.0, 2.0]], [1.0]), ([[1.0, 2.0], [3.0, 4.0]], [1.0]), ([[np.nan]], [1.0]),
                               (np.zeros((0, 0)), [])):
        with pytest.raises(ValueError):
            ol.classify_linear_system(matrix, right_side)


def test_kiusalaas_example_8_4_simply_supported_beam():
    # d^4 y / d xi^4 = xi, y = y'' = 0 at 0 and 1: slopes 19.444e-3 and -22.222e-3, mid-span 6.5104e-3
    step = 1e-6
    load = [1.0, 0.0]
    assert ol.beam_deflection(0.5, load, "supported", "supported") == pytest.approx(6.5104e-3, abs=5e-8)
    assert ol.beam_deflection(0.5, load, "supported", "supported") == pytest.approx(5 / 768)
    assert ol.beam_deflection(step, load, "supported", "supported") / step == pytest.approx(19.444e-3, abs=5e-7)
    left_of_end = ol.beam_deflection(1 - step, load, "supported", "supported")
    assert -left_of_end / step == pytest.approx(-22.222e-3, abs=5e-7)
    xi = np.linspace(0.0, 1.0, 21)
    assert ol.beam_deflection(xi, load, "supported", "supported") == pytest.approx(xi ** 5 / 120 - xi ** 3 / 36 + 7 * xi / 360)


def test_beam_with_a_uniform_load_classical_values_and_units():
    xi = np.linspace(0.0, 1.0, 21)
    supported = ol.beam_deflection(xi, [1.0], "supported", "supported")
    assert supported[10] == pytest.approx(5 / 384) and supported == pytest.approx(supported[::-1])
    assert ol.beam_deflection(1.0, [1.0], "fixed", "free") == pytest.approx(1 / 8)
    assert ol.beam_deflection(0.0, [1.0], "free", "fixed") == pytest.approx(1 / 8)          # the mirror image
    fixed = ol.beam_deflection(xi, [1.0], "fixed", "fixed")
    assert fixed[10] == pytest.approx(1 / 384) and fixed == pytest.approx(xi ** 2 * (1 - xi) ** 2 / 24)
    # real units: w0 = 10 kN/m, L = 6 m, EI = 210e9 * 8.356e-5 N m^2
    w0, length, rigidity = 1.0e4, 6.0, 210e9 * 8.356e-5
    assert ol.beam_deflection(length / 2, [w0], "supported", "supported", length, rigidity) == pytest.approx(
        5 * w0 * length ** 4 / (384 * rigidity))
    assert 5 * w0 * length ** 4 / (384 * rigidity) == pytest.approx(9.6e-3, abs=5e-5)


def test_beam_that_is_not_held_and_invalid_arguments():
    for left, right in (("free", "free"), ("supported", "free"), ("free", "supported")):
        with pytest.raises(ValueError, match="not held"):
            ol.beam_deflection(0.5, [1.0], left, right)
    for arguments in ((0.5, [1.0], "pinned", "fixed"), (0.5, [], "fixed", "fixed"), (0.5, [np.nan], "fixed", "fixed"),
                      (np.inf, [1.0], "fixed", "fixed")):
        with pytest.raises(ValueError):
            ol.beam_deflection(*arguments)
    with pytest.raises(ValueError):
        ol.beam_deflection(0.5, [1.0], ["fixed"], "fixed")                       # a support that is not a word
    for outside in (-0.5, 2.0, [0.0, 1.001]):                                    # the formula must not be extended
        with pytest.raises(ValueError, match="between 0 and the length"):
            ol.beam_deflection(outside, [1.0], "supported", "supported")
    assert ol.beam_deflection(3.0, [1.0], "supported", "supported", length=6.0) == pytest.approx(5 * 6.0 ** 4 / 384)
    assert ol.beam_deflection(0.1 * 3 * 2, 1.0, "fixed", "fixed", length=0.6) == pytest.approx(0.0, abs=1e-12)   # 0.6000000000000001
    with pytest.raises(ValueError):
        ol.beam_deflection(0.5, [1.0], "fixed", "fixed", length=0.0)
    with pytest.raises(ValueError):
        ol.beam_deflection(0.5, [1.0], "fixed", "fixed", rigidity=-1.0)


# --- the student guide and the hints under error messages ------------------------------------
def test_guide_prints_every_part_and_is_plain_text(capsys):
    ol.guide()
    whole = capsys.readouterr().out
    assert whole.isascii()
    for title in ol.GUIDE_TOPICS.values():
        assert title.upper() in whole
    assert max(len(line) for line in whole.splitlines()) <= ol.GUIDE_WIDTH + 30      # tables are not wrapped
    for topic, title in ol.GUIDE_TOPICS.items():
        ol.guide(topic)
        part = capsys.readouterr().out
        assert part.startswith(title.upper()) and part in whole
    ol.guide("everything")                                                        # an unknown part: a line, no error
    assert "the parts are 'start', 'typing', 'errors', 'how', 'words'" in capsys.readouterr().out
    assert len(ol.GUIDE_START) == 5


@pytest.mark.parametrize("statement, words", [
    ("assert 1 == 2", "nothing is broken"),
    ("np_missing.exp(1)", "typing mistake"),
    ("exp(1)", "write np.exp(2) with numbers, or sp.exp(x) in a formula"),
    ("ln(2)", "ln is called log here"),
    ("sqrt(2)", "sp.sqrt(x)"),
    ("Sin(1)", "Sin is called sin here, in small letters"),
    ("sen(1)", "sen is called sin here"),
    ("tg(1)", "np.tan(1)"),
    ("PI / 2", "np.pi"),
    ("oo", "np.inf"),
    ("falso", "write False"),
    ("math.sqrt(2)", "np.sqrt(2), np.pi"),
    ("my_x + 1", "the line my_x = ... is missing, or was typed with two = signs"),
    ("guide()", "ol.guide(\"errors\")"),
    ("3 + C", "the letter C has no value here"),
    ("None(3.5)", "the word None is still there"),
    ("__import__('numpy').sqrt(...)", "replace the three dots"),
    ("bool(__import__('sympy').Symbol('x') > 0)", "a line that starts with return"),
    ("__import__('numpy').Sqrt", "called sqrt, in small letters"),
    ("pi / 4", "np.pi"),
    ("e ** 2", "sp.E"),
    ("true", "write True"),
    ("2(3 + 1)", "2 * (3 + 1)"),
    ("__import__('sympy').Symbol('x')(1)", "x * (x + 1)"),
    ("__import__('sympy').Symbol('x') ^ 3", "two stars"),
    ("__import__('numpy').exp(__import__('sympy').Symbol('x'))", "use sp.exp"),
    ("__import__('sympy').e", "sp.E"),
    ("__import__('numpy').squareroot", "check the spelling"),
])
def test_hint_for_the_usual_mistakes_of_a_beginner(statement, words):
    if "sympy" in statement:
        pytest.importorskip("sympy")                      # the build environment of the site has no SymPy
    with pytest.raises(Exception) as caught, np.errstate(all="ignore"):
        eval(compile(statement, "<cell>", "exec" if statement.startswith("assert") else "eval"), {})
    assert words in ol.hint_for(caught.value)


def test_hint_for_names_of_the_setup_cell_and_for_errors_without_a_hint():
    for name in ("np", "sp", "plt", "ol"):
        assert "run the whole notebook first" in ol.hint_for(NameError(f"name '{name}' is not defined"))
    assert ol.hint_for(AssertionError("the roots must be real")) is None          # a check with its own message
    assert ol.hint_for(ZeroDivisionError("division by zero")) is None
    assert ol.hint_for(TypeError("can only concatenate str to str")) is None
    assert ol.hint_for(ValueError("x must be positive")) is None
    assert ol.HINTS_INSTALLED is False                                            # plain Python: nothing to install


@pytest.mark.parametrize("source, words", [
    ("y = 3x + 1", "3 * x, not 3x"),
    ("y = 2 x", "a product without"),
    ("y = ", "nothing after ="),
    ("y = 50%", "a sign Python does not have"),
    ("y = 3,05", "a decimal comma"),
    ("y = 3.5m", "a unit after a number"),
    ("y = 3 \u2212 2", "pasted from a document"),
    ("y = 2 \u00d7 3", "pasted from a document"),
    ("3.5 = y", "one = sign only"),
    ("y(x) = 2", "one = sign only"),
    ("y = x' + 1", "remove the apostrophe"),
    ("for n in (1, 2)\n    pass", "ends with a colon"),
    (" y = 3.5", "start at the left edge"),
    ("def g(x):\nreturn x", "start at the left edge"),
    ("y = (1 + 2", "every ( needs its )"),
    ("y = 1 + 2)", "every ( needs its )"),
    ("y = 'abc", "has no partner"),
    ("y = 1.5 m", "a word or a sign"),
])
def test_hint_for_lines_that_python_cannot_read(source, words):
    with pytest.raises(SyntaxError) as caught:
        compile(source, "<cell>", "exec")
    assert words in ol.hint_for(caught.value)


def test_check_explains_the_decimal_comma(capsys):
    assert ol.check("value", (1, 5), 1.5) is False
    assert "write 1.5, with a point" in capsys.readouterr().out
    assert ol.check("pair", [(1, 5), 2], [1.5, 2.0]) is False                     # my_x = 1,5 inside the list of answers
    assert "write 1.5, with a point" in capsys.readouterr().out
    assert ol.check("pair", (1, 5), [1, 5]) is True                               # two numbers were asked for
    assert ol.check("list", (0, 2, 1), [0, 2, 1]) is True
    capsys.readouterr()
    assert ol.check("value", (1, 5, 2), 1.5) is False                             # 1,5,2 or 1.000,5: not one comma
    assert "a comma separates two numbers" in capsys.readouterr().out
    assert ol.check("value", (-3, 5), -3.5) is False                              # a negative number
    assert "write -3.5, with a point" in capsys.readouterr().out
    assert ol.check("list", [0, 5, 2, 1], [0.5, 2, 1]) is False                   # [0,5, 2, 1]: no advice is possible
    capsys.readouterr()


def test_check_says_when_a_value_is_probably_rounded(capsys):
    for rounded, exact in ((0.33, 1 / 3), (1.41, np.sqrt(2)), (2.718, np.e), (3.14, np.pi), (0.0318, 0.03183)):
        assert ol.check("value", rounded, exact) is False
        assert "is close: if you rounded it, type the exact expression" in capsys.readouterr().out
    assert ol.check("value", 0.5, 1 / 3) is False                                 # simply wrong: no such advice
    assert "is close" not in capsys.readouterr().out
    assert ol.check("value", 0.333333, 1 / 3) is True                             # six digits are enough
    assert ol.check("list", [0.33, 2.0], [1 / 3, 2.0]) is False                   # a list: the plain message
    assert "is close" not in capsys.readouterr().out
    assert ol.check("answer", "False", False) is False                            # text where True or False is expected
    assert "type your answer without quotes; got 'False'" in capsys.readouterr().out
