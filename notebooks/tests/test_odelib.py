"""Tests of odelib against values printed in the reference textbooks.

Lebl = J. Lebl, Notes on Diffy Qs: Differential Equations for Engineers (free on LibreTexts)
Adkins-Davidson = W.A. Adkins, M.G. Davidson, Ordinary Differential Equations, Springer 2012
Run from the ode/notebooks folder:  python -m pytest
"""

import warnings

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


# --- linear systems x' = A x (notebook 17; Lebl, Sections 3.1-3.8) ------------------------------
SEED = 331                                  # the seed of test_cross_validation.py
SYSTEM_TIMES = np.linspace(0.0, 2.0, 21)
WINDOW_TIMES = np.linspace(0.0, 10.0, 201)  # the window of the continuity tests
AGREEMENT = 1e-9                            # eigen_solution against expm (plan 17, Section 3)


def _is_eigenpair(A, value, vector):
    A = np.asarray(A, dtype=float)
    return np.linalg.norm(vector) > 0 and np.allclose(A @ vector, value * vector, rtol=1e-12, atol=1e-12)


def _parallel(u, v):
    # u is a nonzero (complex) multiple of v
    u, v = np.asarray(u, dtype=complex), np.asarray(v, dtype=complex)
    k = np.vdot(v, u) / np.vdot(v, v)
    return np.linalg.norm(u) > 0 and np.allclose(u, k * v, rtol=1e-12, atol=1e-12)


def _by_expm(A, start, times):
    expm = pytest.importorskip("scipy.linalg").expm
    A, start = np.asarray(A, dtype=float), np.asarray(start, dtype=float)
    return np.array([expm(A * s) @ start for s in times])


def test_companion_matrix_lebl_example_3_1_4_and_notebook_16():
    # Lebl, Example 3.1.4: x''' = 2x'' + 8x' + x, that is x''' - 2x'' - 8x' - x = 0 (forcing t left out)
    assert np.array_equal(ol.companion_matrix([1, -2, -8, -1]), [[0, 1, 0], [0, 0, 1], [1, 8, 2]])
    # notebook 16, Lebl Example 2.3.3: y''' - 3y'' - y' + 3y = 0 has the roots -1, 1, 3
    roots = np.sort(np.linalg.eigvals(ol.companion_matrix([1, -3, -1, 3])).real)
    assert roots == pytest.approx([-1, 1, 3])
    assert ol.companion_matrix([2.0, 3.0]) == pytest.approx(np.array([[-1.5]]))      # order 1: [[-a0 / a1]]
    assert ol.companion_matrix([2, 3, 5]) == pytest.approx(np.array([[0, 1], [-2.5, -1.5]]))


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_companion_matrix_has_the_characteristic_polynomial_of_the_equation(order):
    numbers = np.random.default_rng(SEED + order).uniform(-3.0, 3.0, order + 1)
    numbers[0] = np.copysign(abs(numbers[0]) + 0.5, numbers[0])                  # a_n not too small
    matrix = ol.companion_matrix(numbers)
    assert matrix.shape == (order, order)
    assert np.poly(matrix) == pytest.approx(numbers / numbers[0], abs=1e-10)     # det(lambda I - A) = p / a_n
    eigenvalues = np.sort_complex(np.linalg.eigvals(matrix))
    assert eigenvalues == pytest.approx(np.sort_complex(np.roots(numbers)), abs=1e-8)


def test_companion_matrix_agrees_with_higher_order_solution():
    coefficients, start = [1, -3, -1, 3], [1.0, 2.0, 3.0]                  # notebook 16, Lebl Example 2.3.3
    by_system = _by_expm(ol.companion_matrix(coefficients), start, SHORT)[:, 0]
    assert by_system == pytest.approx(ol.higher_order_solution(SHORT, coefficients, start), rel=1e-12)


@pytest.mark.parametrize("bad", [[0, 1, 1], [1], [], [1, np.nan, 2], [[1, 2], [3, 4]], [1, [2, 3]], "abc",
                                 [1, np.inf]])
def test_companion_matrix_refuses_what_is_not_an_equation(bad):
    with pytest.raises(ValueError, match="coefficients must be"):
        ol.companion_matrix(bad)


ROOT_3 = np.sqrt(3)
# name: matrix, case, eigenvalues, a vector each eigenvector must be parallel to
BOOK_MATRICES = {
    "Lebl, Example 3.8.1": ([[1, 2], [2, 1]], ol.CASE_REAL, (-1, 3), ((1, -1), (1, 1))),
    "Lebl 3.4, complex eigenvalues": ([[1, 1], [-1, 1]], ol.CASE_COMPLEX, (1 + 1j, 1 - 1j),
                                      ((1, 1j), (1, -1j))),
    "Lebl, Exercise 3.4.3": ([[3, 1], [2, 4]], ol.CASE_REAL, (2, 5), ((1, -1), (1, 2))),
    "Lebl, Exercise 3.4.4": ([[1, -2], [2, 1]], ol.CASE_COMPLEX, (1 + 2j, 1 - 2j), ((1j, 1), (-1j, 1))),
    "Lebl, Exercise 3.4.9": ([[1, 1], [-1, 0]], ol.CASE_COMPLEX,          # with the eigenvectors of the book
                             ((1 + ROOT_3 * 1j) / 2, (1 - ROOT_3 * 1j) / 2),
                             ((-2, 1 - ROOT_3 * 1j), (-2, 1 + ROOT_3 * 1j))),
    "Lebl, Example 3.7.1": ([[3, 1], [0, 3]], ol.CASE_REPEATED_ONE, (3, 3), ((1, 0),)),
    "Lebl 3.8, the simple case": ([[5, 4], [-1, 1]], ol.CASE_REPEATED_ONE, (3, 3), ((2, -1),)),
    "Lebl, Exercise 3.7.1": ([[5, -3], [3, -1]], ol.CASE_REPEATED_ONE, (2, 2), ((1, 1),)),
    "Lebl, Exercise 3.8.9": ([[-1, -1], [1, -3]], ol.CASE_REPEATED_ONE, (-2, -2), ((1, 1),)),
    "Lebl, Exercise 3.8.14": ([[3, -1], [1, 1]], ol.CASE_REPEATED_ONE, (2, 2), ((1, 1),)),
    "3 I": ([[3, 0], [0, 3]], ol.CASE_REPEATED_TWO, (3, 3), ((1, 0), (0, 1))),
    "the closed tanks, Q / V = 0.01 per second": ([[-0.01, 0.01], [0.01, -0.01]], ol.CASE_REAL, (-0.02, 0),
                                                  ((1, -1), (1, 1))),
    "Lebl, Example 3.6.2, rail cars": ([[-1, 1], [2, -2]], ol.CASE_REAL, (-3, 0), ((1, -2), (1, 1))),
    "Lebl, Example 3.6.1": ([[-3, 1], [2, -2]], ol.CASE_REAL, (-4, -1), ((1, -1), (1, 2))),
    "diagonal (breaks the shortcut)": ([[2, 0], [0, 3]], ol.CASE_REAL, (2, 3), ((1, 0), (0, 1))),
    "Lebl, Example 3.3.2, triangular": ([[1, 0], [1, -1]], ol.CASE_REAL, (-1, 1), ((0, 1), (2, 1))),
    "almost diagonal, defective": ([[3, 1e-8], [0, 3]], ol.CASE_REPEATED_ONE, (3, 3), ((1, 0),)),
    "zero matrix": ([[0, 0], [0, 0]], ol.CASE_REPEATED_TWO, (0, 0), ((1, 0), (0, 1))),
}


@pytest.mark.parametrize("name", sorted(BOOK_MATRICES))
def test_two_by_two_case_on_the_matrices_of_the_book(name):
    matrix, case, values, directions = BOOK_MATRICES[name]
    found = ol.two_by_two_case(matrix)
    A = np.asarray(matrix, dtype=float)
    assert found.case == case
    assert found.trace == pytest.approx(np.trace(A))
    assert found.determinant == pytest.approx(np.linalg.det(A), abs=1e-14)
    assert found.discriminant == pytest.approx(np.trace(A) ** 2 - 4 * np.linalg.det(A), abs=1e-12)
    expected = np.array(values, dtype=complex)
    assert np.array(found.eigenvalues, dtype=complex) == pytest.approx(expected, abs=1e-12)
    assert len(found.eigenvectors) == len(directions)
    if case != ol.CASE_REPEATED_ONE:
        for value, vector, direction in zip(found.eigenvalues, found.eigenvectors, directions, strict=True):
            assert _is_eigenpair(A, value, vector) and _parallel(vector, direction)
        assert found.generalized is None
    else:
        vector, value = found.eigenvectors[0], found.eigenvalues[0]
        assert _is_eigenpair(A, value, vector) and _parallel(vector, directions[0])
        assert np.array_equal((A - value * np.eye(2)) @ found.generalized, vector)   # exactly, as by hand
    if case == ol.CASE_COMPLEX:
        assert found.eigenvalues[0].imag > 0
    else:
        assert found.eigenvalues[0] <= found.eigenvalues[1]


def test_two_by_two_case_gives_the_vectors_of_lebl_example_3_7_1():
    found = ol.two_by_two_case([[3, 1], [0, 3]])
    assert np.array_equal(found.eigenvectors[0], [1, 0]) and np.array_equal(found.generalized, [0, 1])
    assert repr(found).count("\n") == 0                                              # prints in one line
    tanks = ol.two_by_two_case([[-0.01, 0.01], [0.01, -0.01]])
    assert tanks.eigenvalues[1] == 0.0 and str(tanks.eigenvalues[1]) == "0.0"       # not -0.0


@pytest.mark.parametrize("bad", [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5]],
                                 [[1, np.nan], [0, 1]], [[1, 1j], [0, 1]], [1, 2], "abc", [[1, 2], [3, "x"]]])
def test_two_by_two_case_refuses_what_is_not_a_real_2_by_2_matrix(bad):
    with pytest.raises(ValueError):
        ol.two_by_two_case(bad)


@pytest.mark.parametrize("rtol", [0.0, -1e-12, np.nan])
def test_two_by_two_case_refuses_a_bad_tolerance(rtol):
    with pytest.raises(ValueError):
        ol.two_by_two_case([[1, 2], [2, 1]], rtol)


def test_eigen_solution_reproduces_the_solutions_of_the_book():
    t = SYSTEM_TIMES
    # Lebl, Example 3.3.2: x1 = e^t, x2 = e^t / 2 + 3 e^(-t) / 2
    expected = np.column_stack([np.exp(t), np.exp(t) / 2 + 3 * np.exp(-t) / 2])
    assert ol.eigen_solution(t, [[1, 0], [1, -1]], [1, 2]) == pytest.approx(expected, rel=1e-12)
    # Lebl, Example 3.8.1: x = 3 e^(3t) + e^(-t), y = 3 e^(3t) - e^(-t)
    expected = np.column_stack([3 * np.exp(3 * t) + np.exp(-t), 3 * np.exp(3 * t) - np.exp(-t)])
    assert ol.eigen_solution(t, [[1, 2], [2, 1]], [4, 2]) == pytest.approx(expected, rel=1e-12)
    # Lebl, Example 3.1.6 (x' = 2y - x, y' = x, x(0) = 1, y(0) = 0): x(2) = 2.475 and y(2) = 2.457, as printed
    assert ol.eigen_solution(2.0, [[-1, 2], [1, 0]], [1, 0]) == pytest.approx([2.475, 2.457], abs=5e-4)
    # Lebl, Exercise 3.8.14: x = ((1 - t) e^(2t), (2 - t) e^(2t))
    expected = np.column_stack([(1 - t) * np.exp(2 * t), (2 - t) * np.exp(2 * t)])
    assert ol.eigen_solution(t, [[3, -1], [1, 1]], [1, 2]) == pytest.approx(expected, rel=1e-12, abs=1e-12)
    # Lebl 3.4, complex eigenvalues: C1 (e^t sin t, e^t cos t) + C2 (e^t cos t, -e^t sin t), at 0: (C2, C1)
    C1, C2 = 3.0, 2.0
    expected = np.exp(t)[:, None] * np.column_stack([C1 * np.sin(t) + C2 * np.cos(t),
                                                     C1 * np.cos(t) - C2 * np.sin(t)])
    assert ol.eigen_solution(t, [[1, 1], [-1, 1]], [C2, C1]) == pytest.approx(expected, rel=1e-12, abs=1e-12)
    # Lebl, Example 3.7.1: C1 (1, 0) e^(3t) + C2 ((0, 1) + (1, 0) t) e^(3t), at 0: (C1, C2)
    expected = np.exp(3 * t)[:, None] * np.column_stack([C1 + C2 * t, C2 + 0 * t])
    assert ol.eigen_solution(t, [[3, 1], [0, 3]], [C1, C2]) == pytest.approx(expected, rel=1e-12)


def test_eigen_solution_of_the_closed_tanks_and_its_shapes():
    rate = 0.01                                          # Q / V = (1 L/s) / (100 L)
    tanks = [[-rate, rate], [rate, -rate]]
    t = np.linspace(0.0, 300.0, 31)
    salt = ol.eigen_solution(t, tanks, [20.0, 0.0])
    expected = np.column_stack([10 + 10 * np.exp(-t / 50), 10 - 10 * np.exp(-t / 50)])
    assert salt == pytest.approx(expected, rel=1e-12)
    assert salt.sum(axis=1) == pytest.approx(20.0, rel=1e-13)                         # the total salt stays
    for matrix in ([[1, 2], [2, 1]], [[1, 1], [-1, 1]], [[3, 1], [0, 3]], [[3, 0], [0, 3]]):   # four cases
        assert ol.eigen_solution(0.0, matrix, [1.5, -2.0]) == pytest.approx([1.5, -2.0], rel=1e-14, abs=1e-15)
    assert ol.eigen_solution(1.0, tanks, [1, 2]).shape == (2,)
    assert ol.eigen_solution([0.0, 1.0, 2.0], tanks, [1, 2]).shape == (3, 2)
    assert ol.eigen_solution(np.zeros((4, 5)), tanks, [1, 2]).shape == (4, 5, 2)
    backwards = ol.eigen_solution(-1.0, [[1, 0], [0, 2]], [1, 1])
    assert backwards == pytest.approx([np.exp(-1), np.exp(-2)])                      # to the left of t = 0


@pytest.mark.parametrize("size", [0.1, 1.0, 10.0])
def test_eigen_solution_agrees_with_expm_on_random_matrices(size):
    rng = np.random.default_rng(SEED)
    for _ in range(50):
        A = rng.uniform(-size, size, (2, 2))
        start = rng.uniform(-1.0, 1.0, 2)
        times = SYSTEM_TIMES / size                       # the same number of "time constants" for every size
        reference = _by_expm(A, start, times)
        scale = np.maximum(np.linalg.norm(reference, axis=1), np.linalg.norm(start))
        error = np.linalg.norm(ol.eigen_solution(times, A, start) - reference, axis=1)
        assert np.all(error <= AGREEMENT * scale), A


@pytest.mark.parametrize("gap", [0.0] + [sign * 10.0 ** -power for power in range(3, 15) for sign in (1, -1)])
def test_eigen_solution_is_continuous_across_the_boundaries_of_the_cases(gap):
    # x'' + c x' + x = 0 as a system: complex pair below c = 2, repeated at 2, real distinct above
    damped = [[0.0, 1.0], [-1.0, -(2.0 + gap)]]
    expected = _by_expm(damped, [1.0, 0.0], WINDOW_TIMES)
    assert ol.eigen_solution(WINDOW_TIMES, damped, [1.0, 0.0]) == pytest.approx(expected, abs=AGREEMENT)
    # [[-1, 1], [e, -1]]: real for e > 0, complex for e < 0, one eigenvector at e = 0
    coupled = [[-1.0, 1.0], [gap, -1.0]]
    expected = _by_expm(coupled, [1.0, 2.0], WINDOW_TIMES)
    assert ol.eigen_solution(WINDOW_TIMES, coupled, [1.0, 2.0]) == pytest.approx(expected, abs=AGREEMENT)


def test_eigen_solution_refuses_what_it_cannot_compute():
    with pytest.raises(ValueError, match="scipy.linalg.expm"):              # Lebl, Example 3.4.4: 3 x 3
        ol.eigen_solution(SYSTEM_TIMES, [[2, 1, 1], [1, 2, 0], [0, 0, 2]], [1, 0, 0])
    symmetric = [[1, 2], [2, 1]]
    for matrix, start, times in ((symmetric, [1, 2, 3], 1.0), (symmetric, [np.nan, 0], 1.0),
                                 ([[1, np.nan], [2, 1]], [1, 0], 1.0), (symmetric, [1, 0], np.inf),
                                 (symmetric, [1, 0], 1e4), (symmetric, "ab", 1.0)):
        with pytest.raises(ValueError):
            ol.eigen_solution(times, matrix, start)


LEBL_3_4_EXAMPLE = [[2, 1, 1], [1, 2, 0], [0, 0, 2]]       # Lebl, Examples 3.4.2-3.4.3: (1, 1, 0) for 3


def test_check_vector_accepts_every_multiple_of_an_eigenvector(capsys):
    values, vectors = np.linalg.eig(np.array(LEBL_3_4_EXAMPLE, dtype=float))
    unit = vectors[:, np.argmin(np.abs(values - 3))]                         # (0.707, 0.707, 0), any sign
    for answer in ([2, 2, 0], [-1, -1, 0], unit, (1, 1, 0), np.array([1j, 1j, 0])):
        assert ol.check_vector("v", answer, [1, 1, 0], up_to_scale=True) is True
    assert ol.check_vector("v", [1, -1j], [1j, 1], up_to_scale=True) is True            # a complex multiple
    assert capsys.readouterr().out.count("v: correct") == 6


@pytest.mark.parametrize("answer, words", [
    ([1, 0, 0], "does not point along an eigenvector"),
    ([0, 0, 0], "the zero vector is never an eigenvector"),
    ([1, 1], "3 value(s) expected, got 2"),
    (2.0, "3 value(s) expected, got 1"),                                               # a number, not a list
    ([1, 1 + 1e-3, 0], "does not point along an eigenvector"),
    (None, "no answer yet (replace None with your value)"),
    ("1, 1, 0", "type your answer without quotes"),
    ([(1, 5), 2, 0], "write 1.5, with a point"),
    ([[1, 1, 0]], "one list of 3 numbers is expected, without brackets inside it"),
    ([1, "a", 0], "a list of 3 numbers is expected"),
])
def test_check_vector_refuses_without_revealing_or_raising(capsys, answer, words):
    assert ol.check_vector("v", answer, [1, 1, 0], up_to_scale=True) is False
    out = capsys.readouterr().out
    assert words in out and out.count("\n") == 1


def test_check_vector_refuses_the_conjugate_eigenvector(capsys):
    assert ol.check_vector("v", [-1j, 1], [1j, 1], up_to_scale=True) is False
    assert "does not point along" in capsys.readouterr().out


def test_check_vector_without_scale_is_check(capsys):
    expected = [[0, 1], [-1, -2]]
    for answer in (expected, np.array([[0, 1], [-1, -2.0]]), [[0, 1], [-1, -2.1]], [[0, 1]], None, "A", 2.0):
        result = ol.check("A", answer, expected)
        line = capsys.readouterr().out
        assert ol.check_vector("A", answer, expected) is result
        assert capsys.readouterr().out == line


@pytest.mark.parametrize("expected", [[0, 0], [[1, 0], [0, 1]], [np.nan, 1], "ab"])
def test_check_vector_refuses_a_mistake_of_the_notebook(expected):
    with pytest.raises(ValueError):
        ol.check_vector("v", [1, 1], expected, up_to_scale=True)


# --- notebook 17, after the code review (2026-11-20) ----------------------------------------------
def test_two_by_two_case_judges_tiny_and_huge_matrices_like_ordinary_ones():
    tiny = ol.two_by_two_case([[1e-200, 1e-200], [0.0, 2e-200]])
    assert tiny.case == ol.CASE_REAL and tiny.eigenvalues == pytest.approx((1e-200, 2e-200), rel=1e-12)
    huge = ol.two_by_two_case([[1e155, 1.0], [1.0, 2e155]])
    assert huge.case == ol.CASE_REAL and huge.eigenvalues == pytest.approx((1e155, 2e155), rel=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                           # no overflow warning
        assert ol.two_by_two_case([[1e300, 1e300], [-1e300, 1e300]]).case == ol.CASE_COMPLEX


def test_two_by_two_case_nearly_equal_diagonal_has_two_eigenvectors():
    nearly = [[1.0, 0.0], [0.0, 1.0 + 1e-13]]
    found = ol.two_by_two_case(nearly)
    assert found.case == ol.CASE_REPEATED_TWO
    assert ol.eigen_solution(WINDOW_TIMES, nearly, [1.0, 2.0]) == pytest.approx(
        _by_expm(nearly, [1.0, 2.0], WINDOW_TIMES), rel=1e-12)


def test_companion_matrix_refuses_a_tiny_leading_coefficient():
    with pytest.raises(ValueError, match="too small"):
        ol.companion_matrix([1e-320, 1.0, 1.0])


def test_eigen_solution_terms_with_a_zero_constant_stay_zero():
    assert ol.eigen_solution(1000.0, [[1, 0], [0, -1]], [0, 1]) == pytest.approx([0.0, np.exp(-1000.0)])
    for matrix in ([[1, 2], [2, 1]], [[1, 1], [-1, 1]], [[3, 1], [0, 3]], [[3, 0], [0, 3]]):
        assert np.array_equal(ol.eigen_solution(1e4, matrix, [0, 0]), [0.0, 0.0])


def test_check_vector_takes_tol_by_position_and_a_column(capsys):
    assert ol.check_vector("v", [1.0, 1.001], [1, 1], 1e-2, up_to_scale=True) is True
    assert ol.check_vector("v", np.array([[2.0], [2.0]]), [1, 1], up_to_scale=True) is True   # a column
    assert ol.check_vector("A", [[1, 2]], [[1, 2]], 1e-3) is True
    capsys.readouterr()


# --- the phase plane and nonlinear systems (notebook 18; Lebl, Sections 3.5, 8.1-8.3) ----------------
# Lebl, Section 3.5, Cases 1-6: matrix, kind, stability, eigenvalues (as two_by_two_case orders them)
LEBL_CASES = {
    1: ([[1, 1], [0, 2]], ol.KIND_SOURCE, ol.STABILITY_UNSTABLE, (1, 2)),
    2: ([[-1, -1], [0, -2]], ol.KIND_SINK, ol.STABILITY_ASYMPTOTIC, (-2, -1)),
    3: ([[1, 1], [0, -2]], ol.KIND_SADDLE, ol.STABILITY_UNSTABLE, (-2, 1)),
    4: ([[0, 1], [-4, 0]], ol.KIND_CENTER, ol.STABILITY_STABLE, (2j, -2j)),
    5: ([[1, 1], [-4, 1]], ol.KIND_SPIRAL_SOURCE, ol.STABILITY_UNSTABLE, (1 + 2j, 1 - 2j)),
    6: ([[-1, -1], [4, -1]], ol.KIND_SPIRAL_SINK, ol.STABILITY_ASYMPTOTIC, (-1 + 2j, -1 - 2j)),
}
# Lebl, Exercise 3.5.6 (a)-(e), with the answers printed in the book
EXERCISE_356 = {
    "a": ([[1, 1], [1, -1]], ol.KIND_SADDLE, (-np.sqrt(2), np.sqrt(2))),
    "b": ([[1, 1], [0, 2]], ol.KIND_SOURCE, (1, 2)),
    "c": ([[0, -2], [2, 0]], ol.KIND_CENTER, (2j, -2j)),
    "d": ([[1, 3], [-2, -4]], ol.KIND_SINK, (-2, -1)),
    "e": ([[1, -4], [-4, 1]], ol.KIND_SADDLE, (-3, 5)),
}
JACOBIAN_AGREEMENT = 1e-8                   # relative, against the exact Jacobians (plan 18, Section 3)
TINY, HUGE = 1e-200, 1e150                  # scale factors of the scale-invariance test
PORTRAIT_SEED_POINTS = 10                   # seeded random points for the Jacobian of Example 8.1.2
LV = (0.4, 0.01, 0.003, 0.3)                # Lebl's a, b, c, d (Section 8.3.2), per time unit
LV_START = (50.0, 20.0)                     # 50 hares and 20 foxes


def lotka_volterra(x, y):
    a, b, c, d = LV
    return (a - b * y) * x, (c * x - d) * y


def lotka_volterra_constant(x, y):
    # Lebl's constant of motion C = y^a x^d e^(-c x - b y)
    a, b, c, d = LV
    return y ** a * x ** d * np.exp(-c * x - b * y)


def example_811(x, y):
    return y, -x + x ** 2                   # Lebl, Example 8.1.1: x'' = -x + x^2


def example_823(x, y):
    return y, -x + y ** 3                   # Lebl, Example 8.2.3


def pendulum(theta, omega):
    return omega, -np.sin(theta)            # Lebl, Section 8.3.1 with g / L = 1


def linear_field(matrix):
    (a11, a12), (a21, a22) = np.asarray(matrix, dtype=float)
    return lambda x, y: (a11 * x + a12 * y, a21 * x + a22 * y)


@pytest.mark.parametrize("number", sorted(LEBL_CASES))
def test_classify_linear_2d_gives_the_six_rows_of_lebl_table_3_5_1(number):
    matrix, kind, stability, eigenvalues = LEBL_CASES[number]
    found = ol.classify_linear_2d(matrix)
    assert (found.kind, found.stability) == (kind, stability)
    assert np.array(found.eigenvalues, dtype=complex) == pytest.approx(np.array(eigenvalues, dtype=complex))
    case = ol.two_by_two_case(matrix)                          # a thin wrapper: the same numbers
    assert (found.trace, found.determinant, found.eigenvalues, found.case) == (
        case.trace, case.determinant, case.eigenvalues, case.case)
    assert found.trace == pytest.approx(sum(eigenvalues).real)          # lambda1 + lambda2 = tr A
    assert found.determinant == pytest.approx(np.prod(eigenvalues).real)  # lambda1 lambda2 = det A


@pytest.mark.parametrize("item", sorted(EXERCISE_356))
def test_classify_linear_2d_lebl_exercise_3_5_6(item):
    matrix, kind, eigenvalues = EXERCISE_356[item]
    found = ol.classify_linear_2d(matrix)
    assert found.kind == kind
    assert np.array(found.eigenvalues, dtype=complex) == pytest.approx(np.array(eigenvalues, dtype=complex))


@pytest.mark.parametrize("matrix, kind, stability", [
    ([[2, -1], [1, 2]], ol.KIND_SPIRAL_SOURCE, ol.STABILITY_UNSTABLE),          # Exercise 3.5.7: 2 +- i
    ([[1, 1], [0, 1]], ol.KIND_DEGENERATE_SOURCE, ol.STABILITY_UNSTABLE),      # Exercise 3.5.2
    ([[1, 1], [1, 1]], ol.KIND_LINE_OF_EQUILIBRIA, ol.STABILITY_UNSTABLE),     # Exercise 3.5.3
    ([[0, 1], [0, 0]], ol.KIND_PARALLEL_LINES, ol.STABILITY_UNSTABLE),         # Exercise 3.5.8
    ([[-0.01, 0.01], [0.01, -0.01]], ol.KIND_LINE_OF_EQUILIBRIA, ol.STABILITY_STABLE),   # notebook 17's tanks
    ([[-1, 1], [0, -1]], ol.KIND_DEGENERATE_SINK, ol.STABILITY_ASYMPTOTIC),
    ([[2, 0], [0, 2]], ol.KIND_STAR_SOURCE, ol.STABILITY_UNSTABLE),
    ([[-1, 0], [0, -1]], ol.KIND_STAR_SINK, ol.STABILITY_ASYMPTOTIC),
    ([[0, 0], [0, 0]], ol.KIND_ALL_EQUILIBRIA, ol.STABILITY_STABLE),
])
def test_classify_linear_2d_borderline_cases_of_lebl_exercises(matrix, kind, stability):
    found = ol.classify_linear_2d(matrix)
    assert (found.kind, found.stability) == (kind, stability)


@pytest.mark.parametrize("a, b", [(1, 3), (-1, -3), (1, -3), (-1, 3)])
def test_classify_linear_2d_a_diagonal_matrix_is_never_a_spiral(a, b):
    # Lebl, Exercise 3.5.4: the eigenvalues a and b are real
    kind = ol.classify_linear_2d([[a, 0], [0, b]]).kind
    assert kind not in (ol.KIND_SPIRAL_SOURCE, ol.KIND_SPIRAL_SINK, ol.KIND_CENTER)
    assert kind == (ol.KIND_SADDLE if a * b < 0 else ol.KIND_SOURCE if a > 0 else ol.KIND_SINK)


@pytest.mark.parametrize("p, kind", [
    (-3.0, ol.KIND_SPIRAL_SINK), (-1.0, ol.KIND_SPIRAL_SINK), (0.0, ol.KIND_DEGENERATE_SINK),
    (0.5, ol.KIND_SINK), (1.0, ol.KIND_LINE_OF_EQUILIBRIA), (3.0, ol.KIND_SADDLE),
    (-3.0 + 30 * 0.1, ol.KIND_DEGENERATE_SINK),                       # 0, as the slider gives it
    (-3.0 + 40 * 0.1, ol.KIND_LINE_OF_EQUILIBRIA),                    # 1, as the slider gives it
    (1e-10, ol.KIND_SINK), (-1e-10, ol.KIND_SPIRAL_SINK), (1.0 - 1e-10, ol.KIND_SINK),
    (1.0 + 1e-10, ol.KIND_SADDLE),
])
def test_classify_linear_2d_on_the_family_of_slider_1(p, kind):
    # notebook 18, slider 1: A = [[-1, 1], [p, -1]], tr A = -2, det A = 1 - p, discriminant 4 p
    assert ol.classify_linear_2d([[-1.0, 1.0], [p, -1.0]]).kind == kind


@pytest.mark.parametrize("factor", [TINY, HUGE])
def test_classify_linear_2d_does_not_depend_on_the_unit(factor):
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                       # no overflow warning
        for matrix, kind, stability, _ in LEBL_CASES.values():
            assert ol.classify_linear_2d(np.array(matrix) * factor)[:2] == (kind, stability)


@pytest.mark.parametrize("bad", [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, np.nan], [0, 1]], [[1, 1j], [0, 1]],
                                 "abc", [1, 2]])
def test_classify_linear_2d_refuses_what_is_not_a_real_2_by_2_matrix(bad):
    with pytest.raises(ValueError):
        ol.classify_linear_2d(bad)
    with pytest.raises(ValueError):
        ol.classify_linear_2d([[1, 2], [3, 4]], rtol=0.0)


def test_jacobian_lebl_example_8_1_2():
    for point, exact in (((0, 0), [[0, 1], [-1, 0]]), ((1, 0), [[0, 1], [1, 0]])):
        assert ol.jacobian(example_811, point) == pytest.approx(np.array(exact), abs=JACOBIAN_AGREEMENT)
    points = np.random.default_rng(SEED).uniform(-3.0, 3.0, (PORTRAIT_SEED_POINTS, 2))
    for x, y in points:
        exact = np.array([[0, 1], [-1 + 2 * x, 0]])
        assert ol.jacobian(example_811, (x, y)) == pytest.approx(exact, rel=JACOBIAN_AGREEMENT,
                                                                 abs=JACOBIAN_AGREEMENT)


@pytest.mark.parametrize("field, point, exact", [
    (lambda x, y: (-y - x ** 2, -x + y ** 2), (1, -1), [[-2, -1], [-1, -2]]),            # Example 8.2.1
    (lambda x, y: (y + y ** 2 * np.exp(x), x), (0, -1), [[1, -1], [1, 0]]),                # Example 8.2.2
    (lotka_volterra, (100, 40), [[0, -1], [0.12, 0]]),                                    # Section 8.3.2
    (pendulum, (np.pi, 0), [[0, 1], [1, 0]]),                                             # Section 8.3.1
    (linear_field([[0.3, -7.0], [2.5, 1.1]]), (12.0, -4.0), [[0.3, -7.0], [2.5, 1.1]]),   # a linear field
])
def test_jacobian_on_the_examples_of_the_book(field, point, exact):
    assert ol.jacobian(field, point) == pytest.approx(np.array(exact, dtype=float), rel=JACOBIAN_AGREEMENT,
                                                      abs=JACOBIAN_AGREEMENT)


def test_jacobian_in_three_dimensions_lebl_exercise_8_1_9():
    # x' = x + z^2, y' = z^2 - y, z' = z + x^2 at (-1, 1, -1); the linearization u' = u - 2w, v' = -v - 2w,
    # w' = w - 2u (the text copy prints u' = u - 2: a w is lost)
    def field(x, y, z):
        return x + z ** 2, z ** 2 - y, z + x ** 2

    exact = np.array([[1, 0, -2], [0, -1, -2], [-2, 0, 1]], dtype=float)
    assert ol.jacobian(field, (-1, 1, -1)) == pytest.approx(exact, abs=JACOBIAN_AGREEMENT)
    assert ol.jacobian(field, (-1, 1, -1)).shape == (3, 3)


def test_jacobian_refuses_what_it_cannot_compute():
    with pytest.raises(ValueError, match="finite"):
        ol.jacobian(example_811, (np.nan, 0))
    with pytest.raises(ValueError, match="2 numbers"):
        ol.jacobian(lambda x, y: (x, y, x * y), (1, 1))                           # three values for two
    with pytest.raises(ValueError, match="value number 1 of f is not a finite number at the point"):
        ol.jacobian(lambda x, y: (1 / x, y), (0.0, 1.0))
    with pytest.raises(ValueError, match="coordinate 1"):
        ol.jacobian(lambda x, y: (np.sqrt(x), y), (0.0, 1.0))                     # NaN just left of 0
    for h in (0.0, -1e-5):
        with pytest.raises(ValueError):
            ol.jacobian(example_811, (1, 0), h)
    with pytest.raises(ValueError, match="flat list"):
        ol.jacobian(example_811, [[1, 0]])
    with pytest.raises(ZeroDivisionError):                                       # an error of f is f's own
        ol.jacobian(lambda x, y: (1 / 0, y), (1, 1))


@pytest.mark.parametrize("field, point, kind, stability", [
    (lambda x, y: (-y - x ** 2, -x + y ** 2), (0, 0), ol.KIND_SADDLE, ol.STABILITY_UNSTABLE),      # 8.2.1
    (lambda x, y: (-y - x ** 2, -x + y ** 2), (1, -1), ol.KIND_SINK, ol.STABILITY_ASYMPTOTIC),
    (lambda x, y: (y + y ** 2 * np.exp(x), x), (0, 0), ol.KIND_SADDLE, ol.STABILITY_UNSTABLE),     # 8.2.2
    (lambda x, y: (y + y ** 2 * np.exp(x), x), (0, -1), ol.KIND_SPIRAL_SOURCE, ol.STABILITY_UNSTABLE),
    (example_811, (1, 0), ol.KIND_SADDLE, ol.STABILITY_UNSTABLE),
    (lambda x, y: (-x + x ** 2, y), (0, 0), ol.KIND_SADDLE, ol.STABILITY_UNSTABLE),               # 8.2.8 (a)
    (lambda x, y: (-x + x ** 2, y), (1, 0), ol.KIND_STAR_SOURCE, ol.STABILITY_UNSTABLE),
    (lambda x, y: (y - y ** 2 - x, -x), (0, 0), ol.KIND_SPIRAL_SINK, ol.STABILITY_ASYMPTOTIC),    # 8.2.8 (b)
    (lambda x, y: (y - y ** 2 - x, -x), (0, 1), ol.KIND_SADDLE, ol.STABILITY_UNSTABLE),
    (lambda x, y: (x * y, x + y - 1), (1, 0), ol.KIND_SADDLE, ol.STABILITY_UNSTABLE),             # 8.2.8 (c)
    (lambda x, y: (x * y, x + y - 1), (0, 1), ol.KIND_DEGENERATE_SOURCE, ol.STABILITY_UNSTABLE),
    (lotka_volterra, (0, 0), ol.KIND_SADDLE, ol.STABILITY_UNSTABLE),
    (lotka_volterra, (100, 40), ol.KIND_CENTER, ol.STABILITY_INCONCLUSIVE),
])
def test_classify_equilibrium_2d_on_the_examples_and_exercises_of_the_book(field, point, kind, stability):
    found = ol.classify_equilibrium_2d(field, point)
    assert (found.linear.kind, found.stability) == (kind, stability)
    assert found.point == tuple(float(value) for value in point)


def test_classify_equilibrium_2d_eigenvalues_of_the_book():
    sink = ol.classify_equilibrium_2d(lambda x, y: (-y - x ** 2, -x + y ** 2), (1, -1))   # Example 8.2.1
    assert sink.linear.eigenvalues == pytest.approx((-3, -1))
    spiral = ol.classify_equilibrium_2d(lambda x, y: (y + y ** 2 * np.exp(x), x), (0, -1))   # Example 8.2.2
    assert spiral.linear.eigenvalues[0] == pytest.approx(0.5 + 1j * np.sqrt(3) / 2)
    assert spiral.reason == ol.REASON_LINEAR
    # Lebl, Exercise 8.3.7 (b): foxes hunted at 10 per time unit, the equilibrium (550/3, 40) typed as floats
    a, b, c, d = LV
    hunted = ol.classify_equilibrium_2d(lambda x, y: ((a - b * y) * x, (c * x - d) * y - 10), (550 / 3, 40))
    assert hunted.linear.kind == ol.KIND_SPIRAL_SOURCE
    assert hunted.linear.eigenvalues[0] == pytest.approx((5 + 1j * np.sqrt(327)) / 40, rel=1e-8)
    assert hunted.jacobian == pytest.approx(np.array([[0, -11 / 6], [3 / 25, 1 / 4]]), abs=JACOBIAN_AGREEMENT)


def test_classify_equilibrium_2d_centers_and_singular_points_are_inconclusive():
    for field in (example_811, example_823):                     # the same linearization, a center
        found = ol.classify_equilibrium_2d(field, (0, 0))
        assert found.linear.kind == ol.KIND_CENTER and found.stability == ol.STABILITY_INCONCLUSIVE
        assert found.reason == ol.REASON_CENTER
    for k in range(-2, 3):                                       # the pendulum at (k pi, 0)
        found = ol.classify_equilibrium_2d(pendulum, (k * np.pi, 0))
        expected = ((ol.KIND_CENTER, ol.STABILITY_INCONCLUSIVE) if k % 2 == 0
                    else (ol.KIND_SADDLE, ol.STABILITY_UNSTABLE))
        assert (found.linear.kind, found.stability) == expected
    # Lebl, Exercise 8.1.7 (a) at (1, 0), and x' = x^2, y' = y^2 at (0, 0) (Section 8.2.1): det J = 0
    for field, point in ((lambda x, y: (np.sin(np.pi * y) + (x - 1) ** 2, y ** 2 - y), (1, 0)),
                         (lambda x, y: (x ** 2, y ** 2), (0, 0))):
        found = ol.classify_equilibrium_2d(field, point)
        assert found.stability == ol.STABILITY_INCONCLUSIVE and found.reason == ol.REASON_SINGULAR


def test_classify_equilibrium_2d_refuses_a_point_that_is_not_an_equilibrium():
    with pytest.raises(ValueError, match=r"\(x, y\) = \(0.5, 0\) is not an equilibrium: f = 0, g = -0.25"):
        ol.classify_equilibrium_2d(example_811, (0.5, 0))
    with pytest.raises(ValueError, match="2 coordinates"):
        ol.classify_equilibrium_2d(example_811, (0, 0, 0))


@pytest.mark.parametrize("number", sorted(LEBL_CASES))
def test_trajectories_agree_with_eigen_solution(number):
    matrix = LEBL_CASES[number][0]
    starts = [(1.0, 0.5), (-0.3, 0.8)]
    times, paths = ol.trajectories(linear_field(matrix), starts, (0.0, 2.0), 2000)
    for k, start in enumerate(starts):
        exact = ol.eigen_solution(times, matrix, start)
        error = np.max(np.linalg.norm(paths[:, k] - exact, axis=1) / np.linalg.norm(exact, axis=1))
        assert error < AGREEMENT


def test_trajectories_lotka_volterra_against_solve_ivp_and_the_constant_of_motion():
    solve_ivp = pytest.importorskip("scipy.integrate").solve_ivp
    times, paths = ol.trajectories(lotka_volterra, LV_START, (0.0, 40.0), 1600)
    reference = solve_ivp(lambda s, u: lotka_volterra(*u), (0.0, 40.0), LV_START, t_eval=times,
                          rtol=1e-12, atol=1e-12)
    assert np.max(np.abs(paths[:, 0] - reference.y.T)) < 1e-7 * np.max(np.abs(reference.y))
    constant = lotka_volterra_constant(paths[:, 0, 0], paths[:, 0, 1])
    assert np.max(np.abs(constant - constant[0])) < 1e-8 * constant[0]


def test_trajectories_forward_then_backward_returns_to_the_start():
    end = ol.trajectories(lotka_volterra, LV_START, (0.0, 10.0), 1000)[1][-1, 0]
    times, back = ol.trajectories(lotka_volterra, end, (10.0, 0.0), 1000)
    assert times[0] == 10.0 and times[-1] == 0.0
    assert back[-1, 0] == pytest.approx(LV_START, rel=1e-9)


def test_trajectories_a_path_that_leaves_the_window_becomes_nan():
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                       # no RuntimeWarning
        times, paths = ol.trajectories(example_823, [(0.3, 0.0), (0.0, 0.0)], (0.0, 60.0), 6000,
                                       window=((-1, 1), (-1, 1)))
    left = np.isnan(paths[:, 0, 0])
    assert left.any() and np.all(left[np.argmax(left):])                     # NaN from there on
    assert not np.isinf(paths).any()
    assert np.all(paths[:, 1] == 0.0)                                        # the equilibrium stays
    inside = paths[~left, 0]
    assert np.all(np.abs(inside) <= 3.0)                                     # the window enlarged by 1 size
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        blown = ol.trajectories(lambda x, y: (x ** 2, y), (1.0, 1.0), (0.0, 2.0), 200)[1]   # x blows up at 1
    assert np.isnan(blown[-1, 0]).all() and not np.isinf(blown).any()


def test_trajectories_shapes():
    times, paths = ol.trajectories(example_811, (0.5, 0.0), (0.0, 1.0), 10)
    assert times.shape == (11,) and paths.shape == (11, 1, 2)
    times, paths = ol.trajectories(example_811, [(0.5, 0.0), (0.1, 0.1), (0.0, 0.2)], (0.0, 1.0), 10)
    assert paths.shape == (11, 3, 2) and paths[0, 2] == pytest.approx([0.0, 0.2])
    constant = ol.trajectories(lambda x, y: (1.0, 0.0), [(0.0, 0.0), (1.0, 1.0)], (0.0, 2.0), 4)[1]
    assert constant[-1] == pytest.approx(np.array([[2.0, 0.0], [3.0, 1.0]]))   # constants are broadcast


@pytest.mark.parametrize("arguments, words", [
    (((1, 2, 3),), "starts must be"), (([(1, 2, 3)],), "starts must be"), (((np.nan, 0),), "finite"),
    (((0, 0), (1.0, 1.0)), "two different ends"), (((0, 0), (0.0, np.inf)), "finite"),
    (((0, 0), (0.0, 1.0), 0), "positive whole number"), (((0, 0), (0.0, 1.0), 2.5), "positive whole number"),
    (((0, 0), (0.0, 1.0), ol.MAX_STEPS + 1), "too many steps"),
    (((0, 0), (0.0, 1.0), 10, ((1, 0), (0, 1))), "window"),
])
def test_trajectories_refuse_invalid_input(arguments, words):
    starts, *rest = arguments
    if not rest:
        rest = [(0.0, 1.0)]
    with pytest.raises(ValueError, match=words):
        ol.trajectories(example_811, starts, *rest)


def test_trajectories_refuse_a_field_of_the_wrong_shape():
    with pytest.raises(ValueError, match="must return two values"):
        ol.trajectories(lambda x, y: x + y, [(1, 0), (0, 1), (1, 1)], (0.0, 1.0), 10)    # one array
    with pytest.raises(ValueError, match="must return two values"):
        ol.trajectories(lambda x, y: (x, y, x), (1, 0), (0.0, 1.0), 10)


def _axes():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt.subplots()


def test_direction_field_leaves_out_the_equilibria_and_draws_arrows_of_one_length():
    fig, ax = _axes()
    n = 7                                                   # x = -1, -0.5, ..., 2 and y = -1.5, ..., 1.5
    ol.direction_field(ax, example_811, (-1.0, 2.0), (-1.5, 1.5), n)
    (arrows,) = ax.collections
    assert len(arrows.U) == n * n - 2                       # (0, 0) and (1, 0) are equilibria
    box, (width, height) = ax.get_position(), fig.get_size_inches()
    sx, sy = box.width * width / 3.0, box.height * height / 3.0       # inches per unit, window 3 x 3
    on_screen = np.hypot(np.asarray(arrows.U) * sx, np.asarray(arrows.V) * sy)
    assert on_screen == pytest.approx(np.full(n * n - 2, ol.ARROW_LENGTH * 0.5 * min(sx, sy)))
    assert arrows.units == "inches"                                    # the shaft in proportion to the arrow
    assert arrows.width == pytest.approx(ol.ARROW_WIDTH * ol.ARROW_LENGTH * 0.5 * min(sx, sy))
    assert ax.get_xlim() == (-1.0, 2.0) and ax.get_ylim() == (-1.5, 1.5)
    fig, ax = _axes()
    ax.set_aspect("equal")                                  # equal scales: one length in data units
    ol.direction_field(ax, example_811, (-1.0, 2.0), (-1.5, 1.5), n)
    (arrows,) = ax.collections
    lengths = np.hypot(np.asarray(arrows.U), np.asarray(arrows.V))
    assert lengths == pytest.approx(np.full(n * n - 2, ol.ARROW_LENGTH * 0.5))
    for bad in (1, 0, 2.5, True):
        with pytest.raises(ValueError, match="arrows per side"):
            ol.direction_field(ax, example_811, (-1.0, 2.0), (-1.5, 1.5), bad)
    for window in (((0.0, np.inf), (0.0, 1.0)), (("a", "b"), (0.0, 1.0))):
        with pytest.raises(ValueError, match="increasing"):
            ol.direction_field(ax, example_811, *window)


def test_direction_field_with_nan_values_draws_without_a_warning():
    fig, ax = _axes()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ol.direction_field(ax, lambda x, y: (np.sqrt(x), 1.0), (-1.0, 1.0), (-1.0, 1.0), 5)
    assert len(ax.collections[0].U) == 3 * 5                # the columns x = 0, 0.5, 1 only
    with pytest.raises(ValueError, match="increasing"):
        ol.direction_field(ax, example_811, (1.0, -1.0), (-1.0, 1.0))


def test_phase_portrait_draws_two_lines_per_start_and_checks_its_arguments():
    fig, ax = _axes()
    starts = [(0.5, 0.0), (0.2, 0.0), (-0.3, 0.3)]
    ol.phase_portrait(ax, example_811, starts, (0.0, 5.0), (-1.0, 2.0), (-1.5, 1.5), n_steps=100)
    assert len(ax.lines) == 2 * len(starts)
    assert len(ax.texts) == len(starts)                     # one arrowhead on each forward part
    assert ax.get_xlim() == (-1.0, 2.0) and ax.get_ylim() == (-1.5, 1.5)
    fig, ax = _axes()
    ol.phase_portrait(ax, example_811, starts, (0.0, 5.0), (-1.0, 2.0), (-1.5, 1.5), n_steps=100,
                      both_ways=False)
    assert len(ax.lines) == len(starts)
    for t_span in ((0.0, 0.0), (0.0, -1.0)):
        with pytest.raises(ValueError, match="t1 > t0"):
            ol.phase_portrait(ax, example_811, starts, t_span, (-1.0, 2.0), (-1.5, 1.5))
    with pytest.raises(ValueError, match="outside the window"):
        ol.phase_portrait(ax, example_811, [(3.0, 0.0)], (0.0, 1.0), (-1.0, 2.0), (-1.5, 1.5))
    with pytest.raises(ValueError, match="increasing"):
        ol.phase_portrait(ax, example_811, starts, (0.0, 1.0), (2.0, -1.0), (-1.5, 1.5))


@pytest.mark.parametrize("answer", ["sink", " Sink ", "SINK", "stable node", "Stable-Node", "stable_node"])
def test_check_choice_accepts_the_word_in_every_form(capsys, answer):
    assert ol.check_choice("kind", answer, "sink", ol.PORTRAIT_WORDS, ol.KIND_SYNONYMS) is True
    assert capsys.readouterr().out == "kind: correct\n"


def test_check_choice_other_accepted_forms(capsys):
    assert ol.check_choice("kind", "spiral-sink", "spiral sink", ol.PORTRAIT_WORDS, ol.KIND_SYNONYMS) is True
    assert ol.check_choice("kind", "centre", "center", ol.PORTRAIT_WORDS, ol.KIND_SYNONYMS) is True
    assert ol.check_choice("kind", "Saddle point", "saddle", ol.PORTRAIT_WORDS, ol.KIND_SYNONYMS) is True
    assert ol.check_choice("stability", "Asymptotically  stable", "asymptotically stable",
                           ol.STABILITY_WORDS) is True
    assert capsys.readouterr().out.count("correct") == 4


@pytest.mark.parametrize("answer, words", [
    ("source", "kind: not yet, try again"),
    (None, "kind: no answer yet (replace None with your answer, a word in quotes)"),
    (3, 'kind: not yet (type the word in quotes'),
    (["sink"], 'kind: not yet (type the word in quotes'),
    ({"sink": 1}, 'kind: not yet (type the word in quotes'),
    ("spiral", "kind: not yet (not one of the words: source, sink, saddle, center, spiral source, "
               "spiral sink)"),
])
def test_check_choice_refuses_without_revealing_or_raising(capsys, answer, words):
    assert ol.check_choice("kind", answer, "sink", ol.PORTRAIT_WORDS, ol.KIND_SYNONYMS) is False
    out = capsys.readouterr().out
    assert out.startswith(words) and out.count("\n") == 1 and "sink\"" not in out


def test_check_choice_refuses_a_mistake_of_the_notebook():
    with pytest.raises(ValueError):
        ol.check_choice("kind", "sink", "node", ol.PORTRAIT_WORDS)
    with pytest.raises(ValueError):
        ol.check_choice("kind", "sink", None, ol.PORTRAIT_WORDS)
    with pytest.raises(ValueError, match="list of words"):
        ol.check_choice("kind", "sink", "sink", ["sink", 3])
    with pytest.raises(ValueError, match="map words to words"):
        ol.check_choice("kind", "sink", "sink", ol.PORTRAIT_WORDS, ["stable node"])


def test_check_choice_other_books_names_and_a_read_only_table(capsys):
    assert ol.check_choice("kind", "Stable spiral", "spiral sink", ol.PORTRAIT_WORDS, ol.KIND_SYNONYMS)
    assert ol.check_choice("kind", "unstable focus", "spiral source", ol.PORTRAIT_WORDS, ol.KIND_SYNONYMS)
    capsys.readouterr()
    with pytest.raises(TypeError):
        ol.KIND_SYNONYMS["node"] = ol.KIND_SINK


def test_review_of_notebook_18_library_points():
    # det J = 0 with tr J > 0 is unstable (an eigenvalue with positive real part): x' = x + y^2, y' = x^2
    found = ol.classify_equilibrium_2d(lambda x, y: (x + y ** 2, x ** 2), (0, 0))
    assert found.stability == ol.STABILITY_UNSTABLE and found.reason == ol.REASON_SINGULAR_UNSTABLE
    # a shape that has the right count but not the right form gets its own message
    with pytest.raises(ValueError, match="an array of shape"):
        ol.jacobian(lambda x, y: ([x], [y]), (1, 1))
    # windows that are not two increasing finite ranges
    for window in (((0, 1),), (("a", "b"), (0, 1)), ((0, np.inf), (0, 1))):
        with pytest.raises(ValueError, match="window"):
            ol.trajectories(example_811, (0.5, 0.0), (0.0, 1.0), 10, window)


# --- the Laplace transform (notebook 19; Lebl, Sections 6.1 and 6.2) ---------------------------------
import math  # noqa: E402  (used by the tests of this block only)

LAPLACE_S = (0.5, 1.0, 3.0, 10.0)           # values of s > 0, in 1/s when t is in seconds
LAPLACE_OFFSETS = (0.5, 1.0, 4.0)           # each row of the table is checked at s = s_min + offset
LAPLACE_SEED = 331                          # as SEED in test_cross_validation.py
LAPLACE_DRAWS = 20                          # seeded random sets of constants (a, omega, n, s)
LAPLACE_A_RANGE = (-2.0, 2.0)               # the constants a drawn
LAPLACE_OMEGA_RANGE = (0.2, 3.0)            # the constants omega drawn
LAPLACE_N_RANGE = (0, 8)                    # the powers n drawn: 0, ..., 7
LAPLACE_EDGE_MARGIN = (0.3, 5.0)            # s is drawn this far above the edge of the range of a row
LAPLACE_SCAN = 300                          # seeded sines and cosines (40 000 were run once by hand)
LAPLACE_SCAN_OMEGA = (0.2, 5.0)             # their omega
LAPLACE_SCAN_DECADES = (-1.0, 2.0)          # their s, from 10^-1 to 10^2
S_DEMO = 3.0                                # the s of the first values printed by notebook 19
LOOSE_RTOL = 1e-4                           # a looser rtol than the default
ROW_AGREEMENT = 1e-12                       # two ways of computing one formula of the table (relative)
# functions of t >= 0 with the transforms printed in the book (or found from its table), as (f, F)
BOOK_TRANSFORMS = {
    "Exercise 6.1.12": (lambda t: 4 * (t + 1) ** 2, lambda s: 8 / s ** 3 + 8 / s ** 2 + 4 / s),
    "Exercise 6.1.13": (lambda t: 2 * t ** 2 - 2 * t + 1 - np.exp(-2 * t), lambda s: 8 / (s ** 3 * (s + 2))),
    "Exercise 6.1.14": (lambda t: t * np.exp(-t), lambda s: 1 / (s + 1) ** 2),
    "Exercise 6.1.15": (lambda t: np.sin(t) * np.exp(-t), lambda s: 1 / (s ** 2 + 2 * s + 2)),
    "Example 6.1.6": (lambda t: 1 + np.sin(t), lambda s: (s ** 2 + s + 1) / (s ** 3 + s)),
    "Example 6.1.7": (lambda t: np.exp(-2 * t) * np.sin(2 * t) / 2, lambda s: 1 / (s ** 2 + 4 * s + 8)),
    "Example 6.2.1": (lambda t: np.cos(t) / 3 - np.cos(2 * t) / 3 + np.sin(t),
                      lambda s: s / ((s ** 2 + 1) * (s ** 2 + 4)) + 1 / (s ** 2 + 1)),
    "Example 6.2.4": (lambda t: 1 - np.cos(t), lambda s: 1 / (s * (s ** 2 + 1))),
    # the circuit of notebook 19: q'' + 2 q' + 5 q = 8 e^(-t), q(0) = 1, q'(0) = 0
    "notebook 19, circuit": (lambda t: np.exp(-t) * (2 - np.cos(2 * t) + np.sin(2 * t) / 2),
                             lambda s: (8 / (s + 1) + s + 2) / (s ** 2 + 2 * s + 5)),
    # the same circuit with R = 6 ohm, where the root -1 of the source is a root of the circuit
    "notebook 19, R = 6": (lambda t: (3 / 4 + 2 * t) * np.exp(-t) + np.exp(-5 * t) / 4,
                           lambda s: (8 / (s + 1) + s + 6) / (s ** 2 + 6 * s + 5)),
}


def promised(found, exact, s, rtol=ol.LAPLACE_RTOL):
    """True when found is as near to exact as laplace_numeric promises: rtol max(1 / max(|s|, 1), |F(s)|)."""
    return abs(found - exact) <= rtol * max(1 / max(abs(s), ol.LAPLACE_UNIT_RATE), abs(exact))


def test_laplace_numeric_lebl_examples_6_1_1_to_6_1_3():
    for s in LAPLACE_S:
        assert promised(ol.laplace_numeric(lambda t: 1.0, s), 1 / s, s)                 # Example 6.1.1
        assert promised(ol.laplace_numeric(lambda t: t, s), 1 / s ** 2, s)              # Example 6.1.3
        for a in (2.0, 0.25, -0.25):                                                    # Example 6.1.2
            assert promised(ol.laplace_numeric(lambda t, a=a: np.exp(-a * t), s), 1 / (s + a), s)
    assert isinstance(ol.laplace_numeric(lambda t: 1.0, S_DEMO), float)
    # the three values printed first in notebook 19: 1/s, 1/s^2 and 1/(s + 2) at s = 3
    assert ol.laplace_numeric(lambda t: 1.0, S_DEMO) == pytest.approx(1 / 3)
    assert ol.laplace_numeric(lambda t: t, S_DEMO) == pytest.approx(1 / 9)
    assert ol.laplace_numeric(lambda t: np.exp(-2 * t), S_DEMO) == pytest.approx(1 / 5)


def test_laplace_table_is_lebl_table_6_1_1():
    table = ol.laplace_table()                              # a = 2, omega = 3, n = 3
    assert (ol.LAPLACE_DEMO_A, ol.LAPLACE_DEMO_OMEGA, ol.LAPLACE_DEMO_N) == (2.0, 3.0, 3)
    assert tuple(row.name for row in table) == ol.LAPLACE_ROW_NAMES
    assert ol.LAPLACE_ROW_NAMES == ("1", "t", "t^2", "t^n", "e^(-at)", "sin", "cos", "sinh", "cosh")
    # Table 6.1.1 at s = 5: 1/s, 1/s^2, 2/s^3, 6/s^4, 1/(s + a), omega/(s^2 + omega^2), s/(s^2 + omega^2),
    # omega/(s^2 - omega^2), s/(s^2 - omega^2)
    book = [1 / 5, 1 / 25, 2 / 125, 6 / 625, 1 / 7, 3 / 34, 5 / 34, 3 / 16, 5 / 16]
    assert [row.transform(5.0) for row in table] == pytest.approx(book)
    assert [row.s_min for row in table] == [0, 0, 0, 0, -2, 0, 0, 3, 3]
    # f(t) at t = 0.5: 1, t, t^2, t^3, e^(-2t), sin 3t, cos 3t, sinh 3t, cosh 3t
    values = [1, 0.5, 0.25, 0.125, np.exp(-1), np.sin(1.5), np.cos(1.5), np.sinh(1.5), np.cosh(1.5)]
    assert [row.time(0.5) for row in table] == pytest.approx(values)
    assert [row.time_text for row in table] == ["1", "t", "t^2", "t^n", "e^(-a t)", "sin(omega t)",
                                                "cos(omega t)", "sinh(omega t)", "cosh(omega t)"]
    assert [row.transform_text for row in table] == [
        "1/s", "1/s^2", "2/s^3", "n!/s^(n + 1)", "1/(s + a)", "omega/(s^2 + omega^2)", "s/(s^2 + omega^2)",
        "omega/(s^2 - omega^2)", "s/(s^2 - omega^2)"]
    for row in table:
        assert isinstance(row, ol.LaplaceRow) and isinstance(row.s_min, float)
        assert (row.name + row.time_text + row.transform_text).isascii()


def test_laplace_table_rows_take_numbers_lists_and_arrays_in_float_arithmetic():
    times = np.linspace(0.0, 2.0, 5)
    for row in ol.laplace_table():
        assert isinstance(row.time(1), float) and isinstance(row.transform(row.s_min + 1), float)
        assert isinstance(row.time(np.float64(1.0)), float)                 # a NumPy number in, a float out
        assert np.shape(row.time(times)) == times.shape
        assert np.shape(row.transform(row.s_min + 1.0 + times)) == times.shape
        assert row.time([1, 2]) == pytest.approx([row.time(1.0), row.time(2.0)])          # a list of integers
        assert np.isfinite(row.time(0.0))
    rows = {row.name: row for row in ol.laplace_table()}                    # the idiom of the docstring
    assert rows["1"].time(np.inf) == 1.0
    # integers do not overflow (code review of notebook 19, B3)
    assert ol.laplace_table(n=20)[3].time(np.arange(10))[-1] == pytest.approx(9.0 ** 20)
    assert ol.laplace_table(n=9)[3].transform(np.int32(100)) == pytest.approx(math.factorial(9) / 100.0 ** 10)
    assert rows["t^2"].transform(np.int32(2000)) == pytest.approx(2 / 2000.0 ** 3)
    huge = ol.laplace_table(n=np.int8(127))[3].transform(2.0)
    assert huge == pytest.approx(math.factorial(127) / 2.0 ** 128)
    assert ol.laplace_table(n=70)[3].time(np.int64(10)) == pytest.approx(1e70)
    assert format(ol.laplace_table(a=0)[4].s_min, "g") == "0"                # not "-0"


@pytest.mark.parametrize("name", ol.LAPLACE_ROW_NAMES)
def test_laplace_table_rows_agree_with_the_defining_integral(name):
    (row,) = [row for row in ol.laplace_table() if row.name == name]
    for offset in LAPLACE_OFFSETS:
        s = row.s_min + offset                              # e^(-2t): also s = -1.5 and s = -1
        assert promised(ol.laplace_numeric(row.time, s), row.transform(s), s)


def test_laplace_table_with_other_constants():
    # many books print e^(at) with 1/(s - a): the row of the book with -a in place of a
    rows = {row.name: row for row in ol.laplace_table(a=-3.0, omega=0.5, n=0)}
    assert rows["e^(-at)"].s_min == 3.0 and rows["e^(-at)"].transform(4.0) == pytest.approx(1.0)
    assert rows["e^(-at)"].time(1.0) == pytest.approx(np.exp(3.0))
    assert rows["t^n"].time(7.0) == 1.0 and rows["t^n"].transform(4.0) == pytest.approx(0.25)     # t^0 = 1
    assert rows["cosh"].s_min == 0.5
    assert ol.laplace_table(n=np.int64(5))[3].transform(1.0) == pytest.approx(120.0)             # 5! = 120
    with pytest.raises(TypeError):
        ol.laplace_table(-3.0)                              # the constants are given by name


def test_laplace_table_transform_refuses_s_at_or_below_the_edge_of_its_range():
    rows = {row.name: row for row in ol.laplace_table()}
    for name, s in (("1", 0.0), ("1", np.float64(0.0)), ("1", -1.0), ("sinh", 1.0), ("cosh", 3.0),
                    ("e^(-at)", -2.0), ("t", np.linspace(0.0, 5.0, 3)), ("t", np.nan)):
        with pytest.raises(ValueError, match="holds only for s > "):
            rows[name].transform(s)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                      # no overflow warning: a sentence
        with pytest.raises(ValueError, match="too large for the computer"):
            ol.laplace_table(n=170)[3].transform(0.5)       # 170! / 0.5^171
        with pytest.raises(ValueError, match="too large for the computer"):
            rows["t^2"].transform(1e-200)                   # s^3 is 0 for the computer: no RuntimeWarning
    for s in (2 + 0j, np.complex128(2), "2", True, np.True_, [1.0, "a"]):
        with pytest.raises(ValueError, match="s must be a real number, a list or an array of real numbers"):
            rows["1"].transform(s)


def test_laplace_table_rows_agree_with_the_integral_for_seeded_random_constants():
    rng = np.random.default_rng(LAPLACE_SEED)
    for _ in range(LAPLACE_DRAWS):
        a, omega = rng.uniform(*LAPLACE_A_RANGE), rng.uniform(*LAPLACE_OMEGA_RANGE)
        n, margin = int(rng.integers(*LAPLACE_N_RANGE)), rng.uniform(*LAPLACE_EDGE_MARGIN)
        for row in ol.laplace_table(a=a, omega=omega, n=n):
            s = row.s_min + margin
            assert promised(ol.laplace_numeric(row.time, s), row.transform(s), s), (row.name, a, omega, n, s)


def test_laplace_numeric_keeps_its_promise_on_a_seeded_scan_of_sines_and_cosines():
    # the scan of the code review (B2): 40 000 cases were run once by hand, none beyond the promise
    rng = np.random.default_rng(LAPLACE_SEED)
    for _ in range(LAPLACE_SCAN):
        omega, s = rng.uniform(*LAPLACE_SCAN_OMEGA), 10 ** rng.uniform(*LAPLACE_SCAN_DECADES)
        size = s * s + omega * omega
        assert promised(ol.laplace_numeric(lambda t, omega=omega: math.sin(omega * t), s), omega / size, s)
        assert promised(ol.laplace_numeric(lambda t, omega=omega: math.cos(omega * t), s), s / size, s)
    s = 14.138                                              # where one integral alone was wrong by 6e-8
    assert promised(ol.laplace_numeric(lambda t: math.sin(3 * t), s), 3 / (s * s + 9), s)


@pytest.mark.parametrize("name", sorted(BOOK_TRANSFORMS))
def test_laplace_numeric_gives_the_transforms_of_the_book(name):
    f, transform = BOOK_TRANSFORMS[name]
    for s in LAPLACE_S:
        assert promised(ol.laplace_numeric(f, s), transform(s), s)


def test_laplace_numeric_linearity_and_the_product_warning():
    def f(t):
        return np.exp(-t)

    def g(t):
        return np.cos(2 * t)

    s = 2.0
    F, G = ol.laplace_numeric(f, s), ol.laplace_numeric(g, s)
    # Theorem 6.1.1: L{A f + B g} = A L{f} + B L{g}
    assert ol.laplace_numeric(lambda t: 3 * f(t) - 5 * g(t), s) == pytest.approx(3 * F - 5 * G)
    # Lebl's warning: the transform of a product is not the product of the transforms (notebook 19)
    product = ol.laplace_numeric(lambda t: t * t, s)
    assert product == pytest.approx(1 / 4)
    assert ol.laplace_numeric(lambda t: t, s) ** 2 == pytest.approx(1 / 16)
    # a transform that is 0 is found too (the accuracy is absolute): L{1 - 2t} at s = 2
    assert promised(ol.laplace_numeric(lambda t: 1 - 2 * t, s), 0.0, s)


def test_laplace_numeric_the_rules_of_lebl_section_6_2():
    for s in LAPLACE_S:
        # Table 6.2.1 on g = 1 + sin 2t: L{g'} = s G - g(0), L{g''} = s^2 G - s g(0) - g'(0)
        G = ol.laplace_numeric(lambda t: 1 + np.sin(2 * t), s)
        assert promised(ol.laplace_numeric(lambda t: 2 * np.cos(2 * t), s), s * G - 1, s)
        assert ol.laplace_numeric(lambda t: -4 * np.sin(2 * t), s) == pytest.approx(s ** 2 * G - s * 1 - 2)
        # Exercise 6.2.7 on f = sin t: L{-t f(t)} = F'(s), with F = 1/(s^2 + 1), F' = -2 s/(s^2 + 1)^2
        assert promised(ol.laplace_numeric(lambda t: -t * np.sin(t), s), -2 * s / (s ** 2 + 1) ** 2, s)
        # the first shifting property (Section 6.1.4): L{e^(-at) f(t)} = F(s + a), on f = cos 3t, a = 2
        assert promised(ol.laplace_numeric(lambda t: np.exp(-2 * t) * np.cos(3 * t), s),
                        (s + 2) / ((s + 2) ** 2 + 9), s)


def test_laplace_numeric_the_range_of_s():
    def growing(t):
        return np.exp(2 * t)                                # L{e^(2t)} = 1/(s - 2) for s > 2

    assert ol.laplace_numeric(growing, 3) == pytest.approx(1.0)
    assert ol.laplace_numeric(growing, 2.1) == pytest.approx(10.0)          # 5% above the edge
    with warnings.catch_warnings():
        warnings.simplefilter("error")                      # no overflow warning reaches the student
        for s in (1, 2, 2.01, 2.05, 0, -1):                 # below the edge, on it, too near to it
            with pytest.raises(ValueError, match="does not converge for s = "):
                ol.laplace_numeric(growing, s)
    # a decaying function has a transform at s = 0 and for some s < 0: L{e^(-2t)} = 1/(s + 2), s > -2
    for s in (0, -1, -1.5, -1.9):
        assert ol.laplace_numeric(lambda t: np.exp(-2 * t), s) == pytest.approx(1 / (s + 2))
    for s in (-1.99, -2, -3):
        with pytest.raises(ValueError, match="does not converge"):
            ol.laplace_numeric(lambda t: np.exp(-2 * t), s)
    with pytest.raises(ValueError, match="does not converge for s = 0,"):
        ol.laplace_numeric(lambda t: 1.0, -0.0)             # L{1} needs s > 0; -0.0 is written 0
    # the documented thresholds near the edge 0: 1 and t^3 at s = 1e-4, sin t at s = 1/100
    assert ol.laplace_numeric(lambda t: 1.0, 1e-4) == pytest.approx(1e4)
    assert ol.laplace_numeric(lambda t: t ** 3, 1e-4) == pytest.approx(6e16)
    assert promised(ol.laplace_numeric(math.sin, 0.01), 1 / (1 + 0.01 ** 2), 0.01)
    with pytest.raises(ValueError, match="does not converge"):
        ol.laplace_numeric(math.sin, 1 / 300)               # below about omega / 200: refused
    # at s = 0 only functions that die out like an exponential are computed (documented)
    with pytest.raises(ValueError, match="does not converge"):
        ol.laplace_numeric(lambda t: 1 / (1 + t) ** 2, 0)


def two_to_the_t_squared(t):
    return 2.0 ** (t * t)                                   # raises OverflowError where np.exp gives inf


@pytest.mark.parametrize("name, f, s", [
    ("e^(t^2)", lambda t: np.exp(t * t), 3.0),              # Lebl: no transform
    ("e^(t^2), a larger s", lambda t: np.exp(t * t), 30.0),         # refused by the growth of |f| before T
    ("2^(t^2)", two_to_the_t_squared, 3.0),                 # an OverflowError inside f
    ("1/t", lambda t: 1 / t, 1.0),                          # Lebl: the integral diverges for all s
    ("1/t, a large s", lambda t: 1 / t, 1000.0),
    ("tan t", lambda t: np.tan(t), 1.0),                    # Lebl: no transform
    ("tan t, s = 10", lambda t: np.tan(t), 10.0),
    ("1/(t - 1)", lambda t: 1 / (t - 1), 3.0),
    ("NaN", lambda t: np.nan, 1.0),
    ("ln(t - 1) with np.log", lambda t: np.log(t - 1), 1.0),        # not defined for t < 1: NaN
    ("ln(t - 1) with math.log", lambda t: math.log(t - 1), 1.0),    # a ValueError inside f
    ("1/(t - t)", lambda t: 1 / (t - t), 1.0),              # a ZeroDivisionError inside f
])
def test_laplace_numeric_refuses_functions_without_a_transform(name, f, s):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="does not converge"):
            ol.laplace_numeric(f, s)


def test_laplace_numeric_documented_limits_a_number_where_no_transform_exists():
    # a numerical integral cannot prove convergence: these wrong numbers are documented in the
    # docstring ("what is promised, and what is not"); the test tells if the limits move
    assert ol.laplace_numeric(lambda t: np.exp(t * t), 40.0) == pytest.approx(1 / 40, rel=0.02)
    assert ol.laplace_numeric(lambda t: np.tan(t), 20.0) == pytest.approx(1 / 20 ** 2, rel=0.02)
    assert ol.laplace_numeric(lambda t: float(t) ** t, 10.0) == pytest.approx(0.0833, rel=0.01)
    # the other side of the same limits: refused for a smaller s (1 / (t - 1) at s = 3 is in the list above),
    # and 1 / (t - 1) also at s = 30, where the first look at f falls on t = 1 itself
    for f, s in ((lambda t: float(t) ** t, 4.0), (lambda t: np.exp(t * t), 30.0),
                 (lambda t: np.tan(t), 10.0), (lambda t: 1 / (t - 1), 30.0)):
        with pytest.raises(ValueError, match="does not converge"):
            ol.laplace_numeric(f, s)


def test_laplace_numeric_steps_and_pulses_with_and_without_breaks():
    def pulse(t):
        return 1.0 if 5 <= t < 6 else 0.0                   # code review of notebook 19, B1

    def late_step(t):
        return 1.0 if t >= 300 else 0.0

    def narrow(t):
        return 100.0 if 5 <= t <= 5.01 else 0.0             # 0.01 wide: found without breaks since the scan

    def hair(t):
        return 100.0 if HAIR_START <= t < HAIR_START + HAIR_WIDTH else 0.0   # between two looks at f: missed

    for s in (1.0, 0.5):
        exact = (math.exp(-5 * s) - math.exp(-6 * s)) / s
        assert promised(ol.laplace_numeric(pulse, s), exact, s)
        assert promised(ol.laplace_numeric(pulse, s, breaks=(5, 6)), exact, s)
    exact = math.exp(-300 * 0.01) / 0.01
    assert promised(ol.laplace_numeric(late_step, 0.01), exact, 0.01)
    assert promised(ol.laplace_numeric(late_step, 0.01, breaks=[300]), exact, 0.01)
    assert promised(ol.laplace_numeric(lambda t: 1.0 if t >= 5 else 0.0, 3.0), math.exp(-15) / 3, 3.0)
    exact = 100 * (math.exp(-5 * 0.5) - math.exp(-5.01 * 0.5)) / 0.5
    assert promised(ol.laplace_numeric(narrow, 0.5, breaks=(5, 5.01)), exact, 0.5)
    assert promised(ol.laplace_numeric(narrow, 0.5), exact, 0.5)
    exact = 100 * (math.exp(-HAIR_START * 0.5) - math.exp(-(HAIR_START + HAIR_WIDTH) * 0.5)) / 0.5
    assert promised(ol.laplace_numeric(hair, 0.5, breaks=(HAIR_START, HAIR_START + HAIR_WIDTH)), exact, 0.5)
    assert ol.laplace_numeric(hair, 0.5) == 0.0             # the documented limit: missed without breaks


HAIR_START, HAIR_WIDTH = 5.003, 0.001       # a pulse far narrower than 1/200 of its distance from t = 0
PULSE_CASES = 60                            # seeded pulses that laplace_numeric must find by itself
PULSE_WIDTH_SHARE = (0.01, 0.3)             # their width, as a part of their distance from t = 0


def test_laplace_numeric_finds_switches_and_pulses_by_itself():
    # the review of the limits (2026-11-23): without breaks a pulse was refused or, narrower, returned 0
    rng = np.random.default_rng(SEED)
    for _ in range(PULSE_CASES):
        start = 10 ** rng.uniform(-1, 1)
        width = rng.uniform(*PULSE_WIDTH_SHARE) * start
        s = 10 ** rng.uniform(-0.5, 0.7)
        height = rng.uniform(0.5, 5)
        smooth = float(rng.integers(0, 2))                  # with or without e^(-t) under the pulse

        def f(t, start=start, width=width, height=height, smooth=smooth):
            return smooth * math.exp(-t) + (height if start <= t < start + width else 0.0)

        exact = height * (math.exp(-start * s) - math.exp(-(start + width) * s)) / s + smooth / (s + 1)
        assert promised(ol.laplace_numeric(f, s), exact, s)
    # a sine switched on at pi (Lebl, Section 6.2: u(t - pi) sin t): F(s) = -e^(-pi s) / (s^2 + 1)
    assert promised(ol.laplace_numeric(lambda t: math.sin(t) if t >= math.pi else 0.0, 1.0),
                    -math.exp(-math.pi) / 2, 1.0)
    # the scan must not turn a smooth oscillation into a refusal (1 + sin t at s = 0.5, notebook 19)
    assert promised(ol.laplace_numeric(lambda t: 1 + math.sin(t), 0.5), 1 / 0.5 + 1 / (0.5 ** 2 + 1), 0.5)
    # breaks that change nothing: before 0 is refused, beyond the last time T they are not used
    assert ol.laplace_numeric(lambda t: 1.0, 2.0, breaks=(1.0, 1e9)) == pytest.approx(0.5)
    for bad in ([-1.0], [np.nan], [np.inf], 3, "ab", [None], "56", [True], [np.True_], ["5"]):
        with pytest.raises(ValueError, match="breaks must be a list of finite times"):
            ol.laplace_numeric(lambda t: 1.0, 2.0, breaks=bad)


def test_laplace_numeric_sees_a_fast_feature_at_the_start():
    # mathematics review of notebook 19, finding 3: e^(-at) with a = 1e4 came out as 3e-12
    for a in (1e4, 1e6, 1e8):
        assert ol.laplace_numeric(lambda t, a=a: math.exp(-a * t), 1.0) == pytest.approx(1 / (a + 1))
    assert ol.laplace_numeric(lambda t: 1e6 * math.exp(-1e6 * t), 1.0) == pytest.approx(1e6 / (1e6 + 1))
    assert ol.laplace_numeric(lambda t: math.exp(-1e4 * t), 0) == pytest.approx(1e-4)
    # a singular but integrable start
    assert ol.laplace_numeric(lambda t: 1 / math.sqrt(t), 1.0) == pytest.approx(math.sqrt(math.pi))
    assert ol.laplace_numeric(math.log, 1.0) == pytest.approx(-np.euler_gamma)


@pytest.mark.parametrize("s", [np.nan, np.inf, -np.inf, 1 + 2j, np.complex128(3), "3", None, [1.0, 2.0],
                               True, np.True_])
def test_laplace_numeric_refuses_an_s_that_is_not_a_finite_real_number(s):
    with pytest.raises(ValueError, match="s must be one finite real number"):
        ol.laplace_numeric(lambda t: 1.0, s)


def test_laplace_numeric_refuses_what_is_not_a_function_of_one_real_number():
    with pytest.raises(ValueError, match="f must be a function"):
        ol.laplace_numeric(1.0, 3)
    with pytest.raises(ValueError, match="f must take one number and return one number"):
        ol.laplace_numeric(lambda t, u: t, 3)
    with pytest.raises(ValueError, match="a power is written t "):
        ol.laplace_numeric(lambda t: t ^ 2, 3)              # the hint of the other helpers
    with warnings.catch_warnings():
        warnings.simplefilter("error")                      # no ComplexWarning: a sentence
        for f in (lambda t: "one", lambda t: "1", lambda t: np.exp(1j * t), lambda t: 1 + 2j,
                  lambda t: [1, 2], lambda t: None):
            with pytest.raises(ValueError, match="f must return one real number"):
                ol.laplace_numeric(f, 3)
    # an error of f's own is said in one sentence that quotes it
    with pytest.raises(ValueError, match="f stopped with an error at t = .* \\(KeyError"):
        ol.laplace_numeric(lambda t: {}[t], 3)
    with pytest.raises(ValueError, match="NameError: name 'omega_typo' is not defined"):
        ol.laplace_numeric(lambda t: np.sin(omega_typo * t), 3)  # noqa: F821  (the mistake is the test)


def test_laplace_numeric_refuses_an_rtol_outside_its_range():
    for rtol in (0.0, -1e-8, 1e-15, 1.0, 1e3):
        with pytest.raises(ValueError, match="rtol must be between"):
            ol.laplace_numeric(lambda t: 1.0, 3, rtol=rtol)
    for rtol in (np.nan, None, "1e-8"):
        with pytest.raises(ValueError, match="rtol must be one finite real number"):
            ol.laplace_numeric(lambda t: 1.0, 3, rtol=rtol)
    with pytest.raises(TypeError):
        ol.laplace_numeric(lambda t: 1.0, 3, LOOSE_RTOL)    # rtol is given by name


def test_laplace_numeric_uses_only_times_from_zero_on():
    seen = []

    def recorded(t):
        seen.append(t)
        return np.exp(-t)

    for s in (1.0, 0.0, -0.5, 50.0):
        ol.laplace_numeric(recorded, s, breaks=(0.0, 2.0))
    assert min(seen) >= 0.0


def test_laplace_numeric_a_looser_rtol_costs_no_more_and_keeps_its_own_promise():
    calls = {}
    for rtol in (LOOSE_RTOL, ol.LAPLACE_RTOL):
        count = []

        def counted(t, count=count):
            count.append(t)
            return math.sin(3 * t)

        s = 0.2
        assert promised(ol.laplace_numeric(counted, s, rtol=rtol), 3 / (s * s + 9), s, rtol)
        calls[rtol] = len(count)
    assert calls[LOOSE_RTOL] <= calls[ol.LAPLACE_RTOL]


def test_laplace_numeric_takes_and_returns_numpy_numbers_as_floats():
    found = ol.laplace_numeric(lambda t: np.array(1.0), np.float64(S_DEMO))
    assert isinstance(found, float) and found == pytest.approx(1 / S_DEMO)
    assert ol.laplace_numeric(lambda t: np.float32(1.0), np.int64(2)) == pytest.approx(0.5)


def test_laplace_numeric_leaves_the_warning_filters_and_the_numpy_settings_as_they_were():
    filters, settings = list(warnings.filters), np.geterr()
    ol.laplace_numeric(lambda t: 1.0, S_DEMO)
    with pytest.raises(ValueError):
        ol.laplace_numeric(lambda t: np.exp(2 * t), 1.0)
    assert list(warnings.filters) == filters and np.geterr() == settings


@pytest.mark.parametrize("arguments, words", [
    ({"omega": 0.0}, "omega must be a positive number"), ({"omega": -1.0}, "omega must be a positive number"),
    ({"omega": np.nan}, "omega must be one finite real number"),
    ({"omega": np.inf}, "omega must be one finite real number"),
    ({"omega": [1, 2]}, "omega must be one finite real number"),
    ({"n": -1}, "n must be a whole number"), ({"n": 2.5}, "n must be a whole number"),
    ({"n": 3.0}, "n must be a whole number"), ({"n": True}, "n must be a whole number"),
    ({"n": "3"}, "n must be a whole number"), ({"n": None}, "n must be a whole number"),
    ({"n": 1000}, "is too large"),
    ({"a": np.nan}, "a must be one finite real number"), ({"a": np.inf}, "a must be one finite real number"),
    ({"a": None}, "a must be one finite real number"), ({"a": "2"}, "a must be one finite real number"),
    ({"a": 2 + 0j}, "a must be one finite real number"), ({"a": [2.0]}, "a must be one finite real number"),
    ({"a": True}, "a must be one finite real number"), ({"a": np.True_}, "a must be one finite real number"),
    ({"omega": np.True_}, "omega must be one finite real number"),
])
def test_laplace_table_refuses_invalid_constants(arguments, words):
    with pytest.raises(ValueError, match=words):
        ol.laplace_table(**arguments)
