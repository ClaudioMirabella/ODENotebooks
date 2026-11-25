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


# --- steps, impulses and convolution (notebook 20; Lebl, Sections 6.2 to 6.4) ------------------------
SWITCH_TIMES = (0.0, 0.5, 1.0, 2.5)         # the times a at which a step switches on
CONV_TIMES = np.linspace(0.0, 6.0, 13)      # where a convolution is compared with a formula
CONV_OMEGAS = (0.5, 1.0, 2.0, 7.0)          # the omega of Lebl's Examples 6.3.2 and 6.3.4
CONV_DRAWS = 12                             # seeded random cases per test
TRANSFER_DRAWS = 4                          # for the transfer function (the hammer and dsolve test it too)
EXACT = 1e-12                               # two formulas that agree to rounding (relative)
CONV_PULSES = 60                            # seeded pulses that convolution must find by itself
CONV_PULSE_RATIO = (0.005, 0.2)             # their width, as a part of t (the docstring promises t / 256)
ROCKET_ON, ROCKET_OFF = 1.0, 5.0            # Lebl, Example 6.2.2: the rocket fires for 1 <= t < 5
ROCKET_TIMES = np.linspace(0.0, 12.0, 25)
HAMMER = (1.0, 2.0, 5.0)                    # notebook 20: m = 1 kg, c = 2 kg/s, k = 5 N/m
HAMMER_IMPULSE = 3.0                        # newton seconds
SHORT_PULSE = 1e-4                          # seconds: a pulse this short stands for a blow in solve_ivp
SHORT_PULSE_AGREEMENT = 1e-3                # the response to it against the impulse response
FORCED_AGREEMENT = 1e-7                     # forced_response (one quad at its default tolerance)
DIFFERENCE_STEP = 1e-6                      # the step of a difference quotient
DIFFERENCE_AGREEMENT = 1e-4                 # a difference quotient against the derivative (relative)
IVP_TIGHT = {"rtol": 1e-11, "atol": 1e-13}  # solve_ivp, as TIGHT in test_cross_validation.py
IVP_AGREEMENT = 1e-8                        # solve_ivp at these tolerances against a formula
ROUNDING = 1e-15                            # two ways of writing one function by cases
UNSEEN = 1e-6                               # a pulse that was not seen leaves less than this
ANY_EQUATION = ((0.5, 4.0), (-1.0, 4.0), (-2.0, 9.0))      # ranges in which a, b, c are drawn
DAMPED_EQUATION = ((0.5, 4.0), (0.5, 4.0), (1.0, 9.0))     # the same for a damped spring (b, c > 0)
S_DRAWN = (0.5, 4.0)                        # the range in which s is drawn
DELAY_DRAWN = (0.2, 3.0)                    # the range in which a delay is drawn
RATE_DRAWN = (-1.0, 2.0)                    # the rates p of e^(-pt)
OMEGA_DRAWN = (0.3, 6.0)                    # angular frequencies
FACTOR_DRAWN = (-3.0, 3.0)                  # the constant c of (c f) * g
END_DRAWN = (1.0, 20.0)                     # the time t of a convolution with a pulse
DURATIONS = (0.5, 2.0, 4.0, 2 * np.pi, 8.0)                # seconds for which the rocket fires


def drawn(rng, ranges):
    """One number from each range (low, high), from the seeded generator."""
    return [rng.uniform(low, high) for low, high in ranges]


def near(found, exact, rtol=ol.CONVOLUTION_RTOL):
    """True when found is as near to exact as convolution promises: rtol max(1, |exact|), at every point."""
    found, exact = np.asarray(found, dtype=float), np.asarray(exact, dtype=float)
    return bool(np.all(np.abs(found - exact) <= rtol * np.maximum(1.0, np.abs(exact))))


def rocket_force(t):
    return ol.step(t, ROCKET_ON) - ol.step(t, ROCKET_OFF)


def rocket_solution(t):
    # the answer printed in Example 6.2.2
    return ((1 - np.cos(t - ROCKET_ON)) * ol.step(t, ROCKET_ON)
            - (1 - np.cos(t - ROCKET_OFF)) * ol.step(t, ROCKET_OFF))


def test_step_is_lebl_unit_step_with_the_value_1_at_the_jump():
    # Example 6.1.4: u(t) = 0 if t < 0, 1 if t >= 0
    assert ol.step([-2.0, -1e-300, 0.0, 1e-300, 3.0]).tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]
    assert ol.STEP_START == 0.0
    for a in SWITCH_TIMES:
        before, after = np.nextafter(a, -np.inf), np.nextafter(a, np.inf)
        assert ol.step([before, a, after], a).tolist() == [0.0, 1.0, 1.0]
    assert ol.step(-0.0) == 1.0 and ol.step(0.0, -0.0) == 1.0          # the two zeros of the computer
    assert ol.step(np.inf) == 1.0 and ol.step(-np.inf) == 0.0
    assert ol.step(2, 3) == 0.0 and ol.step(np.int64(3), np.float32(3)) == 1.0


def test_step_shape_and_type_of_the_result():
    one = ol.step(2.0, 1.0)
    assert isinstance(one, np.ndarray) and one.shape == () and one.dtype == float and float(one) == 1.0
    grid = ol.step([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], 2.0)
    assert grid.shape == (2, 3) and grid.dtype == float
    assert grid.tolist() == [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
    assert ol.step(np.array([]), 1.0).shape == (0,)


def test_step_builds_the_functions_of_lebl_section_6_2():
    times = np.linspace(-1.0, 4.0, 501)
    # the window u(t - 1) - u(t - 2) is 1 on [1, 2) and 0 elsewhere
    window = ol.step(times, 1) - ol.step(times, 2)
    assert np.array_equal(window, np.where((times >= 1) & (times < 2), 1.0, 0.0))
    # the signal u(t - pi) sin t
    signal = ol.step(times, np.pi) * np.sin(times)
    assert np.array_equal(signal, np.where(times >= np.pi, np.sin(times), 0.0))
    # t on [0, 1], -t + 2 on [1, 2), 0 otherwise
    tent = (times * (ol.step(times) - ol.step(times, 1))
            + (-times + 2) * (ol.step(times, 1) - ol.step(times, 2)))
    by_cases = np.select([times < 0, times < 1, times < 2], [0.0, times, 2 - times], 0.0)
    assert tent == pytest.approx(by_cases, abs=ROUNDING)
    # Exercise 6.2.13 (answer printed): 0 for t < 1, t - 1 for 1 <= t < 2, 1 for 2 <= t
    printed = (times - 1) * (ol.step(times, 1) - ol.step(times, 2)) + ol.step(times, 2)
    assert printed == pytest.approx(np.select([times < 1, times < 2], [0.0, times - 1], 1.0), abs=ROUNDING)


@pytest.mark.parametrize("a", [0.5, 1.0, 2.5])
def test_the_transform_of_a_step_is_lebl_example_6_1_4(a):
    for s in LAPLACE_S:
        exact = np.exp(-a * s) / s                          # e^(-as)/s
        assert promised(ol.laplace_numeric(lambda t: ol.step(t, a), s, breaks=[a]), exact, s)
        if a == SWITCH_TIMES[-1]:                           # found by itself: for one of the steps only
            assert promised(ol.laplace_numeric(lambda t: ol.step(t, a), s), exact, s)


def test_the_second_shifting_property_on_seeded_delays_and_rows_of_the_table():
    # (6.2.2): L{f(t - a) u(t - a)} = e^(-as) L{f(t)}
    rng = np.random.default_rng(SEED)
    rows = {row.name: row for row in ol.laplace_table()}
    for _ in range(CONV_DRAWS):
        row = rows[str(rng.choice(["1", "t", "t^2", "e^(-at)", "sin", "cos"]))]
        a, s = drawn(rng, (DELAY_DRAWN, S_DRAWN))

        def delayed(t, row=row, a=a):
            return row.time(max(t - a, 0.0)) * float(ol.step(t, a))

        assert promised(ol.laplace_numeric(delayed, s, breaks=[a]), np.exp(-a * s) * row.transform(s), s)


def test_t_times_a_step_is_not_a_delayed_t():
    # notebook 20, Section 3: t u(t - 1) = ((t - 1) + 1) u(t - 1), so the transform is e^(-s) (1/s^2 + 1/s)
    for s in LAPLACE_S:
        found = ol.laplace_numeric(lambda t: t * float(ol.step(t, 1)), s, breaks=[1])
        assert promised(found, np.exp(-s) * (1 / s ** 2 + 1 / s), s)
        assert not promised(found, np.exp(-s) / s ** 2, s)              # the usual mistake


def test_the_rocket_of_lebl_example_6_2_2():
    # x'' + x = u(t - 1) - u(t - 5) from rest: x = (1 - cos(t - 1)) u(t - 1) - (1 - cos(t - 5)) u(t - 5)
    printed = rocket_solution(ROCKET_TIMES)
    by_convolution = ol.convolution(rocket_force, np.sin, ROCKET_TIMES, breaks=[ROCKET_ON, ROCKET_OFF])
    assert near(by_convolution, printed)
    assert near(ol.convolution(rocket_force, np.sin, ROCKET_TIMES), printed)            # no breaks given
    assert near(ol.convolution(np.sin, rocket_force, ROCKET_TIMES), printed)            # g * f
    by_parameters = ol.forced_response(ROCKET_TIMES, 1, 0, 1, lambda t: float(rocket_force(t)))
    assert by_parameters == pytest.approx(printed, abs=FORCED_AGREEMENT)
    for s in LAPLACE_S:                                     # X(s) = (e^(-s) - e^(-5s)) / (s (s^2 + 1))
        exact = (np.exp(-ROCKET_ON * s) - np.exp(-ROCKET_OFF * s)) / (s * (s ** 2 + 1))
        assert promised(ol.laplace_numeric(lambda t: float(rocket_solution(t)), s,
                                           breaks=[ROCKET_ON, ROCKET_OFF]), exact, s)


def test_the_rocket_against_solve_ivp_and_the_amplitude_after_it_stops():
    from scipy.integrate import solve_ivp

    # Exercise 6.2.10: fired from t = 1 for d seconds, the mass then oscillates with amplitude 2 |sin(d/2)|
    for duration in DURATIONS:
        off = ROCKET_ON + duration
        state = np.zeros(2)
        for start, end, force in ((0.0, ROCKET_ON, 0.0), (ROCKET_ON, off, 1.0)):
            state = solve_ivp(lambda t, y, force=force: [y[1], force - y[0]], (start, end), state,
                              **IVP_TIGHT).y[:, -1]
        amplitude = float(np.hypot(*state))                 # afterwards x^2 + x'^2 stays as it is
        assert amplitude == pytest.approx(2 * abs(np.sin(duration / 2)), abs=IVP_AGREEMENT)
        after = off + np.linspace(0.0, 7.0, 15)
        found = ol.convolution(lambda t, off=off: ol.step(t, ROCKET_ON) - ol.step(t, off), np.sin, after,
                               breaks=[ROCKET_ON, off])
        expected = state[0] * np.cos(after - off) + state[1] * np.sin(after - off)
        assert found == pytest.approx(expected, abs=IVP_AGREEMENT)


def test_convolution_lebl_examples_6_3_1_to_6_3_3():
    times = CONV_TIMES
    assert near(ol.convolution(np.exp, lambda t: t, times), np.exp(times) - times - 1)          # 6.3.1
    for omega in CONV_OMEGAS:                                                                   # 6.3.2
        found = ol.convolution(lambda t, w=omega: np.sin(w * t), lambda t, w=omega: np.cos(w * t), times)
        assert near(found, times * np.sin(omega * times) / 2)
    # 6.3.3: the inverse of 1/((s + 1) s^2) is t * e^(-t) = e^(-t) + t - 1
    found = ol.convolution(lambda t: t, lambda t: np.exp(-t), times)
    assert near(found, np.exp(-times) + times - 1)
    for s in LAPLACE_S:
        assert promised(ol.laplace_numeric(lambda t: np.exp(-t) + t - 1, s), 1 / ((s + 1) * s ** 2), s)


def test_convolution_printed_answers_of_lebl_exercises_6_3_10_to_6_3_12_and_6_4_11():
    times = CONV_TIMES
    # 6.3.10: cos t * e^(-t) = (cos t + sin t - e^(-t)) / 2
    assert near(ol.convolution(np.cos, lambda t: np.exp(-t), times),
                (np.cos(times) + np.sin(times) - np.exp(-times)) / 2)
    # 6.3.11: the inverse of 5/(s^4 + s^2) = 5 (1/s^2) (1/(s^2 + 1)) is 5 t * sin t = 5t - 5 sin t
    assert near(ol.convolution(lambda t: 5 * t, np.sin, times), 5 * times - 5 * np.sin(times))
    for s in LAPLACE_S:
        assert promised(ol.laplace_numeric(lambda t: 5 * t - 5 * np.sin(t), s), 5 / (s ** 4 + s ** 2), s)
    # 6.3.12: x'' + x = sin t from rest is sin t * sin t = (sin t - t cos t) / 2
    answer = (np.sin(times) - times * np.cos(times)) / 2
    assert near(ol.convolution(np.sin, np.sin, times), answer)
    assert ol.forced_response(times, 1, 0, 1, np.sin) == pytest.approx(answer, abs=FORCED_AGREEMENT)
    # 6.4.11: the impulse response cos t and the input sin t give cos * sin = (t / 2) sin t
    assert near(ol.convolution(np.cos, np.sin, times), times * np.sin(times) / 2)


def test_lebl_exercise_6_2_14_printed_answer_against_forced_response():
    # x'' - x = (t^2 - 1) u(t - 1), x(0) = 1, x'(0) = 2:
    # x = (2 e^(t - 1) - t^2 - 1) u(t - 1) - e^(-t) / 2 + 3 e^t / 2
    times = np.linspace(0.0, 3.0, 13)
    printed = ((2 * np.exp(times - 1) - times ** 2 - 1) * ol.step(times, 1)
               - np.exp(-times) / 2 + 3 * np.exp(times) / 2)
    found = ol.forced_response(times, 1, 0, -1, lambda t: (t ** 2 - 1) * float(ol.step(t, 1)), 1, 2)
    assert found == pytest.approx(printed, rel=FORCED_AGREEMENT)
    # the part from rest is the input convolved with the impulse response sinh t
    from_rest = ol.convolution(lambda t: (t ** 2 - 1) * ol.step(t, 1),
                               lambda t: ol.impulse_response(t, 1, 0, -1), times, breaks=[1])
    assert near(from_rest, printed - ol.constant_coefficient_solution(times, 1, 0, -1, 1, 2))


def test_impulse_response_lebl_example_6_4_2_and_exercise_6_4_9():
    times = np.linspace(0.0, 8.0, 33)
    for omega_0 in CONV_OMEGAS:                             # x'' + omega_0^2 x = delta(t)
        expected = np.sin(omega_0 * times) / omega_0
        assert ol.impulse_response(times, 1, 0, omega_0 ** 2) == pytest.approx(expected, rel=EXACT)
    assert ol.impulse_response(times, 1, 0, 0) == pytest.approx(times, rel=EXACT)   # Exercise 6.4.9: x = t
    # Exercise 6.4.2: x'' + 2x' + x = delta(t) has the double root -1: t e^(-t)
    assert ol.impulse_response(times, 1, 2, 1) == pytest.approx(times * np.exp(-times), rel=EXACT)
    # Exercise 6.4.10, x' + a x = delta(t), is of first order: not for this function
    with pytest.raises(ValueError, match="a must be nonzero"):
        ol.impulse_response(times, 0, 1, 2)


def test_impulse_response_is_zero_before_the_blow_and_starts_with_velocity_1_over_a():
    assert ol.impulse_response([-3.0, -1e-9, 0.0], *HAMMER).tolist() == [0.0, 0.0, 0.0]
    rng = np.random.default_rng(SEED)
    for _ in range(CONV_DRAWS):
        a, b, c = drawn(rng, ANY_EQUATION)
        slope = float(ol.impulse_response(DIFFERENCE_STEP, a, b, c)) / DIFFERENCE_STEP
        assert slope == pytest.approx(1 / a, rel=DIFFERENCE_AGREEMENT)
        times = rng.uniform(0.0, 3.0, 7)
        assert ol.impulse_response(times, a, b, c) == pytest.approx(
            ol.constant_coefficient_solution(times, a, b, c, 0.0, 1 / a), rel=EXACT)
    one = ol.impulse_response(1.0, *HAMMER)
    assert isinstance(one, np.ndarray) and one.shape == () and one.dtype == float
    assert ol.impulse_response([[0.5, 1.0], [1.5, 2.0]], *HAMMER).shape == (2, 2)


def test_the_hammer_of_notebook_20():
    # x'' + 2x' + 5x = 3 delta(t) from rest: x = (3/2) e^(-t) sin 2t, the free motion with x(0) = 0, x'(0) = 3
    times = np.linspace(0.0, 6.0, 25)
    mass, damper, spring = HAMMER
    printed = 3 / 2 * np.exp(-times) * np.sin(2 * times)
    assert HAMMER_IMPULSE * ol.impulse_response(times, *HAMMER) == pytest.approx(printed, rel=EXACT)
    by_velocity = ol.free_response(times, mass, damper, spring, 0.0, HAMMER_IMPULSE / mass)
    assert by_velocity == pytest.approx(printed, rel=EXACT)
    for s in LAPLACE_S:                                     # X(s) = 3 / (s^2 + 2s + 5)
        found = ol.laplace_numeric(lambda t: HAMMER_IMPULSE * float(ol.impulse_response(t, *HAMMER)), s)
        assert promised(found, 3 / (s ** 2 + 2 * s + 5), s)


def test_impulse_response_is_the_limit_of_short_pulses_by_solve_ivp():
    from scipy.integrate import solve_ivp

    rng = np.random.default_rng(SEED)
    for _ in range(CONV_DRAWS // 2):
        a, b, c = drawn(rng, DAMPED_EQUATION)

        def field(t, y, force, a=a, b=b, c=c):
            return [y[1], (force - b * y[1] - c * y[0]) / a]

        # a pulse of height 1 / SHORT_PULSE for SHORT_PULSE seconds (area 1), then nothing
        end = solve_ivp(field, (0.0, SHORT_PULSE), [0.0, 0.0], args=(1 / SHORT_PULSE,), **IVP_TIGHT)
        times = np.linspace(0.5, 5.0, 10)
        later = solve_ivp(field, (SHORT_PULSE, times[-1]), end.y[:, -1], args=(0.0,), t_eval=times,
                          **IVP_TIGHT)
        assert later.y[0] == pytest.approx(ol.impulse_response(times, a, b, c), abs=SHORT_PULSE_AGREEMENT)


def test_the_transform_of_the_impulse_response_is_the_transfer_function():
    rng = np.random.default_rng(SEED)
    for _ in range(TRANSFER_DRAWS):
        a, b, c = drawn(rng, DAMPED_EQUATION)               # damped: the transform exists for s > 0
        s = rng.uniform(*S_DRAWN)
        found = ol.laplace_numeric(lambda t, a=a, b=b, c=c: float(ol.impulse_response(t, a, b, c)), s)
        assert promised(found, 1 / (a * s ** 2 + b * s + c), s)


def test_convolution_with_the_impulse_response_is_forced_response():
    # Lebl, Section 6.4.3: the solution from rest for any input is its convolution with the impulse response
    rng = np.random.default_rng(SEED)
    times = np.linspace(0.0, 5.0, 6)
    inputs = (np.cos, lambda t: np.exp(-t), lambda t: t ** 2, lambda t: 1.0, lambda t: np.sin(3 * t) + 2)
    for index in range(CONV_DRAWS):
        a, b, c = drawn(rng, ANY_EQUATION)
        f = inputs[index % len(inputs)]
        found = ol.convolution(f, lambda t, a=a, b=b, c=c: ol.impulse_response(t, a, b, c), times)
        expected = ol.forced_response(times, a, b, c, f)
        assert found == pytest.approx(expected, rel=FORCED_AGREEMENT, abs=FORCED_AGREEMENT)


def test_convolution_resonance_of_lebl_example_6_3_4():
    times = CONV_TIMES
    for omega_0 in CONV_OMEGAS:                             # x'' + omega_0^2 x = cos(omega_0 t) from rest
        found = ol.convolution(lambda t, w=omega_0: np.cos(w * t),
                               lambda t, w=omega_0: ol.impulse_response(t, 1, 0, w ** 2), times)
        assert near(found, times * np.sin(omega_0 * times) / (2 * omega_0))
    assert ol.cosine_forced_response(times, 1, 0, 4, 1, 2) == pytest.approx(times * np.sin(2 * times) / 4)


def test_convolution_properties_of_lebl_6_3_3():
    rng = np.random.default_rng(SEED)
    times = np.linspace(0.0, 4.0, 5)
    for _ in range(CONV_DRAWS):
        p, q, w, factor = drawn(rng, (RATE_DRAWN, RATE_DRAWN, OMEGA_DRAWN, FACTOR_DRAWN))

        def f(t, p=p):
            return np.exp(-p * t)

        def g(t, q=q, w=w):
            return np.exp(-q * t) * np.cos(w * t)

        fg = ol.convolution(f, g, times)
        assert near(ol.convolution(g, f, times), fg)                                    # f * g = g * f
        assert near(ol.convolution(lambda t, k=factor: k * f(t), g, times), factor * fg)    # (c f) * g
        assert near(ol.convolution(f, lambda t, k=factor: k * g(t), times), factor * fg)
    # the star is not a product: 1 * 1 = t, and f * 1 is the integral of f from 0 to t
    assert near(ol.convolution(lambda t: 1.0, lambda t: 1.0, times), times)
    assert near(ol.convolution(np.cos, lambda t: 1.0, times), np.sin(times))


def test_convolution_shape_and_type_of_the_result_and_the_time_0():
    one = ol.convolution(np.exp, lambda t: t, 2.0)
    assert isinstance(one, np.ndarray) and one.shape == () and one.dtype == float
    assert float(one) == pytest.approx(np.exp(2.0) - 3.0)
    grid = ol.convolution(lambda t: 1.0, lambda t: 1.0, [[0, 1], [2, 3]])
    assert grid.shape == (2, 2) and grid.ravel() == pytest.approx([0.0, 1.0, 2.0, 3.0])
    assert ol.convolution(np.exp, np.exp, 0.0) == 0.0
    assert ol.convolution(np.exp, np.exp, [0, 0]).tolist() == [0, 0]
    assert ol.convolution(np.exp, np.exp, np.array([])).shape == (0,)
    mixed = ol.convolution(lambda t: np.float32(1.0), lambda t: np.array(2.0), np.int64(3))
    assert float(mixed) == pytest.approx(6)
    assert ol.CONVOLUTION_RTOL == 1e-8 and len(ol.CONVOLUTION_PIECES) == 2


def test_convolution_with_jumps_with_and_without_breaks():
    times = CONV_TIMES
    # Exercise 6.3.1: t^2 * u(t - 1) = (t - 1)^3 / 3 for t >= 1, and 0 before
    expected = np.where(times >= 1, (times - 1) ** 3 / 3, 0.0)
    for breaks in ((), [1], [1.0, 1.0], (1, 4.5)):          # a break that is not a jump costs nothing
        assert near(ol.convolution(lambda t: t ** 2, lambda t: ol.step(t, 1), times, breaks=breaks), expected)
        assert near(ol.convolution(lambda t: ol.step(t, 1), lambda t: t ** 2, times, breaks=breaks), expected)
    # two windows: the convolution is a tent, 0 until 2, up to 4 at t = 6, down to 0 at t = 10
    long = ROCKET_TIMES
    tent = np.clip(long - 2, 0, None) - 2 * np.clip(long - 6, 0, None) + np.clip(long - 10, 0, None)
    assert near(ol.convolution(rocket_force, rocket_force, long), tent)
    assert near(ol.convolution(rocket_force, rocket_force, long, breaks=[ROCKET_ON, ROCKET_OFF]), tent)
    # a switch exactly at a time of t, and a jump on top of an oscillation
    assert near(ol.convolution(lambda t: ol.step(t, 3), np.cos, [3.0, 4.0]), [0.0, np.sin(1.0)])
    found = ol.convolution(lambda t: np.sin(40 * t) * ol.step(t, 2.5), lambda t: 1.0, times)
    assert near(found, np.where(times >= 2.5, (np.cos(100) - np.cos(40 * times)) / 40, 0.0))


def test_convolution_finds_pulses_wider_than_its_first_look_by_itself():
    rng = np.random.default_rng(SEED)
    for index in range(CONV_PULSES):
        end = rng.uniform(*END_DRAWN)
        width = end * rng.uniform(*CONV_PULSE_RATIO)
        on = rng.uniform(0.0, end - width)
        under = index % 2                                   # every second pulse stands on e^(-t)

        def f(t, on=on, width=width, under=under):
            return (ol.step(t, on) - ol.step(t, on + width)) / width + under * np.exp(-t)

        # the pulse of area 1 against sin: (cos(end - on - width) - cos(end - on)) / width
        exact = ((np.cos(end - on - width) - np.cos(end - on)) / width
                 + under * (np.exp(-end) - np.cos(end) + np.sin(end)) / 2)
        assert near(ol.convolution(f, np.sin, end), exact)
        assert near(ol.convolution(np.sin, f, end), exact)


def test_convolution_documented_limit_a_very_short_pulse_needs_breaks():
    # a pulse 0.01 wide at t = 3 in a convolution up to t = 40: shorter than one step of the first look
    # (40 / 256), it is missed and a wrong number comes back with no error. The docstring says so; if this
    # test fails because convolution has become better, correct the docstring.
    on, width, end = 3.0, 0.01, 40.0

    def pulse(t):
        return (ol.step(t, on) - ol.step(t, on + width)) / width

    exact = (np.cos(end - on - width) - np.cos(end - on)) / width
    assert abs(exact) > 0.5
    assert near(ol.convolution(pulse, np.sin, end, breaks=[on, on + width]), exact)
    assert width < end / ol.CONVOLUTION_SCAN
    assert float(ol.convolution(pulse, np.sin, end)) == pytest.approx(0.0, abs=UNSEEN)      # not seen


@pytest.mark.parametrize("arguments, words", [
    ((3, np.sin, 1.0), "f must be a function of one number"),
    ((np.sin, None, 1.0), "g must be a function of one number"),
    ((np.sin, np.cos, -1.0), "each 0 or more"),
    ((np.sin, np.cos, [1.0, -1e-9]), "each 0 or more"),
    ((np.sin, np.cos, np.inf), "finite times"),
    ((np.sin, np.cos, np.nan), "t must be a real number"),
    ((np.sin, np.cos, "2"), "t must be a real number"),
    ((np.sin, np.cos, None), "t must be a real number"),
    ((np.sin, np.cos, True), "t must be a real number"),
    ((np.sin, np.cos, 1 + 2j), "t must be a real number"),
    ((lambda t, extra: t, np.cos, 1.0), "f must take one number and return one number"),
    ((np.sin, lambda: 1.0, 1.0), "g must take one number and return one number"),
    ((lambda t: t ^ 2, np.cos, 1.0), "a power is written"),
    ((lambda t: "one", np.cos, 1.0), "f must return one real number"),
    ((np.sin, lambda t: 1j * t, 1.0), "g must return one real number"),
    ((np.sin, lambda t: [t, t], 1.0), "g must return one real number"),
    ((lambda t: 1 / t, np.cos, 1.0), "f stopped with an error at the time 0 .ZeroDivisionError"),
    ((np.sin, lambda t: np.log(t), 1.0), "g is not a finite number at the time 0"),
    ((lambda t: np.nan, np.cos, 1.0), "f is not a finite number"),
    ((np.sin, lambda t: {}["missing"], 1.0), "g stopped with an error at the time 0 .KeyError"),
])
def test_convolution_refuses_invalid_input(arguments, words):
    with pytest.raises(ValueError, match=words):
        ol.convolution(*arguments)


@pytest.mark.parametrize("breaks", ["1", [-1.0], [np.nan], [np.inf], [None], 5, [True], ["1"], [1 + 0j]])
def test_convolution_refuses_invalid_breaks(breaks):
    with pytest.raises(ValueError, match="breaks must be a list of finite times"):
        ol.convolution(np.sin, np.cos, 1.0, breaks=breaks)


def test_convolution_says_when_it_is_in_doubt():
    # sin(1/t) oscillates without end near t = 0: no 8 digits, and the answer is a sentence
    with pytest.raises(ValueError, match="could not be computed to 8 digits"):
        ol.convolution(lambda t: np.sin(1 / t) if t else 0.0, lambda t: 1.0, 1.0)


def test_convolution_leaves_the_warning_filters_and_the_numpy_settings_as_they_were():
    filters, settings = list(warnings.filters), np.geterr()
    ol.convolution(np.exp, lambda t: t, 1.0)
    with pytest.raises(ValueError):
        ol.convolution(np.sin, lambda t: np.log(t), 1.0)
    assert list(warnings.filters) == filters and np.geterr() == settings


class FakeSymbol:
    """Stands for a SymPy letter where SymPy is not installed: step only looks at the module of the type."""

    __module__ = "sympy.core.symbol"


@pytest.mark.parametrize("call, words", [
    (lambda: ol.step(1.0, None), "a must be one finite real number"),
    (lambda: ol.step(1.0, np.nan), "a must be one finite real number"),
    (lambda: ol.step(1.0, np.inf), "a must be one finite real number"),
    (lambda: ol.step(1.0, "1"), "a must be one finite real number"),
    (lambda: ol.step(1.0, [1.0, 2.0]), "a must be one finite real number"),
    (lambda: ol.step(1.0, 1j), "a must be one finite real number"),
    (lambda: ol.step(1.0, True), "a must be one finite real number"),
    (lambda: ol.step("now"), "t must be a real number"),
    (lambda: ol.step(None), "t must be a real number"),
    (lambda: ol.step([0.0, np.nan]), "t must be a real number"),
    (lambda: ol.step(2j), "t must be a real number"),
    (lambda: ol.step(True), "t must be a real number"),
    (lambda: ol.step(FakeSymbol(), 1), "in a formula typed with SymPy write u"),
    (lambda: ol.impulse_response(1.0, 0, 2, 5), "a must be nonzero"),
    (lambda: ol.impulse_response(1.0, None, 2, 5), "a must be one finite real number"),
    (lambda: ol.impulse_response(1.0, 1, "2", 5), "b must be one finite real number"),
    (lambda: ol.impulse_response(1.0, 1, 2, np.nan), "c must be one finite real number"),
    (lambda: ol.impulse_response(1.0, 1, 2, [5]), "c must be one finite real number"),
    (lambda: ol.impulse_response(1.0, True, 2, 5), "a must be one finite real number"),
    (lambda: ol.impulse_response(np.inf, 1, 2, 5), "t must be made of finite times"),
    (lambda: ol.impulse_response("soon", 1, 2, 5), "t must be a real number"),
    (lambda: ol.impulse_response([0.0, np.nan], 1, 2, 5), "t must be a real number"),
    (lambda: ol.impulse_response(1e6, 1, -5, 1), "too large for the computer"),
])
def test_step_and_impulse_response_refuse_invalid_input(call, words):
    with pytest.raises(ValueError, match=words):
        call()


def test_the_new_functions_do_not_change_the_names_of_the_setup_cell():
    assert ol.SETUP_NAMES == ("np", "sp", "plt", "ol", "solve_ivp", "interact", "FloatSlider", "IntSlider")


# --- corrections after the reviews of notebook 20 (2026-11-24): what would have caught the four bugs ---
RAMP_START = 2.0                            # a switch that closes without a jump: a ramp from 0 to 1
WIDE_RAMP = (0.5, 10.0)                     # (how long the ramp lasts, the time t): 12 steps of the look
STEEP_RAMP = (0.0003, 50.0)                 # far shorter than a step of the first look (50 / 256)
LOW_PULSES = {                              # a low pulse on a changing function: f, ends, t, exact value
    "code review": (lambda t: np.sin(2 * t) + (0.5 if 3 <= t < 3.1 else 0.0), (3.0, 3.1), 20.0,
                    (1 - np.cos(40.0)) / 2 + 0.5 * 0.1),
    "mathematics review": (lambda t: 10 * np.sin(t) + (1.5 if 3.3377 <= t < 3.3877 else 0.0),
                           (3.3377, 3.3877), 10.0, 10 * (1 - np.cos(10.0)) + 1.5 * 0.05),
}
SEEN_AT_ALL = 1e-3                          # a coarse distance: a result farther than this lost a whole pulse
HUGE = 1e200                                # a finite number whose square is not one
TOO_LARGE = 10 ** 400                       # a whole number that no float can hold


def ramp(duration):
    """A function that goes from 0 to 1 in a straight line, from RAMP_START, in the given time."""
    return lambda t: min(max((t - RAMP_START) / duration, 0.0), 1.0)


def one(t):
    return 1.0


def test_convolution_with_a_switch_that_closes_without_a_jump():
    # against 1 the convolution of a ramp is its integral: t - RAMP_START - duration / 2 after the ramp
    duration, end = WIDE_RAMP
    assert near(ol.convolution(ramp(duration), one, end), end - RAMP_START - duration / 2)
    assert near(ol.convolution(one, ramp(duration), end), end - RAMP_START - duration / 2)
    duration, end = STEEP_RAMP
    exact = end - RAMP_START - duration / 2
    assert near(ol.convolution(ramp(duration), one, end, breaks=[RAMP_START, RAMP_START + duration]), exact)


def test_convolution_documented_limit_a_very_steep_ramp_needs_breaks():
    # The docstring: "a switch that closes steeply without a jump ... can be missed". Pinned here: a wrong
    # number with no error. If this test fails because convolution has become better, correct the docstring.
    duration, end = STEEP_RAMP
    exact = end - RAMP_START - duration / 2
    found = float(ol.convolution(ramp(duration), one, end))
    assert not near(found, exact) and found == pytest.approx(exact, abs=SEEN_AT_ALL)


@pytest.mark.parametrize("name", sorted(LOW_PULSES))
def test_convolution_documented_limit_a_low_pulse_on_a_changing_function_needs_breaks(name):
    # The two reproductions of the reviews. With the ends of the pulse in breaks the value is right; without
    # them the pulse is lost, with no error (the docstring says so: the search is a help, not a guarantee).
    f, ends, end, exact = LOW_PULSES[name]
    assert near(ol.convolution(f, one, end, breaks=ends), exact)
    assert near(ol.convolution(one, f, end, breaks=ends), exact)
    lost = float(ol.convolution(f, one, end))
    assert abs(lost - exact) > SEEN_AT_ALL          # if this fails, convolution has become better


def test_convolution_with_a_step_that_has_another_value_at_the_jump():
    # the value 1/2 at the jump (NumPy's and SymPy's own step), and a switch written t > 1: naming the
    # jump in breaks must not make the function fail (two cuts that are neighboring numbers are one cut)
    def half(t):
        return np.heaviside(t - 1, 0.5)

    def late(t):
        return 1.0 if t > 1 else 0.0

    assert near(ol.convolution(half, one, 3.0), 2.0)
    assert near(ol.convolution(half, one, 3.0, breaks=[1]), 2.0)
    assert near(ol.convolution(half, one, 2.0), 1.0)                # the jump on a point of the first look
    assert near(ol.convolution(one, late, 2.0, breaks=[1]), 1.0)
    assert near(ol.convolution(late, np.cos, [2.0, 3.0], breaks=[1, 1.0, 1 + 1e-15]), np.sin([1.0, 2.0]))
    assert ol.CONVOLUTION_NARROW == 1e-12
    assert ol.CONVOLUTION_JUMP_RATIO == ol.LAPLACE_JUMP_RATIO
    assert ol.CONVOLUTION_MAX_JUMPS == ol.LAPLACE_MAX_JUMPS


def test_convolution_refuses_a_product_too_large_for_the_computer():
    with pytest.raises(ValueError, match="too large for the computer between 0 and t = 1"):
        ol.convolution(lambda t: HUGE, lambda t: HUGE, 1.0)         # f and g are finite, their product is not
    # large values with a small result: refused, with a sentence that says why
    with pytest.raises(ValueError, match="so much larger than the result"):
        ol.convolution(lambda t: 1e6 * np.cos(t), one, 2 * np.pi)


def test_convolution_at_the_time_0_does_not_look_at_f_and_g():
    assert float(ol.convolution(lambda t: 1 / t, one, 0)) == 0.0    # as the docstring says
    assert ol.convolution(lambda t: 1 / t, lambda t: {}["missing"], [0.0, 0]).tolist() == [0.0, 0.0]


@pytest.mark.parametrize("call, words", [
    (lambda: ol.step(TOO_LARGE), "t must be a real number"),
    (lambda: ol.step([1, TOO_LARGE]), "t must be a real number"),
    (lambda: ol.impulse_response(TOO_LARGE, 1, 2, 5), "t must be a real number"),
    (lambda: ol.convolution(np.sin, np.cos, TOO_LARGE), "t must be a real number"),
    (lambda: ol.convolution(lambda t: TOO_LARGE, np.cos, 1.0), "f must return one real number"),
    (lambda: ol.convolution(np.sin, np.cos, 1.0, breaks=[TOO_LARGE]), "breaks must be a list of finite"),
    (lambda: ol.convolution(lambda t: ol.step([t], 1), np.cos, 2.0), "f must return one real number"),
    (lambda: ol.convolution(np.sin, lambda t: np.array([1.0]), 2.0), "g must return one real number"),
    (lambda: ol.step([True, False]), "t must be a real number"),
    (lambda: ol.step([1, True]), "t must be a real number"),
    (lambda: ol.step([1.5, "2"]), "t must be a real number"),
    (lambda: ol.step(np.array(["2026-11-24"], dtype="datetime64[D]")), "t must be a real number"),
    (lambda: ol.impulse_response([1, True], 1, 2, 5), "t must be a real number"),
    (lambda: ol.convolution(np.sin, np.cos, [1, True]), "t must be a real number"),
    (lambda: ol.convolution(lambda t: BadFloat(), np.cos, 1.0), "f must return one real number"),
    (lambda: ol.step(["1", "-2"]), "t must be a real number"),
    (lambda: ol.step(np.array([1 + 0j])), "t must be a real number"),
    (lambda: ol.impulse_response([True], 1, 2, 5), "t must be a real number"),
    (lambda: ol.impulse_response(0.0, 5e-324, 1, 1), "or a is too near to 0"),
])
def test_the_new_functions_answer_with_a_valueerror_also_here(call, words):
    with pytest.raises(ValueError, match=words):
        call()


class BadFloat:
    """A value whose float() stops with an error that is not a TypeError, a ValueError or an OverflowError."""

    def __float__(self):
        raise RuntimeError("no number here")


class FakeNumber:
    """Stands for a SymPy number (sp.pi, sp.Integer(3)) where SymPy is not installed: no letter in it."""

    __module__ = "sympy.core.numbers"
    free_symbols = frozenset()

    def __init__(self, value):
        self.value = value

    def __float__(self):
        return float(self.value)


def test_step_takes_a_sympy_number_as_a_number_and_other_kinds_of_numbers():
    from decimal import Decimal
    from fractions import Fraction

    assert float(ol.step(FakeNumber(3), 2)) == 1.0 and float(ol.step(FakeNumber(1.5), 2)) == 0.0
    assert ol.step([Fraction(1, 2), Decimal("1.5"), 2], 1).tolist() == [0.0, 1.0, 1.0]
    assert ol.step(range(3), 1).tolist() == [0.0, 1.0, 1.0] and ol.step(np.uint8(3), 2) == 1.0


# ---------------------------------------------------------------------------
# The fix of 2026-11-24 (notebook 19's plan, Section 8): three refusals that were not needed
# ---------------------------------------------------------------------------
FIX_STEP_AT = 1.0                           # where the step of the first defect jumps
FIX_S_VALUES = (0.5, 1.0, 2.0)
FIX_RAMP_START = 2.0                        # the steep switch of the second defect begins here
FIX_RAMP_WIDTHS = (0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003)
FIX_TOO_LARGE = 10 ** 400                   # an integer that float() cannot hold


def half_value_step(t):
    return float(np.heaviside(t - FIX_STEP_AT, 0.5))        # SymPy's Heaviside has this value at the jump


@pytest.mark.parametrize("s", FIX_S_VALUES)
def test_fix_laplace_numeric_takes_a_step_with_the_value_one_half_at_its_jump(s):
    exact = math.exp(-FIX_STEP_AT * s) / s
    assert promised(ol.laplace_numeric(half_value_step, s), exact, s)
    assert promised(ol.laplace_numeric(half_value_step, s, breaks=[FIX_STEP_AT]), exact, s)
    # the two other values at the jump were right before the fix and must stay so
    for at_jump in (0.0, 1.0):
        def other(t, at_jump=at_jump):
            return float(np.heaviside(t - FIX_STEP_AT, at_jump))
        assert promised(ol.laplace_numeric(other, s), exact, s)
        assert promised(ol.laplace_numeric(other, s, breaks=[FIX_STEP_AT]), exact, s)


def test_fix_laplace_numeric_takes_a_lambdified_sympy_heaviside():
    sp = pytest.importorskip("sympy")
    t = sp.symbols("t")
    switched = sp.lambdify(t, sp.Heaviside(t - FIX_STEP_AT) * sp.exp(-t), "numpy")
    for s in FIX_S_VALUES:
        exact = math.exp(-FIX_STEP_AT * (s + 1)) / (s + 1)
        assert promised(ol.laplace_numeric(lambda time: float(switched(time)), s), exact, s)


@pytest.mark.parametrize("width", FIX_RAMP_WIDTHS)
def test_fix_laplace_numeric_takes_a_steep_switch_without_a_jump(width):
    def ramp(t):
        return min(max((t - FIX_RAMP_START) / width, 0.0), 1.0)

    for s in FIX_S_VALUES:
        exact = (math.exp(-FIX_RAMP_START * s) - math.exp(-(FIX_RAMP_START + width) * s)) / (width * s * s)
        ends = (FIX_RAMP_START, FIX_RAMP_START + width)
        assert promised(ol.laplace_numeric(ramp, s, breaks=ends), exact, s)
        try:                                                # without help: right, or the safe refusal
            found = ol.laplace_numeric(ramp, s)
        except ValueError:
            assert width < FIX_RAMP_WIDTHS[2], "a ramp this wide must be computed without breaks"
        else:
            assert promised(found, exact, s)


def test_fix_a_number_too_large_for_a_float_is_refused_with_a_sentence():
    refusal = "must be one finite real number"
    with pytest.raises(ValueError, match=refusal):
        ol.laplace_numeric(lambda t: 1.0, FIX_TOO_LARGE)
    with pytest.raises(ValueError, match=refusal):
        ol.laplace_numeric(lambda t: 1.0, 1.0, rtol=FIX_TOO_LARGE)
    with pytest.raises(ValueError, match="breaks"):
        ol.laplace_numeric(lambda t: 1.0, 1.0, breaks=[FIX_TOO_LARGE])
    with pytest.raises(ValueError, match=refusal):
        ol.laplace_table(a=FIX_TOO_LARGE)
    with pytest.raises(ValueError, match=refusal):
        ol.laplace_table(omega=FIX_TOO_LARGE)
    with pytest.raises(ValueError, match=refusal):
        ol.step(1.0, FIX_TOO_LARGE)
    with pytest.raises(ValueError, match=refusal):
        ol.impulse_response(1.0, FIX_TOO_LARGE, 1.0, 1.0)


FIX_RAMP_CASES = 60                         # seeded ramps for the documented limit of steep switches
FIX_RAMP_SHARE_SAFE = 1e-3                  # width / starting time at which laplace_numeric is not wrong
FIX_RAMP_SHARE_STEEP = 1e-5                 # and one at which it can be: breaks are needed
FIX_RAMP_WRONG_ALLOWED = 3                  # of FIX_RAMP_CASES at the safe width (measured: 3 in 1500)


def test_fix_laplace_numeric_steep_switches_the_documented_limit():
    rng = np.random.default_rng(SEED)
    limits = ((FIX_RAMP_SHARE_SAFE, FIX_RAMP_WRONG_ALLOWED), (FIX_RAMP_SHARE_STEEP, FIX_RAMP_CASES))
    for share, wrong_allowed in limits:
        wrong = 0
        for _ in range(FIX_RAMP_CASES):
            start = 10 ** rng.uniform(-1, 1)
            width = share * start
            s = 10 ** rng.uniform(-0.7, 1.2)

            def ramp(t, start=start, width=width):
                return min(max((t - start) / width, 0.0), 1.0)

            exact = (math.exp(-start * s) - math.exp(-(start + width) * s)) / (width * s * s)
            assert promised(ol.laplace_numeric(ramp, s, breaks=(start, start + width)), exact, s)
            try:
                wrong += not promised(ol.laplace_numeric(ramp, s), exact, s)
            except ValueError:
                pass                                        # the safe refusal
        assert wrong <= wrong_allowed, f"width / start {share:g}: {wrong} wrong without breaks"


FIX_LAST_EDGE = 2.0 ** 20                   # the last edge of the pieces for |s| at most 1 (LAPLACE_EDGES)
FIX_SMALL_S = 3e-6
FIX_SWITCH_OFF = FIX_LAST_EDGE - 513        # f is switched off shortly before that edge


def test_fix_a_break_next_to_the_last_edge_does_not_replace_the_cut_before_it():
    # the review of the fix: a named break one rounding step below the last edge took the place of the
    # cut that had been found at the switch, and a wrong number came back with no error
    def switched_off(t):
        return 1.0 if t < FIX_SWITCH_OFF else 0.0

    exact = (1 - math.exp(-FIX_SMALL_S * FIX_SWITCH_OFF)) / FIX_SMALL_S
    for breaks in ((), [math.nextafter(FIX_LAST_EDGE, 0.0)], [FIX_SWITCH_OFF]):
        assert promised(ol.laplace_numeric(switched_off, FIX_SMALL_S, breaks=breaks), exact, FIX_SMALL_S)


def test_fix_breaks_at_or_next_to_a_cut_that_is_found_anyway():
    exact = math.exp(-FIX_STEP_AT) / 1.0
    near = (math.nextafter(FIX_STEP_AT, 0.0), math.nextafter(FIX_STEP_AT, 2.0))
    for breaks in ([near[0]], [near[1]], list(near), [FIX_STEP_AT, near[1]], [0.5, FIX_STEP_AT, 2.0]):
        for at_jump in (0.0, 0.5, 1.0):
            def stepped(t, at_jump=at_jump):
                return float(np.heaviside(t - FIX_STEP_AT, at_jump))
            assert promised(ol.laplace_numeric(stepped, 1.0, breaks=breaks), exact, 1.0)


def test_fix_a_square_wave_with_its_switches_named():
    def wave(t):
        return 1.0 if int(t) % 2 == 0 else 0.0              # 1 on [0, 1), 0 on [1, 2), and so on

    for s in FIX_S_VALUES:
        switches = range(1, 60)
        exact = 1 / (s * (1 + math.exp(-s)))
        assert promised(ol.laplace_numeric(wave, s, breaks=switches), exact, s)


# --- series solutions (notebook 21; Lebl, Sections 7.1 to 7.3) ------------------------------------------
SERIES_POINTS = np.linspace(-2.0, 2.0, 17)  # where a partial sum is compared with a function
SERIES_EXACT = 1e-14                        # a partial sum against a formula, well inside the radius
SERIES_TERMS = 40                           # coefficients used for the functions of the notebook
AIRY_POINTS = (1.0, 2.0, -3.0)              # notebook 21: Ai and Bi from the two series, 40 coefficients
AIRY_AGREEMENT = 1e-12                      # plan 21: "to about 1e-12"
BESSEL_POINTS = np.linspace(0.0, 10.0, 21)
BESSEL_AGREEMENT = 1e-8                     # the J_0 of the notebook, 20 terms that are not 0, up to x = 10
BESSEL_ORDERS = (0.0, 1.0, 1.0 / 3.0, 2.5)  # J_nu by its recurrence, against scipy.special.jv
J0_TEN_TERMS = (10.0, -6.2176, 1e-4)        # x, the partial sum of 10 terms there, the digits of the text
J0_ENOUGH, J0_TOO_FEW = 20, 19              # terms within BESSEL_AGREEMENT on [0, 10], and not
ARCTAN_POINT_OUT, ARCTAN_POINT_IN = 1.1, 0.9
ARCTAN_ERROR_10 = (0.17, 0.01)              # at x = 1.1, 10 terms: the docstring's number and its digits
ARCTAN_ERROR_40 = (13.0, 0.5)               # at x = 1.1, 40 terms
ARCTAN_TERMS_IN = (5, 10, 20, 40)           # at x = 0.9 the error shrinks with these numbers of terms
J0_FAR = 20.0                               # where cancellation is measured
J0_TRUNCATION_30 = (1.3e-5, 0.05)           # 30 terms at x = 20: the error and its relative digits
J0_FLOOR = (1e-10, 1e-8)                    # 40, 60, 80 terms at x = 20: the error lies between these
J0_FLOOR_TERMS = (40, 60, 80)
J0_LARGEST_TERM = (7.6e6, 0.01)             # the largest term at x = 20, and its relative digits
INDICIAL_SEED = 331                         # as SEED in test_cross_validation.py
INDICIAL_DRAWS = 40                         # seeded pairs of roots, for each kind
ROOT_RANGE = (-3.0, 3.0)                    # the roots drawn
WHOLE_DIFFERENCES = (1, 2, 3, 5)            # whole-number differences drawn
NOT_WHOLE_SHARE = (0.05, 0.95)              # a difference whole + this share is not a whole number
BETA_RANGE = (0.05, 3.0)                    # the imaginary parts drawn
ROOT_AGREEMENT = 1e-7                       # a seeded root against the root returned
HORNER_RANGE = (-2.0, 2.0)                  # seeded coefficients and points of the Horner test
HORNER_CENTERS = (-1.0, 1.0)
HORNER_SIZES = (1, 12)                      # numbers of coefficients drawn (high excluded)
HORNER_POINTS = 5
HUGE_X = 1e308                              # x - x0 with x0 = -HUGE_X is too large for the computer
LARGE_GAP = (-1999.002, 1000002.0)          # roots 1000.002 and 1000: called the double root 1000.001
LARGE_DOUBLE = 1000.001
AMBIGUOUS_SIDES = (1.5e5, 2.0e5)            # double roots of these sizes: computed, then refused
RECHECK_BAND = (1000.0, (0.0027, 0.0028), (0.0029, 0.0035))     # near 1000: band 1e-6 sqrt(8e6) = 0.0028
RECHECK_SYMMETRIC = 3e5                     # roots +-3e5: refused, with their size named
RECHECK_COMPUTED = 4e5                      # roots 4e5 and 0: (p0 - 1)^2 + 4|q0| = 1.6e11, below 2.5e11
RECHECK_WIDE = (170000.0, (0.4, 1.1))       # just below the limit: 0.4 apart is "double", 1.1 apart is "1"


def cosh_rule(k, a):
    return a[k - 2] / (k * (k - 1))         # Lebl, Example 7.2.1: y'' - y = 0


def airy_rule(k, a):
    return a[k - 3] / (k * (k - 1))         # Lebl, Example 7.2.2: y'' - x y = 0, with a_2 = 0 given


def hermite_rule(n):
    return lambda k, a: 2 * (k - 2 - n) / (k * (k - 1)) * a[k - 2]     # Lebl, Example 7.2.3


def bessel_rule(nu):
    # Bessel's equation of order nu with r = nu: ((k + nu)^2 - nu^2) a_k + a_(k-2) = 0
    return lambda k, a: -a[k - 2] / (k * (k + 2 * nu))


def test_series_example_7_2_1_cosh_and_sinh():
    from math import factorial

    even = ol.series_coefficients(cosh_rule, [1.0, 0.0], SERIES_TERMS)
    odd = ol.series_coefficients(cosh_rule, [0.0, 1.0], SERIES_TERMS)
    for k in range(SERIES_TERMS):
        assert even[k] == pytest.approx(1 / factorial(k) if k % 2 == 0 else 0.0, rel=SERIES_EXACT, abs=0.0)
        assert odd[k] == pytest.approx(1 / factorial(k) if k % 2 == 1 else 0.0, rel=SERIES_EXACT, abs=0.0)
    assert ol.series_value(even, SERIES_POINTS) == pytest.approx(np.cosh(SERIES_POINTS), rel=SERIES_EXACT)
    assert ol.series_value(odd, SERIES_POINTS) == pytest.approx(np.sinh(SERIES_POINTS), rel=SERIES_EXACT)
    assert even.shape == (SERIES_TERMS,) and even.dtype == float


def test_series_example_7_2_2_airy():
    from scipy.special import airy

    y1 = ol.series_coefficients(airy_rule, [1.0, 0.0, 0.0], SERIES_TERMS)
    y2 = ol.series_coefficients(airy_rule, [0.0, 1.0, 0.0], SERIES_TERMS)
    # Lebl, (7.2.6): y1 = 1 + x^3/6 + x^6/180 + ..., y2 = x + x^4/12 + x^7/504 + ...; a_(3n+2) = 0
    assert y1[[0, 3, 6]] == pytest.approx([1.0, 1 / 6, 1 / 180])
    assert y2[[1, 4, 7]] == pytest.approx([1.0, 1 / 12, 1 / 504])
    assert not y1[2::3].any() and not y2[2::3].any()
    ai0, ai0_prime, bi0, bi0_prime = airy(0.0)
    for x in AIRY_POINTS:
        ai, _, bi, _ = airy(x)
        assert ai0 * ol.series_value(y1, x) + ai0_prime * ol.series_value(y2, x) == pytest.approx(
            ai, abs=AIRY_AGREEMENT)
        assert bi0 * ol.series_value(y1, x) + bi0_prime * ol.series_value(y2, x) == pytest.approx(
            bi, rel=AIRY_AGREEMENT)


def test_series_example_7_2_3_hermite():
    # n = 3: the odd series stops, x - (2/3) x^3; n = 4: the even series stops, 1 - 4x^2 + (4/3) x^4 (Lebl)
    odd = ol.series_coefficients(hermite_rule(3), [0.0, 1.0], SERIES_TERMS)
    assert odd[:4] == pytest.approx([0.0, 1.0, 0.0, -2 / 3]) and not odd[4:].any()
    even = ol.series_coefficients(hermite_rule(4), [1.0, 0.0], SERIES_TERMS)
    assert even[:5] == pytest.approx([1.0, 0.0, -4.0, 0.0, 4 / 3]) and not even[5:].any()
    # against NumPy's Hermite polynomials H_n (the same equation): equal up to a constant factor
    for n in range(1, 8):
        start = [1.0, 0.0] if n % 2 == 0 else [0.0, 1.0]
        found = ol.series_coefficients(hermite_rule(n), start, n + 1)
        power = np.polynomial.hermite.herm2poly([0.0] * n + [1.0])
        assert found * power[n] / found[n] == pytest.approx(power, abs=SERIES_EXACT * abs(power).max())


def test_series_example_7_3_2_with_r_inside_the_recurrence():
    # 4x^2 y'' - 4x^2 y' + (1 - 2x) y = 0: r = 1/2, and the coefficients 1/k!: y = x^(1/2) e^x
    from math import factorial

    r = ol.indicial_roots(0.0, 0.25)[0]
    found = ol.series_coefficients(
        lambda k, a: (4 * (k + r - 1) + 2) / (4 * (k + r) * (k + r - 1) + 1) * a[k - 1], [1.0], SERIES_TERMS)
    assert found == pytest.approx([1 / factorial(k) for k in range(SERIES_TERMS)], rel=SERIES_EXACT)
    x = np.linspace(0.1, 2.0, 9)
    assert x ** r * ol.series_value(found, x) == pytest.approx(np.sqrt(x) * np.exp(x), rel=SERIES_EXACT)


@pytest.mark.parametrize("nu", BESSEL_ORDERS)
def test_series_bessel_recurrence_against_scipy(nu):
    from math import gamma

    from scipy.special import jv

    first = 1 / (2 ** nu * gamma(1 + nu))   # Lebl, (7.3.12): the a_0 of J_nu
    found = ol.series_coefficients(bessel_rule(nu), [first, 0.0], SERIES_TERMS)
    x = BESSEL_POINTS[1:]
    assert x ** nu * ol.series_value(found, x) == pytest.approx(jv(nu, x), abs=BESSEL_AGREEMENT)


def test_series_j0_the_numbers_of_notebook_21():
    from scipy.special import jv

    def terms(count):                       # count terms that are not 0: 2 count coefficients
        return ol.series_coefficients(bessel_rule(0.0), [1.0, 0.0], 2 * count)

    at, value, digits = J0_TEN_TERMS
    assert float(ol.series_value(terms(10), at)) == pytest.approx(value, abs=digits)
    assert float(ol.series_value(terms(J0_ENOUGH), at)) == pytest.approx(jv(0, at), abs=BESSEL_AGREEMENT)
    errors = [np.abs(ol.series_value(terms(count), BESSEL_POINTS) - jv(0, BESSEL_POINTS)).max()
              for count in (J0_ENOUGH, J0_TOO_FEW)]
    assert errors[0] < BESSEL_AGREEMENT < errors[1]


def test_series_value_documented_limits():
    # The docstring: outside the radius more terms are worse (arctan at x = 1.1), and cancellation leaves
    # about 1e-9 at x = 20 for J_0 whatever the number of terms (30 terms: about 1e-5, too few)
    from scipy.special import jv

    arctan = ol.series_coefficients(lambda k, a: -(k - 2) / k * a[k - 2], [0.0, 1.0], 2 * SERIES_TERMS)

    def arctan_error(terms, point):
        return abs(float(ol.series_value(arctan[:2 * terms], point)) - np.arctan(point))

    assert arctan_error(10, ARCTAN_POINT_OUT) == pytest.approx(ARCTAN_ERROR_10[0], abs=ARCTAN_ERROR_10[1])
    assert arctan_error(40, ARCTAN_POINT_OUT) == pytest.approx(ARCTAN_ERROR_40[0], abs=ARCTAN_ERROR_40[1])
    inside = [arctan_error(n, ARCTAN_POINT_IN) for n in ARCTAN_TERMS_IN]
    assert inside == sorted(inside, reverse=True)
    j0 = ol.series_coefficients(bessel_rule(0.0), [1.0, 0.0], 4 * SERIES_TERMS)

    def j0_error(terms):
        return abs(float(ol.series_value(j0[:2 * terms], J0_FAR)) - jv(0, J0_FAR))

    assert j0_error(30) == pytest.approx(J0_TRUNCATION_30[0], rel=J0_TRUNCATION_30[1])
    for terms in J0_FLOOR_TERMS:
        assert J0_FLOOR[0] < j0_error(terms) < J0_FLOOR[1]
    largest = max(abs(c) * J0_FAR ** k for k, c in enumerate(j0))
    assert largest == pytest.approx(J0_LARGEST_TERM[0], rel=J0_LARGEST_TERM[1])


def test_series_value_is_horner_and_keeps_the_shape():
    rng = np.random.default_rng(INDICIAL_SEED)
    for _ in range(INDICIAL_DRAWS):
        coefficients = rng.uniform(*HORNER_RANGE, rng.integers(*HORNER_SIZES))
        center = rng.uniform(*HORNER_CENTERS)
        x = rng.uniform(*HORNER_RANGE, HORNER_POINTS)
        expected = np.polyval(coefficients[::-1], x - center)
        assert ol.series_value(coefficients, x, center) == pytest.approx(expected, rel=SERIES_EXACT,
                                                                         abs=SERIES_EXACT)
    one = ol.series_value([1.0, 2.0, 3.0], 2.0)
    assert isinstance(one, np.ndarray) and one.shape == () and float(one) == 17.0
    assert ol.series_value([5.0], [[1.0, 2.0], [3.0, 4.0]]).shape == (2, 2)
    assert ol.series_value([1.0, 1.0], [1.0, 3.0], x0=1.0) == pytest.approx([1.0, 3.0])
    assert ol.series_value((1, 2), np.int64(3)) == 7.0                      # whole numbers and tuples
    assert ol.SERIES_CENTER == 0.0


def test_series_coefficients_whole_numbers_and_the_list_a():
    seen = []

    def rule(k, a):
        seen.append((k, len(a), a[k - 1]))
        return a[k - 1] + 1

    assert ol.series_coefficients(rule, [0], np.int64(4)) == pytest.approx([0.0, 1.0, 2.0, 3.0])
    assert seen == [(1, 1, 0.0), (2, 2, 1.0), (3, 3, 2.0)]
    assert ol.series_coefficients(rule, [1.0, 2.0], 2) == pytest.approx([1.0, 2.0])     # n = len(initial)
    assert ol.series_coefficients(lambda k, a: np.float64(a[k - 1]) / 2, np.array([4.0]), 3) == pytest.approx(
        [4.0, 2.0, 1.0])
    assert ol.series_coefficients(lambda k, a: np.array(a[k - 1]), [3.0], 2) == pytest.approx([3.0, 3.0])


def test_series_coefficients_a_can_be_looped_over_and_sliced():
    # code review, BUG 1: sum(a), for v in a, np.array(a) and slices inside 0, ..., k - 1 work
    fibonacci = ol.series_coefficients(lambda k, a: sum(a[k - 2:k]), [1.0, 1.0], 8)
    assert fibonacci == pytest.approx([1, 1, 2, 3, 5, 8, 13, 21])
    assert ol.series_coefficients(lambda k, a: sum(a) / k, [1.0], 5) == pytest.approx([1.0] * 5)
    assert ol.series_coefficients(lambda k, a: float(np.array(a).sum()), [1.0], 4) == pytest.approx(
        [1, 1, 2, 4])
    assert ol.series_coefficients(lambda k, a: max(v for v in a) + len(a[:k]), [0.0], 4) == pytest.approx(
        [0, 1, 3, 6])
    cut = ol.series_coefficients(lambda k, a: sum(a[:k + 5]), [1.0], 3)      # a slice past k is cut at k
    assert cut == pytest.approx([1, 1, 2])


@pytest.mark.parametrize("bad, words", [
    (None, "recurrence must be a function of k and a"),
    ("a[k - 2]", "recurrence must be a function of k and a"),
])
def test_series_coefficients_refuses_a_recurrence_that_is_not_a_function(bad, words):
    with pytest.raises(ValueError, match=words):
        ol.series_coefficients(bad, [1.0], 3)


@pytest.mark.parametrize("bad", [None, [], "1, 0", [1.0, np.nan], [1.0, np.inf], [[1.0, 0.0]], 1.0,
                                 [True, 0.0], [1.0, 1j], [1.0, None]])
def test_series_coefficients_refuses_bad_initial_values(bad):
    with pytest.raises(ValueError, match="initial must be a list of finite real numbers"):
        ol.series_coefficients(cosh_rule, bad, 5)


@pytest.mark.parametrize("bad", [20.0, True, -1, 1, "20", None, ol.MAX_STEPS + 1, np.float64(5)])
def test_series_coefficients_refuses_bad_n(bad):
    with pytest.raises(ValueError, match=r"n, the number of coefficients, must be a whole number such as 20"):
        ol.series_coefficients(cosh_rule, [1.0, 0.0], bad)


@pytest.mark.parametrize("rule, initial, words", [
    (airy_rule, [1.0, 0.0],
     r"asked for a\[-1\] at k = 2: there is no coefficient with a negative index \(use "
     r"a\[k - 1\], not a\[-1\], .*give at least 3 initial values"),
    (lambda k, a: a[-2], [1.0, 2.0], r"asked for a\[-2\] at k = 2: .*\(use a\[k - 2\], not a\[-2\]"),
    (lambda k, a: a[k], [1.0],
     r"asked for a\[1\] at k = 1: only a\[0\] to a\[0\] are known when a_1 is computed"),
    (lambda k, a: a[k / 2], [1.0, 2.0], r"asked for a\[1.0\] at k = 2: an index of a must be a whole number"),
    (lambda k, a: a[True], [1.0, 2.0], r"an index of a must be a whole number"),
    (lambda k, a: sum(a[-2:]), [1.0, 2.0],
     r"asked for a\[-2:None\] at k = 2: a slice of a must start and stop"),
    (lambda k, a: sum(a[k - 3:k]), [1.0, 2.0], r"asked for a\[-1:2\] at k = 2: a slice of a must start"),
    (lambda k, a: 1 / (k - 3) * a[k - 1], [1.0],
     r"stopped with an error at k = 3 \(ZeroDivisionError: .*\): a division by 0 \(if you wrote a power with "
     r"\^, write \*\* instead\)"),
    (lambda k, a: np.float64(1.0) / (k - 2), [1.0],
     r"gives inf at k = 2, not a finite number \(a division by 0\?\)"),
    (lambda k, a: np.sqrt(2.0 - k), [1.0], r"gives nan at k = 3"),
    (lambda k, a: "1", [1.0], r"must return one real number, got '1' at k = 1"),
    (lambda k, a: [1.0], [1.0], r"must return one real number, got \[1.0\] at k = 1"),
    (lambda k, a: 1j, [1.0], r"must return one real number, got 1j at k = 1"),
    (lambda k, a: True, [1.0], r"must return one real number, got True at k = 1"),
    (lambda k, a: None, [1.0], r"must return one real number, got None at k = 1"),
    (lambda k: k, [1.0], r"stopped at k = 1 with a TypeError .*it must take two arguments, k and the list a"),
    (lambda k, a: a[k - 1] + "x", [1.0], r"stopped with an error at k = 1 \(TypeError: unsupported operand"),
    (lambda k, a: k ^ 2.0, [1.0], r"a power is written k \*\* 2, not k \^ 2"),
    (lambda k, a: undefined_name, [1.0], r"stopped with an error at k = 1 \(NameError"),       # noqa: F821
])
def test_series_coefficients_refuses_a_recurrence_with_a_sentence_that_names_k(rule, initial, words):
    with pytest.raises(ValueError, match=words):
        ol.series_coefficients(rule, initial, 6)


def test_series_coefficients_a_type_error_inside_is_not_called_a_wrong_signature():
    # code review, STYLE 6: only a function that does not take (k, a) is told to take two arguments
    with pytest.raises(ValueError) as caught:
        ol.series_coefficients(lambda k, a: a[k - 1] + "x", [1.0], 3)
    assert "it must take two arguments" not in str(caught.value)


def test_series_coefficients_a_wrong_index_is_never_read_from_the_end():
    # the bug that this guards against: with a plain list, Airy's rule and two initial values would read
    # a[-1] = a_1 as a_(-1), and a_2 would come out wrong with no error
    with pytest.raises(ValueError, match="give at least 3 initial values"):
        ol.series_coefficients(airy_rule, [0.0, 1.0], 10)
    with pytest.raises(ValueError, match="give at least 2 initial values"):
        ol.series_coefficients(cosh_rule, [1.0], 10)


@pytest.mark.parametrize("coefficients", [None, [], "1 2", [1.0, np.nan], [[1.0], [2.0]], [1.0, True], 3.0])
def test_series_value_refuses_bad_coefficients(coefficients):
    with pytest.raises(ValueError, match="coefficients must be a list of finite real numbers"):
        ol.series_value(coefficients, 1.0)


@pytest.mark.parametrize("x, words", [
    ("1", "x must be a real number, a list or an array of real numbers"),
    (np.nan, "x must be a real number, a list or an array of real numbers"),
    (np.inf, "x must be made of finite real numbers"),
    ([1.0, -np.inf], "x must be made of finite real numbers"),
    (1j, "x must be a real number"),
    (None, "x must be a real number"),
])
def test_series_value_refuses_bad_x(x, words):
    with pytest.raises(ValueError, match=words):
        ol.series_value([1.0, 2.0], x)


@pytest.mark.parametrize("x0", [None, np.nan, np.inf, True, "0", 1j, [0.0, 1.0]])
def test_series_value_refuses_bad_x0(x0):
    with pytest.raises(ValueError, match="x0 must be one finite real number"):
        ol.series_value([1.0, 2.0], 1.0, x0)


def test_series_value_too_large_for_the_computer():
    with pytest.raises(ValueError, match="the partial sum is too large for the computer"):
        ol.series_value([1.0, 1.0, 1.0], 1e200)
    with pytest.raises(ValueError, match="the partial sum is too large for the computer"):
        ol.series_value([0.0, 1e300, -1e300], [1.0, 1e10])


def test_series_value_x_minus_x0_too_large_gives_no_warning():
    # code review, RISK 2: x - x0 overflows; with warnings as errors the answer is still the ValueError
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="the partial sum is too large for the computer"):
            ol.series_value([2.0, 1.0], HUGE_X, -HUGE_X)
        assert float(ol.series_value([2.0], HUGE_X, -HUGE_X)) == 2.0     # one coefficient: a_0 itself


# The four cases: (p0, q0) -> (r1, r2, case). Lebl's examples, the notebook's, and exercises with answers.
INDICIAL_BOOK = {
    "Example 7.3.2, 4x^2 y'' - 4x^2 y' + (1 - 2x) y = 0": ((0.0, 0.25), (0.5, 0.5, ol.INDICIAL_DOUBLE)),
    "Example 7.3.3, x^2 y'' + x(1 + x) y' + (pi + x^2) y = 0": (
        (1.0, np.pi), (1j * np.sqrt(np.pi), -1j * np.sqrt(np.pi), ol.INDICIAL_COMPLEX)),
    "Exercise 7.3.8, x^2 y'' - y = 0": (
        (0.0, -1.0), ((1 + np.sqrt(5)) / 2, (1 - np.sqrt(5)) / 2, ol.INDICIAL_NOT_INTEGER)),
    "Exercise 7.3.10, x^2 y'' - x y' + y = 0": ((-1.0, 1.0), (1.0, 1.0, ol.INDICIAL_DOUBLE)),
    "notebook 21, Exercise 2(c), 2x^2 y'' + 3x y' - (1 + x) y = 0": (
        (1.5, -0.5), (0.5, -1.0, ol.INDICIAL_NOT_INTEGER)),
    "Bessel, order 0": ((1.0, 0.0), (0.0, 0.0, ol.INDICIAL_DOUBLE)),
    "Bessel, order 1/2": ((1.0, -0.25), (0.5, -0.5, ol.INDICIAL_INTEGER)),
    "Bessel, order 1": ((1.0, -1.0), (1.0, -1.0, ol.INDICIAL_INTEGER)),
    "Bessel, order 1/3": ((1.0, -1 / 9), (1 / 3, -1 / 3, ol.INDICIAL_NOT_INTEGER)),
    "Euler, x^2 y'' - 2x y' + 2y = 0": ((-2.0, 2.0), (2.0, 1.0, ol.INDICIAL_INTEGER)),
    "Euler, x^2 y'' + 2x y' = 0": ((2.0, 0.0), (0.0, -1.0, ol.INDICIAL_INTEGER)),
}


@pytest.mark.parametrize("name", sorted(INDICIAL_BOOK))
def test_indicial_roots_of_the_book_and_the_notebook(name):
    (p0, q0), (r1, r2, case) = INDICIAL_BOOK[name]
    found = ol.indicial_roots(p0, q0)
    assert found[2] == case
    assert found[0] == pytest.approx(r1, abs=SERIES_EXACT) and found[1] == pytest.approx(r2, abs=SERIES_EXACT)
    for root in found[:2]:                  # each root solves r(r - 1) + p0 r + q0 = 0
        assert root * (root - 1) + p0 * root + q0 == pytest.approx(0.0, abs=SERIES_EXACT)
    kind = complex if case == ol.INDICIAL_COMPLEX else float
    assert all(type(root) is kind for root in found[:2])


def test_indicial_roots_seeded_pairs_of_each_kind():
    # p0 = 1 - (r1 + r2) and q0 = r1 r2 have the roots r1 and r2
    rng = np.random.default_rng(INDICIAL_SEED)
    for _ in range(INDICIAL_DRAWS):
        root = rng.uniform(*ROOT_RANGE)
        whole = int(rng.choice(WHOLE_DIFFERENCES))
        other = root - whole - rng.uniform(*NOT_WHOLE_SHARE)     # a difference that is not a whole number
        cases = {(root, root): ol.INDICIAL_DOUBLE, (root, root - whole): ol.INDICIAL_INTEGER,
                 (root, other): ol.INDICIAL_NOT_INTEGER}
        for (r1, r2), case in cases.items():
            found = ol.indicial_roots(1 - (r1 + r2), r1 * r2)
            assert found[2] == case and found[:2] == pytest.approx((r1, r2), abs=ROOT_AGREEMENT)
        alpha, beta = rng.uniform(*ROOT_RANGE), rng.uniform(*BETA_RANGE)
        found = ol.indicial_roots(1 - 2 * alpha, alpha ** 2 + beta ** 2)
        assert found[2] == ol.INDICIAL_COMPLEX
        assert found[0] == pytest.approx(complex(alpha, beta))
        assert found[1] == pytest.approx(complex(alpha, -beta))


def test_indicial_roots_at_and_near_the_borders_of_the_cases():
    # the rule of the docstring: (r1 - r2)^2 within INDICIAL_TOLERANCE max(1, (p0 - 1)^2 + 4 |q0|) of m^2
    tol = ol.INDICIAL_TOLERANCE
    assert tol == 1e-12
    # a double root typed with decimals: 0.7 and 0.0225 give (r - 0.15)^2 only up to rounding
    r1, r2, case = ol.indicial_roots(0.7, 0.0225)
    assert case == ol.INDICIAL_DOUBLE and r1 == r2 == pytest.approx(0.15)
    # roots 1 +- gap/2 (and 1 +- i gap/2): p0 = -1, q0 = 1 -+ gap^2/4, and max(1, ...) is about 8
    for gap in (1e-7, 1e-6):                # closer than about 1e-6: a double root 1
        assert ol.indicial_roots(-1.0, 1 - gap ** 2 / 4)[:2] == (1.0, 1.0)
        assert ol.indicial_roots(-1.0, 1 - gap ** 2 / 4)[2] == ol.INDICIAL_DOUBLE
        assert ol.indicial_roots(-1.0, 1 + gap ** 2 / 4)[2] == ol.INDICIAL_DOUBLE
    for gap in (1e-5, 1e-3):                # farther: two roots, or complex ones
        assert ol.indicial_roots(-1.0, 1 - gap ** 2 / 4)[2] == ol.INDICIAL_NOT_INTEGER
        assert ol.indicial_roots(-1.0, 1 + gap ** 2 / 4)[2] == ol.INDICIAL_COMPLEX
    for whole in (1, 2, 7):                 # roots 0 and -(whole + off): p0 = 1 + whole + off, q0 = 0
        assert ol.indicial_roots(1.0 + whole, 0.0)[2] == ol.INDICIAL_INTEGER
        offs = {0.1 * tol * whole: ol.INDICIAL_INTEGER, 10 * tol * whole: ol.INDICIAL_NOT_INTEGER,
                1e-6: ol.INDICIAL_NOT_INTEGER, 0.5: ol.INDICIAL_NOT_INTEGER}
        for off, case in offs.items():
            assert ol.indicial_roots(1.0 + whole + off, 0.0)[2] == case
    assert ol.indicial_roots(1.0, 0.0) == (0.0, 0.0, ol.INDICIAL_DOUBLE)
    assert str(ol.indicial_roots(1.0, 0.0)[0]) == "0.0"   # not -0.0
    assert str(ol.indicial_roots(2.0, 0.0)[0]) == "0.0"   # code review, STYLE 4: 0.0 / -1.0 was -0.0
    assert ol.indicial_roots(sum([0.1] * 10), -0.25)[2] == ol.INDICIAL_INTEGER   # 0.999... for 1: +-1/2


def test_indicial_roots_large_and_small_numbers():
    big = 1e4 + 0.5                         # roots near 1e4 and 1e-4: no cancellation
    r1, r2, case = ol.indicial_roots(1 - big, 1.0)
    assert r1 == pytest.approx(big) and r2 == pytest.approx(1 / big, rel=SERIES_EXACT)
    assert case == ol.INDICIAL_NOT_INTEGER
    assert ol.indicial_roots(np.float64(1.0), np.int64(-4)) == (2.0, -2.0, ol.INDICIAL_INTEGER)
    assert ol.indicial_roots(3, 1) == (-1.0, -1.0, ol.INDICIAL_DOUBLE)


def test_indicial_roots_the_documented_band_and_the_refusal_of_large_roots():
    # mathematics review, ERROR 1: the docstring's example (roots 1000 and 1000.002 give the double root
    # 1000.001), and roots of size 1.8e5 and more are refused instead of being guessed
    assert ol.indicial_roots(*LARGE_GAP)[:2] == pytest.approx((LARGE_DOUBLE, LARGE_DOUBLE))
    assert ol.indicial_roots(*LARGE_GAP)[2] == ol.INDICIAL_DOUBLE
    assert ol.INDICIAL_AMBIGUOUS == 0.25
    accepted, refused = AMBIGUOUS_SIDES
    assert ol.indicial_roots(1 - 2 * accepted, accepted ** 2)[2] == ol.INDICIAL_DOUBLE
    with pytest.raises(ValueError, match="too large for the computer to tell a double root from two roots 1"):
        ol.indicial_roots(1 - 2 * refused, refused ** 2)
    # the reviewer's rows: roots 1000001 and 1000000 (and others) were a "double root"; now refused
    for p0, q0 in ((-2000000.0, 1000001000000.0), (-1999999.5, 1000000500000.0), (2000000.5, 999999500000.0)):
        with pytest.raises(ValueError, match="the roots, up to about 1e\\+06 in size, are too large"):
            ol.indicial_roots(p0, q0)


def test_indicial_roots_the_rechecked_sentences_of_the_docstring():
    # re-check, findings 1 to 3: the band 1e-6 sqrt(...), the size named in the refusal, and the wide bands
    # just below the limit (all outside the range of the course, and said in the docstring)
    near, gaps_in, gaps_out = RECHECK_BAND
    for gap in gaps_in:
        assert ol.indicial_roots(1 - (2 * near + gap), near * (near + gap))[2] == ol.INDICIAL_DOUBLE
    for gap in gaps_out:
        assert ol.indicial_roots(1 - (2 * near + gap), near * (near + gap))[2] == ol.INDICIAL_NOT_INTEGER
    with pytest.raises(ValueError, match="the roots, up to about 3e\\+05 in size"):
        ol.indicial_roots(1.0, -RECHECK_SYMMETRIC ** 2)            # roots +-3e5: the mean 0 is not named
    assert ol.indicial_roots(1 - RECHECK_COMPUTED, 0.0)[:2] == (RECHECK_COMPUTED, 0.0)   # 4e5, 0: computed
    low, (close, near_one) = RECHECK_WIDE
    double = ol.indicial_roots(1 - (2 * low + close), low * (low + close))
    assert double[2] == ol.INDICIAL_DOUBLE and double[0] == pytest.approx(low + close / 2)
    assert ol.indicial_roots(1 - (2 * low + near_one), low * (low + near_one))[2] == ol.INDICIAL_INTEGER


@pytest.mark.parametrize("p0, q0, words", [
    (None, 0.0, "p0 must be one finite real number"), ("1", 0.0, "p0 must be one finite real number"),
    (np.nan, 0.0, "p0 must be one finite real number"), (np.inf, 0.0, "p0 must be one finite real number"),
    (True, 0.0, "p0 must be one finite real number"), (1j, 0.0, "p0 must be one finite real number"),
    (10 ** 400, 0.0, "p0 must be one finite real number"),
    ([1.0, 2.0], 0.0, "p0 must be one finite real number"),
    (1.0, None, "q0 must be one finite real number"), (1.0, np.nan, "q0 must be one finite real number"),
    (1.0, False, "q0 must be one finite real number"), (1.0, "0", "q0 must be one finite real number"),
    (1e200, 0.0, "are too large for the computer"), (1.0, -np.inf, "q0 must be one finite real number"),
    (1.0, 1e308, "are too large for the computer"),
    (1.0, -(1e100 ** 2), "too large for the computer to tell"),
])
def test_indicial_roots_refuses_bad_input(p0, q0, words):
    with pytest.raises(ValueError, match=words):
        ol.indicial_roots(p0, q0)


def test_series_functions_leave_warnings_and_numpy_settings_alone():
    filters, settings = list(warnings.filters), np.geterr()
    ol.series_coefficients(cosh_rule, [1.0, 0.0], SERIES_TERMS)
    with pytest.raises(ValueError):
        ol.series_coefficients(lambda k, a: np.float64(1.0) / 0.0, [1.0], 3)
    with pytest.raises(ValueError):
        ol.series_value([1.0, 1.0, 1.0], 1e200)
    ol.indicial_roots(0.7, 0.0225)
    assert list(warnings.filters) == filters and np.geterr() == settings
    assert ol.SETUP_NAMES == ("np", "sp", "plt", "ol", "solve_ivp", "interact", "FloatSlider", "IntSlider")
    assert (ol.INDICIAL_NOT_INTEGER, ol.INDICIAL_DOUBLE, ol.INDICIAL_INTEGER, ol.INDICIAL_COMPLEX) == (
        "two real roots, difference not a whole number", "double root",
        "two real roots, difference a whole number", "complex roots")


# --- numerical methods, part 2 (notebook 22; Lebl, Section 1.7) ------------------------------------------
TABLE_1_7_1_ERRORS = (1.07407, 0.79139, 0.52751, 0.31966, 0.17960, 0.09588, 0.04965, 0.02528)
TABLE_1_7_1_RATIOS = (0.73681, 0.66656, 0.60599, 0.56184, 0.53385, 0.51779, 0.50913)
TABLE_DECIMALS = 1e-5                       # the book prints five decimals
TABLE_STEPS = (2, 4, 8, 16, 32, 64, 128, 256)          # N on [0, 2]: h = 1, 1/2, ..., 1/128
EXERCISE_1_7_7 = ((1, 2, 4), (16.0, 8.0, 4.0))          # Euler for x' = 2t to t = 4: steps, printed errors
COSINE_START = 1.0                          # notebook 22: y' = y cos t, y(0) = 1, exact y = e^(sin t)
COSINE_END = 4.0                            # its convergence study ends at T = 4
COSINE_SPAN = (0.0, COSINE_END)
LOCAL_STEPS = (0.2, 0.1, 0.05, 0.025)       # plan 22: the local errors of one step from t = 0
LOCAL_EULER = (1.978e-2, 4.987e-3, 1.249e-3, 3.125e-4)  # Euler's, printed to 4 digits
GLOBAL_STEPS = (10, 20, 40, 80, 160)        # plan 22: the global errors at T = 4
GLOBAL_EULER = (7.814e-2, 3.281e-2, 1.511e-2, 7.251e-3, 3.553e-3)
PRINTED_DIGITS = 1e-3                       # relative: four printed digits
ORDER_DIGITS = 0.05                         # an observed order against the expected one, on the finest steps
SPRING_RATE = 4.0                           # notebook 22: x'' + 4x = 0, x(0) = 1, x'(0) = 0, x = cos 2t
SPRING_END = 5.0                            # the error of x at T = 5
SPRING_STEPS = (25, 50, 100, 200, 400, 800)
SPRING_ERRORS = {"euler": (5.530, 5.441e-2), "heun": (0.1069, 1.396e-4), "rk4": (5.080e-4, 1.089e-9)}
SPRING_PERIODS = 2.0 * np.pi                # notebook 22: Euler's amplitude after t = 2 pi with N = 40 steps
SPRING_PERIOD_STEPS = 40
SPRING_AMPLITUDE = (6.57, 0.01)             # Euler's amplitude there, and its printed digits
STIFF_RATE = 100.0                          # notebook 22: y' = -100 (y - sin t), y(0) = 1
STIFF_STEPS = ((0.015, -0.5), (0.025, -1.5), (0.03, -2.0))   # Euler's factor 1 - 100 h of the gap
BACKWARD_END = 1.0
BACKWARD_H = (0.5, 0.1, 0.05, 0.02, 0.01)
LARGE_PRODUCTS = (-10.0, -100.0, -1e4, -1e6)    # lambda h for the stability test of backward Euler
CLOSED_FORM = 1e-12                         # a recurrence of the notebook against its closed form (relative)
EMBEDDED_H = (0.1, 0.01, 0.001)             # the estimate against Euler's true local error on y' = y cos t
EMBEDDED_SHARE = (0.15, 0.02, 0.002)        # |estimate / true - 1| below these
SEEDED_CASES = 20                           # seeded linear equations
SEEDED_RATES = (-5.0, 2.0)                  # lambda drawn in this range
SEEDED_STARTS = (-3.0, 3.0)                 # y0 drawn in this range
SEEDED_N = (1, 40)                          # numbers of steps drawn (high excluded)
SEEDED_DIGITS = (1e-9, 1e-15)               # relative and absolute: one step against its formula
STIFF_LONG = (0.0, 3.0)                     # the interval of the test of Euler's factor
STABILITY_SPAN = (0.0, 10.0)                # the interval of the stability test, steps h = 1
NONLINEAR_SPAN = (0.0, 5.0)                 # y' = -y^3 from y(0) = 2
NEWTON_MARGIN = 10                          # a residual within this many NEWTON_TOLERANCE
GROWING_RATE = 10.0                         # y' = 10 y: backward Euler at h lambda = 1 and 2
TOO_MANY_BACKWARD = ((0.0, 20.0), 1e-4)     # 200 000 steps: above BACKWARD_MAX_STEPS
OVERFLOW_SPAN = (0.0, 1000.0)               # y' = y with h = 0.5: the values double and overflow


def cosine_rate(t, y):
    return y * np.cos(t)                    # y' = y cos t


def cosine_exact(t):
    return np.exp(np.sin(t))


def spring_rate(t, u):
    return np.array([u[1], -SPRING_RATE * u[0]])          # x' = v, v' = -4x


def spring_exact(t):
    root = np.sqrt(SPRING_RATE)
    return np.array([np.cos(root * t), -root * np.sin(root * t)])


def stiff_rate(t, y):
    return -STIFF_RATE * (y - np.sin(t))


def stiff_exact(t):
    # y = A e^(-100 t) + (100^2 sin t - 100 cos t) / (100^2 + 1), with y(0) = 1
    slow = (STIFF_RATE ** 2 * np.sin(t) - STIFF_RATE * np.cos(t)) / (STIFF_RATE ** 2 + 1)
    return (1 + STIFF_RATE / (STIFF_RATE ** 2 + 1)) * np.exp(-STIFF_RATE * t) + slow


def lebl_rate(x, y):
    return y ** 2 / 3                       # Lebl, Section 1.7: y' = y^2 / 3, y(0) = 1, y = 3 / (3 - x)


def test_global_error_reproduces_lebl_table_1_7_1():
    sizes, errors = ol.global_error(ol.euler, lebl_rate, (0.0, 2.0), 1.0, lambda x: 3 / (3 - x), TABLE_STEPS)
    assert sizes == pytest.approx([2.0 / n for n in TABLE_STEPS])
    assert errors == pytest.approx(TABLE_1_7_1_ERRORS, abs=TABLE_DECIMALS)
    assert errors[1:] / errors[:-1] == pytest.approx(TABLE_1_7_1_RATIOS, abs=TABLE_DECIMALS)


def test_global_error_lebl_exercise_1_7_7():
    # x' = 2t, x(0) = 0: Euler gives 0, 8, 12 at t = 4 with h = 4, 2, 1; printed errors 16, 8, 4
    steps, printed = EXERCISE_1_7_7
    sizes, errors = ol.global_error(ol.euler, lambda t, x: 2 * t, (0.0, 4.0), 0.0, lambda t: t ** 2, steps)
    assert list(sizes) == [4.0, 2.0, 1.0] and list(errors) == list(printed)
    assert list(ol.observed_order(errors)) == [1.0, 1.0]           # the factors 0.5, 0.5 of the book


def test_local_error_numbers_of_the_notebook():
    errors = [ol.local_error(ol.euler, cosine_rate, 0.0, COSINE_START, h, cosine_exact) for h in LOCAL_STEPS]
    assert errors == pytest.approx(LOCAL_EULER, rel=PRINTED_DIGITS)
    assert errors[0] == pytest.approx(cosine_exact(LOCAL_STEPS[0]) - (1 + LOCAL_STEPS[0]))     # 1.21978 - 1.2


@pytest.mark.parametrize(("method", "order"), [(ol.euler, 1), (ol.heun, 2), (ol.rk4, 4)])
def test_local_and_global_orders_of_the_three_methods(method, order):
    local = [ol.local_error(method, cosine_rate, 0.0, COSINE_START, h, cosine_exact) for h in LOCAL_STEPS]
    assert ol.observed_order(local)[-1] == pytest.approx(order + 1, abs=ORDER_DIGITS)
    _, errors = ol.global_error(method, cosine_rate, COSINE_SPAN, COSINE_START, cosine_exact, GLOBAL_STEPS)
    assert ol.observed_order(errors)[-1] == pytest.approx(order, abs=ORDER_DIGITS)


def test_global_error_numbers_of_the_notebook():
    _, errors = ol.global_error(ol.euler, cosine_rate, COSINE_SPAN, COSINE_START, cosine_exact,
                                GLOBAL_STEPS)
    assert errors == pytest.approx(GLOBAL_EULER, rel=PRINTED_DIGITS)


@pytest.mark.parametrize("name", sorted(SPRING_ERRORS))
def test_global_error_on_the_spring_as_a_system(name):
    # the error of x alone (component 0) at T = 5, as in the notebook; the largest component is never smaller
    method = getattr(ol, name)
    sizes, errors = ol.global_error(method, spring_rate, (0.0, SPRING_END), [1.0, 0.0], spring_exact,
                                    SPRING_STEPS, component=0)
    first, last = SPRING_ERRORS[name]
    assert errors[0] == pytest.approx(first, rel=PRINTED_DIGITS)
    assert errors[-1] == pytest.approx(last, rel=PRINTED_DIGITS)
    _, largest = ol.global_error(method, spring_rate, (0.0, SPRING_END), [1.0, 0.0], spring_exact,
                                 SPRING_STEPS)
    assert np.all(largest >= errors)
    _, second = ol.global_error(method, spring_rate, (0.0, SPRING_END), [1.0, 0.0], spring_exact,
                                SPRING_STEPS, component=1)
    assert largest == pytest.approx(np.maximum(errors, second))     # the largest component, not a sum
    assert sizes == pytest.approx([SPRING_END / n for n in SPRING_STEPS])


def test_euler_amplitude_on_the_spring_grows_as_the_notebook_says():
    h = SPRING_PERIODS / SPRING_PERIOD_STEPS
    _, values = ol.euler(spring_rate, (0.0, SPRING_PERIODS), [1.0, 0.0], h)
    amplitude = np.hypot(values[-1, 0], values[-1, 1] / np.sqrt(SPRING_RATE))
    factor = (1 + SPRING_RATE * h ** 2) ** (SPRING_PERIOD_STEPS / 2)    # sqrt(1 + 4h^2) at each step
    assert amplitude == pytest.approx(factor, rel=CLOSED_FORM)
    assert amplitude == pytest.approx(SPRING_AMPLITUDE[0], abs=SPRING_AMPLITUDE[1])


def test_local_and_global_error_of_a_system_take_the_largest_component():
    h = SPRING_END / SPRING_STEPS[0]
    one_step = ol.euler(spring_rate, (0.0, h), [1.0, 0.0], h)[1][-1]
    expected = np.abs(one_step - spring_exact(h))
    largest = ol.local_error(ol.euler, spring_rate, 0.0, [1.0, 0.0], h, spring_exact)
    assert largest == pytest.approx(expected.max())
    for j in (0, 1):
        assert ol.local_error(ol.euler, spring_rate, 0.0, [1.0, 0.0], h, spring_exact,
                              component=j) == pytest.approx(expected[j])


def test_global_error_accepts_a_method_written_by_a_student():
    def midpoint(f, t_span, y0, h):
        t = np.arange(t_span[0], t_span[1] + h / 2, h)
        y = [float(y0)]
        for t_now in t[:-1]:
            k1 = f(t_now, y[-1])
            y.append(y[-1] + h * f(t_now + h / 2, y[-1] + h / 2 * k1))
        return t, np.array(y)

    _, errors = ol.global_error(midpoint, cosine_rate, COSINE_SPAN, COSINE_START, cosine_exact,
                                GLOBAL_STEPS)
    assert ol.observed_order(errors)[-1] == pytest.approx(2, abs=ORDER_DIGITS)
    local = [ol.local_error(midpoint, cosine_rate, 0.0, COSINE_START, h, cosine_exact) for h in LOCAL_STEPS]
    assert ol.observed_order(local)[-1] == pytest.approx(3, abs=ORDER_DIGITS)


@pytest.mark.parametrize(("h", "factor"), STIFF_STEPS)
def test_euler_factor_on_the_stiff_equation(h, factor):
    # the gap y - (slow part) is multiplied by 1 - 100 h at each Euler step (the notebook's predictions)
    assert 1 - STIFF_RATE * h == pytest.approx(factor)
    stable = abs(factor) < 1
    _, values = ol.euler(stiff_rate, STIFF_LONG, 1.0, h)
    assert (np.max(np.abs(values - stiff_exact(np.arange(values.size) * h))) < 1) == stable


@pytest.mark.parametrize("h", BACKWARD_H)
def test_backward_euler_against_its_closed_form_on_the_stiff_equation(h):
    # linear: y_(n+1) = (y_n + 100 h sin t_(n+1)) / (1 + 100 h), the formula of the notebook
    t, values = ol.backward_euler(stiff_rate, (0.0, BACKWARD_END), 1.0, h)
    expected = [1.0]
    for t_next in t[1:]:
        expected.append((expected[-1] + STIFF_RATE * h * np.sin(t_next)) / (1 + STIFF_RATE * h))
    assert values == pytest.approx(expected, rel=CLOSED_FORM)
    given = ol.backward_euler(stiff_rate, (0.0, BACKWARD_END), 1.0, h, jacobian=lambda t, y: -STIFF_RATE)[1]
    assert given == pytest.approx(expected, rel=CLOSED_FORM)


def test_backward_euler_has_order_one():
    errors = [abs(ol.backward_euler(cosine_rate, COSINE_SPAN, COSINE_START, COSINE_END / n)[1][-1]
                  - cosine_exact(COSINE_END)) for n in GLOBAL_STEPS]
    assert ol.observed_order(errors)[-1] == pytest.approx(1, abs=ORDER_DIGITS)


@pytest.mark.parametrize("product", LARGE_PRODUCTS)
def test_backward_euler_is_stable_for_every_step(product):
    # y' = lambda y with lambda h very negative: the values decay, never change sign; Euler's grow
    rate, h = product, 1.0
    t, values = ol.backward_euler(lambda t, y: rate * y, STABILITY_SPAN, 1.0, h)
    assert np.all(values > 0) and np.all(np.diff(values) < 0)
    assert values == pytest.approx((1 / (1 - product)) ** np.arange(t.size), rel=CLOSED_FORM)
    euler_values = ol.euler(lambda t, y: rate * y, STABILITY_SPAN, 1.0, h)[1]
    assert np.all(np.abs(euler_values[1:]) > 1) and np.all(np.diff(np.abs(euler_values)) > 0)


def test_backward_euler_returns_the_shapes_of_euler():
    t, values = ol.backward_euler(spring_rate, (0.0, 1.0), [1.0, 0.0], 0.1)
    t_euler, values_euler = ol.euler(spring_rate, (0.0, 1.0), [1.0, 0.0], 0.1)
    assert values.shape == values_euler.shape == (11, 2) and np.array_equal(t, t_euler)
    assert ol.backward_euler(stiff_rate, (0.0, 1.0), 1.0, 0.1)[1].shape == (11,)


def test_backward_euler_on_a_nonlinear_equation_solves_each_step():
    # y' = -y^3 (not linear): each value must satisfy its own equation Y = y_n + h f(t, Y)
    h = 0.5
    t, values = ol.backward_euler(lambda t, y: -y ** 3, NONLINEAR_SPAN, 2.0, h)
    residuals = values[1:] - values[:-1] - h * (-values[1:] ** 3)
    assert np.max(np.abs(residuals)) < ol.NEWTON_TOLERANCE * NEWTON_MARGIN


def test_backward_euler_on_a_growing_solution_as_the_docstring_says():
    # lambda > 0: h lambda = 1 cannot be solved; h lambda = 2 gives 1, -1, 1, ... (the solution grows)
    with pytest.raises(ValueError, match="cannot be solved"):
        ol.backward_euler(lambda t, y: GROWING_RATE * y, (0.0, 1.0), 1.0, 1 / GROWING_RATE,
                          jacobian=lambda t, y: GROWING_RATE)
    with pytest.raises(ValueError, match="backward Euler"):
        ol.backward_euler(lambda t, y: GROWING_RATE * y, (0.0, 1.0), 1.0, 1 / GROWING_RATE)
    values = ol.backward_euler(lambda t, y: GROWING_RATE * y, (0.0, 1.0), 1.0, 2 / GROWING_RATE)[1]
    assert values == pytest.approx((-1.0) ** np.arange(values.size))


def test_an_error_inside_the_jacobian_is_its_own_message():
    def broken(t, y):
        return 1 / 0

    with pytest.raises(ZeroDivisionError):
        ol.backward_euler(stiff_rate, (0.0, 1.0), 1.0, 0.1, jacobian=broken)


def test_embedded_step_numbers_and_shapes():
    value, estimate = ol.embedded_step(cosine_rate, 0.0, COSINE_START, LOCAL_STEPS[0])
    h = LOCAL_STEPS[0]
    k1, k2 = 1.0, (1 + h) * np.cos(h)
    assert isinstance(value, float) and value == pytest.approx(1 + h * (k1 + k2) / 2)
    assert estimate == pytest.approx(h / 2 * abs(k2 - k1))
    pair, pair_estimate = ol.embedded_step(spring_rate, 0.0, [1.0, 0.0], 0.1)
    heun_value = ol.heun(spring_rate, (0.0, 0.1), [1.0, 0.0], 0.1)[1][-1]
    assert pair.shape == (2,) and pair == pytest.approx(heun_value)
    assert pair_estimate == pytest.approx(np.max(np.abs(pair - ol.euler(spring_rate, (0.0, 0.1), [1.0, 0.0],
                                                                        0.1)[1][-1])))


@pytest.mark.parametrize(("h", "share"), tuple(zip(EMBEDDED_H, EMBEDDED_SHARE, strict=True)))
def test_embedded_estimate_against_the_true_local_error_of_euler(h, share):
    _, estimate = ol.embedded_step(cosine_rate, 0.0, COSINE_START, h)
    true = ol.local_error(ol.euler, cosine_rate, 0.0, COSINE_START, h, cosine_exact)
    assert abs(estimate / true - 1) < share


def test_seeded_linear_equations():
    # y' = lambda y: one Euler step errs by |y0 (e^(lambda h) - 1 - lambda h)|; backward Euler is
    # y0 / (1 - lambda h)^n; the embedded estimate is |y0| h^2 lambda^2 / 2
    rng = np.random.default_rng(SEED)
    for _ in range(SEEDED_CASES):
        rate, start = rng.uniform(*SEEDED_RATES), rng.uniform(*SEEDED_STARTS)
        n = int(rng.integers(*SEEDED_N))
        h = 1.0 / n

        def f(t, y, rate=rate):
            return rate * y

        def exact(t, rate=rate, start=start):
            return start * np.exp(rate * t)

        one = ol.local_error(ol.euler, f, 0.0, start, h, exact)
        assert one == pytest.approx(abs(start * (np.exp(rate * h) - 1 - rate * h)), rel=SEEDED_DIGITS[0],
                                    abs=SEEDED_DIGITS[1])
        values = ol.backward_euler(f, (0.0, 1.0), start, h)[1]
        assert values == pytest.approx(start / (1 - rate * h) ** np.arange(n + 1), rel=CLOSED_FORM)
        assert ol.embedded_step(f, 0.0, start, h)[1] == pytest.approx(abs(start) * (rate * h) ** 2 / 2)


def euler_local(y0, h=0.1):
    return ol.local_error(ol.euler, cosine_rate, 0.0, y0, h, cosine_exact)


def euler_global(t_span, y0, steps):
    return ol.global_error(ol.euler, cosine_rate, t_span, y0, cosine_exact, steps)


@pytest.mark.parametrize(("call", "words"), [
    (lambda: ol.local_error("euler", cosine_rate, 0.0, 1.0, 0.1, cosine_exact), "method must be a function"),
    (lambda: ol.local_error(ol.euler, None, 0.0, 1.0, 0.1, cosine_exact), "f must be a function"),
    (lambda: ol.local_error(ol.euler, cosine_rate, 0.0, 1.0, 0.1, 2.0), "exact must be a function"),
    (lambda: ol.local_error(ol.euler, cosine_rate, "0", 1.0, 0.1, cosine_exact), "t0 must be one finite"),
    (lambda: ol.local_error(ol.euler, cosine_rate, 0.0, 1.0, 0.0, cosine_exact), "h must be a positive"),
    (lambda: ol.local_error(ol.euler, cosine_rate, 0.0, 1.0, -0.1, cosine_exact), "h must be a positive"),
    (lambda: euler_local(1.0, h=None), "h must be one finite real number"),
    (lambda: euler_local(1.0, h=[0.1]), "h must be one finite real number"),
    (lambda: euler_local(1.0, h="0.1"), "h must be one finite real number"),
    (lambda: euler_local(1.0, h=True), "h must be one finite real number"),
    (lambda: euler_local(2.0), "y0 must be the exact value"),
    (lambda: euler_local(None), "y0 must be a finite real"),
    (lambda: euler_local([[1.0]]), "y0 must be a finite real"),
    (lambda: euler_local(np.nan), "y0 must be a finite real"),
    (lambda: ol.local_error(ol.euler, cosine_rate, 0.0, 1.0, 0.1, lambda t: [1.0, 2.0]),
     "exact\\(t\\) must return 1 finite number"),
    (lambda: ol.local_error(ol.euler, spring_rate, 0.0, [1.0, 0.0], 0.1, spring_exact, component=2),
     "component must be None or a whole number from 0 to 1"),
    (lambda: ol.local_error(ol.euler, spring_rate, 0.0, [1.0, 0.0], 0.1, spring_exact, component=True),
     "component must be None"),
    (lambda: ol.local_error(lambda f, t_span, y0, h: 3.0, cosine_rate, 0.0, 1.0, 0.1, cosine_exact),
     "method must be called as method\\(f, t_span, y0, h\\)"),
    (lambda: euler_global((1.0, 0.0), 1.0, [10]), "t_span must be two"),
    (lambda: euler_global(4.0, 1.0, [10]), "t_span must be two"),
    (lambda: euler_global((0.0, 4.0, 5.0), 1.0, [10]),
     "t_span must be two"),
    (lambda: euler_global((0.0, np.inf), 1.0, [10]), "t1 must be one"),
    (lambda: euler_global((0.0, 4.0), 1.0, []), "steps must be a list"),
    (lambda: euler_global((0.0, 4.0), 1.0, [10.0]), "steps must be a list"),
    (lambda: euler_global((0.0, 4.0), 1.0, [0, 10]), "steps must be a list"),
    (lambda: euler_global((0.0, 4.0), 1.0, 10), "steps must be a list"),
    (lambda: euler_global((0.0, 4.0), 1.0, [True]), "steps must be a list"),
    (lambda: euler_global((0.0, 4.0), 1.0, [10 ** 7]),
     "steps must be a list"),
    (lambda: euler_global((0.0, 4.0), 0.0, [10]), "y0 must be the exact"),
    (lambda: ol.backward_euler("f", (0.0, 1.0), 1.0, 0.1), "f must be a function of t and y"),
    (lambda: ol.backward_euler(stiff_rate, (0.0, 1.0), 1.0, 0.1, jacobian=3.0), "jacobian must be None or"),
    (lambda: ol.backward_euler(stiff_rate, (0.0, 1.0), 1.0, 0.3), "must be a positive integer"),
    (lambda: ol.backward_euler(stiff_rate, (1.0, 0.0), 1.0, 0.1), "t0 < t1"),
    (lambda: ol.backward_euler(stiff_rate, (0.0, 1.0), 1.0, -0.1), "h must be a positive"),
    (lambda: ol.backward_euler(stiff_rate, (0.0, 1.0), "1", 0.1), "y0 must be a finite real"),
    (lambda: ol.backward_euler(lambda t, y: [y[0], y[0]], (0.0, 1.0), 1.0, 0.1),
     "f\\(t, y\\) must return 1 number\\(s\\), one for each unknown"),
    (lambda: ol.backward_euler(lambda t, y: np.log(y - 2), (0.0, 1.0), 1.0, 0.1),
     "f\\(t, y\\) is not a finite number at t = 0"),
    (lambda: ol.backward_euler(stiff_rate, (0.0, 1.0), 1.0, 0.1, jacobian=lambda t, y: [1.0, 2.0]),
     "jacobian\\(t, y\\) must return 1 x 1 finite number"),
    (lambda: ol.backward_euler(lambda t, y: -y, (0.0, 1.0), 1.0, 1.0, jacobian=lambda t, y: 3.0),  # a wrong J
     "Newton's method did not converge in the step to t = 1"),
    (lambda: ol.backward_euler(lambda t, y: y, (0.0, 1.0), 1.0, 1.0, jacobian=lambda t, y: 1.0),
     "cannot be solved"),
    (lambda: ol.backward_euler(lambda t, y: y, OVERFLOW_SPAN, 1.0, 0.5), "too large for the computer"),
    (lambda: ol.backward_euler(stiff_rate, TOO_MANY_BACKWARD[0], 1.0, TOO_MANY_BACKWARD[1]),
     "the limit of backward_euler is 100000"),
    (lambda: ol.embedded_step("f", 0.0, 1.0, 0.1), "f must be a function of t and y"),
    (lambda: ol.embedded_step(cosine_rate, None, 1.0, 0.1), "t must be one finite real number"),
    (lambda: ol.embedded_step(cosine_rate, 0.0, 1.0, 0.0), "h must be a positive"),
    (lambda: ol.embedded_step(cosine_rate, 0.0, 1.0, None), "h must be one finite real number"),
    (lambda: ol.embedded_step(cosine_rate, 0.0, 1.0, [0.1]), "h must be one finite real number"),
    (lambda: ol.embedded_step(cosine_rate, 0.0, [], 0.1), "y must be a finite real"),
    (lambda: ol.embedded_step(lambda t, y: np.inf, 0.0, 1.0, 0.1), "is not a finite number at t = 0"),
])
def test_numerical_methods_part_2_refuse_bad_input(call, words):
    with pytest.raises(ValueError, match=words):
        call()


def test_numerical_methods_part_2_leave_warnings_and_numpy_settings_alone():
    filters, settings = list(warnings.filters), np.geterr()
    ol.backward_euler(stiff_rate, (0.0, 1.0), 1.0, 0.1)
    with pytest.raises(ValueError):
        ol.backward_euler(lambda t, y: np.log(y - 2), (0.0, 1.0), 1.0, 0.1)
    ol.global_error(ol.rk4, spring_rate, (0.0, SPRING_END), [1.0, 0.0], spring_exact, SPRING_STEPS[:2])
    ol.embedded_step(cosine_rate, 0.0, 1.0, 0.1)
    assert list(warnings.filters) == filters and np.geterr() == settings
    assert ol.SETUP_NAMES == ("np", "sp", "plt", "ol", "solve_ivp", "interact", "FloatSlider", "IntSlider")
    assert (ol.NEWTON_TOLERANCE, ol.NEWTON_MAX_ITERATIONS, ol.BACKWARD_MAX_STEPS) == (1e-10, 50, 100_000)
