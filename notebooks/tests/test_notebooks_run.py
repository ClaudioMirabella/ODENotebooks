"""Every numbered notebook runs from top to bottom without errors.

Each notebook is executed in a fresh kernel, as a student would with "Run All Cells": as it is,
with right answers, with wrong answers, and with the slider functions at the ends of their
sliders. That is about a hundred executions of a few seconds each, so they are run **in
parallel** before the first test (see ``_run_everything_first``): about a minute and a half in
all, where one after the other they took more than seven minutes. Skipped where the notebook
tools are not installed (the site's build environment only tests odelib).

Run from the ode/notebooks folder:  python -m pytest tests/test_notebooks_run.py
One notebook only:                  python -m pytest tests/test_notebooks_run.py -k 16
One after the other, as before:     set NOTEBOOK_TEST_WORKERS=1 first
"""

import concurrent.futures
import copy
import json
import os
import pathlib
import re
import sys

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
for _module in ("ipykernel", "sympy", "matplotlib", "scipy", "ipywidgets"):
    pytest.importorskip(_module)

NOTEBOOKS = pathlib.Path(__file__).resolve().parent.parent
PATTERN = "[0-9][0-9]*_*.ipynb"             # the same allow-list as the student site (02_..., 02E_...)
TIMEOUT_SECONDS = 600
PATHS = sorted(NOTEBOOKS.glob(PATTERN))

FILL_IN = "    return None                 # replace None with the formula"

# The exercises of the notebooks built from plans/make_NN.py. For each notebook:
#   answers   what a student types (placeholder -> a correct answer)
#   solved    the lines the notebook must then print
#   unsolved  the lines it prints when the exercises are untouched (and nothing stops)
#   wrong     wrong answers (placeholder -> answer) and the line they must produce
NAME_14 = "14_Nonhomogeneous_Equations.ipynb"
U_DERIVATIVES = "    return -y2 * f / wronskian_of(y1, y2), y1 * f / wronskian_of(y1, y2)"
NAME_15 = "15_Forced_Vibrations.ipynb"
NAME_16 = "16_Higher_Order_BVP.ipynb"
NAME_17 = "17_Linear_Systems.ipynb"
TANK_MATRIX = "    return np.array([[-(Q + Qw), Q], [Q + Qw, -(Q + Qw)]]) / V"
CART_2KG = "my_cart_3kg = sp.Rational(2, 5) * sp.cos(t) + sp.Rational(3, 5) * sp.cos(t / sp.sqrt(6))"
CART_3KG = "my_cart_3kg = sp.Rational(2, 5) * sp.cos(t / sp.sqrt(6)) - sp.Rational(2, 5) * sp.cos(t)"
TANK_WRONG = "Exercise 3: not yet. For Q = "
NAME_18 = "18_Phase_Plane.ipynb"
HARES_JACOBIAN = "    return [[0.4 - 0.01 * y - 2 * gamma * x, -0.01 * x], [0.003 * y, 0.003 * x - 0.3]]"
HARES_TRANSPOSED = "    return [[0.4 - 0.01 * y - 2 * gamma * x, 0.003 * y], [-0.01 * x, 0.003 * x - 0.3]]"
HARES_ARRAY = ("    return np.array([[0.4 - 0.01 * y - 2 * gamma * x, -0.01 * x], "
               "[0.003 * y, 0.003 * x - 0.3]])")
HARES_WRONG = "Exercise 3: not yet. At x = "
NAME_19 = "19_Laplace_Transform.ipynb"
WRONG_FORMULA = "not yet (your formula does not give the right values"
SPRING_NOT_DIVIDED = "my_spring_2d = sp.exp(-t) * (2 * sp.cos(3 * t) + 3 * sp.sin(3 * t))"    # not 3 / omega
SPRING_WITH_PHASE = "my_spring_2d = sp.sqrt(5) * sp.exp(-t) * sp.cos(3 * t - sp.atan(sp.Rational(1, 2)))"
NAME_20 = "20_Steps_Impulses_Convolution.ipynb"
CASES_BOOK = "my_cases_2a = (t - 1) * (u(t - 1) - u(t - 2)) + u(t - 2)"            # Lebl's printed answer
CASES_TWO_RAMPS = "my_cases_2a = (t - 1) * u(t - 1) - (t - 2) * u(t - 2)"          # the same function
CASES_HEAVISIDE = ("my_cases_2a = (t - 1) * (sp.Heaviside(t - 1) - sp.Heaviside(t - 2)) "
                   "+ sp.Heaviside(t - 2)")                                         # SymPy's own step
CASES_PIECEWISE = "my_cases_2a = sp.Piecewise((0, t < 1), (t - 1, t < 2), (1, True))"
BLOW_CUT_OFF = "my_blow_2c = 2 * sp.sin(2 * t) * u(t - 1)"                         # cut off, not delayed
BLOW_DOUBLE_ANGLE = "my_blow_2c = 4 * sp.sin(t - 1) * sp.cos(t - 1) * u(t - 1)"    # the same as the answer
BLOW_TWICE = "my_blow_2c = 4 * sp.sin(2 * (t - 1)) * u(t - 1)"                     # the 1/2 of h forgotten
PRODUCT_2D = "my_convolution_2d = sp.cos(t) * sp.exp(-t)"                          # the ordinary product
NO_STEP_HINT = "not yet (before t = 1 nothing has moved: multiply by u(t - 1))"
CUT_OFF_HINT = "cut off before t = 1, not delayed: write t - 1 in place of every t)"
SWITCH_LATE_2B = "my_switch_2b = (1 - sp.cos(t - 1)) * u(t - 1.2)"       # the switch too late
BLOW_LATE_2C = "my_blow_2c = 2 * sp.sin(2 * (t - 1)) * u(t - 1.4)"
HAMMER_NUMBERS = "MY_MASS, MY_DAMPER, MY_SPRING, MY_BLOW = 1.0, 2.0, 5.0, 3.0"
CONVOLUTION_ONE_SINE = "my_convolution_2d = sp.sqrt(2) * sp.sin(t + sp.pi / 4) / 2 - sp.exp(-t) / 2"
EXERCISES = {
    NAME_20: {
        "answers": {
            "my_window_at_4 = None": "my_window_at_4 = 1",
            "my_window_at_6 = None": "my_window_at_6 = 0",
            "my_shift_rule = None": "my_shift_rule = False",
            "my_velocity = None": "my_velocity = 3",
            "my_delta_at_0 = None": "my_delta_at_0 = 1",
            "my_cases_2a = None": CASES_BOOK,
            "my_switch_2b = None": "my_switch_2b = (1 - sp.cos(t - 1)) * u(t - 1)",
            "my_blow_2c = None": "my_blow_2c = 2 * sp.sin(2 * (t - 1)) * u(t - 1)",
            "my_convolution_2d = None": "my_convolution_2d = (sp.cos(t) + sp.sin(t) - sp.exp(-t)) / 2",
        },
        "solved": ("Exercise 1a, at t = 4: correct", "Exercise 1a, at t = 6: correct",
                   "Exercise 1b: correct", "Exercise 1c: correct", "Exercise 1d: correct",
                   "Exercise 2a: correct", "Exercise 2b: correct", "Exercise 2c: correct",
                   "Exercise 2d: correct"),
        "unsolved": ("Exercise 1a, at t = 4: no answer yet", "Exercise 1b: no answer yet",
                     "Exercise 1d: no answer yet", "Exercise 2a: no answer yet",
                     "Exercise 2d: no answer yet"),
        "wrong": ({"my_window_at_4 = None": "my_window_at_4 = 0",
                   "my_window_at_6 = None": "my_window_at_6 = 1",
                   "my_shift_rule = None": "my_shift_rule = True",
                   "my_velocity = None": "my_velocity = 12",                        # P times m
                   "my_delta_at_0 = None": "my_delta_at_0 = 0",
                   "my_cases_2a = None": "my_cases_2a = (t - 1) * u(t - 1) + u(t - 2)",     # no window
                   "my_switch_2b = None": "my_switch_2b = 1 - sp.cos(t - 1)",       # the step forgotten
                   "my_blow_2c = None": BLOW_CUT_OFF,
                   "my_convolution_2d = None": PRODUCT_2D},
                  "Exercise 2d: " + WRONG_FORMULA),
    },
    NAME_19: {
        "answers": {
            "my_transform_1a = None": "my_transform_1a = 1 / (s - 3)",
            "my_edge_1a = None": "my_edge_1a = 3",
            "my_product_rule = None": "my_product_rule = False",
            "my_form = None": "my_form = 2",
            "my_rest = None": "my_rest = -2 * s + 1",
            "my_transform_2a = None": "my_transform_2a = 2 / s ** 3",
            "my_inverse_2b = None": "my_inverse_2b = 2 * t ** 2 - 2 * t + 1 - sp.exp(-2 * t)",
            "my_transform_2c = None": "my_transform_2c = 1 / (s ** 2 + 2 * s + 2)",
            "my_spring_2d = None": "my_spring_2d = sp.exp(-t) * (2 * sp.cos(3 * t) + sp.sin(3 * t))",
        },
        "solved": ("Exercise 1a, transform: correct", "Exercise 1a, smallest c: correct",
                   "Exercise 1b: correct", "Exercise 1c: correct", "Exercise 1d: correct",
                   "Exercise 2a: correct", "Exercise 2b: correct", "Exercise 2c: correct",
                   "Exercise 2d: correct"),
        "unsolved": ("Exercise 1a, transform: no answer yet", "Exercise 1a, smallest c: no answer yet",
                     "Exercise 1c: no answer yet", "Exercise 2d: no answer yet"),
        "wrong": ({"my_transform_1a = None": "my_transform_1a = 1 / (s + 3)",       # the sign of a
                   "my_edge_1a = None": "my_edge_1a = -3",
                   "my_product_rule = None": "my_product_rule = True",
                   "my_form = None": "my_form = 1",
                   "my_rest = None": "my_rest = -2 * s - 1",                        # the sign of x'(0)
                   "my_transform_2a = None": "my_transform_2a = 1 / s ** 3",        # n! forgotten
                   "my_inverse_2b = None": "my_inverse_2b = 4 * t ** 2 - 2 * t + 1 - sp.exp(-2 * t)",
                   "my_transform_2c = None": "my_transform_2c = 1 / ((s - 1) ** 2 + 1)",    # the wrong shift
                   "my_spring_2d = None": SPRING_NOT_DIVIDED},
                  "Exercise 2d: " + WRONG_FORMULA),
    },
    NAME_18: {
        "answers": {
            "my_kind_357 = None": 'my_kind_357 = "spiral source"',
            "my_diagonal_spiral = None": "my_diagonal_spiral = False",
            "my_number_of_equilibria = None": "my_number_of_equilibria = 0",
            "my_conservative_sink = None": "my_conservative_sink = False",
            "my_tr_det_d = None": "my_tr_det_d = [-3, 2]",
            "my_kind_d = None": 'my_kind_d = "sink"',
            "my_tr_det_e = None": "my_tr_det_e = [2, -15]",
            "my_kind_e = None": 'my_kind_e = "saddle"',
            "my_point_b = None": "my_point_b = (1 / 2, -1 / 4)",
            "my_jacobian_b = None": "my_jacobian_b = [[-1, 1], [1, 1]]",
            "my_kind_b = None": 'my_kind_b = "saddle"',
            "my_other_point = None": "my_other_point = (0, 1)",
            "my_kind_origin = None": 'my_kind_origin = "spiral sink"',
            "my_stability_origin = None": 'my_stability_origin = "asymptotically stable"',
            "my_kind_other = None": 'my_kind_other = "saddle"',
            "my_stability_other = None": 'my_stability_other = "unstable"',
            "my_speed_over_the_top = None": "my_speed_over_the_top = 2",
            FILL_IN: HARES_JACOBIAN,
        },
        "solved": ("Exercise 1a: correct", "Exercise 1b: correct", "Exercise 1c: correct",
                   "Exercise 1d: correct",
                   "Exercise 2a (d), tr and det: correct", "Exercise 2a (d), picture: correct",
                   "Exercise 2a (e), tr and det: correct", "Exercise 2a (e), picture: correct",
                   "Exercise 2b, equilibrium: correct", "Exercise 2b, J: correct",
                   "Exercise 2b, picture: correct",
                   "Exercise 2c, the other equilibrium: correct", "Exercise 2c, (0, 0), picture: correct",
                   "Exercise 2c, (0, 0), stability: correct", "Exercise 2c, the other, picture: correct",
                   "Exercise 2c, the other, stability: correct", "Exercise 2d: correct",
                   "Exercise 3: correct at all 15 points",
                   "gamma = 0.001: (0, 0) saddle, eigenvalues -0.3 and 0.4; (400, 0) saddle, eigenvalues "
                   "-0.4 and 0.9; (100, 30) spiral sink, eigenvalues -0.05 +- 0.296i",
                   "gamma = 0.01: (0, 0) saddle, eigenvalues -0.3 and 0.4; (40, 0) sink, eigenvalues "
                   "-0.4 and -0.18; (100, -60) is outside the quarter",
                   "gamma = 0.001: at t = 200, 100.0 hares and 30.0 foxes"),
        "unsolved": ("Exercise 1a: no answer yet", "Exercise 2a (d), picture: no answer yet",
                     "Exercise 2c, the other, stability: no answer yet", "Exercise 2d: no answer yet",
                     "Exercise 3: complete hares_jacobian first"),
        "wrong": ({"my_kind_357 = None": 'my_kind_357 = "spiral sink"',
                   "my_diagonal_spiral = None": "my_diagonal_spiral = True",
                   "my_number_of_equilibria = None": "my_number_of_equilibria = 1",
                   "my_conservative_sink = None": "my_conservative_sink = True",
                   "my_tr_det_d = None": "my_tr_det_d = [2, -3]",                 # det and tr exchanged
                   "my_kind_d = None": 'my_kind_d = "source"',                    # the sign of tr A
                   "my_tr_det_e = None": "my_tr_det_e = [2, 15]",
                   "my_kind_e = None": 'my_kind_e = "center"',
                   "my_point_b = None": "my_point_b = (-1 / 4, 1 / 2)",           # y and x exchanged
                   "my_jacobian_b = None": "my_jacobian_b = [[1, 1], [1, 1]]",    # 2(x - 1) taken as 2x
                   "my_kind_b = None": 'my_kind_b = "sink"',
                   "my_other_point = None": "my_other_point = (1, 0)",
                   "my_kind_origin = None": 'my_kind_origin = "center"',          # not a spiral sink
                   "my_stability_origin = None": 'my_stability_origin = "stable"',
                   "my_kind_other = None": 'my_kind_other = "source"',
                   "my_stability_other = None": 'my_stability_other = "stable"',
                   "my_speed_over_the_top = None": "my_speed_over_the_top = 4",   # 2 rad/s, squared
                   FILL_IN: HARES_TRANSPOSED},
                  "Exercise 3: not yet. At x = 50 hares, y = 20 foxes, gamma = 0.001: row 1, column 2"),
    },
    NAME_17: {
        "answers": {
            "my_Av = None": "my_Av = [3, -3, 9]",
            "my_equations = None": "my_equations = 4",
            "my_triangular = None": "my_triangular = [-3, 2, 4]",
            "my_one_cart_moves = None": "my_one_cart_moves = False",
            "my_eigenvalues = None": "my_eigenvalues = [2, 5]",
            "my_v_for_low = None": "my_v_for_low = [-2, 2]",                 # other forms of the eigenvectors
            "my_v_for_high = None": "my_v_for_high = [3, 6]",
            "my_complex_eigenvalue = None": "my_complex_eigenvalue = 1 + 2j",
            "my_complex_vector = None": "my_complex_vector = [1, -1j]",      # -i times (i, 1)
            "my_x1 = None": "my_x1 = sp.exp(t) * sp.cos(2 * t)",
            "my_x2 = None": "my_x2 = sp.exp(t) * sp.sin(2 * t)",
            "my_repeated = None": "my_repeated = 2",
            "my_v = None": "my_v = [1, 1]",
            "my_w = None": "my_w = [0, -1 / 3]",
            "my_cart_3kg = None": CART_3KG,
            FILL_IN: TANK_MATRIX,
        },
        "solved": ("Exercise 1a: correct", "Exercise 1b: correct", "Exercise 1c: correct",
                   "Exercise 1d: correct", "Exercise 2a, eigenvalues: correct",
                   "Exercise 2a, eigenvector for the smaller: correct",
                   "Exercise 2a, eigenvector for the larger: correct", "Exercise 2b, eigenvalue: correct",
                   "Exercise 2b, eigenvector: correct", "Exercise 2b, x1: correct",
                   "Exercise 2b, x2: correct", "Exercise 2c, eigenvalue: correct", "Exercise 2c, v: correct",
                   "Exercise 2c, w: correct", "Exercise 2d: correct",
                   "Exercise 3: correct on all 3 sets of pipes",
                   "The open tanks: eigenvalues -0.06 and -0.02 per second, eigenvectors [1.0, -2.0] and "
                   "[1.0, 2.0]; tank 2 holds at most 7.70 g, at t = 27.5 s"),
        "unsolved": ("Exercise 1a: no answer yet", "Exercise 2b, x1: no answer yet",
                     "Exercise 2c, w: no answer yet", "Exercise 2d: no answer yet",
                     "Exercise 3: complete tank_matrix first"),
        "wrong": ({"my_Av = None": "my_Av = [1, -1, 3]",                             # v itself
                   "my_equations = None": "my_equations = 2",                        # the number of equations
                   "my_triangular = None": "my_triangular = [2, -3, 4]",             # not in order
                   "my_one_cart_moves = None": "my_one_cart_moves = True",
                   "my_eigenvalues = None": "my_eigenvalues = [5, 2]",
                   "my_v_for_low = None": "my_v_for_low = [1, 2]",                   # for the other one
                   "my_v_for_high = None": "my_v_for_high = [0, 0]",                 # the zero vector
                   "my_complex_eigenvalue = None": "my_complex_eigenvalue = 1 - 2j",
                   "my_complex_vector = None": "my_complex_vector = [-1j, 1]",       # the conjugate
                   "my_x1 = None": "my_x1 = sp.exp(t) * sp.sin(2 * t)",              # x1 and x2 exchanged
                   "my_x2 = None": "my_x2 = sp.exp(t) * sp.cos(2 * t)",
                   "my_repeated = None": "my_repeated = -2",
                   "my_v = None": "my_v = [1, -1]",
                   "my_w = None": "my_w = [1, -1]",
                   "my_cart_3kg = None": CART_2KG,                                   # the 2 kg cart
                   FILL_IN: "    return np.array([[-(Q + Qw), Q + Qw], [Q, -(Q + Qw)]]) / V"},   # transposed
                  "Exercise 3: not yet. For Q = 1 L/s, Qw = 3 L/s, V = 100 L"),
    },
    NAME_16: {
        "answers": {
            "my_value_at_1 = None": "my_value_at_1 = 0",
            "my_constants = None": "my_constants = 5",
            "my_polynomials = None": "my_polynomials = 4",
            "my_numbers_of_solutions = None": "my_numbers_of_solutions = [np.inf, 0, 1]",
            "my_roots = None": "my_roots = [0, 0, 2, 3]",
            "my_coefficients = None": "my_coefficients = [1, -3, 4, -12]",
            "my_x = None": "my_x = -sp.sin(t)",
            "my_y = None": "my_y = sp.sqrt(1 - x ** 2)",
            FILL_IN: "    return 5 * w0 * L ** 4 / (384 * EI)",
        },
        "solved": ("Exercise 1, y(1): correct", "Exercise 1, constants: correct", "Exercise 1, polynomials: correct",
                   "Exercise 1, numbers of solutions: correct", "Exercise 2a: correct", "Exercise 2b: correct",
                   "Exercise 2c: correct", "Exercise 2d: correct", "Exercise 3: correct on all 3 beams",
                   "The steel beam goes down 9.6 mm at mid-span; the limit L / 300 is 20 mm: acceptable"),
        "unsolved": ("Exercise 1, y(1): no answer yet", "Exercise 1, numbers of solutions: no answer yet",
                     "Exercise 2a: no answer yet", "Exercise 2d: no answer yet", "Exercise 3: complete midspan_deflection first"),
        "wrong": ({"my_value_at_1 = None": "my_value_at_1 = 1",
                   "my_constants = None": "my_constants = 2",                           # two different roots
                   "my_polynomials = None": "my_polynomials = 1",
                   "my_numbers_of_solutions = None": "my_numbers_of_solutions = [1, 1, 1]",   # two conditions, one solution
                   "my_roots = None": "my_roots = [0, 2, 3]",                           # the double root counted once
                   "my_coefficients = None": "my_coefficients = [1, -3, -4, 12]",       # r^2 - 4 in place of r^2 + 4
                   "my_x = None": "my_x = sp.sin(t)",                                   # x'(pi) = -1
                   "my_y = None": "my_y = x",                                           # the solution that was given
                   FILL_IN: "    return w0 * L ** 4 / (8 * EI)"},                       # the cantilever
                  "Exercise 3: not yet. For w0 = 10000 N/m, L = 6 m"),
    },
    NAME_15: {
        "answers": {
            "my_omega = None": "my_omega = 2",
            "my_growth_per_second = None": "my_growth_per_second = 0.25",
            "my_size_at_100 = None": "my_size_at_100 = 25",
            "my_below_omega_0 = None": "my_below_omega_0 = True",
            "my_higher_peak = None": "my_higher_peak = False",
            "my_resonance = None": "my_resonance = 0.984",
            "my_amplitude = None": "my_amplitude = 2.016",
            "my_natural = None": "my_natural = 0.316",
            "my_displacement = None": "my_displacement = 2.020",
            "my_collapses = None": "my_collapses = True",
            "my_charge = None": "my_charge = 2 + 4 * sp.exp(-3 * t) - 6 * sp.exp(-2 * t)",
            FILL_IN: "    return F0 / np.sqrt((k - m * omega ** 2) ** 2 + (c * omega) ** 2)",
        },
        "solved": ("Exercise 1, omega: correct", "Exercise 1, growth: correct", "Exercise 1, size at t = 100: correct",
                   "Exercise 1, below omega_0: correct", "Exercise 1, higher peak: correct",
                   "Exercise 2a, omega: correct", "Exercise 2a, amplitude: correct", "Exercise 2b, natural frequency: correct",
                   "Exercise 2b, displacement: correct", "Exercise 2b, collapse: correct", "Exercise 2c: correct",
                   "Exercise 3: correct on all 5 cases"),
        "unsolved": ("Exercise 1, omega: no answer yet", "Exercise 2a, amplitude: no answer yet", "Exercise 2c: no answer yet",
                     "Exercise 3: complete amplitude first"),
        "wrong": ({"my_omega = None": "my_omega = 4",                                   # k / m without the square root
                   "my_growth_per_second = None": "my_growth_per_second = 0.5",         # F0 / (m omega)
                   "my_size_at_100 = None": "my_size_at_100 = 50",
                   "my_below_omega_0 = None": "my_below_omega_0 = False",
                   "my_higher_peak = None": "my_higher_peak = True",
                   "my_resonance = None": "my_resonance = 1.0",                         # omega_0, not the practical resonance
                   "my_amplitude = None": "my_amplitude = 2.0",                         # the amplitude at omega_0
                   "my_natural = None": "my_natural = 0.1",                             # k / m without the square root
                   "my_displacement = None": "my_displacement = 1.01",                  # half: the size of one cosine
                   "my_collapses = None": "my_collapses = False",
                   "my_charge = None": "my_charge = 2 - 2 * sp.exp(-2 * t)",             # q(0) = 0 only
                   FILL_IN: "    return F0 / np.sqrt((k / m - omega ** 2) ** 2 + (c * omega / m) ** 2)"},   # the factor m forgotten
                  "Exercise 3: not yet. For omega = 0.984, m = 4, c = 1, k = 4, F0 = 2"),
    },
    NAME_14: {
        "answers": {
            "my_powers = None": "my_powers = [0, 2, 1, 1, 1, 0]",
            "my_guessing_works = None": "my_guessing_works = False",
            "my_a = None": "my_a = -sp.exp(2 * x) / 4",
            "my_b = None": "my_b = x ** 2 * sp.exp(2 * x) / 2",
            "my_c = None": "my_c = (-16 * sp.sin(3 * x) + 6 * sp.cos(3 * x)) / 73",
            "my_d = None": "my_d = x ** 2 - 4 * x + 6 + sp.exp(-x) * (x - 5)",
            FILL_IN: U_DERIVATIVES,
        },
        "solved": ("Exercise 1, powers: correct", "Exercise 1, tan x: correct", "Exercise 2a: correct",
                   "Exercise 2b: correct", "Exercise 2c: correct", "Exercise 2d: correct",
                   "Exercise 3, y'' + y = tan x (Lebl): correct", "Exercise 3, y'' - y = exp(x) (Lebl, Exercise 2.5.9): correct",
                   "Exercise 3, x y'' + 2y' + x y = 1 (Adkins-Davidson 5.6, Example 4): correct", "Exercise 3: correct"),
        "unsolved": ("Exercise 1, powers: no answer yet", "Exercise 2a: no answer yet", "Exercise 2d: no answer yet",
                     "Exercise 3: complete u_derivatives first"),
        "wrong": ({"my_powers = None": "my_powers = [0, 1, 1, 1, 1, 1]",                 # the double root and the trap
                   "my_guessing_works = None": "my_guessing_works = True",
                   "my_a = None": "my_a = sp.exp(2 * x) / 4",                           # a lost sign
                   "my_b = None": "my_b = x * sp.exp(2 * x)",                           # multiplied by x once only
                   "my_c = None": "my_c = -2 * sp.sin(3 * x) / 8",                      # the cosine forgotten
                   "my_d = None": "my_d = x ** 2 - 4 * x + 6",                          # a particular solution only
                   FILL_IN: "    return y2 * f / wronskian_of(y1, y2), -y1 * f / wronskian_of(y1, y2)"},   # both signs wrong
                  "Exercise 2d: not yet (the equation is satisfied, but not the condition on y(0), y'(0))"),
    },
    "07_Mass_Spring_Damper.ipynb": {
        "answers": {
            "my_omega_0 = None": "my_omega_0 = 2",
            "my_frequency_factor = None": "my_frequency_factor = 0.5",
            "my_period_changes = None": "my_period_changes = False",
            "my_p = None": "my_p = 1.25",
            "my_omega_1 = None": "my_omega_1 = 2.165",
            "my_amplitude = None": "my_amplitude = 0.902",
            "my_phase = None": "my_phase = 0.983",
            "my_critical = None": "my_critical = 20",
            "my_time_between_zeros = None": "my_time_between_zeros = 1.451",
            FILL_IN: ("    return ('undamped' if c == 0 else 'critically damped' if np.isclose(c * c, 4 * m * k) "
                      "else 'underdamped' if c * c < 4 * m * k else 'overdamped')"),
        },
        "solved": ("Exercise 1, omega_0: correct", "Exercise 1, factor: correct", "Exercise 1, period: correct",
                   "Exercise 2a, p: correct", "Exercise 2a, omega_1: correct", "Exercise 2a, amplitude: correct",
                   "Exercise 2a, phase: correct", "Exercise 2b, critical damping: correct",
                   "Exercise 2b, time between zeros: correct", "Exercise 3: correct on all 8 cases"),
        "unsolved": ("Exercise 1, omega_0: no answer yet", "Exercise 2a, phase: no answer yet",
                     "Exercise 2b, time between zeros: no answer yet", "Exercise 3: complete kind_of_motion first"),
        "wrong": ({"my_omega_0 = None": "my_omega_0 = 4",                               # the square root forgotten
                   "my_frequency_factor = None": "my_frequency_factor = 0.25",
                   "my_period_changes = None": "my_period_changes = True",
                   "my_omega_1 = None": "my_omega_1 = 2.5",                             # omega_0 instead of omega_1
                   "my_amplitude = None": "my_amplitude = 0.707",
                   "my_phase = None": "my_phase = 0.785",
                   FILL_IN: "    return 'underdamped' if c * c < 4 * m * k else 'overdamped'"},
                  "Exercise 1, omega_0: not yet"),
    },
    "06_Characteristic_Equation.ipynb": {
        "answers": {
            "my_oscillate = None": "my_oscillate = [True, True, False, False, False, True]",
            "my_tend_to_zero = None": "my_tend_to_zero = [False, True, True, False, True, False]",
            "my_a = None": "my_a = 2 * sp.exp(5 * x) + 3 * sp.exp(-2 * x)",
            "my_b = None": "my_b = sp.sqrt(2) * sp.exp(-2 * x) * sp.cos(3 * x + sp.pi / 4)",   # the one-cosine form
            "my_c = None": "my_c = (1 + 4 * x) * sp.exp(-4 * x)",
            FILL_IN: "    d = np.sqrt(complex(b * b - 4 * a * c)); return (-b + d) / (2 * a), (-b - d) / (2 * a)",
        },
        "solved": ("Exercise 1, oscillate: correct", "Exercise 1, tend to zero: correct", "Exercise 2a: correct",
                   "Exercise 2b: correct", "Exercise 2c: correct", "Exercise 3: correct"),
        "unsolved": ("Exercise 1, oscillate: no answer yet", "Exercise 2a: no answer yet", "Exercise 2c: no answer yet",
                     "Exercise 3: complete roots first"),
        "wrong": ({"my_oscillate = None": "my_oscillate = [True, True, True, False, False, True]",
                   "my_a = None": "my_a = 2 * sp.exp(5 * x) + 3 * sp.exp(2 * x)",          # a lost minus sign
                   "my_b = None": "my_b = sp.exp(-2 * x) * (sp.cos(3 * x) + sp.sin(3 * x))",
                   "my_c = None": "my_c = sp.exp(-4 * x)"},
                  "Exercise 1, oscillate: not yet"),
    },
    "05_Wronskian_Abel.ipynb": {
        "answers": {
            "my_independent = None": "my_independent = [True, False, True, False]",
            "my_b_and_c = None": "my_b_and_c = [4, -21]",
            "my_K = None": "my_K = -10",
            "my_wronskian = None": "my_wronskian = x ** -2",
            "my_constants = None": "my_constants = [1, -1]",
            FILL_IN: "    return sp.simplify(first * sp.diff(second, x) - sp.diff(first, x) * second)",
        },
        "solved": ("Exercise 1: correct", "Exercise 2a, b and c: correct", "Exercise 2a, K: correct",
                   "Exercise 2b: correct", "Exercise 2c: correct", "Exercise 3: correct"),
        "unsolved": ("Exercise 1: no answer yet", "Exercise 2a, K: no answer yet", "Exercise 2b: no answer yet",
                     "Exercise 3: complete wronskian first"),
        "wrong": ({"my_independent = None": "my_independent = [True, True, True, True]",
                   "my_b_and_c = None": "my_b_and_c = [-4, -21]", "my_K = None": "my_K = 10",   # the lost minus signs
                   "my_wronskian = None": "my_wronskian = 1 / x", "my_constants = None": "my_constants = [1, 1]"},
                  "Exercise 1: not yet"),
    },
    "04_Euler_Method.ipynb": {
        "answers": {
            "my_value = None": "my_value = 8",
            "my_value_is_below = None": "my_value_is_below = True",
            "my_second_row = None": "my_second_row = 2.209",
            "my_error_estimate = None": "my_error_estimate = 0.024",
            "my_thresholds = None": "my_thresholds = [0.2, 0.4]",
            FILL_IN: "    return y + h * f(x, y)",
        },
        "solved": ("Exercise 1, value: correct", "Exercise 1, below: correct", "Exercise 2a: correct",
                   "Exercise 2b: correct", "Exercise 2c: correct", "Exercise 3: correct"),
        "unsolved": ("Exercise 1, value: no answer yet", "Exercise 2a: no answer yet", "Exercise 2c: no answer yet",
                     "Exercise 3: complete euler_step first"),
        "wrong": ({"my_value = None": "my_value = 20.1", "my_value_is_below = None": "my_value_is_below = False",
                   "my_second_row = None": "my_second_row = 1.926", "my_error_estimate = None": "my_error_estimate = 0.5",
                   "my_thresholds = None": "my_thresholds = [0.4, 0.2]"},
                  "Exercise 1, value: not yet"),
    },
    "03_Logistic_Equation.ipynb": {
        "answers": {
            "my_limits = None": "my_limits = [5, 5, 0]",
            "my_equilibria = None": "my_equilibria = [1.55, 6.45]",
            "my_critical_harvest = None": "my_critical_harvest = 1.6",
            "my_fastest_time = None": "my_fastest_time = 2.77",
            "my_fastest_rate = None": "my_fastest_rate = 0.625",
            FILL_IN: "    return k * x * (M - x) - h",
        },
        "solved": ("Exercise 1: correct", "Exercise 2a, equilibria: correct", "Exercise 2a, critical harvest: correct",
                   "Exercise 2b, time: correct", "Exercise 2b, rate: correct", "Exercise 3: correct"),
        "unsolved": ("Exercise 1: no answer yet", "Exercise 2a, equilibria: no answer yet",
                     "Exercise 2b, rate: no answer yet", "Exercise 3: complete harvested first"),
        "wrong": ({"my_limits = None": "my_limits = [5, 5, 5]",            # x = 0 is an equilibrium
                   "my_equilibria = None": "my_equilibria = [6.45, 1.55]", # wrong order
                   "my_critical_harvest = None": "my_critical_harvest = 2", "my_fastest_time = None": "my_fastest_time = 2.5",
                   "my_fastest_rate = None": "my_fastest_rate = 0.5"},
                  "Exercise 1: not yet"),
    },
    "02_Radioactive_Decay.ipynb": {
        "answers": {
            "my_fractions = None": "my_fractions = [0.125, 0.37]",
            "my_age_third = None": "my_age_third = 9082",
            "my_age_tenth = None": "my_age_tenth = 19000",                  # rounded: accepted
            "my_half_life = None": "my_half_life = 31",
            FILL_IN: "    return -half_life * np.log(fraction) / np.log(2)",
        },
        "solved": ("Exercise 1: correct", "Exercise 2a: correct", "Exercise 2b: correct", "Exercise 2c: correct",
                   "Exercise 3: correct"),
        "unsolved": ("Exercise 1: no answer yet", "Exercise 2a: no answer yet", "Exercise 2c: no answer yet",
                     "Exercise 3: complete age first"),
        "wrong": ({"my_fractions = None": "my_fractions = [0.33, 0.5]",     # "three half-lives: one third"
                   "my_age_third = None": "my_age_third = 17190", "my_age_tenth = None": "my_age_tenth = 57300",
                   "my_half_life = None": "my_half_life = 25"},
                  "Exercise 1: not yet"),
    },
    "01_Exponential_Growth.ipynb": {
        "answers": {
            "my_doubling_years = None": "my_doubling_years = 23",         # a whole number: accepted
            "my_factor = None": "my_factor = 8",
            "my_rate = None": "my_rate = 0.018",
            "my_population = None": "my_population = 144",
            "my_time_of_maximum = None": "my_time_of_maximum = 20",
            FILL_IN: "    return np.log(P2 / P1) / (t2 - t1)",
        },
        "solved": ("Exercise 1, doubling time: correct", "Exercise 1, factor: correct", "Exercise 2a, rate: correct",
                   "Exercise 2a, population: correct", "Exercise 2b: correct", "Exercise 3: correct"),
        "unsolved": ("Exercise 1, doubling time: no answer yet", "Exercise 2a, rate: no answer yet",
                     "Exercise 2b: no answer yet", "Exercise 3: complete growth_rate first"),
        "wrong": ({"my_doubling_years = None": "my_doubling_years = 33",   # 100 / 3: a frequent slip
                   "my_factor = None": "my_factor = 6", "my_rate = None": "my_rate = 0.02",
                   "my_population = None": "my_population = 140", "my_time_of_maximum = None": "my_time_of_maximum = 2"},
                  "Exercise 1, doubling time: not yet"),
    },
    "08_First_Steps_in_Python.ipynb": {
        "answers": {
            "my_x = None": "my_x = 1",
            "my_y = None": "my_y = 2",
            "my_derivative = None": "my_derivative = x**2 * (3 - 2*x) * sp.exp(-2*x)",
            FILL_IN: "    return x * np.exp(-2 * x)",
        },
        "solved": ("Exercise 2a: correct", "Exercise 2b: correct", "Exercise 3: correct"),
        "unsolved": ("Exercise 2a: no answer yet", "Exercise 2b: no answer yet",
                     "Exercise 3: complete g(x) first"),
        "wrong": ({"my_x = None": "my_x = 2", "my_y = None": "my_y = 1"}, "Exercise 2a: not yet"),
    },
    "09_Slope_Fields.ipynb": {
        "answers": {
            "my_A = None": "my_A = 3",
            "my_B = None": "my_B = 1",
            "my_C = None": "my_C = 2",
            "my_orders = None": "my_orders = [2, 1, 3, 1]",
            "my_linear = None": "my_linear = [True, False, True, False]",
            "my_C1 = None": "my_C1 = 20/3",
            "my_C2 = None": "my_C2 = 3.333",                      # rounded: accepted
            FILL_IN: "    return np.exp(x - y)",
            "my_solution = None": "my_solution = x",
        },
        "solved": ("Exercise 1: correct", "Exercise 2a, orders: correct", "Exercise 2a, linear: correct",
                   "Exercise 2b: correct", "Exercise 3: correct", "Exercise 3, your solution: correct"),
        "unsolved": ("Exercise 1: no answer yet", "Exercise 2a, orders: no answer yet",
                     "Exercise 2b: no answer yet", "Exercise 3: complete f(x, y) first",
                     "Exercise 3, your solution: no answer yet"),
        "wrong": ({"my_A = None": "my_A = 1", "my_B = None": "my_B = 2", "my_C = None": "my_C = 3",
                   "my_solution = None": "my_solution = 2 * x"},
                  "Exercise 1: not yet"),
    },
    "10_Separable_Equations.ipynb": {
        "answers": {
            "my_x = None": "my_x = sp.exp(t**3) + 1",
            "my_y = None": "my_y = 1 / (1 - sp.log(x))",
            "my_right_end = None": "my_right_end = 2.718",        # rounded: accepted
            "my_left_end = None": "my_left_end = -1",
            FILL_IN: "    return (tank_radius / hole_radius) ** 2 * np.sqrt(2 * h0 / G)",
        },
        "solved": ("Exercise 2a: correct", "Exercise 2b, solution: correct", "Exercise 2b, right end: correct",
                   "Exercise 2c: correct", "Exercise 3: correct"),
        "unsolved": ("Exercise 2a: no answer yet", "Exercise 2b, solution: no answer yet",
                     "Exercise 2b, right end: no answer yet", "Exercise 2c: no answer yet",
                     "Exercise 3: complete drain_time first"),
        "wrong": ({"my_x = None": "my_x = sp.exp(x ** 3) + 1",        # right formula, wrong letter: a hint
                   "my_y = None": "my_y = 1 / (1 - sp.log(t))",
                   "my_right_end = None": "my_right_end = 1", "my_left_end = None": "my_left_end = 0"},
                  "Exercise 2a: not yet (write the formula with the letter t only"),
    },
    "11_Linear_First_Order.ipynb": {
        "answers": {
            "my_guarantees = None": "my_guarantees = [True, True, False, False]",
            "my_y = None": "my_y = 2 * sp.exp(sp.cos(2 * x) + 1) + 1",
            "my_P5 = None": "my_P5 = 6.31e6",                     # three digits: accepted
            "my_interval = None": "my_interval = [-1, 3]",
            FILL_IN: "    return 2 * (1 - np.exp(-2 * x))",
        },
        "solved": ("Exercise 1: correct", "Exercise 2a: correct", "Exercise 2b: correct",
                   "Exercise 2c: correct", "Exercise 3: correct"),
        "unsolved": ("Exercise 1: no answer yet", "Exercise 2a: no answer yet", "Exercise 2b: no answer yet",
                     "Exercise 2c: no answer yet", "Exercise 3: complete y_exact first"),
        "wrong": ({"my_guarantees = None": "my_guarantees = [True, True, True, True]",
                   "my_P5 = None": "my_P5 = 6.9e6", "my_interval = None": "my_interval = [0, 3]"},
                  "Exercise 1: not yet"),
    },
    "12_Exact_Equations.ipynb": {
        "answers": {
            "my_exact = None": "my_exact = [True, False, False, True]",
            "my_F = None": "my_F = x**3 + 3*x*y + y**3 + 5",      # shifted by a constant: accepted
            "my_mu = None": "my_mu = 2 * x",                      # a constant multiple: accepted
            "my_y = None": "my_y = 2 / (3 * x - 2)",
            FILL_IN: "    return sp.simplify(sp.diff(M, y) - sp.diff(N, x)) == 0",
        },
        "solved": ("Exercise 1: correct", "Exercise 2a: correct", "Exercise 2b: correct",
                   "Exercise 2c: correct", "Exercise 3: correct"),
        "unsolved": ("Exercise 1: no answer yet", "Exercise 2a: no answer yet", "Exercise 2b: no answer yet",
                     "Exercise 2c: no answer yet", "Exercise 3: complete is_exact first"),
        "wrong": ({"my_exact = None": "my_exact = [True, True, True, True]",
                   "my_F = None": "my_F = x**3 + y**3", "my_mu = None": "my_mu = y",
                   "my_y = None": "my_y = 2 / x"},
                  "Exercise 2b: not yet"),
    },
    "13_First_Order_Models.ipynb": {
        "answers": {
            "my_fractions = None": "my_fractions = [0.37, 0.05]",         # two decimals: accepted
            "my_rc_factors = None": "my_rc_factors = [2, 1]",
            "my_rl_factors = None": "my_rl_factors = [0.5, 0.5]",
            "my_minutes = None": "my_minutes = 3.73",
            "my_concentration = None": "my_concentration = 1.73",
            "my_room = None": "my_room = 10",
            "my_parachute_speed = None": "my_parachute_speed = 5.1",
            FILL_IN: "    return PUMP_IN * concentration_in(t) - PUMP_OUT * x / tank_volume(t)",
        },
        "solved": ("Exercise 1, fractions: correct", "Exercise 1, RC: correct", "Exercise 1, RL: correct",
                   "Exercise 2a: correct", "Exercise 2b: correct", "Exercise 2c: correct",
                   "Exercise 2d: correct", "Exercise 3: correct"),
        "unsolved": ("Exercise 1, fractions: no answer yet", "Exercise 1, RC: no answer yet",
                     "Exercise 2a: no answer yet", "Exercise 2d: no answer yet",
                     "Exercise 3: complete rate_of_change first"),
        "wrong": ({"my_fractions = None": "my_fractions = [0.63, 0.95]",       # closed, not left
                   "my_rc_factors = None": "my_rc_factors = [2, 2]",
                   "my_rl_factors = None": "my_rl_factors = [2, 0.5]",
                   "my_minutes = None": "my_minutes = 6.67", "my_room = None": "my_room = 20",
                   "my_parachute_speed = None": "my_parachute_speed = 56.0"},
                  "Exercise 1, fractions: not yet"),
    },
}


# --- executing the notebooks, in parallel ------------------------------------------------
MAX_WORKERS = 8                             # kernels running at the same time (each is a Python process)
WORKERS = int(os.environ.get("NOTEBOOK_TEST_WORKERS", min(MAX_WORKERS, os.cpu_count() or 1)))
_DONE = {}                                  # what has been executed: key of a notebook -> ("ok", notebook) or ("error", exception)
_WANTED = None                              # while the tests are only asked which notebooks they need: the list of them


class _Asked(BaseException):
    """Raised by execute() while the notebooks are being collected (not an Exception: no test catches it)."""


def _key(nb) -> str:
    return json.dumps([(cell.cell_type, cell.source) for cell in nb.cells])


def _execute_now(nb):
    client = nbclient.NotebookClient(nb, timeout=TIMEOUT_SECONDS, kernel_name="python3",
                                     resources={"metadata": {"path": str(NOTEBOOKS)}})
    client.execute()                         # raises CellExecutionError on the first failing cell
    return nb


def _outcome(nb):
    try:
        return "ok", _execute_now(nb)
    except Exception as problem:             # kept, and raised again in the test that asked for this notebook
        return "error", problem


def execute(nb):
    """The notebook after "Run All Cells"; raises what the first failing cell raised."""
    if _WANTED is not None:
        _WANTED.append(copy.deepcopy(nb))
        raise _Asked
    key = _key(nb)
    if key not in _DONE:                     # not prepared in advance: run it now
        _DONE[key] = _outcome(copy.deepcopy(nb))
    state, result = _DONE[key]
    if state == "error":
        raise result
    return copy.deepcopy(result)


@pytest.fixture(scope="session", autouse=True)
def _run_everything_first(request):
    """Execute in parallel, once, every notebook that the selected tests of this file will ask for.

    Each test builds a notebook and calls execute() on it. Here every selected test is called
    once in a mode where execute() only takes note of the notebook and stops the test; then
    the notebooks are executed together, WORKERS at a time, and the real tests find them ready.
    A test that takes anything but its own parameters is left alone and runs as usual.
    """
    global _WANTED
    if WORKERS <= 1:
        yield
        return
    _WANTED = []
    try:
        for item in request.session.items:
            if getattr(item, "module", None) is not sys.modules[__name__]:
                continue
            arguments = dict(getattr(getattr(item, "callspec", None), "params", {}))
            if set(item.fixturenames) - set(arguments) - {"_run_everything_first", "request"}:
                continue
            try:
                item.obj(**arguments)
            except _Asked:
                pass
            except Exception:                # it fails before executing anything: the real run will say why
                pass
        unique = {}
        for nb in _WANTED:
            unique.setdefault(_key(nb), nb)
    finally:
        _WANTED = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:   # the work is done by the kernels
        for key, outcome in zip(unique, pool.map(_outcome, unique.values())):
            _DONE[key] = outcome
    yield


def printed(nb) -> str:
    return "".join(output.get("text", "") for cell in nb.cells if cell.cell_type == "code"
                   for output in cell.outputs if output.output_type == "stream")


def with_answers(name: str, answers: dict):
    """The notebook with each placeholder (found exactly once) replaced by an answer."""
    nb = nbformat.read(NOTEBOOKS / name, as_version=4)
    for placeholder, answer in answers.items():
        cells = [cell for cell in nb.cells if cell.cell_type == "code" and placeholder in cell.source]
        assert len(cells) == 1, f"{name}: placeholder not found exactly once: {placeholder!r}"
        cells[0].source = cells[0].source.replace(placeholder, answer)
    return nb


def test_there_are_notebooks():
    assert PATHS, f"no notebook matches {PATTERN} in {NOTEBOOKS}"
    assert set(EXERCISES) <= {path.name for path in PATHS}


@pytest.mark.parametrize("path", PATHS, ids=lambda p: p.stem)
def test_notebook_runs_without_errors(path):
    text = printed(execute(nbformat.read(path, as_version=4)))
    for line in EXERCISES.get(path.name, {}).get("unsolved", ()):
        assert line in text


@pytest.mark.parametrize("name", sorted(EXERCISES))
def test_exercises_accept_the_right_answers(name):
    text = printed(execute(with_answers(name, EXERCISES[name]["answers"])))
    for line in EXERCISES[name]["solved"]:
        assert line in text


@pytest.mark.parametrize("name", sorted(EXERCISES))
def test_exercises_reject_wrong_answers(name):
    answers, line = EXERCISES[name]["wrong"]
    text = printed(execute(with_answers(name, answers)))
    assert line in text
    assert "correct" not in text.replace("not yet", "")


def test_notebook_14_accepts_every_particular_solution():
    # a particular solution plus any solution of the homogeneous equation is a particular solution
    answers = dict(EXERCISES[NAME_14]["answers"])
    answers["my_a = None"] = "my_a = -sp.exp(2 * x) / 4 + 3 * sp.exp(3 * x) - sp.exp(-2 * x)"
    answers["my_b = None"] = "my_b = (x ** 2 / 2 + 7 * x - 1) * sp.exp(2 * x)"
    answers["my_c = None"] = ("my_c = (-16 * sp.sin(3 * x) + 6 * sp.cos(3 * x)) / 73 "
                              "+ sp.exp(x / 2) * sp.cos(sp.sqrt(3) * x / 2)")
    text = printed(execute(with_answers(NAME_14, answers)))
    for line in ("Exercise 2a: correct", "Exercise 2b: correct", "Exercise 2c: correct"):
        assert line in text


@pytest.mark.parametrize("formula, message", [
    ("    return y1 * f / wronskian_of(y1, y2), -y2 * f / wronskian_of(y1, y2)", "Exercise 3: not yet (check the signs"),
    ("    return -y2 * f / wronskian_of(y1, y2)", "Exercise 3: not yet (return the two formulas"),
    ("    return -y2 * f, y1 * f", "Exercise 3: not yet (check the signs"),            # the Wronskian forgotten
])
def test_notebook_14_wrong_formulas_of_variation_of_parameters(formula, message):
    text = printed(execute(with_answers(NAME_14, {FILL_IN: formula})))
    assert message in text and "Exercise 3: correct" not in text


@pytest.mark.parametrize("formula", [
    "    return F0 / ((k - m * omega ** 2) ** 2 + (c * omega) ** 2)",                   # the square root forgotten
    "    return F0 / np.sqrt((k - m * omega ** 2) ** 2 + c * omega ** 2)",              # c omega not squared
    "    return F0 / abs(k - m * omega ** 2)",                                          # the formula without damping
    "    return ol.steady_state(m, c, k, F0, omega)",                                   # amplitude and phase: two numbers
])
def test_notebook_15_wrong_amplitudes_are_caught_by_the_measurement(formula):
    text = printed(execute(with_answers(NAME_15, {FILL_IN: formula})))
    assert "Exercise 3: not yet" in text and "Exercise 3: correct" not in text


def test_notebook_15_section_5_can_be_run_again_after_exercise_3():
    # found by the review: the cell of Section 5 used the name of the function of Exercise 3
    nb = with_answers(NAME_15, EXERCISES[NAME_15]["answers"])
    section_5 = next(cell for cell in nb.cells if cell.cell_type == "code" and "FIGURE_4 = " in cell.source)
    check = next(cell for cell in nb.cells if cell.cell_type == "code" and "PERIODS_MEASURED = " in cell.source)
    nb.cells.extend([nbformat.v4.new_code_cell(section_5.source), nbformat.v4.new_code_cell(check.source)])
    assert printed(execute(nb)).count("Exercise 3: correct on all 5 cases") == 2


@pytest.mark.parametrize("answers, message", [
    ({"my_numbers_of_solutions = None": "my_numbers_of_solutions = [0, np.inf, 1]"}, "Exercise 1, numbers of solutions: not yet"),
    ({"my_numbers_of_solutions = None": "my_numbers_of_solutions = 3"}, None),          # not a list: no traceback
    ({"my_y = None": "my_y = -sp.sqrt(1 - x ** 2)"}, "Exercise 2d: not yet (the equation is satisfied, but not the condition on y(0))"),
    ({"my_y = None": "my_y = 0"}, "Exercise 2d: not yet (the equation is satisfied, but not the condition on y(0))"),
    ({"my_x = None": "my_x = sp.sin(t) * 0"}, "Exercise 2c: not yet (the equation is satisfied, but not the condition on y'("),
    ({FILL_IN: "    return 5 * w0 * L ** 3 / (384 * EI)"}, "Exercise 3: not yet. For w0"),
    ({FILL_IN: "    return 5 / 384"}, "Exercise 3: not yet. For w0"),
    ({FILL_IN: "    return [5 * w0 * L ** 4 / (384 * EI), 0]"}, "Exercise 3: not yet (midspan_deflection must return one number"),
    ({FILL_IN: "    return 5 * w0 * L ** 4 / (384 * E * I)"}, "Exercise 3: not yet (midspan_deflection must return one number"),
    ({FILL_IN: "    return 0.013021 * w0 * L ** 4 / EI"}, "Exercise 3: correct on all 3 beams"),       # the value printed in Section 6
    ({FILL_IN: "    return 0.01302 * w0 * L ** 4 / EI"}, "Exercise 3: correct on all 3 beams"),
    ({FILL_IN: "    return w0 * L ** 4 / (384 * EI)"}, "Exercise 3: not yet. For w0"),                 # fixed at both ends
    ({"my_numbers_of_solutions = None": "my_numbers_of_solutions = np.array([np.inf, 0, 1])"},
     "Exercise 1, numbers of solutions: correct"),
])
def test_notebook_16_particular_wrong_answers(answers, message):
    try:
        text = printed(execute(with_answers(NAME_16, answers)))
    except Exception as problem:                               # a wrong answer must never stop the notebook
        pytest.fail(f"the notebook stopped: {str(problem)[-300:]}")
    assert message is None or message in text


@pytest.mark.parametrize("answers, message", [
    # other accepted forms
    ({"my_v_for_low = None": "my_v_for_low = [1, -1]", "my_v_for_high = None": "my_v_for_high = [1, 2]"},
     "Exercise 2a, eigenvector for the larger: correct"),
    ({"my_complex_vector = None": "my_complex_vector = [1j, 1]"}, "Exercise 2b, eigenvector: correct"),
    ({"my_v = None": "my_v = [1, 1]", "my_w = None": "my_w = [1 / 3, 0]"}, "Exercise 2c, w: correct"),
    ({"my_v = None": "my_v = [2, 2]", "my_w = None": "my_w = [2 / 3, 0]"}, "Exercise 2c, w: correct"),
    ({FILL_IN: "    return [[-(Q + Qw) / V, Q / V], [(Q + Qw) / V, -(Q + Qw) / V]]"},
     "Exercise 3: correct on all 3 sets of pipes"),
    # wrong answers, each with its message
    ({"my_v_for_low = None": "my_v_for_low = [1, 2]"},
     "Exercise 2a, eigenvector for the smaller: not yet (your vector does not point along an eigenvector"),
    ({"my_v_for_high = None": "my_v_for_high = [0, 0]"},
     "Exercise 2a, eigenvector for the larger: not yet (the zero vector is never an eigenvector)"),
    ({"my_complex_vector = None": "my_complex_vector = [-1j, 1]"},
     "Exercise 2b, eigenvector: not yet (your vector does not point along an eigenvector"),
    ({"my_v = None": "my_v = [1, 1]", "my_w = None": "my_w = [1, 1]"},                        # w = v
     "Exercise 2c, w: not yet ((A - lambda I) w must be your v"),
    ({"my_w = None": "my_w = [1, 0]"}, "Exercise 2c, w: not yet (find an eigenvector v first"),
    ({"my_cart_3kg = None": CART_2KG}, "Exercise 2d: not yet (your formula does not give the right values"),
    ({FILL_IN: "    return np.array([[-(Q + Qw), Q + Qw], [Q, -(Q + Qw)]]) / V"}, TANK_WRONG),   # transposed
    ({FILL_IN: "    return np.array([[-(Q + Qw), Q], [Q + Qw, -(Q + Qw)]])"}, TANK_WRONG),       # without / V
    ({FILL_IN: "    return np.array([[-Q, Q], [Q, -(Q - Qw)]]) / V"}, TANK_WRONG),               # Q - Qw
    ({FILL_IN: "    return Q / V"}, "Exercise 3: not yet (tank_matrix must return a 2 x 2 matrix"),
    ({FILL_IN: "    return np.full((2, 2), np.nan)"}, TANK_WRONG),                  # NaN is never right
    ({FILL_IN: "    return q / V"}, "got: NameError: name 'q' is not defined"),
    # a complex eigenvector, its w; the right values in the wrong order
    ({"my_v = None": "my_v = [1j, 1j]", "my_w = None": "my_w = [1j / 3, 0]"}, "Exercise 2c, w: correct"),
    ({"my_v = None": "my_v = sp.Matrix([3, 3])", "my_w = None": "my_w = sp.Matrix([1, 0])"},   # SymPy columns
     "Exercise 2c, w: correct"),
    ({"my_eigenvalues = None": "my_eigenvalues = [5, 2]"},
     "Exercise 2a, eigenvalues: not yet (right values: write them in increasing order)"),
    # a single number where a list is expected does not stop the notebook
    ({"my_Av = None": "my_Av = 3"}, "Exercise 1a: not yet (3 value(s) expected, got 1)"),
    ({"my_v_for_low = None": "my_v_for_low = 1"},
     "Exercise 2a, eigenvector for the smaller: not yet (2 value(s) expected"),
    ({"my_v = None": "my_v = [1, 1]", "my_w = None": "my_w = 2"},
     "Exercise 2c, w: not yet (a list of two numbers"),
])
def test_notebook_17_particular_answers(answers, message):
    try:
        text = printed(execute(with_answers(NAME_17, answers)))
    except Exception as problem:                               # a wrong answer must never stop the notebook
        pytest.fail(f"the notebook stopped: {str(problem)[-300:]}")
    assert message in text


def test_notebook_17_section_6_can_be_run_again_after_exercise_3():
    # the lesson of notebook 15: no global name of the sections is redefined by an exercise
    nb = with_answers(NAME_17, EXERCISES[NAME_17]["answers"])
    section_6 = [cell.source for cell in nb.cells if cell.cell_type == "code"
                 and ("COMPLEX_EXAMPLE = " in cell.source or "def plot_cases" in cell.source)]
    check = next(cell for cell in nb.cells if cell.cell_type == "code" and "PIPE_SETS = " in cell.source)
    assert len(section_6) == 2
    nb.cells.extend([nbformat.v4.new_code_cell(source) for source in section_6]
                    + [nbformat.v4.new_code_cell(check.source)])
    assert printed(execute(nb)).count("Exercise 3: correct on all 3 sets of pipes") == 2


@pytest.mark.parametrize("answers, message", [
    # other accepted forms: capitals, Lebl's synonyms, a tuple, np.array, 2.0 for 2
    ({"my_kind_d = None": 'my_kind_d = "Sink"'}, "Exercise 2a (d), picture: correct"),
    ({"my_kind_d = None": 'my_kind_d = "stable node"'}, "Exercise 2a (d), picture: correct"),
    ({"my_kind_357 = None": 'my_kind_357 = " Spiral-Source "'}, "Exercise 1a: correct"),
    ({"my_kind_other = None": 'my_kind_other = "saddle point"'},
     "Exercise 2c, the other, picture: correct"),
    ({"my_point_b = None": "my_point_b = (0.5, -0.25)"}, "Exercise 2b, equilibrium: correct"),
    ({"my_point_b = None": "my_point_b = np.array([0.5, -0.25])"}, "Exercise 2b, equilibrium: correct"),
    ({"my_jacobian_b = None": "my_jacobian_b = np.array([[-1.0, 1.0], [1.0, 1.0]])"},
     "Exercise 2b, J: correct"),
    ({"my_tr_det_e = None": "my_tr_det_e = (2.0, -15.0)"}, "Exercise 2a (e), tr and det: correct"),
    ({"my_speed_over_the_top = None": "my_speed_over_the_top = 2.0"}, "Exercise 2d: correct"),
    ({FILL_IN: HARES_ARRAY}, "Exercise 3: correct at all 15 points"),
    ({FILL_IN: "    return [[LV_A - LV_B * y - 2 * gamma * x, -LV_B * x], [LV_C * y, LV_C * x - LV_D]]"},
     "Exercise 3: correct at all 15 points"),
    # wrong answers, each with its message
    ({"my_kind_d = None": 'my_kind_d = "source"'}, "Exercise 2a (d), picture: not yet, try again"),
    ({"my_kind_origin = None": 'my_kind_origin = "center"'},
     "Exercise 2c, (0, 0), picture: not yet, try again"),
    ({"my_stability_origin = None": 'my_stability_origin = "stable"'},
     "Exercise 2c, (0, 0), stability: not yet, try again"),
    ({"my_kind_357 = None": 'my_kind_357 = "spiral"'},
     "Exercise 1a: not yet (not one of the words: source, sink"),
    ({"my_kind_357 = None": "my_kind_357 = 3"}, "Exercise 1a: not yet (type the word in quotes"),
    ({"my_point_b = None": "my_point_b = (-1 / 4, 1 / 2)"},
     "Exercise 2b, equilibrium: not yet (right values: write them in the order asked)"),
    ({"my_tr_det_d = None": "my_tr_det_d = [2, -3]"},
     "Exercise 2a (d), tr and det: not yet (right values: write them in the order asked)"),
    ({"my_jacobian_b = None": "my_jacobian_b = [[-1, 1]]"},
     "Exercise 2b, J: not yet (4 value(s) expected, got 2)"),
    ({"my_speed_over_the_top = None": "my_speed_over_the_top = 4"}, "Exercise 2d: not yet"),
    ({"my_other_point = None": "my_other_point = 1"},
     "Exercise 2c, the other equilibrium: not yet (2 value(s)"),
    ({FILL_IN: "    return [[0.4 - 0.01 * y - gamma * x, -0.01 * x], [0.003 * y, 0.003 * x - 0.3]]"},
     HARES_WRONG),                                                          # -gamma x for -2 gamma x
    ({FILL_IN: "    return [[0.4 - 0.01 * y - 2 * gamma * x, 0.01 * x], [0.003 * y, 0.003 * x - 0.3]]"},
     HARES_WRONG),                                                          # +0.01 x for -0.01 x
    ({FILL_IN: "    return [[0.4 - 0.01 * y, -0.01 * x], [0.003 * y, 0.003 * x - 0.3]]"},
     HARES_WRONG),                                                          # Lotka-Volterra, without gamma
    ({FILL_IN: "    return np.full((2, 2), np.nan)"}, HARES_WRONG),        # NaN is never right
    ({FILL_IN: "    return [0.4 - 0.01 * y - 2 * gamma * x, -0.01 * x, 0.003 * y, 0.003 * x - 0.3]"},
     "Exercise 3: not yet (hares_jacobian must return a 2 x 2 matrix"),
    ({FILL_IN: "    return [[q, 0], [0, 0]]"}, "got: NameError: name 'q' is not defined"),
    ({FILL_IN: "    return [[((0.4 - 0.01 * y) * x - gamma * x ** 2) / x - gamma * x, -0.01 * x], "
               "[0.003 * y, 0.003 * x - 0.3]]"},                    # right for x > 0, fails at x = 0
     "got: ZeroDivisionError"),
])
def test_notebook_18_particular_answers(answers, message):
    try:
        text = printed(execute(with_answers(NAME_18, answers)))
    except Exception as problem:                               # a wrong answer must never stop the notebook
        pytest.fail(f"the notebook stopped: {str(problem)[-300:]}")
    assert message in text


def test_notebook_18_section_7_can_be_run_again_after_exercise_3():
    # the lesson of notebook 15: no global name of the sections is redefined by an exercise
    nb = with_answers(NAME_18, EXERCISES[NAME_18]["answers"])
    section_7 = [cell.source for cell in nb.cells if cell.cell_type == "code"
                 and ("def pendulum" in cell.source or "def lotka_volterra" in cell.source
                      or "def plot_foxes" in cell.source)]
    check = next(cell for cell in nb.cells if cell.cell_type == "code" and "CHECK_POINTS = " in cell.source)
    assert len(section_7) == 3
    nb.cells.extend([nbformat.v4.new_code_cell(source) for source in section_7]
                    + [nbformat.v4.new_code_cell(check.source)])
    assert printed(execute(nb)).count("Exercise 3: correct at all 15 points") == 2


@pytest.mark.parametrize("answers, message", [
    # other accepted forms: any equivalent formula, 3.0 for 3, an amplitude and a phase
    ({"my_transform_1a = None": "my_transform_1a = (s - 3) ** -1"}, "Exercise 1a, transform: correct"),
    ({"my_edge_1a = None": "my_edge_1a = 3.0"}, "Exercise 1a, smallest c: correct"),
    ({"my_rest = None": "my_rest = 1 - 2 * s"}, "Exercise 1d: correct"),
    ({"my_transform_2a = None": "my_transform_2a = sp.factorial(2) * s ** -3"}, "Exercise 2a: correct"),
    ({"my_inverse_2b = None": "my_inverse_2b = 1 - 2 * t + 2 * t ** 2 - 1 / sp.exp(2 * t)"},
     "Exercise 2b: correct"),
    ({"my_transform_2c = None": "my_transform_2c = 1 / ((s + 1) ** 2 + 1)"}, "Exercise 2c: correct"),
    ({"my_spring_2d = None": SPRING_WITH_PHASE}, "Exercise 2d: correct"),
    # wrong answers, each with its message
    ({"my_transform_1a = None": "my_transform_1a = 1 / (t - 3)"},
     "Exercise 1a, transform: not yet (write the formula with the letter s only; it also contains t)"),
    ({"my_transform_1a = None": 'my_transform_1a = "1 / (s - 3)"'},
     "Exercise 1a, transform: not yet (type the formula without quotes)"),
    ({"my_edge_1a = None": "my_edge_1a = -3"}, "Exercise 1a, smallest c: not yet (your value -3, try again)"),
    ({"my_product_rule = None": "my_product_rule = True"}, "Exercise 1b: not yet"),
    ({"my_form = None": "my_form = 3"}, "Exercise 1c: not yet (your value 3, try again)"),
    ({"my_form = None": 'my_form = "2"'}, "Exercise 1c: not yet (type your answer without quotes"),
    ({"my_rest = None": "my_rest = s ** 2 * X - 2 * s + 1"},
     "Exercise 1d: not yet (write the formula with the letter s only; it also contains X)"),
    ({"my_rest = None": "my_rest = 2 * s - 1"}, "Exercise 1d: " + WRONG_FORMULA),
    ({"my_transform_2a = None": "my_transform_2a = 2 / s ** 2"}, "Exercise 2a: " + WRONG_FORMULA),
    ({"my_inverse_2b = None": "my_inverse_2b = 8 / (s ** 3 * (s + 2))"},            # the question typed back
     "Exercise 2b: not yet (write the formula with the letter t only; it also contains s)"),
    ({"my_transform_2c = None": "my_transform_2c = 1 / (s ** 2 + 1)"}, "Exercise 2c: " + WRONG_FORMULA),
    ({"my_spring_2d = None": "my_spring_2d = 0"}, "Exercise 2d: " + WRONG_FORMULA),
    ({"my_spring_2d = None": "my_spring_2d = sp.exp(-t) * (2 * sp.cos(3 * t) - sp.sin(3 * t))"},
     "Exercise 2d: " + WRONG_FORMULA),
])
def test_notebook_19_particular_answers(answers, message):
    try:
        text = printed(execute(with_answers(NAME_19, answers)))
    except Exception as problem:                               # a wrong answer must never stop the notebook
        pytest.fail(f"the notebook stopped: {str(problem)[-300:]}")
    assert message in text


def test_notebook_19_sections_can_be_run_again_after_the_exercises():
    # the lesson of notebook 15: no global name of the sections is redefined by an exercise
    nb = with_answers(NAME_19, EXERCISES[NAME_19]["answers"])
    again = [cell.source for cell in nb.cells if cell.cell_type == "code"
             and "import odelib as ol" not in cell.source and "interact(" not in cell.source]
    nb.cells.extend(nbformat.v4.new_code_cell(source) for source in again)
    text = printed(execute(nb))
    assert text.count("Exercise 2d: correct") == 2 and text.count("step 4: q(t) =") == 2


@pytest.mark.parametrize("changes, line", [
    # the numbers that the comments of the cells invite a student to change (clarity review of notebook 19)
    ({"S_ONE = 1.0 ": "S_ONE = -1.0 "}, "for s = -1 "),
    ({"S_ONE = 1.0 ": "S_ONE = 0.1 "}, "for s = 0.1 "),
    ({"S_DEMO = 3.0 ": "S_DEMO = 1.0 ", "DECAY = 2.0 ": "DECAY = 5.0 "},
     "L{e^(-5t)} at s = 1: the integral gives 0.16666667"),
    ({"S_INSIDE, S_OUTSIDE = 3.0, 1.0": "S_INSIDE, S_OUTSIDE = 40.0, 2.5"},     # no bare AssertionError
     "L{e^(t^2)} at s = 40: a number came out"),
    ({"S_INSIDE, S_OUTSIDE = 3.0, 1.0": "S_INSIDE, S_OUTSIDE = 40.0, 2.5"},
     "L{e^(2t)} at s = 2.5: a number came out, 2."),
    ({"g = 1 + sp.sin(2 * t)": "g = sp.exp(-t) + t ** 2"}, "g(0) = 1, g'(0) = -1"),
])
def test_notebook_19_survives_the_numbers_a_student_changes(changes, line):
    try:
        text = printed(execute(with_answers(NAME_19, changes)))
    except Exception as problem:
        pytest.fail(f"the notebook stopped: {str(problem)[-300:]}")
    assert line in text


@pytest.mark.parametrize("answers, message", [
    # other accepted forms: any equivalent formula, SymPy's own step, a function by cases, 6 / 2 for 3
    ({"my_cases_2a = None": CASES_TWO_RAMPS}, "Exercise 2a: correct"),
    ({"my_cases_2a = None": CASES_HEAVISIDE}, "Exercise 2a: correct"),
    ({"my_cases_2a = None": CASES_PIECEWISE}, "Exercise 2a: correct"),
    ({"my_switch_2b = None": "my_switch_2b = u(t - 1) - sp.cos(t - 1) * u(t - 1)"}, "Exercise 2b: correct"),
    ({"my_blow_2c = None": BLOW_DOUBLE_ANGLE}, "Exercise 2c: correct"),
    ({"my_convolution_2d = None": CONVOLUTION_ONE_SINE}, "Exercise 2d: correct"),
    ({"my_velocity = None": "my_velocity = 6 / 2"}, "Exercise 1c: correct"),
    ({"my_delta_at_0 = None": "my_delta_at_0 = np.exp(0)"}, "Exercise 1d: correct"),
    # wrong answers, each with its message
    ({"my_window_at_4 = None": 'my_window_at_4 = "1"'},
     "Exercise 1a, at t = 4: not yet (type your answer without quotes"),
    ({"my_window_at_6 = None": "my_window_at_6 = 1"},
     "Exercise 1a, at t = 6: not yet (your value 1, try again)"),
    ({"my_shift_rule = None": "my_shift_rule = True"}, "Exercise 1b: not yet"),
    ({"my_shift_rule = None": "my_shift_rule = True"}, "(is t^2 u(t - 1) a delayed t^2, or t^2 cut off?"),
    ({"my_velocity = None": "my_velocity = 12"}, "Exercise 1c: not yet (your value 12, try again)"),
    ({"my_velocity = None": "my_velocity = 0"}, "Exercise 1c: not yet (your value 0, try again)"),   # x'(0)
    ({"my_delta_at_0 = None": "my_delta_at_0 = 0"}, "Exercise 1d: not yet (your value 0, try again)"),
    ({"my_cases_2a = None": "my_cases_2a = t - 1"}, "Exercise 2a: " + WRONG_FORMULA),
    ({"my_cases_2a = None": 'my_cases_2a = "(t - 1) * u(t - 1)"'},
     "Exercise 2a: not yet (type the formula without quotes)"),
    # the mistakes that the notebook teaches get a sentence of their own (clarity review of notebook 20)
    ({"my_cases_2a = None": "my_cases_2a = (t - 1) * u(t - 1) + u(t - 2)"},
     "Exercise 2a: not yet (the piece t - 1 is never switched off: give it a window, u(t - 1) - u(t - 2))"),
    ({"my_switch_2b = None": "my_switch_2b = 1 - sp.cos(t - 1)"}, "Exercise 2b: " + NO_STEP_HINT),
    ({"my_switch_2b = None": "my_switch_2b = (1 - sp.cos(t)) * u(t - 1)"},
     "Exercise 2b: not yet (this is 1 - cos t " + CUT_OFF_HINT),
    ({"my_blow_2c = None": "my_blow_2c = 2 * sp.sin(2 * (t - 1))"}, "Exercise 2c: " + NO_STEP_HINT),
    # a switch at the wrong time is not accepted (code review, RISK 4)
    ({"my_cases_2a = None": "my_cases_2a = (t - 1) * (u(t - 1) - u(t - 1.8)) + u(t - 1.8)"},
     "Exercise 2a: " + WRONG_FORMULA),
    ({"my_switch_2b = None": SWITCH_LATE_2B}, "Exercise 2b: " + WRONG_FORMULA),
    ({"my_blow_2c = None": BLOW_LATE_2C}, "Exercise 2c: " + WRONG_FORMULA),
    # u with a number inside a formula: u(3) is 1
    ({"my_switch_2b = None": "my_switch_2b = (1 - sp.cos(t - 1)) * u(t - 1) * u(3)"}, "Exercise 2b: correct"),
    ({"my_switch_2b = None": "my_switch_2b = sp.exp(-s) / (s * (s ** 2 + 1))"},     # X(s) typed for x(t)
     "Exercise 2b: not yet (write the formula with the letter t only; it also contains s)"),
    ({"my_blow_2c = None": BLOW_CUT_OFF}, "Exercise 2c: not yet (this is 2 sin 2t " + CUT_OFF_HINT),
    ({"my_blow_2c = None": BLOW_TWICE},
     "Exercise 2c: not yet (the row of sin 2t has a 2 on top: 4/(s^2 + 4) is 2 times 2/(s^2 + 4))"),
    ({"my_convolution_2d = None": PRODUCT_2D}, "Exercise 2d: " + WRONG_FORMULA),
    ({"my_convolution_2d = None": "my_convolution_2d = (sp.cos(t) + sp.sin(t) + sp.exp(-t)) / 2"},
     "Exercise 2d: " + WRONG_FORMULA),
])
def test_notebook_20_particular_answers(answers, message):
    try:
        text = printed(execute(with_answers(NAME_20, answers)))
    except Exception as problem:                               # a wrong answer must never stop the notebook
        pytest.fail(f"the notebook stopped: {str(problem)[-300:]}")
    assert message in text


def test_notebook_20_sections_can_be_run_again_after_the_exercises():
    # the lesson of notebook 15: no global name of the sections is redefined by an exercise
    nb = with_answers(NAME_20, EXERCISES[NAME_20]["answers"])
    again = [cell.source for cell in nb.cells if cell.cell_type == "code"
             and "import odelib as ol" not in cell.source and "interact(" not in cell.source]
    nb.cells.extend(nbformat.v4.new_code_cell(source) for source in again)
    text = printed(execute(nb))
    assert text.count("Exercise 2d: correct") == 2 and text.count("step 4, by SymPy: x(t) =") == 2


@pytest.mark.parametrize("changes, line", [
    # the numbers that the comments of the cells invite a student to change: no bare AssertionError
    ({"A_SWITCH = 2.0 ": "A_SWITCH = 0.5 "}, "L{u(t - 0.5)}:"),
    ({"A_SWITCH = 2.0 ": "A_SWITCH = 0.0 "}, "L{u(t - 0)}:"),
    ({"A_SWITCH = 2.0 ": "A_SWITCH = 7.0 "}, "L{u(t - 7)}:"),                       # beyond the picture
    ({"DELAY = 2.0 ": "DELAY = 3.5 "}, "0.01509869"),                               # e^(-3.5) / 2, at s = 1
    ({"DELAY = 2.0 ": "DELAY = 0.0 "}, "0.50000000"),                               # sin t itself, at s = 1
    ({"S_PULSE = 1.0 ": "S_PULSE = 3.0 "}, "the transform of the unit pulse at s = 3"),
    ({"SIFT_AT = 1.0 ": "SIFT_AT = 0.0 "}, "cos(0) = 1.000000"),
    ({"SIFT_AT = 1.0 ": "SIFT_AT = -2.5 "}, "from a = -2.5 to a + b"),
    ({"ROCKET_ON, ROCKET_OFF = 1.0, 5.0 ": "ROCKET_ON, ROCKET_OFF = 2.0, 6.0 "},    # the checks hold for any
     "while the rocket fires: x between 0.00 and 2.00; at the stop x = 1.65"),      # firing of 4 seconds
    ({"CUT = 1.0 ": "CUT = 0.0 "}, "the right formula   the wrong formula"),        # nothing is cut off
    ({HAMMER_NUMBERS: "MY_MASS, MY_DAMPER, MY_SPRING, MY_BLOW = 2.0, 0.0, 10.0, 5.0"},
     "velocity just after the blow: P/m = 2.5 m/second"),
    ({HAMMER_NUMBERS: "MY_MASS, MY_DAMPER, MY_SPRING, MY_BLOW = 0.5, 6.0, 1.0, -1.0"},
     "velocity just after the blow: P/m = -2 m/second"),
    # the first time at which the largest |x| is reached (re-check of notebook 20, finding 3)
    ({HAMMER_NUMBERS: "MY_MASS, MY_DAMPER, MY_SPRING, MY_BLOW = 2.0, 0.0, 5.0, 3.0"},
     "largest |x| 0.949 m, first at t = 0.99 seconds"),
    ({HAMMER_NUMBERS: "MY_MASS, MY_DAMPER, MY_SPRING, MY_BLOW = 1.0, 2.0, 5.0, -3.0"},
     "largest |x| 0.771 m, first at t = 0.55 seconds"),
])
def test_notebook_20_survives_the_numbers_a_student_changes(changes, line):
    try:
        text = printed(execute(with_answers(NAME_20, changes)))
    except Exception as problem:
        pytest.fail(f"the notebook stopped: {str(problem)[-300:]}")
    assert line in text


@pytest.mark.parametrize("changes, sentence", [
    # the numbers of the text: a cell that checks them stops with a sentence, not with a bare AssertionError
    ({"BLOW = 3.0 ": "BLOW = 5.0 "}, "put those numbers back, and try your own in the cell after the figure"),
    ({"MASS, DAMPER, SPRING = 1.0, 2.0, 5.0 ": "MASS, DAMPER, SPRING = 2.0, 0.0, 10.0 "},
     "put those numbers back, and try your own in the cell after the figure"),
    ({"FORCE_START, FORCE_RATE = 8.0, 1.0 ": "FORCE_START, FORCE_RATE = 4.0, 2.0 "}, "put 8.0 and 1.0 back"),
    ({"S_PULSE = 1.0 ": "S_PULSE = 0.0 "}, "S_PULSE must be larger than 0"),
    ({"ROCKET_ON, ROCKET_OFF = 1.0, 5.0 ": "ROCKET_ON, ROCKET_OFF = 1.0, 1.0 + 2 * np.pi "},
     "put 1.0 and 5.0 back, and use the slider below for other firing times"),
])
def test_notebook_20_cells_of_the_text_stop_with_a_sentence(changes, sentence):
    with pytest.raises(nbclient.exceptions.CellExecutionError, match="AssertionError") as stopped:
        execute(with_answers(NAME_20, changes))
    assert sentence in str(stopped.value)


def test_notebook_20_u_is_one_step_for_formulas_and_for_numbers():
    nb = nbformat.read(NOTEBOOKS / NAME_20, as_version=4)
    lines = ["print(u(np.array([1.0, 2.0, 3.0]) - 2), u(2.5 - 2), u(t - 2), u(sp.Integer(3) - 2))",
             "t = np.linspace(0.0, 3.0, 4)", "u = 5.0"]            # what a student may do in Exercise 3
    nb.cells.append(nbformat.v4.new_code_cell(chr(10).join(lines)))
    reset = [cell for cell in nb.cells if cell.cell_type == "code" and "u = step_in_formulas" in cell.source]
    assert len(reset) == 2                                  # the cell before the answers, and the answer cell
    answers = EXERCISES[NAME_20]["answers"]
    for cell in reset:                                      # Exercise 2 run again, with the right answers
        source = cell.source
        for placeholder, answer in answers.items():
            source = source.replace(placeholder, answer)
        nb.cells.append(nbformat.v4.new_code_cell(source))
    text = printed(execute(nb))
    assert "[0. 1. 1.] 1.0 Heaviside(t - 2, 1) 1" in text
    assert "Exercise 2a: correct" in text and "Exercise 2c: correct" in text


def test_is_exact_written_with_an_equation_gets_a_hint_not_a_traceback():
    name = "12_Exact_Equations.ipynb"
    text = printed(execute(with_answers(name, {FILL_IN: "    return sp.Eq(sp.diff(M, y), sp.diff(N, x))"})))
    assert "Exercise 3: not yet (is_exact must return True or False" in text


# --- help for a student who has never used a notebook (review of 2026-11-17) ---------------
ANSWER_MARKER = "# ---- your answers go above this line; the lines below check them (nothing to change) ----"
HINT = "Hint (MATH 331): "
GUIDE_NOTEBOOK = "00_Index.ipynb"


def sources(name, kind):
    nb = nbformat.read(NOTEBOOKS / name, as_version=4)
    return [cell.source for cell in nb.cells if cell.cell_type == kind]


@pytest.mark.parametrize("name", sorted(EXERCISES))
def test_answer_cells_say_where_the_answers_go_and_exercises_say_what_to_do_first(name):
    def is_answer(line):                                    # my_x = None, with or without a comment
        return re.fullmatch(r"my_\w+ = None\s*(#.*)?", line) is not None

    answer_cells = [source for source in sources(name, "code") if is_answer(source.split("\n")[0])]
    assert answer_cells, f"{name}: no answer cell found"
    for source in answer_cells:
        lines = source.split("\n")
        at = lines.index(ANSWER_MARKER)                     # fails if the marker is missing
        assert all(is_answer(line) for line in lines[:at]), f"{name}: something else above the marker"
        assert not any(is_answer(line) for line in lines[at:]), f"{name}: an answer below the marker"
    for source in sources(name, "code"):                    # and no answer line anywhere without a marker
        assert ANSWER_MARKER in source or not any(is_answer(line) for line in source.split("\n")), f"{name}: {source[:60]}"
    heading = [source for source in sources(name, "markdown") if source.startswith("## ") and ". Exercises\n" in source]
    assert len(heading) == 1 and "**Before you start:** run the whole notebook (" in heading[0]


def test_the_first_predict_exercise_does_not_show_its_answer():
    nb = nbformat.read(NOTEBOOKS / "08_First_Steps_in_Python.ipynb", as_version=4)
    cell = next(cell for cell in nb.cells if cell.cell_type == "code" and cell.source.startswith("print(2 + 3 * 4, 2 ** 3 ** 2"))
    assert cell.outputs == []


def test_the_guide_notebook_has_the_text_of_the_library():
    import odelib as ol

    text = "\n".join(sources(GUIDE_NOTEBOOK, "markdown"))
    for step, words in ol.GUIDE_START:
        assert step in text and words in text
    for message, meaning, remedy in ol.GUIDE_ERRORS:
        assert meaning in text and remedy in text
    for question, answer in ol.GUIDE_HOW:
        assert question in text and answer[1:] in text
    for word, meaning in ol.GUIDE_WORDS:
        assert meaning in text
    assert ol.GUIDE_TYPING_RULE in text
    names = {path.name for path in PATHS} - {GUIDE_NOTEBOOK}
    assert all(f"]({name})" in text for name in names), "a notebook is missing from the table of notebook 00"


MISTAKES = (("print(sqrt(16))", "write np.sqrt(2) with numbers"),
            ("my_y = 2\nmy_x == 3.5", "the line my_x = ... is missing, or was typed with two = signs"),
            ("y = 3.5m", "a unit after a number"),
            ("y = 1\n y = 3.5", "start at the left edge"),      # (one line alone with a space in front is accepted)
            ("value = 2(3 + 1)", "2 * (3 + 1)"),
            ("x = sp.symbols('x')\nformula = x^3", "two stars"),
            ("x = sp.symbols('x')\nformula = np.exp(-2 * x)", "use sp.exp"),
            ("FACTOR = 3\nassert FACTOR * 2 == 4", "nothing is broken"),
            ("print(minuts)", "Python does not know the name 'minuts'"),
            ("my_answer = true", "write True"),
            ("y = 3x + 1", "3 * x, not 3x"),
            ("y = (1 + 2", "every ( needs its )"))


def test_a_beginners_mistake_gets_a_hint_under_the_red_message():
    setup = next(source for source in sources("08_First_Steps_in_Python.ipynb", "code") if "import odelib as ol" in source)
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(setup)]
                                  + [nbformat.v4.new_code_cell(source) for source, _ in MISTAKES]
                                  + [nbformat.v4.new_code_cell("1 / 0"), nbformat.v4.new_code_cell("print('still running')")])
    nbclient.NotebookClient(nb, timeout=TIMEOUT_SECONDS, kernel_name="python3", allow_errors=True,
                            resources={"metadata": {"path": str(NOTEBOOKS)}}).execute()
    def kinds(cell):
        return [output.output_type for output in cell.outputs]

    def red_text(cell):                                     # what is written on the error stream, under the message
        return "".join(output.text for output in cell.outputs if output.output_type == "stream" and output.name == "stderr")

    for cell, (source, words) in zip(nb.cells[1:], MISTAKES):
        # (Python itself may print a warning first, as for 2(3 + 1))
        assert kinds(cell)[-2:] == ["error", "stream"], f"{source}: the red message, then the hint, were expected; got {kinds(cell)}"
        last = cell.outputs[-1]
        assert last.name == "stderr" and last.text.startswith(HINT) and words in last.text, f"wrong hint for: {source}"
    assert kinds(nb.cells[-2]) == ["error"] and nb.cells[-2].outputs[0].ename == "ZeroDivisionError"   # no hint: only the message
    assert "still running" in printed(nb)


def test_an_error_still_stops_a_notebook_that_uses_the_library():
    # the first way of adding hints (IPython's set_custom_exc) made the kernel swallow errors: never again
    setup = next(source for source in sources("08_First_Steps_in_Python.ipynb", "code") if "import odelib as ol" in source)
    for failing in ("assert 1 == 2", "print(sqrt(16))", "y = 3x", "1 / 0"):
        nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(setup), nbformat.v4.new_code_cell(failing),
                                             nbformat.v4.new_code_cell("print('must not be reached')")])
        with pytest.raises(nbclient.exceptions.CellExecutionError):
            _execute_now(nb)
        assert "must not be reached" not in printed(nb)


# Slider functions at the ends of their sliders: the cases that used to stop with a traceback
# or draw an empty picture (review of 2026-11-17). Each line is run in a cell added at the end.
EXTREMES = {
    NAME_20: ("plot_rocket(d=0.5)", "plot_rocket(d=8.0)", "plot_rocket(d=4.0)",
              "plot_rocket(d=0.5 + 75 * 0.1)",                        # 8.0 as the slider gives it
              "plot_rocket(d=0.5 + 58 * 0.1)",                        # 6.3, the position nearest to 2 pi
              "plot_rocket(d=2 * np.pi)", "plot_rocket(d=np.pi)",     # at rest; the largest amplitude
              "plot_pulse(b=0.05)", "plot_pulse(b=2.0)", "plot_pulse(b=1.0)",
              "plot_pulse(b=0.05 + 39 * 0.05)",                       # 2.0 as the slider gives it
              "plot_pulse(b=0.05 + 1 * 0.05)", "plot_pulse(b=0.05 + 9 * 0.05)"),    # other values
    NAME_19: ("plot_area(s=0.5)", "plot_area(s=5.0)", "plot_area(s=1.0)",
              "plot_area(s=0.5 + 45 * 0.1)",                          # 5.0 as the slider gives it
              "plot_circuit(R=0.5)", "plot_circuit(R=8.0)",
              "plot_circuit(R=0.5 + 75 * 0.1)",                       # 8.0 as the slider gives it
              "plot_circuit(R=4.4)", "plot_circuit(R=4.5)",           # each side of the critical resistance
              "plot_circuit(R=20 ** 0.5)", "plot_circuit(R=20 ** 0.5 - 1e-10)",
              "plot_circuit(R=20 ** 0.5 + 1e-10)",
              "plot_circuit(R=6.0)", "plot_circuit(R=0.5 + 55 * 0.1)"),   # other values
    NAME_18: ("plot_family(p=-3.0)", "plot_family(p=3.0)", "plot_family(p=0.0)", "plot_family(p=1.0)",
              "plot_family(p=-3.0 + 30 * 0.1)", "plot_family(p=-3.0 + 40 * 0.1)",     # 0, 1 from the slider
              "plot_family(p=1e-10)", "plot_family(p=-1e-10)", "plot_family(p=1.0 - 1e-10)",
              "plot_family(p=1.0 + 1e-10)",
              "plot_foxes(foxes=10.0)", "plot_foxes(foxes=200.0)",
              "plot_foxes(foxes=40.0)",                               # the start (50, 40): the fewest hares
              "plot_foxes(foxes=10.0 + 19 * 10.0)"),                  # 200 as the slider gives it
    NAME_17: ("plot_cases(c=0.0)", "plot_cases(c=4.0)", "plot_cases(c=2.0)",
              "plot_cases(c=0.0 + 20 * 0.1)",                         # 2.0 as the slider gives it
              "plot_cases(c=2.0 - 1e-10)", "plot_cases(c=2.0 + 1e-10)",
              "plot_modes(x2_start=-2.0)", "plot_modes(x2_start=2.0)", "plot_modes(x2_start=-1.0)",
              "plot_modes(x2_start=-2.0 + 10 * 0.1)",                 # -1.0 as the slider gives it
              "plot_modes(x2_start=0.0)"),
    NAME_16: ("plot_bvp(lam=1.0)", "plot_bvp(lam=4.0)", "plot_bvp(lam=9.0)", "plot_bvp(lam=0.2)", "plot_bvp(lam=10.0)",
              "plot_bvp(lam=0.9999999999)", "plot_bvp(lam=1.1)", "plot_bvp(lam=0.2 + 8 * 0.1)",     # 1.0 as the slider gives it
              "plot_beam(holding=0, uneven=-1.0)", "plot_beam(holding=0, uneven=1.0)", "plot_beam(holding=1, uneven=-1.0)",
              "plot_beam(holding=1, uneven=1.0)", "plot_beam(holding=2, uneven=-1.0)", "plot_beam(holding=2, uneven=1.0)"),
    NAME_15: ("plot_beats(omega=4.0)", "plot_beats(omega=3.95)", "plot_beats(omega=4.05)", "plot_beats(omega=2.5)",
              "plot_beats(omega=5.0)", "plot_beats(omega=4.0000000001)",
              "plot_response(c=0.1, omega=1.0)", "plot_response(c=2.0, omega=0.1)", "plot_response(c=0.1, omega=3.0)",
              "plot_response(c=2.0, omega=3.0)", "plot_response(c=1.4, omega=1.0)", "plot_response(c=1.5, omega=0.1)"),
    NAME_14: ("plot_family(c=3.0)", "plot_family(c=2.0)", "plot_family(c=4.0)", "plot_family(c=2.95)", "plot_family(c=3.05)",
              "plot_family(c=3.0000000001)",
              "plot_transient(y0=-2.0, v0=-5.0)", "plot_transient(y0=2.0, v0=5.0)", "plot_transient(y0=-0.1, v0=0.4)"),
    "06_Characteristic_Equation.ipynb": ("plot_merging_roots(gap=0.01)", "plot_merging_roots(gap=0.9)",
                                         "plot_roots(b=-2.0, c=-3.0)", "plot_roots(b=6.0, c=9.0)", "plot_roots(b=0.0, c=0.0)",
                                         "plot_roots(b=4.0, c=4.0)", "plot_roots(b=-2.0, c=9.0)", "plot_roots(b=0.0, c=4.0)"),
    "05_Wronskian_Abel.ipynb": ("abel_numeric(alpha=-1.0, x_end=40.0)", "abel_numeric(alpha=3.0, x_end=5.0)",
                                "abel_numeric(alpha=0.0, x_end=40.0)"),
    "05E_Wronskian_Further.ipynb": ("plot_parallelogram(x_now=0.0, zeta=0.0)", "plot_parallelogram(x_now=12.0, zeta=0.5)",
                                    "plot_sturm(angle=0.0)", "plot_sturm(angle=180.0)", "plot_sturm(angle=5.0)"),
    "04_Euler_Method.ipynb": ("plot_step(h=0.02, x_end=10.0)", "plot_step(h=0.6, x_end=10.0)", "plot_step(h=0.6, x_end=1.0)",
                              "compare(model=list(MODELS)[0], n_steps=2)", "compare(model=list(MODELS)[1], n_steps=200)",
                              "compare(model=list(MODELS)[2], n_steps=2)",
                              "plot_stability(h=0.6, rate=-20.0, x_end=10.0)", "plot_stability(h=0.02, rate=-0.5, x_end=1.0)"),
    "02_Radioactive_Decay.ipynb": ("plot_decay(half_life=500.0, N0=0.1, n_half_lives=1)",
                                   "plot_decay(half_life=10000.0, N0=10.0, n_half_lives=10)",
                                   "plot_dating(fraction=0.02, delta_f=0.05)", "plot_dating(fraction=0.99, delta_f=0.05)",
                                   "plot_dating(fraction=0.5, delta_f=0.001)"),
    "02E_Decay_Chains.ipynb": ("plot_chain(half_parent=50.0, half_daughter=50.0)",        # the equal case
                               "plot_chain(half_parent=1.0, half_daughter=200.0, t_end=600.0)",
                               "plot_chain(half_parent=200.0, half_daughter=1.0, t_end=10.0)"),
    "01_Exponential_Growth.ipynb": ("plot_growth(P0=-100.0)", "plot_growth(P0=0.0)", "plot_growth(k=-0.1, P0=500.0)",
                                    "plot_growth(k=0.0)", "plot_growth(k=0.15, P0=500.0, t_end=120.0)"),
    "03_Logistic_Equation.ipynb": ("plot_logistic(k=0.5, M=1.0, x0=12.0, t_end=50.0)",
                                   "plot_logistic(k=0.01, M=10.0, x0=0.1, t_end=2.0)",
                                   "plot_harvest(h=0.0)", "plot_harvest(h=1.6)", "plot_harvest(h=2.5, t_end=80.0)",
                                   "plot_harvest(h=2.5, k=0.05, M=2.0)", "plot_harvest(h=0.0, k=0.3, M=12.0, t_end=5.0)"),
    "03E_Logistic_Further.ipynb": ("plot_negative_start(x0=-3.0, t_end=20.0)", "plot_negative_start(x0=-0.05, t_end=2.0)",
                                   "plot_euler_logistic(product=2.9, x0=9.0, n_steps=100)",
                                   "plot_euler_logistic(product=0.1, x0=0.1, n_steps=5)",
                                   "plot_bifurcation(h=0.0)", "plot_bifurcation(h=1.6)", "plot_bifurcation(h=2.0)"),
    "07_Mass_Spring_Damper.ipynb": ("plot_response(c=0.0)", "plot_response(c=4.0)", "plot_response(c=3.9)",
                                    "plot_response(c=12.0, m=0.25, k=0.5)", "plot_response(c=0.0, x0=0.0, v0=0.0)",
                                    "plot_response(c=12.0, m=4.0, k=16.0, x0=-2.0, v0=5.0)",
                                    "plot_response(c=0.1, m=4.0, k=16.0, x0=2.0, v0=-5.0)"),
    "07E_Vibrations_Further.ipynb": ("plot_root_locus(c=0.0)", "plot_root_locus(c=4.0)",
                                     "plot_root_locus(c=12.0, m=0.25, k=0.5)", "plot_root_locus(c=0.1, m=4.0, k=16.0)",
                                     "plot_cloud(c=0.0, t_now=0.0)", "plot_cloud(c=0.0, t_now=8.0)",
                                     "plot_cloud(c=2.0, m=0.5, k=10.0, t_now=8.0)", "plot_cloud(c=2.0, m=0.5, k=1.0, t_now=8.0)",
                                     "plot_map(b=0.0, c=0.0)", "plot_map(b=-4.0, c=-4.0)", "plot_map(b=4.0, c=8.0)",
                                     "plot_map(b=4.0, c=4.0)", "plot_map(b=-4.0, c=8.0)", "plot_map(b=0.0, c=4.0)"),
}


@pytest.mark.parametrize("name", sorted(EXTREMES))
def test_slider_functions_survive_the_ends_of_their_sliders(name):
    nb = nbformat.read(NOTEBOOKS / name, as_version=4)
    lines = ["import warnings", "warnings.simplefilter('error', RuntimeWarning)", *EXTREMES[name], "print('extremes done')"]
    nb.cells.append(nbformat.v4.new_code_cell(chr(10).join(lines)))
    assert "extremes done" in printed(execute(nb))
