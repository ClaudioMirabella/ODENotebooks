"""Tests of odelib.check_formula (it needs SymPy, which the site's build environment does not have:
kept in its own file so that the other tests of odelib still run there).

Run from the ode/notebooks folder:  python -m pytest tests/test_check_formula.py
"""

import pytest

import odelib as ol

sp = pytest.importorskip("sympy")
X, T = sp.symbols("x t")
INTERVAL = (0.5, 2.5)


def test_check_formula_accepts_equivalent_forms(capsys):
    expected = 1 / (1 - sp.log(X))
    assert ol.check_formula("y", -1 / (sp.log(X) - 1), expected, X, INTERVAL) is True
    assert ol.check_formula("y", 2 * sp.exp(2 * sp.cos(X) ** 2) + 1,
                            2 * sp.exp(sp.cos(2 * X) + 1) + 1, X, INTERVAL) is True
    assert ol.check_formula("constant", 3, sp.Integer(3), X, INTERVAL) is True
    assert capsys.readouterr().out.count("correct") == 3


def test_check_formula_rejects_without_revealing_or_raising(capsys):
    expected = sp.exp(T ** 3) + 1
    assert ol.check_formula("x", None, expected, T, INTERVAL) is False
    assert "no answer yet" in capsys.readouterr().out
    assert ol.check_formula("x", sp.exp(T) + 1, expected, T, INTERVAL) is False
    out = capsys.readouterr().out
    assert "not yet" in out and "exp(t**3)" not in out
    assert ol.check_formula("x", sp.exp(X ** 3) + 1, expected, T, INTERVAL) is False       # wrong letter
    assert "letter t only" in capsys.readouterr().out
    assert ol.check_formula("x", "exp(t**3) + 1", expected, T, INTERVAL) is False          # typed as text
    assert "without quotes" in capsys.readouterr().out
    assert ol.check_formula("x", [1, 2], expected, T, INTERVAL) is False                   # not a formula
    assert ol.check_formula("x", sp.log(T - 10), expected, T, INTERVAL) is False           # complex values
    assert ol.check_formula("x", 1 / (T - T), expected, T, INTERVAL) is False              # not finite


# --- check_solution: answers that are not unique (notebook 14) -----------------------------
def hiccup(y):
    """Residual of Lebl's equation y'' - 9y = e^(3x)."""
    return sp.diff(y, X, 2) - 9 * y - sp.exp(3 * X)


def test_check_solution_accepts_every_particular_solution(capsys):
    particular = X * sp.exp(3 * X) / 6
    assert ol.check_solution("yp", particular, hiccup, X, INTERVAL) is True
    assert ol.check_solution("yp", particular + 5 * sp.exp(3 * X) - 2 * sp.exp(-3 * X), hiccup, X, INTERVAL) is True
    assert ol.check_solution("yp", sp.exp(3 * X) * (6 * X - 1) / 36, hiccup, X, INTERVAL) is True
    assert capsys.readouterr().out.count("correct") == 3


def test_check_solution_is_not_fooled_by_large_homogeneous_solutions(capsys):
    # found by the review of notebook 14: with a tolerance relative to the answer these passed
    for wrong in (10 ** 6 * sp.exp(3 * X), 10 ** 7 * sp.exp(-3 * X) + sp.exp(3 * X),
                  -X * sp.exp(3 * X) / 5 + 10 ** 6 * sp.exp(3 * X)):
        assert ol.check_solution("yp", wrong, hiccup, X, INTERVAL) is False
    assert ol.check_solution("yp", X * sp.exp(3 * X) / 6 + 10 ** 6 * sp.exp(3 * X), hiccup, X, INTERVAL) is True
    assert capsys.readouterr().out.count("correct") == 1


def test_check_solution_accepts_an_absolute_value(capsys):
    # Lebl's printed answer for y'' + y = tan x
    answer = sp.cos(X) * sp.log(sp.Abs((sp.sin(X) - 1) / (sp.sin(X) + 1))) / 2
    assert ol.check_solution("yp", answer, lambda y: sp.diff(y, X, 2) + y - sp.tan(X), X, (-1.0, 1.0)) is True
    assert ol.check_solution("yp", -answer, lambda y: sp.diff(y, X, 2) + y - sp.tan(X), X, (-1.0, 1.0)) is False


def test_check_solution_rejects_without_revealing_or_raising(capsys):
    assert ol.check_solution("yp", None, hiccup, X, INTERVAL) is False
    assert "no answer yet" in capsys.readouterr().out
    for wrong in (X * sp.exp(3 * X) / 3,                 # a wrong coefficient
                  -X * sp.exp(3 * X) / 6,                # a lost sign
                  sp.exp(3 * X) / 6,                     # the guess that was not modified
                  sp.Integer(0), 0,                      # the zero function
                  sp.exp(3 * X) + sp.exp(-3 * X)):       # a solution of the homogeneous equation
        assert ol.check_solution("yp", wrong, hiccup, X, INTERVAL) is False
        out = capsys.readouterr().out
        assert "does not satisfy the equation" in out and "/6" not in out
    assert ol.check_solution("yp", T * sp.exp(3 * T) / 6, hiccup, X, INTERVAL) is False     # wrong letter
    assert "letter x only" in capsys.readouterr().out
    assert ol.check_solution("yp", "x*exp(3*x)/6", hiccup, X, INTERVAL) is False            # typed as text
    assert "without quotes" in capsys.readouterr().out
    assert ol.check_solution("yp", [1, 2], hiccup, X, INTERVAL) is False                    # not a formula
    assert ol.check_solution("yp", 1 / (X - X), hiccup, X, INTERVAL) is False               # not finite
    assert ol.check_solution("yp", X, lambda y: 1 / 0, X, INTERVAL) is False                # a residual that fails


def test_check_solution_with_initial_conditions(capsys):
    # Lebl, Exercise 2.5.13: y'' + 2y' + y = x^2, y(0) = 1, y'(0) = 2
    def residual(y):
        return sp.diff(y, X, 2) + 2 * sp.diff(y, X) + y - X ** 2

    conditions = ((0, 1), (0, 2, 1))
    answer = X ** 2 - 4 * X + 6 + sp.exp(-X) * (X - 5)
    assert ol.check_solution("y", answer, residual, X, INTERVAL, conditions) is True
    capsys.readouterr()
    assert ol.check_solution("y", X ** 2 - 4 * X + 6, residual, X, INTERVAL, conditions) is False
    out = capsys.readouterr().out
    assert "y(0)" in out and "y'(0)" in out and "equation is satisfied" in out
    assert ol.check_solution("y", answer + sp.exp(-X) * X, residual, X, INTERVAL, conditions) is False
    out = capsys.readouterr().out
    assert "y'(0)" in out and "y(0)" not in out


def test_check_solution_names_a_condition_given_at_a_sympy_number(capsys):
    # found while building notebook 16: a condition at sp.pi that is not met must be named, not blamed on the formula
    def residual(u):
        return sp.diff(u, T, 2) + u

    conditions = ((0, 0), (sp.pi, 1, 1))                 # Lebl, Exercise 4.1.8: x(0) = 0, x'(pi) = 1
    assert ol.check_solution("x", -sp.sin(T), residual, T, (0.0, 3.0), conditions) is True
    capsys.readouterr()
    assert ol.check_solution("x", sp.sin(T), residual, T, (0.0, 3.0), conditions) is False
    assert "not the condition on y'(3.14159)" in capsys.readouterr().out
    assert ol.check_solution("x", sp.Integer(0), residual, T, (0.0, 3.0), conditions) is False
    assert "not the condition on y'(3.14159)" in capsys.readouterr().out
