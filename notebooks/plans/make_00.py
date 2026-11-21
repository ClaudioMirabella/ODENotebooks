"""Build notebook 00, "Start here: how to use these notebooks", from the cell sources below.

    python plans/make_00.py              # write and execute (outputs are stored)
    python plans/make_00.py --no-run     # write only

Run from the notebooks folder. The guide (five steps, how to type mathematics, what the red
messages mean, words) is not written here: it is the text of odelib (GUIDE_START, GUIDE_TYPING,
GUIDE_ERRORS, GUIDE_WORDS), which is also printed by ol.guide() and shown on the home page of
the site, so the three places always say the same thing. The table of the notebooks is written
here: add a row when a notebook is added.
"""

import sys

from nbbuild import NOTEBOOKS, Builder

sys.path.insert(0, str(NOTEBOOKS))
import odelib as ol  # noqa: E402

NOTEBOOK = Builder("00_Index.ipynb", __doc__)
md, code = NOTEBOOK.md, NOTEBOOK.code


def table(header, rows):
    """A Markdown table; a cell written between backquotes is shown as code."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def as_code(text):
    return "`" + text + "`" if text else ""


steps = "\n".join(f"{number}. **{step}.** {text}" for number, (step, text) in enumerate(ol.GUIDE_START, start=1))
typing = table(("On paper", "In Python", "The usual mistake"),
               [(paper, as_code(python).replace("   or, in a formula, ", "` or, in a formula, `").replace("   or ", "` or `"),
                 mistake) for paper, python, mistake in ol.GUIDE_TYPING])
errors = table(("The last line of the red message", "What it means", "What to do"),
               [(as_code(message.split("   ")[0]) + (" " + " ".join(message.split("   ")[1:]) if "   " in message else ""),
                 meaning, remedy) for message, meaning, remedy in ol.GUIDE_ERRORS])
words = table(("Word", "Meaning"), [(as_code(word) if word[0] in "[#" else "**" + word + "**", meaning)
                                    for word, meaning in ol.GUIDE_WORDS])
how = "\n".join(f"* **... {question}?** {answer[:1].upper() + answer[1:]}" for question, answer in ol.GUIDE_HOW)


# ================================================================================================
md(r"""
# 00 - Start here: how to use these notebooks

**MATH 331 - Differential Equations.** Lecture plan: Read this first (ten minutes; nothing to install, no Python needed).

You do not need to know Python to use the notebooks of this course. You need **five steps**,
which are on this page, and ten minutes to try them here. Everything else is explained where it
is used; notebook 08, *First steps in Python*, is the first lesson.

This guide is always at hand: in any notebook, type `ol.guide()` in a code cell and run it.
""")

md("## 1. Five steps\n\n" + steps + r"""

You cannot break anything: the notebooks run only inside your browser, and the original can
always be opened again.
""")

code(r"""
# --- Setup ---------------------------------------------------------------------
# odelib.py lives next to the notebooks: it contains the tested helper functions
# of the course (here: check, which compares your answer with the expected one, and guide).
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path.cwd()))

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp                      # noqa: F401  (not used in every notebook)
from IPython.display import Math, display                 # noqa: F401
from ipywidgets import FloatSlider, IntSlider, interact    # noqa: F401

import odelib as ol

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "font.size": 11, "legend.fontsize": 9})
print("numpy", np.__version__, "| sympy", sp.__version__)
""")

md(r"""
## 2. Try it now (two minutes)

The grey box above is the first code cell of every notebook: it loads the tools. Run it (step
2 or step 3). When it has finished, it prints one line with two version numbers.

**An answer.** In the next cell, type `4` in place of `None` and press Shift+Enter. Then try a
wrong answer, to see what happens.
""")

code(r"""
my_first_answer = None          # type 4 in place of the first None, then press Shift+Enter

ol.check("Practice", my_first_answer, 2 + 2);
""")

md(r"""
**A slider.** Run the next cell and move the slider (with the mouse, or click it and use the
arrow keys). A picture with a slider works only after its cell has run.
""")

code(r"""
FIGURE_SIZE = (7.0, 3.0)
N_POINTS = 200
RATE_RANGE = (-1.0, 1.0, 0.1)               # minimum, maximum, step
DEFAULT_RATE = 0.5


def plot_exponential(k=DEFAULT_RATE):
    # the function exp(k x) for 0 <= x <= 3
    points = np.linspace(0.0, 3.0, N_POINTS)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(points, np.exp(k * points), "-", linewidth=2)
    ax.set_ylim(0.0, np.exp(RATE_RANGE[1] * 3.0))
    ax.set_xlabel("x")
    ax.set_title(f"exp(k x) with k = {k:g}")
    plt.show()


interact(plot_exponential,
         k=FloatSlider(value=DEFAULT_RATE, min=RATE_RANGE[0], max=RATE_RANGE[1], step=RATE_RANGE[2], description="k",
                       continuous_update=False));
""")

md(r"""
**A mistake, on purpose.** Click in the empty grey box below, type `print(sqrt(16))` and run it. A
red message appears. Next to the red box (just above it or just below it) there is a line that
starts with *Hint (MATH 331)*: it is written for this course and says what to do. Correct the
line as the hint says, and run it again.
""")

code("")

md(r"""
**Done?** You know the mechanics. Go back to the home page and open **08 - First steps in
Python**, the first lesson. The rest of this page is for looking things up.
""")

md("## 3. How to type mathematics\n\n" + typing + "\n\n" + ol.GUIDE_TYPING_RULE)

md("## 4. When you see red\n\nA red box is an error message, not a disaster. Read its **last line**, and the line "
   "that starts with *Hint (MATH 331)* just above or below the box, when there is one.\n\n" + errors + r"""

Still stuck? Copy the last line of the message and bring it to class or to office hours.
""")

md("## 5. How do I ...\n\n" + how)

md("## 6. Words\n\n" + words)

md(r"""
## 7. Good to know

* **Your copy.** Once you have run or changed a notebook, your browser keeps **your** version and
  shows it again next time, on this computer only. It is lost if you clear the browser data or
  use a private window: download what you want to keep (*File > Download*).
* **Every exercise has four steps**: predict (no computer); by hand, then check; fill in a
  function; explore. Final answers are at the end of each notebook, closed: try first.
* **The pictures.** Under every plot a sentence that starts with *The plot shows* says what there
  is to see; curves differ by line style, not only by colour.
* **Privacy.** No login, no cookies, no trackers: your code and your answers never leave your
  computer.
* **On your own laptop** (optional, for the projects): ask your instructor for the file
  `environment-student.yml`; the notebooks you download run there unchanged.
""")

md(r"""
## 8. The notebooks

Start with **08**. The numbers are those of the files; the order of the course is the one of the
home page, by week.

| # | Notebook | Equation | Session | Sections of the course text |
|---|---|---|---|---|
| 08 | [First steps in Python](08_First_Steps_in_Python.ipynb) | the diagnostic exam, checked by the computer | 2 | 0.2 |
| 09 | [Slope fields and what a solution is](09_Slope_Fields.ipynb) | $y' = f(x, y)$ | 3 | 0.2, 0.3, 1.2 |
| 10 | [Separable equations](10_Separable_Equations.ipynb) | $y' = f(x)\,g(y)$ | 4 | 1.3 |
| 01 | [Exponential growth](01_Exponential_Growth.ipynb) | $P' = kP$ | 3, 4, 7 | 0.2, 0.3, 1.1-1.3 |
| 02 | [Radioactive decay and half-life](02_Radioactive_Decay.ipynb) | $N' = -\lambda N$ | 4, 7 | 1.3 |
| 11 | [Linear first-order equations; existence and uniqueness](11_Linear_First_Order.ipynb) | $y' + p(x)y = f(x)$ | 5 | 1.2, 1.4 |
| 12 | [Exact equations and substitutions](12_Exact_Equations.ipynb) | $M\,dx + N\,dy = 0$ | 6 | 1.5, 1.8 |
| 02E | [Going further: decay chains](02E_Decay_Chains.ipynb) (optional) | $N_D' = \lambda_P N_P - \lambda_D N_D$ | after 5 | 1.4 |
| 03 | [The logistic equation](03_Logistic_Equation.ipynb) | $x' = kx(M - x)$ | 7 | 1.3, 1.5, 1.6 |
| 13 | [First-order models](13_First_Order_Models.ipynb) | $x' = (x_\infty - x)/\tau$ | 7-8 | 1.3, 1.4, 1.6, 1.7 |
| 04 | [Euler's method and beyond](04_Euler_Method.ipynb) | $y' = f(x, y)$ | 8, 29 | 1.7 |
| 03E | [Going further: the logistic equation](03E_Logistic_Further.ipynb) (optional) | blow-up; Euler's false oscillations | after 7-8 | 1.5, 1.6, 1.7 |
| 05 | [Superposition, the Wronskian and Abel's theorem](05_Wronskian_Abel.ipynb) | $y'' + p(x)y' + q(x)y = 0$ | 11 | 2.1 |
| 06 | [The characteristic equation](06_Characteristic_Equation.ipynb) | $ay'' + by' + cy = 0$ | 11-12 | 2.2, 2.3 |
| 05E | [Going further: the Wronskian](05E_Wronskian_Further.ipynb) (optional) | areas, special functions, Sturm's theorem | after 11-12 | 2.1, 2.3 |
| 14 | [Non-homogeneous equations](14_Nonhomogeneous_Equations.ipynb) | $ay'' + by' + cy = f(x)$ | 13-14 | 2.5 |
| 07 | [Free vibrations: the mass-spring-damper system](07_Mass_Spring_Damper.ipynb) | $mx'' + cx' + kx = 0$ | 14 | 2.4 |
| 07E | [Going further: vibrations](07E_Vibrations_Further.ipynb) (optional) | roots, energy, areas in the plane of states | after 14 | 2.4 |
| 15 | [Forced vibrations and resonance](15_Forced_Vibrations.ipynb) | $mx'' + cx' + kx = F_0\cos\omega t$ | 15 | 2.6 |
| 16 | [Higher-order equations and boundary value problems](16_Higher_Order_BVP.ipynb) | $y^{(n)} + \dots = 0$; $EIy^{(4)} = w(x)$ | 16 | 2.3, 4.1 |
| 17 | [Linear systems and the eigenvalue method](17_Linear_Systems.ipynb) | $\vec{x}\,' = A\vec{x}$ | 17-18 | 3.1-3.4, 3.6-3.8 |
| 18 | [The phase plane and nonlinear systems](18_Phase_Plane.ipynb) | $x' = f(x, y),\ y' = g(x, y)$ | 19-20 | 3.5, 8.1-8.3 |

The **course text** is J. Lebl, *Notes on Diffy Qs: Differential Equations for Engineers*, free
on LibreTexts; the section numbers above are those of that edition.
""")

md(r"""
## 9. For later: conventions of the notebooks

* **Letters.** The unknown is $y(x)$ when an equation is studied for itself, as in the course text;
  it is a function of the time $t$ in the models ($x(t)$, $v(t)$, $T(t)$, $h(t)$).
* **Integrating factor.** $r(x)$ for a linear equation (Lebl, Section 1.4) and $\mu$ for an exact
  equation: the first is a special case of the second (notebook 12).
* **Logistic equation.** $x' = kx(M - x)$, as in Lebl, Section 1.6.
* **Gravity.** $g = 9.81$ m/s$^2$ everywhere (`ol.GRAVITY`).
* **Exercise numbers** of the course text are those of the lists at the end of each chapter of the
  LibreTexts edition, unless "in the text" is said.
* **Checks.** `ol.check` compares a number, `ol.check_formula` a formula typed with SymPy,
  `ol.check_solution` substitutes a formula in an equation; all answer *correct* or *not yet* and
  never show the answer.
* **Self-checks.** Worked examples are compared with the values printed in the books by lines that
  start with `assert`: the notebook stops if a number does not match.

## 10. For later: the books behind the notebooks

| Book | Used for |
|---|---|
| J. Lebl, *Notes on Diffy Qs* (the course text) | everything |
| W.A. Adkins and M.G. Davidson, *Ordinary Differential Equations* (Springer, 2012) | decay and dating, the Wronskian and Abel's formula, springs and circuits, more examples |
| G.B. Arfken, H.J. Weber and F.E. Harris, *Mathematical Methods for Physicists*, 7th ed. (2013) | the Wronskian (optional notebook 05E) |
| NIST *Digital Library of Mathematical Functions* (dlmf.nist.gov) | Airy and Bessel functions (05E) |
| J.D. Logan, *An Introduction to Nonlinear Partial Differential Equations*, 2nd ed. (Wiley, 2008) | harvesting and bifurcation (03E) |
| A. Meseguer, *Fundamentals of Numerical Mathematics for Physicists and Engineers* (Wiley, 2020) | errors and stability of Euler's method (04) |
| J. Kiusalaas, *Numerical Methods in Engineering with Python 3* (Cambridge University Press, 2013) | the deflection of a beam (16) |

Two small slips found in the course text while checking are pointed out where they occur
(notebook 04). Checking every number is part of doing mathematics.
""")


# ================================================================================================
if __name__ == "__main__":
    NOTEBOOK.main()
