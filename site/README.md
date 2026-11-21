# MATH 331 student site

The course notebooks, running in the browser (JupyterLite): no installation, no account.

| File | What it is |
|---|---|
| `build_site.py` | builds the site into `_output/` from `../notebooks/` |
| `serve_site.py` | serves `_output/` from this computer (testing, or a classroom without internet) |
| `landing.html` | the home page: list of notebooks and the student guide |
| `check_site.py` | runs every notebook of the built site in a headless browser and fails on any error |
| `requirements-site.txt` | the build tools, pinned |
| `.venv/`, `.cache/`, `_output/` | created by the steps below; not in git |

The home page lists the notebooks **by week of the lecture plan** (`READING_ORDER` in `build_site.py`),
with the optional "going further" notebooks (`NNE_*.ipynb`) marked. A new notebook that the list does
not mention yet appears at the end under "More notebooks"; a name in the list that does not exist
stops the build.

## Build

Once:

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-site.txt
```

Every time the notebooks change:

```
.venv\Scripts\python build_site.py
```

The script (1) runs the tests of `odelib` and stops if one fails; (2) copies **only** the numbered
notebooks and `odelib.py` (an allow-list: nothing else of the repository can reach the
site); (3) downloads, once, the Python runtime (Pyodide 314.0.6) and the 29 packages the notebooks
need, checking each file against its published SHA-256; (4) builds JupyterLite; (5) makes the site
self-contained (runtime and packages served by the site, fall-back to PyPI switched off);
(6) writes the home page. Result: about 135 MB of static files (23 notebooks, 2026-11-20).

Stop `serve_site.py` before rebuilding: on Windows a running server (or a terminal whose current
folder is `_output`) keeps the folder locked, and the script says so.

## Try it, and use it in class without internet

```
python serve_site.py            # http://localhost:8000, this computer only
python serve_site.py --lan      # also for the laptops on the same network or hotspot
```

`--lan` prints the address to give to the class (for example `http://192.168.1.20:8000`); Windows
may ask once to allow Python through the firewall. Use `serve_site.py`, not `python -m http.server`:
on Windows the latter labels `.mjs` files wrongly and Python never starts in the browser.

## Publish

The whole site is the folder `_output/`: static files, which any web host can serve. In the
public repository of the course the workflow `.github/workflows/site.yml` builds it and publishes
it with GitHub Pages at every push to `main` (once, in the settings of the repository: *Pages >
Source: GitHub Actions*).

## Correct a notebook after the students have started

A browser that has run a notebook keeps its own copy and goes on showing it. To make a
correction reach everybody, give the notebook a higher revision in `revisions.json` (the text at
the top of that file says how) and build again: the notebook is published under a new name
(`..._r2.ipynb`), the links to it are updated, and the home page says what changed and where the
student's earlier copy is. `test_build_site.py` tests this, and the home page.

## What the students get

* Home page: five steps at the top, the notebooks by week (notebook 00, the guide with a
  two-minute practice, comes first), then the whole guide: how to type mathematics, what the
  red messages mean, "how do I ...", words, sliders, keeping your work, privacy, slow devices.
  The five steps and the tables are filled in by `build_site.py` from the text of
  `../notebooks/odelib.py`: change them there.
* In every notebook, a note at the top (run everything first; where the guide is; download your
  work) and, in the setup cell, six lines that install the sliders in the browser. In an
  ordinary Jupyter both do nothing, so a downloaded notebook also runs on a laptop with Python
  installed. A stored slider, which a browser would show as "Loading widget...", is replaced
  by a line that says to run the notebook.
* Known limit: a browser that has run a notebook keeps its own copy and goes on showing it, also
  after the notebook on the site has been corrected (the guide says how to get the new one).
* First visit: about 60 MB are downloaded (Python 12 MB, packages 36 MB, the notebook interface);
  later visits use the browser's cache.

## Tested (2026-11-17, one Chromium browser, Windows, served by `serve_site.py`)

All cells run, without errors, using only files of the site:

| Notebook | Code cells | Plots | Sliders |
|---|---|---|---|
| 01 Exponential growth | 6 | 3 | 3 |
| 02 Radioactive decay | 8 | 6 | 8 |
| 03 Logistic equation | 9 | 9 | 13 |
| 04 Euler's method | 10 | 5 | 4 |
| 05 Wronskian and Abel | 15 | 8 | 5 |
| 06 Characteristic equation | 11 | 5 | 3 |
| 07 Mass-spring-damper | 11 | 8 | 9 |
| 08 First steps in Python (2026-11-17) | 22 | 5 | 2 |
| 09 Slope fields (2026-11-17) | 21 | 9 | 3 |
| 10 Separable equations (2026-11-17) | 16 | 7 | 2 |
| 11 Linear first-order equations (2026-11-17) | 19 | 9 | 1 |
| 12 Exact equations and substitutions (2026-11-17) | 18 | 5 | 1 |
| 13 First-order models (2026-11-17) | 17 | 9 | 4 |
| 14 Non-homogeneous equations (2026-11-20) | 16 | 9 | 3 |
| 15 Forced vibrations (2026-11-20) | 15 | 8 | 3 |
| 16 Higher order, boundary values (2026-11-20) | 15 | 7 | 3 |
| 17 Linear systems (2026-11-20) | 19 | 12 | 2 |
| 18 The phase plane and nonlinear systems (2026-11-20) | 20 | 13 | 2 |

Moving a slider from the keyboard redraws its plot (notebooks 04 and 08 to 13). Notebook 08 is listed
first on the home page ("Start here if you are new to Python"); its answers block is collapsed. The server log shows every Python
package served by the site itself; the only missing file is the browser's request for `favicon.ico`.

## Still to check (phase B, by the instructor, on real devices)

For Chrome, Edge, Firefox and Safari; a Windows laptop, a Mac, a Chromebook, a tablet, a phone:

1. open the home page in a fresh browser profile and time the first run of notebook 03;
2. run all cells, move a slider, download the notebook (File > Download);
3. reload: is an edit still there? Then follow "Start again from the original" on the home page and
   check that the steps are right in that browser (they were written from the menus, not tried);
4. repeat on a slow connection (phone hotspot);
5. repeat from another laptop with `serve_site.py --lan`, the internet switched off.

## Automatic check

```
.venv\Scripts\python check_site.py            # all notebooks, with Microsoft Edge
.venv\Scripts\python check_site.py --browser chrome
```

Every notebook is opened in a fresh browser profile (a first visit), all cells are run by the
Python in the browser, and the check fails if a cell stops with an error, if a notebook does not
finish, or if the page asks another host for a file. It drives a browser already installed
(Edge or Chrome): nothing is downloaded. Last run (2026-11-20): 23 of 23 notebooks (the 00 guide,
01-18 and the four optional E notebooks), 14-19 s each, 6.8 min in all.
Run it after every build.
