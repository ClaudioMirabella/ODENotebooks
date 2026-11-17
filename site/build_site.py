"""
Build the MATH 331 student site: the course notebooks, running in the browser (JupyterLite).

    .venv/Scripts/python build_site.py            # -> _output/   (the whole site, static files)
    .venv/Scripts/python build_site.py --no-checks  # skip the tests of odelib before building

What the script guarantees:

* **allow-list**: only the numbered notebooks and odelib.py are copied from ../notebooks; nothing
  else of the repository can reach the site;
* **self-contained**: the Python runtime (Pyodide), the scientific libraries and the slider package
  are downloaded once into .cache/ (each file checked against its published SHA-256) and served by
  the site itself; the fall-back to PyPI is switched off, so a browser contacts no third party;
* **same files everywhere**: two cells are added to the copies (a reminder to download your work,
  and a guard that installs the sliders in the browser); in an ordinary Jupyter both do nothing;
* a landing page (index.html) lists the notebooks and explains the site to the students.

Serve the result in class with:  python -m http.server 8000 --directory _output
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import pathlib
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
NOTEBOOKS = HERE.parent / "notebooks"
sys.path.insert(0, str(NOTEBOOKS))
import odelib as guide  # noqa: E402  (the text of the student guide: GUIDE_START, GUIDE_TYPING, ...)
CACHE = HERE / ".cache"
OUTPUT = HERE / "_output"
LANDING = HERE / "landing.html"
REVISIONS = HERE / "revisions.json"     # which notebooks were corrected after the students started (see the file)

# --- what goes on the site (the allow-list) ---------------------------------------------------
NOTEBOOK_PATTERN = "[0-9][0-9]*_*.ipynb"
NOTEBOOK_NAME = re.compile(r"\d\d(E?)_\w+\.ipynb")      # 03_Name.ipynb, or 03E_Name.ipynb (optional)
OPTIONAL_LABEL = "Optional."
# The home page follows the lecture plan of the course, not the numbers of the files.
# Each entry: heading, then the notebooks by the first part of their file name.
READING_ORDER = (
    ("Before you start - how to use the notebooks", ("00",)),
    ("Week 1 - Getting ready (Session 2)", ("08",)),
    ("Week 2 - What a solution is; separable equations (Sessions 3-4)", ("09", "10", "01", "02")),
    ("Week 3 - Linear equations, existence and uniqueness; exact equations (Sessions 5-6)", ("11", "12", "02E")),
    ("Week 4 - Models; a first look at numerical methods (Sessions 7-8)", ("03", "13", "04", "03E")),
    ("Week 6 - Second-order linear equations (Sessions 11-12)", ("05", "06", "05E")),
    ("Week 7 - Non-homogeneous equations and vibrations (Sessions 13-14)", ("14", "07", "07E")),
    ("Week 8 - Forced vibrations; higher order and boundary values (Sessions 15-16)", ("15", "16")),
)
UNPLACED_HEADING = "More notebooks"         # anything new that READING_ORDER does not mention yet
EXTRA_FILES = ("odelib.py",)
START_HERE = "08_First_Steps_in_Python.ipynb"      # listed first on the home page
START_LABEL = "The first lesson, if you are new to Python."
GUIDE_NOTEBOOK = "00_Index.ipynb"       # the student guide as a notebook; it gets no reminder cell
CORRECTED_NOTE = ("This is a new copy, without what you had typed: your earlier copy is still in your browser "
                  "(menu File > Open...), under the old name.")

# --- the Python runtime ---------------------------------------------------------------------------
PYODIDE_VERSION = "314.0.6"             # the version jupyterlite-pyodide-kernel 0.8.7 expects
PYODIDE_CORE = (f"https://github.com/pyodide/pyodide/releases/download/{PYODIDE_VERSION}/"
                f"pyodide-core-{PYODIDE_VERSION}.tar.bz2")
PYODIDE_FILES = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/"
# Packages of the Pyodide distribution the notebooks and the kernel need; their dependencies are
# added from pyodide-lock.json. (Found by serving the site offline and reading the missing files.)
PYODIDE_PACKAGES = ("numpy", "scipy", "sympy", "matplotlib", "ipython", "traitlets", "micropip",
                    "jedi")           # jedi (and parso): code completion, asked for by the kernel
# Pure-Python wheels served by the site's own package index (piplite), with exact versions.
PIPLITE_WHEELS = ("ipywidgets==8.1.9", "jupyterlab_widgets==3.0.17", "comm==0.2.3")

KERNEL_PLUGIN = "@jupyterlite/pyodide-kernel-extension:kernel"
SETUP_MARKER = "import sys"             # the guard goes just before this line of the setup cell

GUARD = [
    "# In the browser version of the course (JupyterLite) the sliders are installed on first use.\n",
    "try:\n",
    "    import ipywidgets  # noqa: F401\n",
    "except ModuleNotFoundError:\n",
    "    import piplite\n",
    "    await piplite.install(\"ipywidgets\")\n",
    "\n",
]
REMINDER = [
    "> **New here?** First run everything: the `>>` button in the bar above, then **Confirm Kernel\n",
    "> Restart**; or the menu **Run > Run All Cells** (the first start takes about a minute). What you\n",
    "> see before that was stored when the notebook was made.  \n",
    "> **Stuck?** Open [00 - Start here](00_Index.ipynb), the guide of the course (five steps, how to\n",
    "> type mathematics, what the red messages mean). Or, after the notebook has run: click `+` in the\n",
    "> bar above, type `ol.guide()` in the new grey box and press Shift+Enter.  \n",
    "> **Your copy.** Your changes are saved only in this browser, on this computer. To keep your work:\n",
    "> **File > Download**.\n",
]
WIDGET_MIME = "application/vnd.jupyter.widget-view+json"
SLIDER_NOTE = ("[Sliders appear here after the notebook has run: the >> button at the top, then "
               "Confirm Kernel Restart; or the menu Run > Run All Cells.]\n")


# ================================================================================================
def log(message: str) -> None:
    print(message, flush=True)


def fetch(url: str, dest: pathlib.Path, sha256: str | None = None) -> pathlib.Path:
    """Download once into the cache; verify the checksum when one is published."""
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        log(f"  download {url}")
        request = urllib.request.Request(url, headers={"User-Agent": "course-site-build"})
        with urllib.request.urlopen(request) as response, open(dest, "wb") as out:
            shutil.copyfileobj(response, out)
    if sha256:
        found = hashlib.sha256(dest.read_bytes()).hexdigest()
        if found != sha256:
            dest.unlink()
            raise RuntimeError(f"checksum mismatch for {dest.name}: expected {sha256}, found {found}")
    return dest


def pyodide_closure(lock: dict, seeds) -> list[str]:
    """The seeds and everything they depend on, by the names of pyodide-lock.json."""
    packages, todo, seen = lock["packages"], list(seeds), []
    while todo:
        name = todo.pop()
        key = name if name in packages else name.replace("_", "-").lower()
        if key not in packages:
            raise KeyError(f"{name} is not in the Pyodide distribution")
        if key not in seen:
            seen.append(key)
            todo.extend(packages[key].get("depends", []))
    return sorted(seen)


def pyodide_runtime(extra_packages=()) -> pathlib.Path:
    """The Pyodide runtime with only the packages we need, in the cache."""
    root = CACHE / f"pyodide-{PYODIDE_VERSION}"
    runtime = root / "pyodide"
    if not (runtime / "pyodide-lock.json").exists():
        tarball = fetch(PYODIDE_CORE, CACHE / f"pyodide-core-{PYODIDE_VERSION}.tar.bz2")
        root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tarball) as tar:
            tar.extractall(root, filter="data")
    lock = json.loads((runtime / "pyodide-lock.json").read_text(encoding="utf-8"))
    names = pyodide_closure(lock, [*PYODIDE_PACKAGES, *extra_packages])
    for name in names:
        entry = lock["packages"][name]
        fetch(PYODIDE_FILES + entry["file_name"], runtime / entry["file_name"], entry["sha256"])
    log(f"  Pyodide {PYODIDE_VERSION}: {len(names)} packages")
    return runtime


def piplite_wheels() -> list[pathlib.Path]:
    wheels = CACHE / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    wanted = {spec.split("==")[0].lower().replace("-", "_"): spec for spec in PIPLITE_WHEELS}
    have = {p.name.split("-")[0].lower(): p for p in wheels.glob("*.whl")}
    missing = [spec for name, spec in wanted.items() if name not in have]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
                        "--disable-pip-version-check", "--quiet", "-d", str(wheels), *missing], check=True)
    return sorted(wheels.glob("*.whl"))


# ================================================================================================
# Corrections after the students have started. A browser that has run a notebook keeps its own copy
# under the notebook's file name and shows it instead of the one of the site. A corrected notebook
# is therefore published under a new name, and every link to it is rewritten.
def read_revisions(path: pathlib.Path = REVISIONS) -> dict:
    """{"10": {"revision": 2, "date": "2027-02-03", "note": "..."}}, checked; {} if there is no file."""
    if not path.exists():
        return {}
    entries = json.loads(path.read_text(encoding="utf-8")).get("notebooks", {})
    for prefix, entry in entries.items():
        revision = entry.get("revision")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
            raise SystemExit(f"{path.name}: the revision of notebook {prefix} must be a whole number, 1 or more")
        if revision > 1 and not (str(entry.get("date", "")).strip() and str(entry.get("note", "")).strip()):
            raise SystemExit(f"{path.name}: notebook {prefix} has revision {revision}: say when (\"date\") and what "
                             "changed (\"note\"), for the students")
    return entries


def published_name(name: str, revisions: dict) -> str:
    """The file name under which a notebook is published: 10_Name.ipynb, or 10_Name_r2.ipynb once corrected."""
    revision = revisions.get(name.split("_")[0], {}).get("revision", 1)
    return name if revision == 1 else f"{name[:-len('.ipynb')]}_r{revision}.ipynb"


def with_published_links(nb: dict, names: dict) -> dict:
    """The notebook with its links to other notebooks, "](10_Name.ipynb)", pointing to the published files."""
    changed = {old: new for old, new in names.items() if old != new}
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown" or not changed:
            continue
        source = "".join(cell["source"])
        for old, new in changed.items():
            source = source.replace(f"]({old})", f"]({new})")
        cell["source"] = source.splitlines(keepends=True)
    return nb


def student_notebook(source: pathlib.Path) -> dict:
    """A copy of a course notebook with the reminder cell and the slider guard."""
    nb = json.loads(source.read_text(encoding="utf-8"))
    cells = nb["cells"]
    code = [c for c in cells if c["cell_type"] == "code"]
    if code:                                    # the index notebook has no code
        setup = code[0]
        lines = setup["source"] if isinstance(setup["source"], list) else setup["source"].splitlines(keepends=True)
        at = next((k for k, line in enumerate(lines) if line.startswith(SETUP_MARKER)), None)
        if at is None:
            raise ValueError(f"{source.name}: no '{SETUP_MARKER}' line in the first code cell")
        setup["source"] = lines[:at] + GUARD + lines[at:]
        for cell in code:                       # a stored slider is only the words "Loading widget..."
            cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": [SLIDER_NOTE]}
                               if WIDGET_MIME in output.get("data", {}) else output for output in cell.get("outputs", [])]
        if source.name != GUIDE_NOTEBOOK:
            reminder = {"cell_type": "markdown", "metadata": {}, "source": REMINDER}
            cells.insert(cells.index(setup), reminder)
    return nb


def describe(source: pathlib.Path) -> dict:
    """Title and sessions of a notebook, read from its first cell."""
    nb = json.loads(source.read_text(encoding="utf-8"))
    first = "".join(nb["cells"][0]["source"])
    title = re.search(r"^#\s+(.*)$", first, re.M)
    sessions = re.search(r"Lecture plan:\s*([^.\n]*)", first)
    return {"file": source.name, "source": source.name, "title": title.group(1).strip() if title else source.stem,
            "sessions": sessions.group(1).strip() if sessions else "", "correction": None}


def stage_contents() -> tuple[pathlib.Path, list[dict]]:
    stage = CACHE / "stage"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    listing = []
    sources = course_notebooks()
    revisions = read_revisions()
    unknown = set(revisions) - {source.name.split("_")[0] for source in sources}
    if unknown:
        raise SystemExit(f"{REVISIONS.name} names notebooks that do not exist: {sorted(unknown)}")
    names = {source.name: published_name(source.name, revisions) for source in sources}
    for source in sources:
        copy = with_published_links(student_notebook(source), names)
        (stage / names[source.name]).write_text(json.dumps(copy, indent=1), encoding="utf-8")
        entry = revisions.get(source.name.split("_")[0], {})
        listing.append({**describe(source), "file": names[source.name],
                        "correction": entry if entry.get("revision", 1) > 1 else None})
    for name in EXTRA_FILES:
        shutil.copyfile(NOTEBOOKS / name, stage / name)
    staged = sorted(p.name for p in stage.iterdir())
    allowed = sorted([d["file"] for d in listing] + list(EXTRA_FILES))
    if staged != allowed:                       # belt and braces: nothing but the allow-list
        raise RuntimeError(f"unexpected files staged: {set(staged) - set(allowed)}")
    return stage, listing


def course_notebooks() -> list[pathlib.Path]:
    """The notebooks that go on the site, in order: 02 before its optional companion 02E."""
    found = [p for p in NOTEBOOKS.glob(NOTEBOOK_PATTERN) if NOTEBOOK_NAME.fullmatch(p.name)]
    return sorted(found, key=lambda p: (p.name[:2], bool(NOTEBOOK_NAME.fullmatch(p.name).group(1)), p.name))


def landing_page(listing: list[dict]) -> str:
    rows = []
    by_prefix = {item["file"].split("_")[0]: item for item in listing}
    placed = [prefix for _, prefixes in READING_ORDER for prefix in prefixes]
    missing = [prefix for prefix in placed if prefix not in by_prefix]
    if missing:
        raise RuntimeError(f"READING_ORDER names notebooks that do not exist: {missing}")
    groups = [(heading, [by_prefix[prefix] for prefix in prefixes]) for heading, prefixes in READING_ORDER]
    unplaced = [item for prefix, item in by_prefix.items() if prefix not in placed]
    if unplaced:
        groups.append((UNPLACED_HEADING, unplaced))
    for heading, items in groups:
        rows.append(f'<li class="week">{html.escape(heading)}</li>')
        rows.extend(notebook_row(item) for item in items)
    page = LANDING.read_text(encoding="utf-8")
    guide_file = next(item["file"] for item in listing if item["source"] == GUIDE_NOTEBOOK)
    page = page.replace("?path=" + GUIDE_NOTEBOOK, "?path=" + guide_file)       # the link of the quick start
    parts = {"NOTEBOOKS": "\n".join(rows), **guide_parts()}
    for name, content in parts.items():
        marker = f"<!--{name}-->"
        if page.count(marker) != 1:
            raise RuntimeError(f"landing.html must contain {marker} exactly once")
        page = page.replace(marker, content)
    return page


def guide_parts() -> dict:
    """The quick start and the three tables of the guide as HTML, from the text of odelib."""
    def text(words: str) -> str:
        return html.escape(words)

    def python(words: str) -> str:              # "np.sqrt(2)   or, in a formula, sp.sqrt(2)": code, words, code
        pieces = words.split("   ")
        return " ".join(f"<code>{text(piece)}</code>" if number % 2 == 0 else text(piece)
                        for number, piece in enumerate(pieces))

    def table(header, rows) -> str:
        head = "".join(f"<th>{text(title)}</th>" for title in header)
        body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>\n" for row in rows)
        return f"<table>\n<tr>{head}</tr>\n{body}</table>"

    steps = "\n".join(f"      <li><b>{text(step)}.</b> {text(words)}</li>" for step, words in guide.GUIDE_START)
    typing = table(("On paper", "In Python", "The usual mistake"),
                   [(text(paper), python(code), text(mistake)) for paper, code, mistake in guide.GUIDE_TYPING])
    errors = table(("The last line of the red message", "What it means", "What to do"),
                   [(python(message), text(meaning), text(remedy)) for message, meaning, remedy in guide.GUIDE_ERRORS])
    words = table(("Word", "Meaning"), [(f"<b>{text(word)}</b>", text(meaning)) for word, meaning in guide.GUIDE_WORDS])
    how = "\n".join(f"      <li><b>... {text(question)}?</b> {text(answer[:1].upper() + answer[1:])}</li>"
                    for question, answer in guide.GUIDE_HOW)
    return {"QUICK_START": steps, "TYPING": typing, "TYPING_RULE": text(guide.GUIDE_TYPING_RULE), "HOW": how,
            "ERRORS": errors.replace("<td><code>", '<td class="wrap"><code>'), "WORDS": words}


def notebook_row(item: dict) -> str:
    """One line of the home page: the title as a link, and a note (sessions, optional, start here)."""
    link = "notebooks/index.html?path=" + item["file"]
    sessions = item["sessions"]
    if NOTEBOOK_NAME.fullmatch(item["file"]).group(1):               # an optional notebook
        note = OPTIONAL_LABEL + " " + sessions[:1].upper() + sessions[1:]
    else:
        note = " ".join(part for part in (START_LABEL if item["source"] == START_HERE else "", sessions) if part)
    when = f'<span class="when">{html.escape(note)}</span>' if note else ""
    if item["correction"]:
        correction = item["correction"]
        when += ('<span class="when corrected">' + html.escape(f"Corrected on {correction['date']}: {correction['note']} ")
                 + html.escape(CORRECTED_NOTE) + "</span>")
    return f'<li><a href="{link}">{html.escape(item["title"])}</a>{when}</li>'


ICON_SIZE = 32                          # pixels
ICON_BACKGROUND = (0x00, 0x20, 0x60)    # navy of the course slides (red, green, blue)
ICON_CURVE = (0xD9, 0x5F, 0x02)         # orange
ICON_GROWTH = 2.5                       # the curve is exp(ICON_GROWTH * x) on 0 <= x <= 1


def favicon() -> bytes:
    """A small icon for the browser tab, drawn here: a rising exponential on a navy square.

    Written by hand in the ICO format (one 32-bit image), so the build needs no image library
    and the browser no longer asks for a file the site does not have.
    """
    n = ICON_SIZE
    top = math.exp(ICON_GROWTH)
    rows = []
    for row in range(n):                                    # ICO stores the rows from the bottom
        pixels = bytearray()
        for column in range(n):
            height = (math.exp(ICON_GROWTH * column / (n - 1)) - 1) / (top - 1) * (n - 3) + 1
            red, green, blue = ICON_CURVE if abs(row - height) <= 1.5 else ICON_BACKGROUND
            pixels += bytes((blue, green, red, 0xFF))
        rows.append(bytes(pixels))
    image = b"".join(rows)
    mask = bytes(n * ((n + 31) // 32 * 4))                  # nothing transparent
    header = struct.pack("<IiiHHIIiiII", 40, n, 2 * n, 1, 32, 0, len(image) + len(mask), 0, 0, 0, 0)
    data = header + image + mask
    directory = struct.pack("<HHH", 0, 1, 1) + struct.pack("<BBBBHHII", n, n, 0, 0, 1, 32, len(data), 6 + 16)
    return directory + data


def jupyter_lite() -> str:
    exe = pathlib.Path(sys.executable).parent / ("jupyter-lite.exe" if sys.platform == "win32" else "jupyter-lite")
    if not exe.exists():
        raise FileNotFoundError("jupyter-lite not found: run this script with the site's .venv")
    return str(exe)


def site_size(root: pathlib.Path) -> float:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) / 1e6


def build(checks: bool = True) -> pathlib.Path:
    if checks:
        log("1. tests of odelib")
        tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(NOTEBOOKS / "tests")],
                               cwd=NOTEBOOKS)
        if tests.returncode:
            raise SystemExit("the tests of odelib fail: the site is not built")
    log("2. student files (allow-list)")
    stage, listing = stage_contents()
    log("3. Python runtime and packages")
    runtime = pyodide_runtime()
    wheels = piplite_wheels()
    log("4. JupyterLite")
    try:
        if OUTPUT.exists():
            shutil.rmtree(OUTPUT)
    except PermissionError:
        raise SystemExit(f"cannot replace {OUTPUT}: a program is using it. Stop the local web server "
                         "(the http.server window) and close any file open from that folder, then build again.")
    work = CACHE / "lite"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    command = [jupyter_lite(), "build", "--contents", str(stage), "--output-dir", str(OUTPUT)]
    for wheel in wheels:
        command += ["--piplite-wheels", str(wheel)]
    result = subprocess.run(command, cwd=work, capture_output=True, text=True)
    if result.returncode:
        raise SystemExit("jupyter lite build failed:\n" + (result.stdout + result.stderr)[-3000:])
    log("5. self-contained: runtime served by the site, no fall-back to PyPI")
    shutil.copytree(runtime, OUTPUT / "static" / "pyodide")
    config_path = OUTPUT / "jupyter-lite.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    kernel = config["jupyter-config-data"].setdefault("litePluginSettings", {}).setdefault(KERNEL_PLUGIN, {})
    kernel["pyodideUrl"] = "./static/pyodide/pyodide.mjs"     # the kernel imports Pyodide as a module
    kernel["disablePyPIFallback"] = True
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    log("6. landing page")
    (OUTPUT / "index.html").write_text(landing_page(listing), encoding="utf-8")
    (OUTPUT / "favicon.ico").write_bytes(favicon())
    log(f"-> {OUTPUT}  ({len(listing)} notebooks, {site_size(OUTPUT):.0f} MB)")
    return OUTPUT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the MATH 331 student site.")
    parser.add_argument("--no-checks", action="store_true", help="skip the tests of odelib")
    build(checks=not parser.parse_args().no_checks)
