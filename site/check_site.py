"""
Automatic check of the built student site: every notebook is opened in a real browser (headless),
all its cells are run by the Python that runs in the browser, and the check fails if a cell stops
with an error, if a notebook does not finish, or if the page asks any other host for a file.

    .venv\\Scripts\\python check_site.py                  # all notebooks, with Microsoft Edge
    .venv\\Scripts\\python check_site.py 08 13            # only the notebooks whose name starts so
    .venv\\Scripts\\python check_site.py --browser chrome

It uses a browser already installed on this computer (Edge or Chrome): nothing is downloaded.
The site is served by serve_site.py on a free port of this computer for the time of the check.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import socket
import subprocess
import sys
import time
from urllib.parse import urlparse

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).resolve().parent
SITE = HERE / "_output"
FILES = SITE / "files"
BROWSERS = {"edge": "msedge", "chrome": "chrome"}
SECOND = 1000                               # Playwright counts in milliseconds
START_TIMEOUT = 120 * SECOND                # the notebook page and its toolbar
RUN_TIMEOUT = 420 * SECOND                  # Python start, packages and all cells of one notebook
SERVER_WAIT = 15.0                          # seconds
POLL = 1.0                                  # seconds between two looks at the page
RUN_ALL = "Restart the kernel and run all cells"
ERROR_WORDS = ("Traceback (most recent call last)", "Error:")

# What the page is asked after "run all": the prompts of the code cells and the red outputs.
STATE = """
() => {
  // an empty code cell (a place for the student to type) is never run: it is not counted
  const cells = [...document.querySelectorAll('.jp-CodeCell')].filter(
      c => (c.querySelector('.cm-content')?.textContent || '').trim() !== '');
  const prompts = cells.map(c => (c.querySelector('.jp-InputPrompt')?.textContent || '').trim());
  const red = [...document.querySelectorAll(
      '.jp-OutputArea-output[data-mime-type="application/vnd.jupyter.stderr"]')].map(e => e.textContent);
  return {prompts, red, images: document.querySelectorAll('.jp-OutputArea img').length,
          sliders: document.querySelectorAll('[role=slider]').length};
}
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(port: int) -> subprocess.Popen:
    server = subprocess.Popen([sys.executable, str(HERE / "serve_site.py"), "--port", str(port)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + SERVER_WAIT
    while time.monotonic() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return server
        time.sleep(POLL / 4)
    server.terminate()
    raise SystemExit("the local web server did not start")


def code_cells(notebook: pathlib.Path) -> int:
    """The code cells that "run all" executes: an empty one (a place for the student to type) is not."""
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    return sum(cell["cell_type"] == "code" and bool("".join(cell["source"]).strip()) for cell in cells)


def run_notebook(page, base: str, name: str, expected_cells: int) -> dict:
    """Open one notebook, run all cells, wait until every cell has run; return what was seen."""
    foreign: list[str] = []
    page.on("request", lambda request: foreign.append(request.url)
            if urlparse(request.url).scheme in ("http", "https") and not request.url.startswith(base) else None)
    page.goto(f"{base}/notebooks/index.html?path={name}", timeout=START_TIMEOUT)
    page.get_by_title(RUN_ALL).first.click(timeout=START_TIMEOUT)
    page.get_by_role("button", name="Confirm Kernel Restart").click(timeout=START_TIMEOUT)

    started = time.monotonic()
    seen_running = False
    state: dict = {}
    while (time.monotonic() - started) * SECOND < RUN_TIMEOUT:
        time.sleep(POLL)
        state = page.evaluate(STATE)
        prompts = state["prompts"]
        waiting = [p for p in prompts if p in ("[ ]:", "[*]:")]
        seen_running = seen_running or bool(waiting)           # the stored prompts were cleared
        if seen_running and not waiting and len(prompts) == expected_cells:
            break
        if seen_running and any(word in text for text in state["red"] for word in ERROR_WORDS):
            break                                              # a cell failed: the others will not run
    else:
        state["problem"] = "did not finish in time"
    state["seconds"] = time.monotonic() - started
    state["foreign"] = sorted(set(foreign))
    errors = [text[:300] for text in state.get("red", []) if any(word in text for word in ERROR_WORDS)]
    if errors:
        state["problem"] = "a cell stopped with an error: " + errors[0].strip().splitlines()[-1]
    elif state["foreign"]:
        state["problem"] = "the page asked another host: " + state["foreign"][0]
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prefixes", nargs="*", help="only the notebooks whose file name starts with one of these")
    parser.add_argument("--browser", choices=sorted(BROWSERS), default="edge")
    args = parser.parse_args()
    if not (SITE / "index.html").exists():
        raise SystemExit(f"no site in {SITE}: run build_site.py first")
    notebooks = [p for p in sorted(FILES.glob("[0-9][0-9]*_*.ipynb"))
                 if code_cells(p) and (not args.prefixes or p.name.startswith(tuple(args.prefixes)))]
    if not notebooks:
        raise SystemExit("no notebook to check")

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    server = start_server(port)
    failed = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel=BROWSERS[args.browser], headless=True)
            for notebook in notebooks:
                context = browser.new_context()              # a fresh profile: a first visit, every time
                page = context.new_page()
                try:
                    state = run_notebook(page, base, notebook.name, code_cells(notebook))
                except PlaywrightTimeout as problem:
                    state = {"problem": f"the page did not respond ({str(problem).splitlines()[0]})",
                             "prompts": [], "images": 0, "sliders": 0, "seconds": 0.0}
                context.close()
                verdict = "FAILED  " + state["problem"] if "problem" in state else "ok"
                print(f"{notebook.name:36s} {len(state['prompts']):2d} cells  {state['images']:2d} plots  "
                      f"{state['sliders']:2d} sliders  {state['seconds']:5.0f} s  {verdict}", flush=True)
                if "problem" in state:
                    failed.append(notebook.name)
            browser.close()
    finally:
        server.terminate()
    print(f"{len(notebooks) - len(failed)} of {len(notebooks)} notebooks run in the browser without errors")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
