"""Shared by the make_NN.py scripts: build a course notebook from readable cell sources.

    nb = Builder("09_Slope_Fields.ipynb", __doc__)
    nb.md(r'''...''')          # a text cell
    nb.code(r'''...''')        # a code cell
    nb.main()                  # write and execute (--no-run: write only)

Rules enforced here: plain ASCII sources and plain ASCII file; fixed cell ids (so a rebuild
changes only what was edited); the notebook is executed and its outputs are stored, as for
notebooks 01-07, so the plots can be read without a running kernel.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import textwrap

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NOTEBOOKS = pathlib.Path(__file__).resolve().parent.parent
TIMEOUT_SECONDS = 600
ELLIPSIS = chr(0x2026)          # ipywidgets describes a slider with an ellipsis


# --- help for a student who has never used a notebook (review of 2026-11-17) ---------------
# Under the heading of the exercises: what to do before the first answer.
EXERCISES_HEADING = re.compile(r"\A## \d+\. Exercises\n\n.+?\n\n", re.S)       # the heading and its first paragraph
EXERCISES_NOTE = ("**Before you start:** run the whole notebook (the `>>` button at the top, then *Confirm Kernel "
                  "Restart*; or the menu *Run > Run All Cells*), every time you open it: the exercises use what the\n"
                  "cells above define. In an answer cell change only the lines above the line that starts with "
                  "`# ----`, then press Shift+Enter. If Python does not accept what you type, see \"How to type\n"
                  "mathematics\" in notebook 00, or click `+` at the top, type `ol.guide(\"typing\")` in the new "
                  "cell and press Shift+Enter.\n\n")
# In an answer cell: where the student's lines end and the checking code begins.
ANSWER_LINE = re.compile(r"my_\w+ = None\s*(#.*)?")      # the whole line
ANSWER_MARKER = "# ---- your answers go above this line; the lines below check them (nothing to change) ----"


def with_exercises_note(text: str) -> str:
    """The text of a cell, with the note for beginners after the first paragraph of "N. Exercises"."""
    found = EXERCISES_HEADING.match(text)
    return text if found is None else text[:found.end()] + EXERCISES_NOTE + text[found.end():]


def with_answer_marker(source: str) -> str:
    """The source of a code cell, with a dashed comment after its first lines "my_... = None"."""
    lines = source.split("\n")
    count = 0
    while count < len(lines) and ANSWER_LINE.fullmatch(lines[count]):
        count += 1
    if count == 0 or not any(line.strip() for line in lines[count:]):
        return source                                    # not an answer cell, or nothing is checked in it
    return "\n".join(lines[:count] + [ANSWER_MARKER] + lines[count:])


class Builder:
    def __init__(self, file_name: str, description: str = ""):
        self.target = NOTEBOOKS / file_name
        self.description = description
        self.cell_id = "nb" + file_name.split("_")[0] + "-{:02d}"       # nb08-00, nb02E-00, ...
        self.cells: list[tuple[str, str]] = []
        self.no_output: set[int] = set()                 # cells whose output is not stored

    def md(self, text: str) -> None:
        self.cells.append(("markdown", with_exercises_note(textwrap.dedent(text).strip("\n"))))

    def code(self, text: str, keep_output: bool = True) -> None:
        """A code cell. ``keep_output=False``: its output is not stored (it would give an answer away)."""
        if not keep_output:
            self.no_output.add(len(self.cells))
        self.cells.append(("code", with_answer_marker(textwrap.dedent(text).strip("\n"))))

    def build(self) -> nbformat.NotebookNode:
        nb = new_notebook()
        nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
        nb.metadata["language_info"] = {"name": "python"}
        for index, (kind, source) in enumerate(self.cells):
            if not source.isascii():
                raise ValueError(f"cell {index} is not plain ASCII")
            cell = new_markdown_cell(source) if kind == "markdown" else new_code_cell(source)
            cell["id"] = self.cell_id.format(index)
            nb.cells.append(cell)
        return nb

    def main(self) -> None:
        parser = argparse.ArgumentParser(description=self.description,
                                         formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument("--no-run", action="store_true", help="write the notebook without executing it")
        args = parser.parse_args()
        nb = self.build()
        if not args.no_run:
            from nbclient import NotebookClient
            NotebookClient(nb, timeout=TIMEOUT_SECONDS, kernel_name="python3",
                           resources={"metadata": {"path": str(NOTEBOOKS)}}).execute()
        for index in self.no_output:
            nb.cells[index]["outputs"] = []
            nb.cells[index]["execution_count"] = None
        for cell in nb.cells:
            for output in cell.get("outputs", []):
                plain = output.get("data", {}).get("text/plain")
                if plain is not None:
                    output["data"]["text/plain"] = plain.replace(ELLIPSIS, "...")
        nbformat.validate(nb)
        nbformat.write(nb, self.target)
        if not self.target.read_text(encoding="utf-8").isascii():
            raise ValueError(f"{self.target.name} is not plain ASCII")
        print(f"wrote {self.target.name}: {len(nb.cells)} cells" + ("" if args.no_run else ", executed"))
