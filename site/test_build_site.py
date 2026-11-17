"""Tests of the parts of build_site.py that do not need the network: corrections and the home page.

    .venv/Scripts/python -m pytest test_build_site.py        (from the site folder)
"""

import json

import pytest

import build_site as site

NAME = "10_Separable_Equations.ipynb"
ENTRY = {"revision": 2, "date": "2027-02-03", "note": "Exercise 2(b): the sign of the answer was wrong."}


def test_a_notebook_keeps_its_name_until_it_is_corrected():
    assert site.published_name(NAME, {}) == NAME
    assert site.published_name(NAME, {"10": {"revision": 1}}) == NAME
    assert site.published_name(NAME, {"10": ENTRY}) == "10_Separable_Equations_r2.ipynb"
    assert site.published_name(NAME, {"10": {**ENTRY, "revision": 7}}) == "10_Separable_Equations_r7.ipynb"
    assert site.published_name("02E_Decay_Chains.ipynb", {"02": ENTRY}) == "02E_Decay_Chains.ipynb"      # 02 is not 02E
    assert site.NOTEBOOK_NAME.fullmatch(site.published_name(NAME, {"10": ENTRY}))                     # still on the allow-list


def test_links_to_a_corrected_notebook_are_rewritten_everywhere_else():
    nb = {"cells": [{"cell_type": "markdown", "source": ["See [separable equations](10_Separable_Equations.ipynb) and\n",
                                                         "[the guide](00_Index.ipynb).\n"]},
                    {"cell_type": "code", "source": ["name = '](10_Separable_Equations.ipynb)'\n"]}]}
    names = {NAME: "10_Separable_Equations_r2.ipynb", "00_Index.ipynb": "00_Index.ipynb"}
    changed = site.with_published_links(nb, names)
    text = "".join(changed["cells"][0]["source"])
    assert "](10_Separable_Equations_r2.ipynb)" in text and "](00_Index.ipynb)" in text
    assert "](10_Separable_Equations.ipynb)" not in text
    assert changed["cells"][1]["source"] == ["name = '](10_Separable_Equations.ipynb)'\n"]              # code is not touched


def test_revisions_file_is_checked(tmp_path):
    path = tmp_path / "revisions.json"
    assert site.read_revisions(path) == {}                                                            # no file: no corrections
    path.write_text(json.dumps({"notebooks": {"10": ENTRY}}), encoding="utf-8")
    assert site.read_revisions(path) == {"10": ENTRY}
    for wrong in ({"revision": 0}, {"revision": "2"}, {"revision": 2.0}, {"revision": True},
                  {"revision": 2}, {"revision": 2, "date": "2027-02-03"}, {"revision": 2, "date": " ", "note": "x"}):
        path.write_text(json.dumps({"notebooks": {"10": wrong}}), encoding="utf-8")
        with pytest.raises(SystemExit):
            site.read_revisions(path)
    assert site.read_revisions() == {} or all(entry["revision"] >= 1 for entry in site.read_revisions().values())


def test_the_home_page_says_what_was_corrected():
    plain = {"file": NAME, "source": NAME, "title": "10 - Separable equations", "sessions": "Session 4", "correction": None}
    row = site.notebook_row(plain)
    assert "?path=10_Separable_Equations.ipynb" in row and "Corrected" not in row
    corrected = {**plain, "file": "10_Separable_Equations_r2.ipynb", "correction": ENTRY}
    row = site.notebook_row(corrected)
    assert "?path=10_Separable_Equations_r2.ipynb" in row
    assert "Corrected on 2027-02-03: Exercise 2(b): the sign of the answer was wrong." in row
    assert "your earlier copy is still in your browser" in row


def test_the_guide_of_the_home_page_comes_from_the_library():
    parts = site.guide_parts()
    assert set(parts) == {"QUICK_START", "TYPING", "TYPING_RULE", "ERRORS", "HOW", "WORDS"}
    assert parts["QUICK_START"].count("<li>") == len(site.guide.GUIDE_START)
    assert parts["ERRORS"].count("<tr>") == len(site.guide.GUIDE_ERRORS) + 1
    page = site.LANDING.read_text(encoding="utf-8")
    for name in list(parts) + ["NOTEBOOKS"]:
        assert page.count(f"<!--{name}-->") == 1
