"""The nav is a promise that a page exists. Assert it keeps the promise.

A nav entry pointing at a missing file does not fail the build by default --
MkDocs renders it as a dead link and moves on -- so a typo ships silently.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_AREAS = [
    "Home",
    "Getting started",
    "Correctness",
    "Performance",
    "Numeric formats",
    "API reference",
    "Background",
]


class _IgnoreUnknownTags(yaml.SafeLoader):
    """MkDocs Material configs carry `!!python/name:` tags in some fields.

    ``safe_load`` raises on those. Ignoring unknown tags is the narrow fix;
    ``unsafe_load`` would execute whatever the file names, in a test that only
    wants to read a list of page paths.
    """


_IgnoreUnknownTags.add_multi_constructor("", lambda loader, suffix, node: None)


def _config() -> dict:
    return yaml.load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"), _IgnoreUnknownTags)


def _walk(entry):
    if isinstance(entry, str):
        yield entry
    elif isinstance(entry, dict):
        for value in entry.values():
            yield from _walk(value)
    elif isinstance(entry, list):
        for value in entry:
            yield from _walk(value)


def test_nav_is_grouped_into_the_expected_areas():
    nav = _config()["nav"]
    areas = [next(iter(e)) if isinstance(e, dict) else e for e in nav]
    assert areas == EXPECTED_AREAS


def test_every_nav_page_exists_on_disk():
    missing = [p for p in _walk(_config()["nav"]) if not (ROOT / "docs" / p).is_file()]
    assert missing == [], f"nav references missing pages: {missing}"


def test_correctness_hub_links_every_page_beside_it():
    """The hub is the entry point, so an unlinked sibling is unreachable.

    Asserted against what is on disk rather than a hardcoded list, so each rung
    of the ladder is forced to add its own row when it adds its own page --
    and the test never promises a page that does not exist yet.
    """
    correctness = ROOT / "docs" / "correctness"
    hub = (correctness / "index.md").read_text(encoding="utf-8")

    siblings = sorted(p.name for p in correctness.glob("*.md") if p.name != "index.md")
    unlinked = [name for name in siblings if name not in hub]
    assert unlinked == [], f"correctness hub does not link: {unlinked}"
