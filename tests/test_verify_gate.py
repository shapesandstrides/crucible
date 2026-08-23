"""The gate: exit codes from the CLI, and collection from pytest.

The plugin is never enabled explicitly in these tests. It is registered as a
pytest11 entry point, so installing the package has to be sufficient — and
passing -p would both hide that and double-register the plugin.

A verification tool that always exits 0 is a report. The exit code is the
product feature here, so it gets tested as carefully as the numerics.
"""

import textwrap

import pytest
from typer.testing import CliRunner

torch = pytest.importorskip("torch")

from shapesandstrides.cli import (  # noqa: E402
    EXIT_FAILED,
    EXIT_NOTHING_FOUND,
    EXIT_OK,
    app,
)

pytest_plugins = ["pytester"]

runner = CliRunner()

GOOD = """
from shapesandstrides.verify import verify

@verify(against="torch.add", dtypes=["float32"], device="cpu")
def good_add(a, b):
    return a + b
"""

BAD = """
from shapesandstrides.verify import verify

@verify(against="torch.add", dtypes=["float32"], device="cpu")
def subtracts_instead(a, b):
    return a - b
"""


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


# ------------------------------------------------------------------- the CLI


def test_all_passing_exits_zero(tmp_path):
    _write(tmp_path, "my_kernels.py", GOOD)
    res = runner.invoke(app, ["verify", str(tmp_path), "--device", "cpu"])
    assert res.exit_code == EXIT_OK, res.output
    assert "CORRECT" in res.output
    assert "good_add" in res.output


def test_a_failing_kernel_exits_nonzero(tmp_path):
    _write(tmp_path, "my_kernels.py", BAD)
    res = runner.invoke(app, ["verify", str(tmp_path), "--device", "cpu"])
    assert res.exit_code == EXIT_FAILED, res.output
    assert "INCORRECT" in res.output


def test_one_failure_among_several_still_fails_the_run(tmp_path):
    _write(tmp_path, "a_kernels.py", GOOD)
    _write(tmp_path, "b_kernels.py", BAD)
    res = runner.invoke(app, ["verify", str(tmp_path), "--device", "cpu"])
    assert res.exit_code == EXIT_FAILED
    assert "1 failed" in res.output


def test_the_minimal_failing_case_is_reported(tmp_path):
    _write(tmp_path, "my_kernels.py", BAD)
    res = runner.invoke(app, ["verify", str(tmp_path), "--device", "cpu"])
    # A verdict with no reproducible case is not actionable.
    assert "float32" in res.output


def test_finding_nothing_is_not_a_pass(tmp_path):
    """Pointing the gate at the wrong directory and getting a green tick is
    the worst available outcome, so an empty scan is its own exit code."""
    _write(tmp_path, "my_kernels.py", "x = 1\n")
    res = runner.invoke(app, ["verify", str(tmp_path), "--device", "cpu"])
    assert res.exit_code == EXIT_NOTHING_FOUND
    assert "No @verify-marked kernels" in res.output


def test_missing_path_is_reported_not_silently_passed(tmp_path):
    res = runner.invoke(app, ["verify", str(tmp_path / "nope")])
    assert res.exit_code == EXIT_NOTHING_FOUND


def test_a_kernel_that_raises_is_an_error_not_a_pass(tmp_path):
    _write(
        tmp_path,
        "my_kernels.py",
        """
        from shapesandstrides.verify import verify

        @verify(against="torch.add", dtypes=["float32"], device="cpu")
        def explodes(a, b):
            raise RuntimeError("boom")
        """,
    )
    res = runner.invoke(app, ["verify", str(tmp_path), "--device", "cpu"])
    assert res.exit_code == EXIT_FAILED
    # The kernel raising on every shape is INCORRECT, not a crash of the tool.
    assert "INCORRECT" in res.output or "ERROR" in res.output


def test_quiet_hides_passes_but_still_exits_zero(tmp_path):
    _write(tmp_path, "my_kernels.py", GOOD)
    res = runner.invoke(app, ["verify", str(tmp_path), "--device", "cpu", "-q"])
    assert res.exit_code == EXIT_OK
    assert "good_add" not in res.output


# -------------------------------------------------------------- pytest plugin


def test_plugin_collects_a_marked_kernel_and_passes(pytester):
    pytester.makepyfile(my_kernels=textwrap.dedent(GOOD))
    res = pytester.runpytest("--sas-device", "cpu")
    res.assert_outcomes(passed=1)


def test_plugin_fails_a_broken_kernel_with_a_replay_line(pytester):
    pytester.makepyfile(my_kernels=textwrap.dedent(BAD))
    res = pytester.runpytest("--sas-device", "cpu")
    res.assert_outcomes(failed=1)
    res.stdout.fnmatch_lines(["*is INCORRECT*"])
    res.stdout.fnmatch_lines(["*replay*shapesandstrides replay*"])


def test_plugin_ignores_ordinary_test_files(pytester):
    """The plugin must never shadow normal test collection."""
    pytester.makepyfile(
        test_kernels=textwrap.dedent(
            """
            def test_something():
                assert True
            """
        )
    )
    res = pytester.runpytest()
    res.assert_outcomes(passed=1)


def test_plugin_pattern_is_configurable(pytester):
    pytester.makepyfile(weird_name=textwrap.dedent(GOOD))
    pytester.makeini(
        """
        [pytest]
        sas_kernel_files = weird_name.py
        """
    )
    res = pytester.runpytest("--sas-device", "cpu")
    res.assert_outcomes(passed=1)
