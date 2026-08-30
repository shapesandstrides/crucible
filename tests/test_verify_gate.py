"""The gate: exit codes from the CLI, and collection from pytest.

The plugin is never enabled explicitly in these tests. It is registered as a
pytest11 entry point, so installing the package has to be sufficient — and
passing -p would both hide that and double-register the plugin.

A verification tool that always exits 0 is a report. The exit code is the
product feature here, so it gets tested as carefully as the numerics.
"""

import json
import textwrap

import pytest
from typer.testing import CliRunner

torch = pytest.importorskip("torch")

from shapesandstrides.cli import (  # noqa: E402
    EXIT_FAILED,
    EXIT_NOTHING_FOUND,
    EXIT_OK,
    _verify_json,
    app,
)
from shapesandstrides.correctness import CorrectnessReport  # noqa: E402
from shapesandstrides.reference import OracleKind  # noqa: E402
from shapesandstrides.types import CheckKind, OracleTier  # noqa: E402

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


def test_plugin_fails_a_broken_kernel_with_a_reproduce_line(pytester):
    pytester.makepyfile(my_kernels=textwrap.dedent(BAD))
    res = pytester.runpytest("--sas-device", "cpu")
    res.assert_outcomes(failed=1)
    res.stdout.fnmatch_lines(["*is INCORRECT*"])
    res.stdout.fnmatch_lines(["*reproduce*shape=*seed=*"])
    assert "shapesandstrides replay" not in res.stdout.str()


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


# --------------------------------------------------- the tier, and --json
#
# `verify` is the command an agent runs most, so it is the one that must not
# have to be screen-scraped out of a coloured table. And whatever it prints has
# to carry the tier: a PASS that does not say what adjudicated it invites the
# reader to assume the strongest reading.


def test_the_human_table_shows_the_oracle_tier(tmp_path):
    _write(tmp_path, "my_kernels.py", GOOD)
    res = runner.invoke(app, ["verify", str(tmp_path), "--device", "cpu"])
    assert res.exit_code == EXIT_OK, res.output
    assert "A:" in res.output, (
        f"the oracle column must carry the tier letter, got:\n{res.output}"
    )


def test_json_output_is_parseable_and_carries_the_tier(tmp_path):
    _write(tmp_path, "my_kernels.py", GOOD)
    res = runner.invoke(app, ["verify", str(tmp_path), "--device", "cpu", "--json"])
    assert res.exit_code == EXIT_OK, res.output
    payload = json.loads(res.output)
    assert payload["failed"] == 0
    assert payload["exit_code"] == EXIT_OK
    k = payload["kernels"][0]
    assert k["kernel"] == "good_add"
    assert k["verdict"] == "CORRECT"
    assert k["oracle_tier"] == "A"
    assert k["oracle_kind"] == "torch_op"
    assert k["oracle_label"] == "torch.add"
    assert k["checks"] == ["reference"]
    assert k["shapes_passed"] == k["shapes_total"]
    assert k["minimal_failure"] is None
    assert k["error"] is None


def test_json_output_reports_a_failure_with_its_minimal_case(tmp_path):
    _write(tmp_path, "my_kernels.py", BAD)
    res = runner.invoke(app, ["verify", str(tmp_path), "--device", "cpu", "--json"])
    assert res.exit_code == EXIT_FAILED, res.output
    payload = json.loads(res.output)
    assert payload["failed"] == 1
    k = payload["kernels"][0]
    assert k["verdict"] == "INCORRECT"
    assert k["minimal_failure"] is not None
    assert k["minimal_failure"]["spec"]["label"]


def test_json_distinguishes_an_error_from_an_incorrect_result(tmp_path):
    """Rule 7: ERROR and INCORRECT are different things, so a boolean verdict
    would be a lie. A kernel that raises did not produce a wrong answer."""
    _write(tmp_path, "my_kernels.py", """
        from shapesandstrides.verify import verify

        @verify(against="torch.add", dtypes=["float32"], device="cpu")
        def explodes(a, b):
            raise RuntimeError("boom")
    """)
    res = runner.invoke(app, ["verify", str(tmp_path), "--device", "cpu", "--json"])
    assert res.exit_code == EXIT_FAILED, res.output
    k = json.loads(res.output)["kernels"][0]
    assert k["verdict"] == "INCORRECT", "a raising kernel is still a wrong answer per shape"
    assert k["oracle_tier"] == "A"


def test_json_stays_json_when_nothing_was_found(tmp_path):
    """Exit 5 already says 'nothing found'. Emitting prose on stdout under
    --json would make an agent parse an error out of a sentence."""
    _write(tmp_path, "not_kernels.py", "x = 1\n")
    res = runner.invoke(app, ["verify", str(tmp_path), "--device", "cpu", "--json"])
    assert res.exit_code == EXIT_NOTHING_FOUND
    payload = json.loads(res.output)
    assert payload["kernels"] == []
    assert payload["exit_code"] == EXIT_NOTHING_FOUND


def test_plugin_failure_output_names_the_oracle_tier(pytester):
    """The pytest path must not be the honest one's poor relation: a developer
    reading a failure here needs to know what adjudicated it too."""
    pytester.makepyfile(my_kernels=textwrap.dedent(BAD))
    res = pytester.runpytest("--sas-device", "cpu")
    res.assert_outcomes(failed=1)
    res.stdout.fnmatch_lines(["*oracle*A:torch_op:torch.add*"])


def test_json_exposes_whether_the_verdict_is_a_correctness_claim(tmp_path):
    """An agent should not have to know the tier semantics to act on this."""
    _write(tmp_path, "my_kernels.py", GOOD)
    res = runner.invoke(app, ["verify", str(tmp_path), "--device", "cpu", "--json"])
    assert json.loads(res.output)["kernels"][0]["correctness_valid"] is True


def test_json_does_not_advertise_a_replay_command(tmp_path):
    """`shapesandstrides replay` does not exist and is no longer printed
    anywhere. The shape label and seed under minimal_failure carry the same
    information without promising a command that fails. This test guards the
    machine-readable path specifically, because an agent will act on a command
    string it is handed."""
    _write(tmp_path, "my_kernels.py", BAD)
    res = runner.invoke(app, ["verify", str(tmp_path), "--device", "cpu", "--json"])
    k = json.loads(res.output)["kernels"][0]
    assert "replay_command" not in k
    assert "replay_hint" not in k
    assert "shapesandstrides replay" not in res.output
    assert k["minimal_failure"]["seed"] is not None
    assert k["minimal_failure"]["spec"]["label"]


def test_every_json_kernel_entry_has_the_same_keys():
    """A schema that changes shape per outcome forces defensive .get() calls on
    the reader, and the ERROR row is exactly the one nobody tests against.

    Unit-tested rather than driven through the CLI because an ERROR entry is
    defensive: verify_kernel only raises if check() fails outside its per-shape
    try/except, which no realistic kernel triggers. Untestable end to end is
    not the same as unreachable, so the shape still has to be pinned.
    """
    report = CorrectnessReport(
        oracle_kind=OracleKind.TORCH_OP,
        oracle_label="torch.add",
        oracle_tier=OracleTier.A,
        checks=[CheckKind.REFERENCE],
        passed=True, total=4, failed_count=0,
    )
    entries = [
        _verify_json("ok", report, None),
        _verify_json("bad", report.model_copy(update={"passed": False, "failed_count": 1}), None),
        _verify_json("boom", None, "RuntimeError: boom"),
    ]
    assert [e["verdict"] for e in entries] == ["CORRECT", "INCORRECT", "ERROR"]
    shapes = {frozenset(e) for e in entries}
    assert len(shapes) == 1, f"entries disagree on their keys: {shapes}"


def test_an_error_entry_claims_no_correctness():
    """No tier, and not a valid verdict: nothing adjudicated this kernel at
    all, which is a different thing from adjudicating it and finding it wrong."""
    e = _verify_json("boom", None, "RuntimeError: boom")
    assert e["oracle_tier"] is None
    assert e["correctness_valid"] is False
    assert e["error"] == "RuntimeError: boom"
