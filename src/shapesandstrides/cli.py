"""The local catalog CLI.

A thin client over the library. It contains no measurement logic, no
correctness logic, and no metric collection — it lists, renders and
compares records that already exist.
"""

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from shapesandstrides.records import CorruptRecordError, list_runs, load_run

app = typer.Typer(add_completion=False, help="Honest correctness and timing for kernels.")
# Rich auto-detects terminal width and falls back to a narrow default when
# stdout isn't a tty (e.g. under CliRunner or when piped). That truncates
# table columns and can hide substrings a caller expects to find in output.
# Force a wide, non-interactive layout whenever we're not attached to a real
# terminal, so rendered output is stable regardless of how it's invoked.
console = Console(width=200 if not sys.stdout.isatty() else None)

ROOT = typer.Option(None, "--root", help="Record store (default: ~/.shapesandstrides)")


def _verdict(rec) -> str:
    if rec.correctness and not rec.correctness.passed:
        return "[red]INCORRECT[/red]"
    if rec.comparison is None:
        return "[dim]no timing[/dim]"
    c = rec.comparison
    if not c.is_performance_valid:
        # Tier C: the measurement was too unstable to support any verdict.
        # Withholding one is the entire point of the tier system.
        return f"[yellow]UNSTABLE[/yellow] [dim](tier {c.tier.value})[/dim]"
    if c.speedup_ci_lo > 1.2:
        return f"[green]FASTER {c.speedup:.2f}x[/green]"
    if c.speedup_ci_hi < 0.95:
        return f"[red]SLOWER {c.speedup:.2f}x[/red]"
    return f"[yellow]PARITY {c.speedup:.2f}x[/yellow]"


@app.command()
def runs(
    root: Path = ROOT,
    limit: int = typer.Option(20, "--limit", "-n"),
    as_json: bool = typer.Option(False, "--json"),
):
    """List stored runs, newest first."""
    records = list_runs(root=root, limit=limit)
    if as_json:
        print(json.dumps([r.model_dump(mode="json") for r in records], indent=2))
        return
    if not records:
        console.print("[dim]no runs stored yet[/dim]")
        return

    t = Table(box=None, pad_edge=False)
    for col in ("run", "kernel", "gpu", "shapes", "verdict"):
        t.add_column(col)
    for r in records:
        shapes = f"{r.correctness.total - r.correctness.failed_count}/{r.correctness.total}" if r.correctness else "-"
        t.add_row(r.run_id[-13:], r.kernel_name, r.device.gpu_name or "-", shapes, _verdict(r))
    console.print(t)


@app.command()
def show(run_id: str, root: Path = ROOT):
    """Show one run in detail."""
    try:
        r = load_run(run_id, root=root)
    except FileNotFoundError:
        console.print(f"[red]run not found:[/red] {run_id}")
        raise typer.Exit(1)
    except CorruptRecordError as e:
        console.print(f"[red]run record is corrupt:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[bold]{r.kernel_name}[/bold]  {r.run_id}")
    console.print(f"  gpu       {r.device.gpu_name} (sm_{(r.device.compute_capability or '').replace('.', '')})")
    console.print(f"  torch     {r.device.torch_version}  cuda {r.device.cuda_version}  triton {r.device.triton_version}")
    if r.context.sm_clock_mhz:
        console.print(
            f"  context   {r.context.sm_clock_mhz:.0f}/{r.context.max_sm_clock_mhz:.0f} MHz  "
            f"{r.context.temperature_c:.0f}C  {r.context.power_draw_w:.1f}W"
        )
    if r.correctness:
        c = r.correctness
        console.print(f"\n  [bold]correctness[/bold]  {c.total - c.failed_count}/{c.total} shapes")
        if c.minimal_failure:
            m = c.minimal_failure
            console.print(f"    [red]minimal failure[/red]  {m.spec.label}  seed {m.seed}")
            console.print(f"    [dim]reproduce with[/dim]  {c.replay_hint}")
    else:
        console.print("\n  [bold]correctness[/bold]  [dim]not run[/dim]")
    if r.comparison:
        c = r.comparison
        console.print(f"\n  [bold]performance[/bold]  {_verdict(r)}")
        console.print(f"    candidate {c.candidate.median_ms:.4f} ms   baseline {c.baseline.median_ms:.4f} ms")
        console.print(f"    95% CI    [{c.speedup_ci_lo:.3f}, {c.speedup_ci_hi:.3f}]   tier {c.tier.value}")
        if not c.is_performance_valid:
            console.print(
                f"    [yellow]tier {c.tier.value} — measurement too unstable for a performance verdict[/yellow]"
            )
    else:
        console.print("\n  [bold]performance[/bold]  [dim]no timing[/dim]")
    if r.memory.peak_allocated_bytes:
        console.print(f"\n  [bold]memory[/bold]    peak {r.memory.peak_allocated_bytes / 1024**2:.1f} MB")


@app.command()
def compare(run_a: str, run_b: str, root: Path = ROOT):
    """Compare two stored runs side by side."""
    try:
        a, b = load_run(run_a, root=root), load_run(run_b, root=root)
    except FileNotFoundError as e:
        console.print(f"[red]run not found:[/red] {e}")
        raise typer.Exit(1)
    except CorruptRecordError as e:
        console.print(f"[red]run record is corrupt:[/red] {e}")
        raise typer.Exit(1)

    if (a.device.gpu_name, a.device.compute_capability, a.device.torch_version, a.device.triton_version) != (
        b.device.gpu_name, b.device.compute_capability, b.device.torch_version, b.device.triton_version
    ):
        console.print(
            "[yellow]environments differ — these runs are not directly comparable[/yellow]\n"
        )

    t = Table(box=None)
    t.add_column(""); t.add_column(a.kernel_name); t.add_column(b.kernel_name)
    t.add_row("run", a.run_id[-13:], b.run_id[-13:])
    t.add_row("gpu", a.device.gpu_name or "-", b.device.gpu_name or "-")
    t.add_row("torch", a.device.torch_version or "-", b.device.torch_version or "-")
    t.add_row(
        "shapes",
        f"{a.correctness.total - a.correctness.failed_count}/{a.correctness.total}" if a.correctness else "-",
        f"{b.correctness.total - b.correctness.failed_count}/{b.correctness.total}" if b.correctness else "-",
    )
    t.add_row(
        "median ms",
        f"{a.comparison.candidate.median_ms:.4f}" if a.comparison else "-",
        f"{b.comparison.candidate.median_ms:.4f}" if b.comparison else "-",
    )
    t.add_row("verdict", _verdict(a), _verdict(b))
    console.print(t)


@app.command()
def rm(run_id: str, root: Path = ROOT):
    """Delete a stored run."""
    from shapesandstrides.records import _runs_dir

    p = _runs_dir(root) / f"{run_id}.json"
    if not p.exists():
        console.print(f"[red]run not found:[/red] {run_id}")
        raise typer.Exit(1)
    p.unlink()
    console.print(f"removed {run_id}")


if __name__ == "__main__":
    app()


# Exit codes are the contract. A gate that always exits 0 is a report.
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_NOTHING_FOUND = 5  # mirrors pytest's "no tests collected"


def _oracle_cell(report) -> str:
    """Tier first, because it is the part that says how much the verdict means.

    A tier-C pass and a tier-A pass are different claims, and a reader who sees
    only the label will assume the stronger one.
    """
    return f"{report.oracle_tier.value}:{report.oracle_kind.value}:{report.oracle_label}"


def _verify_json(name: str, report, error: str | None) -> dict:
    """One kernel's result, machine-readable.

    `verdict` is a three-way string rather than a boolean: ERROR (never
    adjudicated) is not INCORRECT (adjudicated and wrong), and collapsing them
    would violate rule 7. An ERROR carries no tier because no verdict exists
    to grade.

    Deliberately omits report.replay_hint. minimal_failure already carries the
    shape and the seed as structured fields, so repeating them as one string
    would be redundant here. The hint exists for the human path, where a single
    line reads better than two fields.

    Note this used to omit a `replay_command` that named `shapesandstrides
    replay`, a subcommand that does not exist. That string is gone from every
    path now, not just this one.
    """
    if error is not None:
        return {
            "kernel": name, "verdict": "ERROR",
            "oracle_tier": None, "oracle_kind": None, "oracle_label": None,
            "checks": [], "correctness_valid": False,
            "shapes_passed": 0, "shapes_total": 0,
            "minimal_failure": None, "error": error,
        }
    return {
        "kernel": name,
        "verdict": "CORRECT" if report.passed else "INCORRECT",
        "oracle_tier": report.oracle_tier.value,
        "oracle_kind": report.oracle_kind.value,
        "oracle_label": report.oracle_label,
        "checks": [c.value for c in report.checks],
        "correctness_valid": report.is_correctness_valid,
        "shapes_passed": report.total - report.failed_count,
        "shapes_total": report.total,
        "minimal_failure": (
            report.minimal_failure.model_dump(mode="json")
            if report.minimal_failure else None
        ),
        "error": None,
    }


@app.command()
def verify(
    target: Path = typer.Argument(
        Path("."), help="File or directory to scan for @verify-marked kernels."
    ),
    device: str = typer.Option(None, "--device", help="Override the decorator's device."),
    tier: str = typer.Option(None, "--tier", help="Override the shape tier (fast/canonical/full)."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only print failures."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable output."),
):
    """Verify every @verify-marked kernel under TARGET.

    Exits 1 if any kernel fails, 5 if none were found, 0 otherwise. The exit
    code is the point: it is what lets this block a merge rather than produce
    a report nobody reads.
    """
    from shapesandstrides.shapes import ShapeTier
    from shapesandstrides.verify import discover_in_path, spec_of, verify_kernel

    def _emit_nothing_found(message: str) -> None:
        """Exit 5, in whichever format the caller asked for.

        Under --json this still has to be JSON. Printing prose on stdout would
        force whoever is parsing it to read an error out of a sentence.
        """
        if as_json:
            print(json.dumps(
                {"kernels": [], "device": device or "?", "failed": 0,
                 "exit_code": EXIT_NOTHING_FOUND, "message": message},
                indent=2,
            ))
        else:
            console.print(message)
        raise typer.Exit(EXIT_NOTHING_FOUND)

    try:
        found = discover_in_path(target)
    except FileNotFoundError as e:
        _emit_nothing_found(f"[red]{e}[/red]" if not as_json else str(e))

    if not found:
        # Deliberately not a pass. Pointing the gate at the wrong path and
        # getting a green tick is the worst possible outcome here.
        _emit_nothing_found(
            f"No @verify-marked kernels found under {target}. "
            f"Mark a kernel with @verify(against=...) so it can be checked."
            if as_json else
            f"[yellow]No @verify-marked kernels found under {target}.[/yellow]\n"
            f"[dim]Mark a kernel with @verify(against=...) so it can be checked.[/dim]"
        )

    resolved_tier = ShapeTier(tier) if tier else None

    # Run everything first, render second. The two output formats then cannot
    # drift apart, and neither can decide a verdict the other would not.
    results = []
    for name, fn in found:
        try:
            results.append((name, verify_kernel(fn, device=device, tier=resolved_tier), None))
        except Exception as e:
            # The kernel never got adjudicated at all: no report, so no tier
            # and no verdict. ERROR, not INCORRECT (rule 7).
            results.append((name, None, f"{type(e).__name__}: {e}"))

    failures = sum(1 for _, r, err in results if err is not None or not r.passed)
    dev = device or spec_of(found[0][1]).device
    exit_code = EXIT_FAILED if failures else EXIT_OK

    if as_json:
        print(json.dumps({
            "kernels": [_verify_json(name, r, err) for name, r, err in results],
            "device": dev,
            "failed": failures,
            "exit_code": exit_code,
        }, indent=2))
        raise typer.Exit(exit_code)

    table = Table(box=None, pad_edge=False)
    table.add_column("kernel")
    table.add_column("verdict")
    table.add_column("shapes", justify="right")
    table.add_column("oracle")
    table.add_column("minimal failing case")

    for name, report, err in results:
        if err is not None:
            table.add_row(name, "[red]ERROR[/red]", "-", "-", err)
            continue
        if report.passed and quiet:
            continue
        table.add_row(
            name,
            "[green]CORRECT[/green]" if report.passed else "[red]INCORRECT[/red]",
            f"{report.total - report.failed_count}/{report.total}",
            _oracle_cell(report),
            "" if report.passed
            else (report.minimal_failure.spec.label if report.minimal_failure else "?"),
        )

    console.print(table)
    console.print(
        f"\n{len(found)} kernel(s) on device={dev}, "
        f"[{'red' if failures else 'green'}]{failures} failed[/]"
    )
    raise typer.Exit(exit_code)
