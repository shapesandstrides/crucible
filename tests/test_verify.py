"""The one-line surface: a decorator, discovery, and a verdict.

Adoption is the whole strategy, so the cost of adding this to an existing
codebase has to be one line that does not change how the kernel behaves.
"""

import sys
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from shapesandstrides.shapes import ShapeTier  # noqa: E402
from shapesandstrides.verify import (  # noqa: E402
    VerifySpec,
    discover_in_module,
    discover_in_path,
    is_marked,
    spec_of,
    verify,
    verify_kernel,
)


# ------------------------------------------------------------- the decorator


def test_decorator_does_not_change_the_function():
    """A kernel must stay callable exactly as before. If adding verification
    alters behaviour, nobody will put it on a production kernel."""

    @verify(against="torch.add")
    def add(a, b):
        return a + b

    out = add(torch.ones(3), torch.ones(3))
    assert torch.equal(out, torch.full((3,), 2.0))
    assert add.__name__ == "add"


def test_decorator_attaches_a_spec():
    @verify(against="torch.add", dtypes=["float32"])
    def add(a, b):
        return a + b

    assert is_marked(add)
    s = spec_of(add)
    assert isinstance(s, VerifySpec)
    assert s.against == "torch.add"
    assert s.dtypes == ("float32",)


def test_unmarked_function_has_no_spec():
    def plain(a, b):
        return a + b

    assert not is_marked(plain)
    assert spec_of(plain) is None


def test_op_name_is_derived_from_a_torch_operator():
    """op_name drives tolerance selection, so a sensible default matters.
    'torch.add' should mean the 'add' tolerance profile without being told."""

    @verify(against="torch.add")
    def my_fused_thing(a, b):
        return a + b

    assert spec_of(my_fused_thing).resolved_op_name() == "add"


def test_op_name_falls_back_to_the_function_name_for_an_expression():
    @verify(against=lambda a, b: a + b)
    def rmsnorm(a, b):
        return a + b

    assert spec_of(rmsnorm).resolved_op_name() == "rmsnorm"


def test_explicit_op_name_wins():
    @verify(against="torch.add", op_name="gemm")
    def f(a, b):
        return a + b

    assert spec_of(f).resolved_op_name() == "gemm"


def test_decorator_rejects_a_bad_reference_at_decoration_time():
    """Fail when the file is imported, not hours later inside CI."""
    with pytest.raises(Exception):

        @verify(against="torch.no_such_operator")
        def f(a, b):
            return a + b


# --------------------------------------------------------------- running it


def test_verify_kernel_returns_a_report():
    @verify(against="torch.add", dtypes=["float32"], device="cpu")
    def add(a, b):
        return a + b

    r = verify_kernel(add)
    assert r.passed
    assert r.total > 0


def test_verify_kernel_catches_a_broken_kernel():
    @verify(against="torch.add", dtypes=["float32"], device="cpu")
    def broken(a, b):
        return a - b

    r = verify_kernel(broken)
    assert not r.passed
    assert r.minimal_failure is not None


def test_verify_kernel_refuses_an_unmarked_function():
    def plain(a, b):
        return a + b

    with pytest.raises(ValueError, match="not marked"):
        verify_kernel(plain)


def test_device_can_be_overridden_at_call_time():
    """CI runners and laptops differ; the decorator's device must not be a
    hard commitment."""

    @verify(against="torch.add", dtypes=["float32"], device="cuda")
    def add(a, b):
        return a + b

    r = verify_kernel(add, device="cpu")
    assert r.passed


# ---------------------------------------------------------------- discovery


def test_discover_in_module_finds_only_marked_functions():
    mod = sys.modules[__name__]

    @verify(against="torch.add")
    def marked(a, b):
        return a + b

    # Injected onto this module so discovery has something module-level.
    mod.marked_kernel_for_discovery = marked
    try:
        found = dict(discover_in_module(mod))
        assert "marked_kernel_for_discovery" in found
        assert all(is_marked(f) for f in found.values())
    finally:
        del mod.marked_kernel_for_discovery


def test_discover_in_path_imports_a_file_and_finds_kernels(tmp_path):
    f = tmp_path / "my_kernels.py"
    f.write_text(
        textwrap.dedent(
            """
            from shapesandstrides.verify import verify

            @verify(against="torch.add", dtypes=["float32"], device="cpu")
            def good(a, b):
                return a + b

            @verify(against="torch.add", dtypes=["float32"], device="cpu")
            def bad(a, b):
                return a - b

            def not_a_kernel(a, b):
                return a + b
            """
        ),
        encoding="utf-8",
    )
    found = discover_in_path(f)
    names = sorted(n for n, _ in found)
    assert names == ["bad", "good"]


def test_discover_in_path_walks_a_directory(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text(
        "from shapesandstrides.verify import verify\n"
        "@verify(against='torch.add', device='cpu')\n"
        "def k1(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    (tmp_path / "pkg" / "b.py").write_text("x = 1\n", encoding="utf-8")
    found = discover_in_path(tmp_path)
    assert [n for n, _ in found] == ["k1"]


def test_discover_in_path_on_a_missing_target_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_in_path(tmp_path / "nope.py")
