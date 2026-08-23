"""Reference resolution: what the user has to hand us, and how little of it.

The whole point of this module is that supplying a reference implementation is
the single biggest tax on adopting the tool, so the common cases should cost
the user nothing. `against="torch.add"` must need no hand-written reference at
all, and a fused kernel must be satisfiable with a one-line lambda.
"""

import math

import pytest
import torch

from shapesandstrides.reference import (
    OracleKind,
    ReferenceResolutionError,
    infer_arity,
    probe_arity,
    resolve,
)


# --------------------------------------------------------------- torch ops


def test_resolves_a_torch_operator_by_name():
    r = resolve("torch.add")
    assert r.kind is OracleKind.TORCH_OP
    assert r.fn is torch.add
    assert r.label == "torch.add"


def test_resolves_a_nested_torch_path():
    r = resolve("torch.nn.functional.gelu")
    assert r.kind is OracleKind.TORCH_OP
    assert r.fn is torch.nn.functional.gelu


def test_resolved_torch_op_actually_computes():
    r = resolve("torch.add")
    a, b = torch.ones(4), torch.ones(4)
    assert torch.equal(r.fn(a, b), torch.full((4,), 2.0))


def test_unknown_torch_attribute_is_a_clear_error():
    with pytest.raises(ReferenceResolutionError) as e:
        resolve("torch.no_such_operator")
    # The message has to name what failed and where, or the user is left
    # guessing whether they mistyped or we are broken.
    assert "torch.no_such_operator" in str(e.value)
    assert "no_such_operator" in str(e.value)


def test_non_callable_target_is_rejected():
    with pytest.raises(ReferenceResolutionError) as e:
        resolve("torch.pi")
    assert "not callable" in str(e.value).lower()


def test_dotted_path_outside_torch_is_allowed_but_not_labelled_a_torch_op():
    r = resolve("math.sqrt")
    assert r.fn is math.sqrt
    assert r.kind is OracleKind.USER_CALLABLE


def test_unimportable_module_is_a_clear_error():
    with pytest.raises(ReferenceResolutionError):
        resolve("definitely_not_a_module.thing")


def test_bare_name_with_no_dot_is_rejected():
    with pytest.raises(ReferenceResolutionError):
        resolve("add")


# ------------------------------------------------------- callables and None


def test_lambda_is_an_expression():
    r = resolve(lambda x, y: x + y)
    assert r.kind is OracleKind.EXPRESSION
    assert r.arity == 2


def test_named_function_is_a_user_callable_and_keeps_its_name():
    def my_slow_reference(x, y):
        return x + y

    r = resolve(my_slow_reference)
    assert r.kind is OracleKind.USER_CALLABLE
    assert r.label == "my_slow_reference"


def test_none_resolves_to_no_reference_rather_than_raising():
    # A kernel with no torch equivalent is the interesting case, not an error.
    # It degrades to a weaker oracle; it does not stop the run.
    r = resolve(None)
    assert r.kind is OracleKind.NONE
    assert r.fn is None
    assert r.arity is None


def test_an_already_resolved_reference_passes_through_unchanged():
    r = resolve("torch.add")
    assert resolve(r) is r


def test_rejects_a_type_it_cannot_interpret():
    with pytest.raises(ReferenceResolutionError):
        resolve(42)


# ------------------------------------------------------------------ arity


def test_infer_arity_counts_required_positional_parameters():
    assert infer_arity(lambda x, y: None) == 2
    assert infer_arity(lambda x: None) == 1


def test_infer_arity_ignores_parameters_with_defaults():
    def f(x, y, alpha=1.0):
        return None

    assert infer_arity(f) == 2


def test_infer_arity_returns_none_when_it_cannot_tell():
    # C-bound torch ops usually have no introspectable signature. Returning
    # None is the honest answer; guessing 2 would be a silent wrong default.
    assert infer_arity(torch.add) in (2, None)


def test_probe_arity_finds_binary_and_unary_torch_ops():
    assert probe_arity(torch.add) == 2
    assert probe_arity(torch.relu) == 1


def test_probe_arity_gives_up_rather_than_guessing():
    def never_works(*args):
        raise RuntimeError("nope")

    assert probe_arity(never_works) is None


def test_resolve_populates_arity_for_a_torch_op():
    # This is the payoff: the user should never have to tell us n_inputs.
    assert resolve("torch.add").arity == 2
    assert resolve("torch.relu").arity == 1
