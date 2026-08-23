import pytest

from shapesandstrides.tolerance import DEFAULT_TOLERANCES, tolerance_for


def test_fp16_is_looser_than_fp32():
    a16, _ = tolerance_for("add", "float16")
    a32, _ = tolerance_for("add", "float32")
    assert a16 > a32, "half precision must not be held to single-precision tolerance"


def test_bf16_is_loosest_of_the_three():
    a16, _ = tolerance_for("add", "float16")
    abf, _ = tolerance_for("add", "bfloat16")
    assert abf > a16, "bfloat16 has fewer mantissa bits than float16"


def test_reduction_ops_get_looser_tolerance_than_elementwise():
    """Accumulation error grows with the number of terms summed."""
    a_add, _ = tolerance_for("add", "float16")
    a_sum, _ = tolerance_for("sum", "float16")
    assert a_sum > a_add


def test_matmul_is_treated_as_a_reduction():
    a_mm, _ = tolerance_for("matmul", "float16")
    a_add, _ = tolerance_for("add", "float16")
    assert a_mm > a_add


def test_unknown_op_falls_back_to_dtype_default():
    known = tolerance_for("add", "float16")
    unknown = tolerance_for("some_custom_kernel", "float16")
    assert unknown == known


def test_unknown_dtype_raises_rather_than_guessing():
    with pytest.raises(KeyError):
        tolerance_for("add", "float8_e4m3fn")


def test_table_is_declarative_not_hardcoded_in_logic():
    assert "float16" in DEFAULT_TOLERANCES
    assert isinstance(DEFAULT_TOLERANCES["float16"], tuple)


def test_a_fused_kernel_gets_more_slack_than_its_widest_single_stage():
    """Error compounds through a chain; no single stage's budget describes it."""
    single, _ = tolerance_for("matmul", "float16")
    fused, _ = tolerance_for("fused", "float16", fused_ops=["layernorm", "matmul", "gelu"])
    assert fused > single


def test_more_fused_stages_means_more_slack():
    two, _ = tolerance_for("f", "float16", fused_ops=["add", "mul"])
    four, _ = tolerance_for("f", "float16", fused_ops=["add", "mul", "sub", "div"])
    assert four > two


def test_a_single_stage_fused_list_matches_the_plain_op():
    assert tolerance_for("f", "float16", fused_ops=["matmul"]) == tolerance_for("matmul", "float16")


def test_override_wins_outright():
    assert tolerance_for("matmul", "float16", override=(1.0, 2.0)) == (1.0, 2.0)


def test_override_bypasses_unknown_dtype_check():
    """An author who supplies an explicit tolerance does not need our table."""
    assert tolerance_for("x", "float8_e4m3fn", override=(0.5, 0.5)) == (0.5, 0.5)
