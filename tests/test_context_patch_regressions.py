"""Regression tests for resolve_context and Context parameter validation.

Verifies:
1. P1: Legitimate narrow integer tensors (int8, uint8, int16, int32, int64) are correctly accepted
   without false-positive overflow or truncation during boundary checks.
2. P2: Lists, tuples, and column vectors recursively enforce integer bounds, boolean rejection,
   and complex rejection across arbitrary allowed nestings.
3. P2: Extreme integer boundary values (e.g. -2^63) are strictly rejected without abs overflow.
4. Legitimate sequences and column vector formats remain fully functional.
"""

import numpy as np
import pytest
import torch

from src.contracts.context import Context, PhysicalContext, resolve_context, MAX_EXACT_INT


# ==============================================================================
# 1. P1: Legitimate Narrow Integer Tensor Compatibility
# ==============================================================================


@pytest.mark.parametrize(
    "dtype, value",
    [
        (torch.int8, 100),
        (torch.uint8, 100),
        (torch.int16, 1000),
        (torch.int32, 1000),
        (torch.int64, 1000),
    ],
)
def test_narrow_integer_tensor_accepted(dtype, value):
    """Verify that legitimate narrow integer tensors are correctly accepted without miscomparison."""
    tensor_val = torch.tensor([value], dtype=dtype)

    # 1. Context has float, legacy has narrow int tensor
    ctx_float = Context.from_re_sc(re=float(value), sc=1.0)
    res1 = resolve_context(context=ctx_float, re=tensor_val, sc=1.0)
    assert res1 is ctx_float

    # 2. Context has narrow int tensor, legacy has float/int
    ctx_narrow = Context(physical=PhysicalContext(re=tensor_val, sc=torch.tensor([1.0])))
    res2 = resolve_context(context=ctx_narrow, re=float(value), sc=1.0)
    assert res2 is ctx_narrow

    # 3. Context.from_re_sc factory accepts narrow int tensor
    ctx_factory = Context.from_re_sc(re=tensor_val, sc=1.0)
    assert ctx_factory is not None
    assert ctx_factory.re is not None


# ==============================================================================
# 2. P2: Lists and Tuples Element-Wise Validation (Integer bounds, booleans, complex)
# ==============================================================================


def test_sequence_integer_range_rejection():
    """Verify that integers exceeding 2^53 are strictly rejected when passed inside lists or tuples."""
    ctx_valid = Context.from_re_sc(re=1000.0, sc=1.0)
    exceeded = MAX_EXACT_INT + 1

    # Flat list
    with pytest.raises(ValueError, match="exceeds maximum lossless float64 representation limit"):
        resolve_context(context=ctx_valid, re=[exceeded])

    # Flat tuple
    with pytest.raises(ValueError, match="exceeds maximum lossless float64 representation limit"):
        resolve_context(context=ctx_valid, re=(exceeded,))

    # Nested column vector list [[2^53 + 1]]
    with pytest.raises(ValueError, match="exceeds maximum lossless float64 representation limit"):
        resolve_context(context=ctx_valid, re=[[exceeded]])

    # Context containing out-of-range integer list
    ctx_list = Context(physical=PhysicalContext(re=[exceeded], sc=1.0))
    with pytest.raises(ValueError, match="exceeds maximum lossless float64 representation limit"):
        resolve_context(context=ctx_list, re=[exceeded])

    # Context.from_re_sc factory rejects out-of-range integer sequence
    with pytest.raises(ValueError, match="exceeds maximum lossless float64 representation limit"):
        Context.from_re_sc(re=[exceeded], sc=1.0)


def test_sequence_boolean_and_complex_rejection():
    """Verify that booleans and complex numbers inside nested sequences are strictly rejected."""
    ctx_valid = Context.from_re_sc(re=1.0, sc=1.0)

    # Nested boolean list [[True]]
    with pytest.raises(TypeError, match="must be numeric, got boolean|sequence contains boolean element"):
        resolve_context(context=ctx_valid, re=[[True]])

    # Flat boolean list [True]
    with pytest.raises(TypeError, match="must be numeric, got boolean|sequence contains boolean element"):
        resolve_context(context=ctx_valid, re=[True])

    # Nested boolean tuple ((False,),)
    with pytest.raises(TypeError, match="must be numeric, got boolean|sequence contains boolean element"):
        resolve_context(context=ctx_valid, re=((False,),))

    # Nested complex list [[1000.0 + 5.0j]]
    with pytest.raises(TypeError, match="must be real-valued, got complex|sequence contains complex element"):
        resolve_context(context=ctx_valid, re=[[1000.0 + 5.0j]])

    # Context containing nested boolean list
    ctx_nested_bool = Context(physical=PhysicalContext(re=[[True]], sc=1.0))
    with pytest.raises(TypeError, match="sequence contains boolean element|got boolean"):
        resolve_context(context=ctx_nested_bool, re=1.0)

    # Context.from_re_sc factory rejects nested boolean/complex
    with pytest.raises(TypeError, match="sequence contains boolean element"):
        Context.from_re_sc(re=[[True]], sc=1.0)
    with pytest.raises(TypeError, match="sequence contains complex element"):
        Context.from_re_sc(re=[[1.0 + 2.0j]], sc=1.0)


# ==============================================================================
# 3. P2: Int64 Extreme Boundary Values Without abs Overflow
# ==============================================================================


@pytest.mark.parametrize(
    "extreme_val",
    [
        -2**63,
        2**63 - 1,
        -(MAX_EXACT_INT + 1),
        MAX_EXACT_INT + 1,
    ],
)
def test_extreme_integer_boundary_rejection_no_abs_overflow(extreme_val):
    """Verify that int64 minimum (-2^63) and out-of-range values are strictly rejected without abs overflow."""
    ctx_valid = Context.from_re_sc(re=1000.0, sc=1.0)

    # PyTorch int64 tensor
    with pytest.raises(ValueError, match="exceeds maximum lossless float64 representation limit"):
        resolve_context(context=ctx_valid, re=torch.tensor([extreme_val], dtype=torch.int64))

    # NumPy int64 array
    with pytest.raises(ValueError, match="exceeds maximum lossless float64 representation limit"):
        resolve_context(context=ctx_valid, re=np.array([extreme_val], dtype=np.int64))

    # Python int scalar
    with pytest.raises(ValueError, match="exceeds maximum lossless float64 representation limit"):
        resolve_context(context=ctx_valid, re=extreme_val)

    # Context.from_re_sc factory rejects extreme values
    with pytest.raises(ValueError, match="exceeds maximum lossless float64 representation limit"):
        Context.from_re_sc(re=torch.tensor([extreme_val], dtype=torch.int64), sc=1.0)


def test_numpy_unsigned_integer_overflow_rejection():
    """Verify that large unsigned numpy integers exceeding 2^53 are rejected without loss."""
    ctx_valid = Context.from_re_sc(re=1000.0, sc=1.0)
    large_uint = np.uint64(2**64 - 1)

    with pytest.raises(ValueError, match="exceeds maximum lossless float64 representation limit"):
        resolve_context(context=ctx_valid, re=np.array([large_uint]))


# ==============================================================================
# 4. Legitimate Sequence and Column Vector Formats Preserved
# ==============================================================================


def test_legitimate_column_vectors_and_sequences():
    """Verify that legitimate column vectors and sequences continue to function smoothly."""
    ctx_base = Context.from_re_sc(re=1000.0, sc=1.0)

    # 2D single-column float list [[1000.0]]
    res1 = resolve_context(context=ctx_base, re=[[1000.0]], sc=1.0)
    assert res1 is ctx_base

    # 2D single-column int list [[1000]]
    res2 = resolve_context(context=ctx_base, re=[[1000]], sc=1.0)
    assert res2 is ctx_base

    # 1D int list [1000]
    res3 = resolve_context(context=ctx_base, re=[1000], sc=1.0)
    assert res3 is ctx_base

    # Tuple (1000.0,)
    res4 = resolve_context(context=ctx_base, re=(1000.0,), sc=1.0)
    assert res4 is ctx_base

    # Batch column vector
    ctx_batch = Context.from_re_sc(
        re=torch.tensor([[1000.0], [2000.0]]),
        sc=torch.tensor([[1.0], [2.0]]),
    )
    res_batch = resolve_context(
        context=ctx_batch,
        re=np.array([[1000.0], [2000.0]]),
        sc=np.array([[1.0], [2.0]]),
    )
    assert res_batch is ctx_batch
