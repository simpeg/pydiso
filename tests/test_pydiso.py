import numpy as np
import scipy.sparse as sp
from pydiso.mkl_solver import (
    MKLPardisoSolver as Solver,
    MatrixType,
    MATRIX_TYPES,
    FillReducingOrdering,
    OutOfCoreMode,
    get_mkl_max_threads,
    get_mkl_pardiso_max_threads,
    get_mkl_version,
    set_mkl_threads,
    set_mkl_pardiso_threads,
)
from concurrent.futures import ThreadPoolExecutor
import pytest
import sys

np.random.seed(12345)
n = 40
L = sp.diags([-1, 1], [-1, 0], (n, n))
U = sp.diags([2, -1], [0, 1], (n, n))
e = np.ones(n)
e[0] = -1
D = sp.diags(e)  # diagonal matrix of 1 and -1
U2 = sp.diags([2, -1], [0, 2], (n, n))

Lc = sp.diags([-(1+1j), (1+1j)], [-1, 0], (n, n))
Uc = sp.diags([(2+2j), -(1+1j)], [0, 1], (n, n))
U2c = sp.diags([(2+2j), -(1+1j)], [0, 2], (n, n))

xr = np.linspace(-10, 10, n)
xc = np.linspace(-20, 20, n) + np.linspace(20, -20, n)*1j

A_real_dict = {'real_structurally_symmetric': L@U,
               'real_symmetric_positive_definite': L@L.T,
               'real_symmetric_indefinite': L@D@L.T,
               'real_nonsymmetric': L@U2
               }
A_complex_dict = {'complex_structurally_symmetric': Lc@Uc,
                  'complex_hermitian_positive_definite': Lc@Lc.T.conjugate(),
                  'complex_hermitian_indefinite': Lc@D@Lc.T.conjugate(),
                  'complex_symmetric': Lc@Lc.T,
                  'complex_nonsymmetric': Lc@U2c
                  }


@pytest.mark.xfail(sys.platform == "darwin", reason="Unexpected Thread bug in third party library")
def test_thread_setting():
    n1 = get_mkl_max_threads()
    n2 = get_mkl_pardiso_max_threads()
    assert n1 == n2

    if n1 > 2:
        set_mkl_threads(n1-1)
        assert get_mkl_max_threads() == n1-1

    set_mkl_pardiso_threads(1)
    assert get_mkl_pardiso_max_threads() == 1

    if n1 > 3:
        assert get_mkl_pardiso_max_threads() != get_mkl_max_threads()


def test_version():
    version_info = get_mkl_version()
    assert "MajorVersion" in version_info
    assert "MinorVersion" in version_info
    assert "UpdateVersion" in version_info
    assert "ProductStatus" in version_info
    assert "Build" in version_info
    assert "Processor" in version_info
    assert "Platform" in version_info

    for item in version_info:
        print(item, version_info[item])


# generate the input lists...
inputs = []
for dtype in (np.float32, np.float64):
    for key, item in A_real_dict.items():
        inputs.append((item.astype(dtype), key))

for dtype in (np.complex64, np.complex128):
    for key, item in A_complex_dict.items():
        inputs.append((item.astype(dtype), key))


@pytest.mark.parametrize("A, matrix_type", inputs)
def test_solver(A, matrix_type):
    dtype = A.dtype
    if np.issubdtype(dtype, np.complexfloating):
        x = xc.astype(dtype)
    else:
        x = xr.astype(dtype)
    b = A@x

    solver = Solver(A, matrix_type=matrix_type)
    A_valid = sp.csr_matrix((solver._data, solver._indices, solver._indptr))
    x2 = solver.solve(b)

    eps = np.finfo(dtype).eps
    np.testing.assert_allclose(x, x2, atol=2E3*eps)

@pytest.mark.parametrize("A, matrix_type", inputs)
def test_transpose_solver(A, matrix_type):
    dtype = A.dtype
    if np.issubdtype(dtype, np.complexfloating):
        x = xc.astype(dtype)
    else:
        x = xr.astype(dtype)
    b = A.T @ x

    solver = Solver(A, matrix_type=matrix_type)
    x2 = solver.solve(b, transpose=True)

    eps = np.finfo(dtype).eps
    np.testing.assert_allclose(x, x2, atol=2E3*eps)

def test_multiple_RHS():
    A = A_real_dict["real_symmetric_positive_definite"]
    x = np.c_[xr, xr]
    b = A @ x

    solver = Solver(A, "real_symmetric_positive_definite")
    x2 = solver.solve(b)

    eps = np.finfo(np.float64).eps
    np.testing.assert_allclose(x, x2, atol=2E3*eps)


def test_matrix_type_errors():
    A = A_real_dict["real_symmetric_positive_definite"]
    with pytest.raises(ValueError):
        solver = Solver(A, matrix_type="complex_hermitian_positive_definite")

    A = A_complex_dict["complex_structurally_symmetric"]
    with pytest.raises(ValueError):
        solver = Solver(A, matrix_type="real_symmetric_positive_definite")


def test_matrix_type_enum():
    """Test that MatrixType, plain matching ints, and legacy strings are all accepted,
    and that they're all equivalent."""
    A = A_real_dict["real_symmetric_positive_definite"]

    solver_enum = Solver(A, matrix_type=MatrixType.REAL_SYMMETRIC_POSITIVE_DEFINITE)
    solver_int = Solver(A, matrix_type=2)
    solver_str = Solver(A, matrix_type="real_symmetric_positive_definite")

    assert solver_enum.matrix_type == MatrixType.REAL_SYMMETRIC_POSITIVE_DEFINITE
    assert solver_int.matrix_type == MatrixType.REAL_SYMMETRIC_POSITIVE_DEFINITE
    assert solver_str.matrix_type == MatrixType.REAL_SYMMETRIC_POSITIVE_DEFINITE

    # legacy dict form still matches the enum's values.
    assert MATRIX_TYPES["real_symmetric_positive_definite"] == MatrixType.REAL_SYMMETRIC_POSITIVE_DEFINITE

    # string lookup is case-insensitive (matches the enum member name, not just the dict key).
    solver_upper = Solver(A, matrix_type="REAL_SYMMETRIC_POSITIVE_DEFINITE")
    assert solver_upper.matrix_type == MatrixType.REAL_SYMMETRIC_POSITIVE_DEFINITE


def test_matrix_type_unrecognized():
    """Test that an unrecognized matrix_type still raises TypeError."""
    A = A_real_dict["real_symmetric_positive_definite"]
    with pytest.raises(TypeError):
        Solver(A, matrix_type="not_a_real_matrix_type")
    with pytest.raises(TypeError):
        Solver(A, matrix_type=999)



def test_rhs_size_error():
    A = A_real_dict["real_symmetric_positive_definite"]
    solver = Solver(A, "real_symmetric_positive_definite")
    n = A.shape[0]
    x = np.random.rand(n)
    b = np.random.rand(n)
    b_bad = np.random.rand(n-1)
    x_bad = np.random.rand(n-1)
    with pytest.raises(ValueError):
        solver.solve(b_bad)
    with pytest.raises(ValueError):
        solver.solve(b, x_bad)

def test_named_iparm_kwargs():
    """Test that the named iparm keyword arguments are applied correctly."""
    A = A_real_dict["real_nonsymmetric"]
    x = xr.copy()
    b = A @ x

    solver = Solver(
        A,
        matrix_type="real_nonsymmetric",
        fill_reducing_ordering=FillReducingOrdering.MINIMUM_DEGREE,
        max_iterative_refinement_steps=4,
        pivoting_perturbation=6,
        scaling=False,
        weighted_matching=False,
        cnr_threads=2,
        low_rank_update=True,
        report_nnz=False,
        report_mflops=True,
        parallel_factorization=True,
        parallel_solve=True,
        out_of_core_mode=OutOfCoreMode.IN_CORE,
    )
    assert solver.iparm[1] == FillReducingOrdering.MINIMUM_DEGREE
    assert solver.iparm[7] == 4
    assert solver.iparm[9] == 6
    assert solver.iparm[10] == 0
    assert solver.iparm[12] == 0
    assert solver.iparm[23] == 1
    assert solver.iparm[24] == 1
    assert solver.iparm[33] == 2
    assert solver.iparm[38] == 1
    assert solver.iparm[59] == OutOfCoreMode.IN_CORE

    x2 = solver.solve(b)
    eps = np.finfo(np.float64).eps
    np.testing.assert_allclose(x, x2, atol=2E3*eps)


def test_named_iparm_kwargs_bunch_kaufman():
    """Test bunch_kaufman_pivoting, which only has an effect on symmetric indefinite matrices."""
    A = A_real_dict["real_symmetric_indefinite"]
    x = xr.copy()
    b = A @ x

    solver = Solver(A, matrix_type="real_symmetric_indefinite", bunch_kaufman_pivoting=False)
    assert solver.iparm[20] == 0

    x2 = solver.solve(b)
    eps = np.finfo(np.float64).eps
    np.testing.assert_allclose(x, x2, atol=2E3*eps)

    # Default (None) leaves the library's matrix-type-dependent default in place.
    solver = Solver(A, matrix_type="real_symmetric_indefinite")
    assert solver.iparm[20] == 1


def test_named_iparm_kwargs_report_nnz_mflops():
    """Test report_nnz/report_mflops, which toggle iparm[17]/iparm[18] reporting."""
    A = A_real_dict["real_nonsymmetric"]
    x = xr.copy()
    b = A @ x

    # Default report_nnz=True: nnz reflects a real, positive count.
    solver = Solver(A, matrix_type="real_nonsymmetric")
    assert solver.nnz > 0

    # report_nnz=False disables the report; iparm[17] stays at the "don't report" value.
    solver2 = Solver(A, matrix_type="real_nonsymmetric", report_nnz=False)
    assert solver2.iparm[17] == 0

    # report_mflops=True shouldn't affect correctness (the exact reported magnitude/sign is an
    # MKL implementation detail we don't assert on here).
    solver3 = Solver(A, matrix_type="real_nonsymmetric", report_mflops=True)
    x2 = solver3.solve(b)
    eps = np.finfo(np.float64).eps
    np.testing.assert_allclose(x, x2, atol=2E3*eps)

    # Default report_mflops=False: no report requested.
    assert solver.iparm[18] == 0


def test_named_iparm_kwargs_plain_int():
    """Test that enum-backed kwargs also accept plain matching ints."""
    A = A_real_dict["real_nonsymmetric"]
    solver = Solver(
        A,
        matrix_type="real_nonsymmetric",
        fill_reducing_ordering=0,
        out_of_core_mode=0,
    )
    assert solver.iparm[1] == 0
    assert solver.iparm[59] == 0
    # the int/enum form of fill_reducing_ordering must not touch iparm[4].
    assert solver.iparm[4] == 2


def test_fill_reducing_ordering_perm_identity():
    """Test that an array passed to fill_reducing_ordering is used as a user-supplied perm."""
    A = A_real_dict["real_nonsymmetric"]
    x = xr.copy()
    b = A @ x

    solver = Solver(A, matrix_type="real_nonsymmetric", fill_reducing_ordering=np.arange(n))
    assert solver.iparm[4] == 1
    np.testing.assert_array_equal(solver.perm, np.arange(n))

    x2 = solver.solve(b)
    eps = np.finfo(np.float64).eps
    np.testing.assert_allclose(x, x2, atol=2E3*eps)


def test_fill_reducing_ordering_perm_custom():
    """Test a non-trivial user-supplied permutation still solves correctly."""
    A = A_real_dict["real_nonsymmetric"]
    x = xr.copy()
    b = A @ x

    perm = np.arange(n)[::-1].copy()
    solver = Solver(A, matrix_type="real_nonsymmetric", fill_reducing_ordering=perm)
    assert solver.iparm[4] == 1
    np.testing.assert_array_equal(solver.perm, perm)

    x2 = solver.solve(b)
    eps = np.finfo(np.float64).eps
    np.testing.assert_allclose(x, x2, atol=2E3*eps)


def test_fill_reducing_ordering_perm_invalid():
    """Test that an invalid permutation array is rejected."""
    A = A_real_dict["real_symmetric_positive_definite"]
    with pytest.raises(ValueError):
        Solver(A, matrix_type="real_symmetric_positive_definite", fill_reducing_ordering=np.arange(n - 1))
    with pytest.raises(ValueError):
        bad_perm = np.arange(n)
        bad_perm[0] = bad_perm[1]
        Solver(A, matrix_type="real_symmetric_positive_definite", fill_reducing_ordering=bad_perm)


def test_named_iparm_kwargs_invalid():
    """Test that invalid values for named iparm kwargs are rejected."""
    A = A_real_dict["real_symmetric_positive_definite"]
    with pytest.raises(ValueError):
        Solver(A, matrix_type="real_symmetric_positive_definite", fill_reducing_ordering=5)
    with pytest.raises(ValueError):
        Solver(A, matrix_type="real_symmetric_positive_definite", out_of_core_mode=99)
    with pytest.raises(ValueError):
        Solver(A, matrix_type="real_symmetric_positive_definite", pivoting_perturbation=-1)
    with pytest.raises(ValueError):
        Solver(A, matrix_type="real_symmetric_positive_definite", pivoting_perturbation=1.5)
    with pytest.raises(TypeError):
        Solver(A, matrix_type="real_symmetric_positive_definite", scaling=1)
    with pytest.raises(ValueError):
        Solver(A, matrix_type="real_symmetric_positive_definite", cnr_threads=-1)
    with pytest.raises(TypeError):
        Solver(A, matrix_type="real_symmetric_positive_definite", bunch_kaufman_pivoting=1)
    with pytest.raises(TypeError):
        Solver(A, matrix_type="real_symmetric_positive_definite", report_nnz=1)
    with pytest.raises(TypeError):
        Solver(A, matrix_type="real_symmetric_positive_definite", report_mflops=1)


def test_named_iparm_kwargs_default_none():
    """Test that leaving the named iparm kwargs as None preserves existing behavior."""
    A = A_real_dict["real_nonsymmetric"]
    solver = Solver(A, matrix_type="real_nonsymmetric")
    assert solver.iparm[10] == 1
    assert solver.iparm[12] == 1
    assert solver.iparm[20] == 0
    # iparm[26] (matrix checker) is no longer user-configurable; it's left entirely to
    # whatever Pardiso itself does with it, since _validate_csr_matrix already guarantees
    # canonical CSR form.
    assert solver.iparm[26] == 0
    # Pardiso reports back -1 in iparm[33] when CNR mode was requested automatically (0)
    # but not actually active.
    assert solver.iparm[33] == -1
    assert solver.iparm[38] == 0
    # report_nnz defaults to True (requested via -1), so after factorization iparm[17] holds
    # the actual reported count rather than the -1 request sentinel.
    assert solver.iparm[17] > 0
    assert solver.iparm[18] == 0


def test_refactor_pivoting_perturbation():
    """Test that pivoting_perturbation can be tuned on refactor() without a new analysis."""
    A = A_real_dict["real_nonsymmetric"]
    x = xr.copy()

    solver = Solver(A, matrix_type="real_nonsymmetric")
    solver.refactor(A * 1.0001, pivoting_perturbation=6)
    assert solver.iparm[9] == 6

    b = (A * 1.0001) @ x
    x2 = solver.solve(b)
    eps = np.finfo(np.float64).eps
    np.testing.assert_allclose(x, x2, atol=2E3*eps)


def test_refactor_bunch_kaufman_pivoting():
    """Test bunch_kaufman_pivoting on refactor(), which only matters for symmetric indefinite matrices."""
    A = A_real_dict["real_symmetric_indefinite"]
    x = xr.copy()

    solver = Solver(A, matrix_type="real_symmetric_indefinite")
    solver.refactor(A * 1.0001, bunch_kaufman_pivoting=False)
    assert solver.iparm[20] == 0

    b = (A * 1.0001) @ x
    x2 = solver.solve(b)
    eps = np.finfo(np.float64).eps
    np.testing.assert_allclose(x, x2, atol=2E3*eps)


def test_refactor_preconditioned_cgs():
    """Test that preconditioned_cgs sets iparm[3] with the given (L, K) encoding."""
    A = A_real_dict["real_nonsymmetric"]
    x = xr.copy()

    solver = Solver(A, matrix_type="real_nonsymmetric")
    solver.refactor(A * 1.0001, preconditioned_cgs=(6, 1))
    assert solver.iparm[3] == 61

    b = (A * 1.0001) @ x
    x2 = solver.solve(b)
    eps = np.finfo(np.float64).eps
    np.testing.assert_allclose(x, x2, atol=2E3*eps)

    A2 = A_real_dict["real_symmetric_positive_definite"]
    solver2 = Solver(A2, matrix_type="real_symmetric_positive_definite")
    solver2.refactor(A2 * 1.0001, preconditioned_cgs=(6, 2))
    assert solver2.iparm[3] == 62

    b2 = (A2 * 1.0001) @ x
    x2_2 = solver2.solve(b2)
    np.testing.assert_allclose(x, x2_2, atol=2E3*eps)


def test_refactor_report_mflops():
    """Test that report_mflops can be tuned on refactor()."""
    A = A_real_dict["real_nonsymmetric"]
    x = xr.copy()
    b = (A * 1.0001) @ x

    solver = Solver(A, matrix_type="real_nonsymmetric")
    solver.refactor(A * 1.0001, report_mflops=True)
    assert solver.iparm[18] != 0

    x2 = solver.solve(b)
    eps = np.finfo(np.float64).eps
    np.testing.assert_allclose(x, x2, atol=2E3*eps)


def test_refactor_kwargs_invalid():
    """Test that invalid values for refactor() kwargs are rejected."""
    A = A_real_dict["real_symmetric_positive_definite"]
    solver = Solver(A, matrix_type="real_symmetric_positive_definite")
    with pytest.raises(ValueError):
        solver.refactor(A, pivoting_perturbation=-1)
    with pytest.raises(TypeError):
        solver.refactor(A, bunch_kaufman_pivoting=1)
    with pytest.raises(ValueError):
        solver.refactor(A, preconditioned_cgs=(0, 1))
    with pytest.raises(ValueError):
        solver.refactor(A, preconditioned_cgs=(10, 1))
    with pytest.raises(ValueError):
        solver.refactor(A, preconditioned_cgs=(5, 3))
    with pytest.raises(TypeError):
        solver.refactor(A, report_mflops=1)


def test_refactor_kwargs_default_none():
    """Test that leaving refactor()'s named iparm kwargs as None preserves existing values."""
    A = A_real_dict["real_nonsymmetric"]
    solver = Solver(A, matrix_type="real_nonsymmetric")
    iparm3_before = solver.iparm[3]
    iparm9_before = solver.iparm[9]
    iparm20_before = solver.iparm[20]
    iparm18_before = solver.iparm[18]

    solver.refactor(A * 1.0001)
    assert solver.iparm[3] == iparm3_before
    assert solver.iparm[9] == iparm9_before
    assert solver.iparm[20] == iparm20_before
    assert solver.iparm[18] == iparm18_before


def test_solve_max_iterative_refinement_steps():
    """Test that max_iterative_refinement_steps can be tuned per solve() call, and is sticky."""
    A = A_real_dict["real_nonsymmetric"]
    x = xr.copy()
    b = A @ x

    solver = Solver(A, matrix_type="real_nonsymmetric")
    x2 = solver.solve(b, max_iterative_refinement_steps=4)
    assert solver.iparm[7] == 4
    eps = np.finfo(np.float64).eps
    np.testing.assert_allclose(x, x2, atol=2E3*eps)

    # Omitting the kwarg on a subsequent call leaves the previous value in place.
    x3 = solver.solve(b)
    assert solver.iparm[7] == 4
    np.testing.assert_allclose(x, x3, atol=2E3*eps)


def test_solve_max_iterative_refinement_steps_invalid():
    """Test that a non-int max_iterative_refinement_steps is rejected."""
    A = A_real_dict["real_symmetric_positive_definite"]
    solver = Solver(A, matrix_type="real_symmetric_positive_definite")
    b = A @ xr
    with pytest.raises(TypeError):
        solver.solve(b, max_iterative_refinement_steps=1.5)


def test_threading():
    """
    Here we test that calling the solver is safe from multiple threads.
    There isn't actually any speedup because it acquires a lock on each call
    to pardiso internally (because those calls are not thread safe).
    """
    n = 200
    n_rhs = 75
    A = sp.diags([-1, 2, -1], (-1, 0, 1), shape=(n, n), format='csr')
    Ainv = Solver(A)

    x_true = np.random.rand(n, n_rhs)
    rhs = A @ x_true

    with ThreadPoolExecutor() as pool:
        x_sol = np.stack(
            list(pool.map(lambda i: Ainv.solve(rhs[:, i]), range(n_rhs))),
            axis=1
        )

    np.testing.assert_allclose(x_true, x_sol)
