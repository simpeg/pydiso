import numpy as np
import scipy.sparse as sp
from pydiso.mkl_solver import (
    MKLPardisoSolver as Solver,
    get_mkl_max_threads,
    get_mkl_pardiso_max_threads,
    get_mkl_version,
    set_mkl_threads,
    set_mkl_pardiso_threads,
)
import pytest

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

xr = np.random.rand(n)
xc = np.random.rand(n) + np.random.rand(n)*1j

A_real_dict = {'real_structurally_symmetric': L@U,
               'real_symmetric_positive_definite': L@L.T,
               'real_symmetric_indefinite': L@D@L.T,
               'real_nonsymmetric': L@U2
               }
A_complex_dict = {'complex_structurally_symmetric': Lc@Uc,
                  'complex_hermitian_positive_definite': Lc@Lc.H,
                  'complex_hermitian_indefinite': Lc@D@Lc.H,
                  'complex_symmetric': Lc@Lc.T,
                  'complex_nonsymmetric': Lc@U2c
                  }

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
    print(get_mkl_version())

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
    x2 = solver.solve(b)

    eps = np.finfo(dtype).eps
    rel_err = np.linalg.norm(x-x2)/np.linalg.norm(x)
    assert rel_err < 1E3*eps
    return rel_err


def test_matrix_type_errors():
    A = A_real_dict["real_symmetric_positive_definite"]
    with pytest.raises(TypeError):
        solver = Solver(A, matrix_type="complex_hermitian_positive_definite")

    A = A_complex_dict["complex_structurally_symmetric"]
    with pytest.raises(TypeError):
        solver = Solver(A, matrix_type="real_symmetric_positive_definite")



if __name__ == '__main__':
    for A, type in inputs:
        try:
            print(test_solver(A, type))
        except:
            pass
