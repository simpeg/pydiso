import numpy as np
import scipy.sparse as sp
from pardiso_solver import PardisoSolver, set_mkl_parallelism, get_mkl_max_threads, get_mkl_version

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

print(get_mkl_max_threads())
print(get_mkl_version())


def test_solver(A, matrix_type):
    dtype = A.dtype
    if np.issubdtype(dtype, np.complexfloating):
        x = xc.astype(dtype)
    else:
        x = xr.astype(dtype)
    b = A@x

    solver = PardisoSolver(A, matrix_type=matrix_type, verbose=True)
    x2 = solver.solve(b)

    eps = np.finfo(dtype).eps
    rel_err = np.linalg.norm(x-x2)/np.linalg.norm(x)
    print(matrix_type, A.dtype, rel_err, rel_err < 1E3*eps)


for dtype in (np.float32, np.float64):
    for key, item in A_real_dict.items():
        item = item.astype(dtype)
        test_solver(item, key)

for dtype in (np.complex64, np.complex128):
    for key, item in A_complex_dict.items():
        item = item.astype(dtype)
        test_solver(item, key)
