#cython: language_level=3
cimport numpy as np
import cython
from cython cimport numeric
from cpython.pythread cimport (
    PyThread_type_lock,
    PyThread_allocate_lock,
    PyThread_acquire_lock,
    PyThread_release_lock,
    PyThread_free_lock
)

import warnings
import numpy as np
import scipy.sparse as sp
import os

cdef extern from 'mkl.h':
    ctypedef long long MKL_INT64
    ctypedef unsigned long long MKL_UINT64
    ctypedef int MKL_INT

ctypedef MKL_INT int_t
ctypedef MKL_INT64 long_t

cdef extern from 'mkl.h':
    int MKL_DOMAIN_PARDISO

    ctypedef struct MKLVersion:
        int MajorVersion
        int MinorVersion
        int UpdateVersion
        char * ProductStatus
        char * Build
        char * Processor
        char * Platform

    void mkl_get_version(MKLVersion* pv)

    void mkl_set_num_threads(int nth)
    int mkl_domain_set_num_threads(int nt, int domain)
    int mkl_get_max_threads()
    int mkl_domain_get_max_threads(int domain)

    ctypedef int (*ProgressEntry)(int* thread, int* step, char* stage, int stage_len) except? -1;
    ProgressEntry mkl_set_progress(ProgressEntry progress);

    ctypedef void * _MKL_DSS_HANDLE_t

    void pardiso(_MKL_DSS_HANDLE_t, const int_t*, const int_t*, const int_t*,
                 const int_t *, const int_t *, const void *, const int_t *,
                 const int_t *, int_t *, const int_t *, int_t *,
                 const int_t *, void *, void *, int_t *) nogil

    void pardiso_64(_MKL_DSS_HANDLE_t, const long_t *, const long_t *, const long_t *,
                    const long_t *, const long_t *, const void *, const long_t *,
                    const long_t *, long_t *, const long_t *, long_t *,
                    const long_t *, void *, void *, long_t *) nogil


#call pardiso (pt, maxfct, mnum, mtype, phase, n, a, ia, ja, perm, nrhs, iparm, msglvl, b, x, error)
cdef int mkl_progress(int *thread, int* step, char* stage, int stage_len) nogil:
    # must be a nogil process to pass to mkl pardiso progress reporting
    with gil:
        # must reacquire the gil to print out back to python.
        print(thread[0], step[0], stage, stage_len)
    return 0

cdef int mkl_no_progress(int *thread, int* step, char* stage, int stage_len) nogil:
    return 0

MATRIX_TYPES ={
    'real_structurally_symmetric': 1,
    'real_symmetric_positive_definite': 2,
    'real_symmetric_indefinite': -2,
    'complex_structurally_symmetric': 3,
    'complex_hermitian_positive_definite': 4,
    'complex_hermitian_indefinite': -4,
    'complex_symmetric': 6,
    'real_nonsymmetric': 11,
    'complex_nonsymmetric': 13}
"""dict : matrix type string keys and corresponding integer value to describe
the different supported matrix types.
"""

_err_messages = {0:"no error",
                -1:'input inconsistent',
                -2:'not enough memory',
                -3:'reordering problem',
                -4:'zero pivot, numerical factorization or iterative refinement problem',
                -5:'unclassified (internal) error',
                -6:'reordering failed',
                -7:'diagonal matrix is singular',
                -8:'32-bit integer overflow problem',
                -9:'not enough memory for OOC',
                -10:'error opening OOC files',
                -11:'read/write error with OOC files',
                -12:'pardiso_64 called from 32-bit library',
                }

class PardisoError(Exception):
    pass

class PardisoWarning(UserWarning):
    pass

class PardisoTypeConversionWarning(
        PardisoWarning, sp.SparseEfficiencyWarning):
    pass

def _ensure_csr(A, sym=False):
    if not sp.issparse(A):
        raise(PardisoError("Matrix is not sparse."))
    if not (sp.isspmatrix_csr(A)):
        if sym and sp.isspmatrix_csc(A):
            A = A.T
        else:
            warnings.warn("Converting %s matrix to CSR format."
                         %A.__class__.__name__, PardisoTypeConversionWarning)
            A = A.tocsr()
    return A

def get_mkl_max_threads():
    """
    Returns the current number of openMP threads available to the MKL Library
    """
    return mkl_get_max_threads()

def get_mkl_pardiso_max_threads():
    """
    Returns the current number of openMP threads available to the Pardiso functions
    """
    return mkl_domain_get_max_threads(MKL_DOMAIN_PARDISO)

def set_mkl_threads(num_threads=None):
    """
    Sets the number of openMP threads available to the MKL library.

    Parameters
    ----------
    num_threads : None or int
        number of threads to use for the MKL library.
        None will set the number of threads to that returned by `os.cpu_count()`.
    """
    if num_threads is None:
        num_threads = os.cpu_count()
    elif num_threads<=0:
        raise PardisoError('Number of threads must be greater than 0')
    mkl_set_num_threads(num_threads)

def set_mkl_pardiso_threads(num_threads=None):
    """
    Sets the number of openMP threads available to the Pardiso functions

    Parameters
    ----------
    num_threads : None or int
        Number of threads to use for the MKL Pardiso routines.
        None (or 0) will set the number of threads to `get_mkl_max_threads`
    """
    if num_threads is None:
        num_threads = 0
    elif num_threads<0:
        raise PardisoError('Number of threads must be greater than 0')
    mkl_domain_set_num_threads(num_threads, MKL_DOMAIN_PARDISO)

def get_mkl_version():
    """
    Returns a dictionary describing the version of Intel Math Kernel Library used
    """
    cdef MKLVersion vers
    mkl_get_version(&vers)
    return vers

cdef class _PardisoParams:
    cdef int_t iparm[64]
    cdef int_t n, mtype, maxfct, mnum, msglvl
    cdef int_t[:] ia, ja, perm

cdef class _PardisoParams64:
    cdef long_t iparm[64]
    cdef long_t n, mtype, maxfct, mnum, msglvl
    cdef long_t[:] ia, ja, perm

ctypedef fused _par_params:
    _PardisoParams
    _PardisoParams64

cdef class MKLPardisoSolver:
    cdef _MKL_DSS_HANDLE_t handle[64]
    cdef _PardisoParams _par
    cdef _PardisoParams64 _par64
    cdef int_t _is_32
    cdef int_t mat_type
    cdef int_t _factored
    cdef size_t shape[2]
    cdef PyThread_type_lock lock
    cdef void * a

    cdef object _data_type
    cdef object _Adata # a reference to make sure the pointer "a" doesn't get destroyed

    def __cinit__(self, *args, **kwargs):
        self.lock = PyThread_allocate_lock()

        for i in range(64):
            self.handle[i] = NULL

    def __init__(self, A, matrix_type=None, factor=True, verbose=False):
        '''ParidsoSolver(A, matrix_type=None, factor=True, verbose=False)
        An interface to the intel MKL pardiso sparse matrix solver.

        This is a solver class for a scipy sparse matrix using the Pardiso sparse
        solver in the Intel Math Kernel Library.

        It will factorize the sparse matrix in three steps: a symbolic
        factorization stage, a numerical factorization stage, and a solve stage.

        The purpose is to construct a sparse factorization that can be repeatedly
        called to solve for multiple right-hand sides.

        Parameters
        ----------
        A : scipy.sparse.spmatrix
            A sparse matrix preferably in a CSR format.
        matrix_type : str, int, or None, optional
            A string describing the matrix type, or it's corresponding int code.
            If None, then assumed to be nonsymmetric matrix.
        factor : bool, optional
            Whether to perform the factorization stage upon instantiation of the class.
        verbose : bool, optional
            Enable verbose output from the pardiso solver.

        Notes
        -----

        The supported matrix types are: real symmetric positive definite, real
        symmetric indefinite, real structurally symmetric, real nonsymmetric,
        complex hermitian positive definite, complex hermitian indefinite, complex
        symmetric, complex structurally symmetric, and complex nonsymmetric.
        The solver supports both single and double precision matrices.

        Examples
        --------

        Solve a symmetric positive definite system by first forming a simple 5 point
        laplacian stencil with a zero boundary condition. Then we create a known
        solution vector to compare to result with.

        >>> import scipy.sparse as sp
        >>> from pydiso.mkl_solver import MKLPardisoSolver
        >>> nx, ny = 5, 7
        >>> Dx = sp.diags((-1, 1), (-1, 0), (nx+1, nx))
        >>> Dy = sp.diags((-1, 1), (-1, 0), (ny+1, ny))
        >>> A = sp.kron(sp.eye(nx), Dy.T @ Dy) + sp.kron(Dx.T @ Dx, sp.eye(ny))
        >>> x = np.linspace(-10, 10, nx*ny)
        >>> b = A @ x

        Next we create the solver object using pardiso

        >>> Ainv = MKLPardisoSolver(A, matrix_type='real_symmetric_positive_definite')
        >>> x_solved = Ainv.solve(b)
        >>> np.allclose(x, x_solved)
        True
        '''
        n_row, n_col = A.shape
        if n_row != n_col:
            raise ValueError("Matrix is not square")
        self.shape = n_row, n_col

        self._data_type = A.dtype
        if matrix_type is None:
            if np.issubdtype(self._data_type, np.complexfloating):
                matrix_type = MATRIX_TYPES['complex_nonsymmetric']
            elif np.issubdtype(self._data_type, np.floating):
                matrix_type = MATRIX_TYPES['real_nonsymmetric']
            else:
                raise(PardisoError('Unrecognized matrix data type'))
        if not(matrix_type in MATRIX_TYPES or matrix_type in MATRIX_TYPES.values()):
            raise PardisoError('Unrecognized matrix type')
        if matrix_type in MATRIX_TYPES:
            matrix_type = MATRIX_TYPES[matrix_type]
        self.mat_type = matrix_type

        if self.mat_type in [1, 2, -2, 11]:
            if not np.issubdtype(self._data_type, np.floating):
                raise TypeError(
                    "matrix dtype and matrix_type not consistent, expected a real matrix"
                )
        else:
            if not np.issubdtype(self._data_type, np.complexfloating):
                raise TypeError(
                    "matrix dtype and matrix_type not consistent, expected a complex matrix"
                )

        if self.mat_type in [-2, 2, -4, 4, 6]:
            A = sp.triu(A, format='csr')
        A = _ensure_csr(A)
        A.sort_indices()

        #set integer length
        integer_len = A.indices.itemsize
        #self._is_32 = integer_len == sizeof(int_t)
        self._is_32 = sizeof(int_t) == 8 or integer_len == sizeof(int_t)

        if self._is_32:
            self._par = _PardisoParams()
            self._initialize(self._par, A, matrix_type, verbose)
        else:
            self._par64 = _PardisoParams64()
            self._initialize(self._par64, A, matrix_type, verbose)

        if verbose:
            #for reporting factorization progress via python's `print`
            mkl_set_progress(mkl_progress)
        else:
            mkl_set_progress(mkl_no_progress)

        self._set_A(A.data)
        self._analyze()
        self._factored = False
        if factor:
            self._factor()

    def refactor(self, A):
        """solver.refactor(A)
        re-use a symbolic factorization with a new `A` matrix.

        Note
        ----
        Must have the same non-zero pattern as the initial `A` matrix.
        If `full_refactor=False`, the initial factorization is used as a
        preconditioner to a Krylov subspace solver in the solve step.

        Parameters
        ----------
        A : scipy.sparse.spmatrix
            A sparse matrix preferably in a CSR format.
        """
        #Assumes that the matrix A has the same non-zero pattern and ordering
        #as the initial A matrix
        if self.mat_type in [-2, 2, -4, 4, 6]:
            A = sp.triu(A, format='csr')
        A = _ensure_csr(A)
        A.sort_indices()

        self._set_A(A.data)
        self._factor()

    cdef _initialized(self):
        cdef int i
        for i in range(64):
            if self.handle[i]:
                return 1
        return 0

    def __call__(self, b):
        return self.solve(b)

    def solve(self, b, x=None, transpose=False):
        """solve(self, b, x=None, transpose=False)
        Solves the equation AX=B using the factored A matrix

        Note
        ----
        The data will be copied if not contiguous in all cases. If multiple rhs
        are given, the input arrays will be copied if not in a contiguous
        Fortran order.

        Parameters
        ----------
        b : numpy.ndarray
            array of shape 1D or 2D for the right hand side of the equation
            (of the same data type as A).
        x : numpy.ndarray, optional
            A pre-allocated output array (of the same data type as A).
            If None, a new array is constructed.
        transpose : bool, optional
            If True, it will solve A^TX=B using the factored A matrix.

        Returns
        -------
        numpy array
            array containing the solution (in Fortran ordering)
        """
        if b.dtype != self._data_type:
            warnings.warn("rhs does not have the same data type as A",
                            PardisoTypeConversionWarning)
            b = b.astype(self._data_type)
        b = np.atleast_1d(b)
        if b.shape[0] != self.shape[0]:
            raise ValueError(f"incorrect length of b, expected {self.shape[0]}, got {b.shape[0]}")
        b = np.require(b, requirements='F')

        if x is None:
            x = np.empty_like(b)
        if(x.dtype!=self._data_type):
            warnings.warn("output does not have the same data type as A",
                            PardisoTypeConversionWarning)
            x = x.astype(self._data_type)
        x = np.atleast_1d(x)
        if x.shape[0] != self.shape[0]:
            raise ValueError(f"incorrect length of x, expected {self.shape[0]}, got {x.shape[0]}")
        x = np.require(x, requirements='F')

        cdef void * bp = np.PyArray_DATA(b)
        cdef void * xp = np.PyArray_DATA(x)

        if bp == xp:
            raise PardisoError("b and x must be different arrays")

        cdef int_t nrhs = b.shape[1] if b.ndim == 2 else 1

        if transpose:
            self.set_iparm(11, 2)
        else:
            self.set_iparm(11, 0)
        self._solve(bp, xp, nrhs)
        return x

    @property
    def perm(self):
        """ Fill-reducing permutation vector used inside pardiso.
        """
        if self._is_32:
            return np.array(self._par.perm)
        else:
            return np.array(self._par64.perm)

    @property
    def iparm(self):
        """ Parameter options for the pardiso solver.
        """
        if self._is_32:
            return np.array(self._par.iparm)
        else:
            return np.array(self._par64.iparm)

    def set_iparm(self, int_t i, int_t val):
        if i > 63 or i < 0:
            raise IndexError(f"index {i} is out of bounds for size 64 array")
        if i not in [
            1, 3, 4, 5, 7, 9, 10, 11, 12, 17, 18, 20, 23,
            24, 26, 30, 33, 34, 35, 36, 38, 42, 55, 59
        ]:
            raise PardisoError(f"cannot set parameter {i} of the iparm array")
        if self._is_32:
            self._par.iparm[i] = val
        else:
            self._par64.iparm[i] = val

    @property
    def nnz(self):
        return self.iparm[17]

    cdef _initialize(self, _par_params par, A, matrix_type, verbose):

        if _par_params is _PardisoParams:
            int_dtype = f'i{sizeof(int_t)}'
        else:
            int_dtype = 'i8'
        par.n = A.shape[0]
        par.perm = np.empty(par.n, dtype=int_dtype)

        par.maxfct = 1
        par.mnum = 1

        par.mtype = matrix_type
        par.msglvl = verbose

        for i in range(64):
            par.iparm[i] = 0  # ensure these all start at 0

        # set default parameters
        par.iparm[0] = 1  # tell pardiso to not reset these values on the first call
        par.iparm[1] = 2  # The nested dissection algorithm from the METIS
        par.iparm[3] = 0  # The factorization is always computed as required by phase.
        par.iparm[4] = 2  # fill perm with computed permutation vector
        par.iparm[5] = 0  # The array x contains the solution; right-hand side vector b is kept unchanged.
        par.iparm[7] = 0  # The solver automatically performs two steps of iterative refinement when perterbed pivots are obtained
        par.iparm[9] = 13 if matrix_type in [11, 13] else 8
        par.iparm[10] = 1 if matrix_type in [11, 13] else 0
        par.iparm[11] = 0  # Solve a linear system AX = B (as opposed to A.T or A.H)
        par.iparm[12] = 1 if matrix_type in [11, 13] else 0
        par.iparm[17] = -1  # Return the number of non-zeros in this value after first call
        par.iparm[18] = 0  # do not report flop count
        par.iparm[20] = 1 if matrix_type in [-2, -4, 6] else 0
        par.iparm[23] = 0  # classic (not parallel) factorization
        par.iparm[24] = 0  # default behavoir of parallel solving
        par.iparm[26] = 0  # Do not check the input matrix
        #set precision
        if self._data_type==np.float64 or self._data_type==np.complex128:
            par.iparm[27] = 0
        elif self._data_type==np.float32 or self._data_type==np.complex64:
            par.iparm[27] = 1
        else:
            raise TypeError("Unsupported data type")
        par.iparm[30] = 0  # this would be used to enable sparse input/output for solves
        par.iparm[33] = 0  # optimal number of thread for CNR mode
        par.iparm[34] = 1  # zero based indexing
        par.iparm[35] = 0  # Do not compute schur complement
        par.iparm[36] = 0  # use CSR storage format
        par.iparm[38] = 0  # Do not use low rank update
        par.iparm[42] = 0  # Do not compute the diagonal of the inverse
        par.iparm[55] = 0  # Internal function used to work with pivot and calculation of diagonal arrays turned off.
        par.iparm[59] = 0  # operate in-core mode

        par.ia = np.require(A.indptr, dtype=int_dtype)
        par.ja = np.require(A.indices, dtype=int_dtype)

    cdef _set_A(self, data):
        self._Adata = data
        self.a = np.PyArray_DATA(data)

    def __dealloc__(self):
        # Need to call pardiso with phase=-1 to release memory
        cdef int_t phase=-1, nrhs=0, error=0
        cdef long_t phase64=-1, nrhs64=0, error64=0

        if self._initialized():
            with nogil:
                PyThread_acquire_lock(self.lock, 1)
                if self._is_32:
                    pardiso(
                        self.handle, &self._par.maxfct, &self._par.mnum, &self._par.mtype,
                        &phase, &self._par.n, NULL, NULL, NULL, NULL, &nrhs, self._par.iparm,
                        &self._par.msglvl, NULL, NULL, &error
                    )
                else:
                    pardiso_64(
                        self.handle, &self._par64.maxfct, &self._par64.mnum, &self._par64.mtype,
                        &phase64, &self._par64.n, NULL, NULL, NULL, NULL, &nrhs64,
                        self._par64.iparm, &self._par64.msglvl, NULL, NULL, &error64
                    )
                PyThread_release_lock(self.lock)
            err = error or error64
            if err!=0:
                raise PardisoError("Memory release error "+_err_messages[err])
            for i in range(64):
                self.handle[i] = NULL

        if self.lock:
            #dealloc lock
            PyThread_free_lock(self.lock)

    cdef _analyze(self):
        #phase = 11
        with nogil:
            err = self._run_pardiso(11)
        if err!=0:
            raise PardisoError("Analysis step error, "+_err_messages[err])

    cdef _factor(self):
        #phase = 22
        self._factored = False

        with nogil:
            err = self._run_pardiso(22)

        if err!=0:
            raise PardisoError("Factor step error, "+_err_messages[err])

        self._factored = True

    cdef _solve(self, void* b, void* x, int_t nrhs_in):
        #phase = 33
        if(not self._factored):
            raise PardisoError("Cannot solve without a previous factorization.")

        with nogil:
            err = self._run_pardiso(33, b, x, nrhs_in)

        if err!=0:
            raise PardisoError("Solve step error, "+_err_messages[err])

    @cython.boundscheck(False)
    cdef int _run_pardiso(self, int_t phase, void* b=NULL, void* x=NULL, int_t nrhs=0) noexcept nogil:
        cdef int_t error=0
        cdef long_t error64=0, phase64=phase, nrhs64=nrhs

        PyThread_acquire_lock(self.lock, 1)
        if self._is_32:
            pardiso(self.handle, &self._par.maxfct, &self._par.mnum, &self._par.mtype,
                    &phase, &self._par.n, self.a, &self._par.ia[0], &self._par.ja[0],
                    &self._par.perm[0], &nrhs, self._par.iparm, &self._par.msglvl, b, x, &error)
        else:
            pardiso_64(self.handle, &self._par64.maxfct, &self._par64.mnum, &self._par64.mtype,
                    &phase64, &self._par64.n, self.a, &self._par64.ia[0], &self._par64.ja[0],
                    &self._par64.perm[0], &nrhs64, self._par64.iparm, &self._par64.msglvl, b, x, &error64)
        PyThread_release_lock(self.lock)
        error = error or error64
        return error