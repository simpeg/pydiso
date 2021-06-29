#cython: language_level=3
cimport numpy as np
from cython cimport numeric

import warnings
import numpy as np
import scipy.sparse as sp
import os

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

    ctypedef int (*ProgressEntry)(int* thread, int* step, char* stage, int stage_len);
    ProgressEntry mkl_set_progress(ProgressEntry progress);

    ctypedef void * _MKL_DSS_HANDLE_t

    void pardisoinit(_MKL_DSS_HANDLE_t, const int *, int *)

    void pardiso(_MKL_DSS_HANDLE_t, const int*, const int*, const int*,
                 const int *, const int *, const void *, const int *,
                 const int *, int *, const int *, int *,
                 const int *, void *, void *, int *)

    void pardiso_64(_MKL_DSS_HANDLE_t, const long_t *, const long_t *, const long_t *,
                    const long_t *, const long_t *, const void *, const long_t *,
                    const long_t *, long_t *, const long_t *, long_t *,
                    const long_t *, void *, void *, long_t *)


#call pardiso (pt, maxfct, mnum, mtype, phase, n, a, ia, ja, perm, nrhs, iparm, msglvl, b, x, error)
cdef int mkl_progress(int_t *thread, int_t* step, char* stage, int_t stage_len):
    print(thread[0], step[0], stage, stage_len)
    return 0

cdef int mkl_no_progress(int_t *thread, int_t* step, char* stage, int_t stage_len) nogil:
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
            warnings.warn("Converting %s matrix to CSR format, will slow down."
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

def set_mkl_paradiso_threads(num_threads=None):
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

cdef inline void * _array_pointer(numeric[:] arr):
    return <void *> &arr[0]

cdef class _PardisoParams:
    cdef int_t iparm[64]
    cdef int_t n, mtype, maxfct, mnum, msglvl
    cdef int_t[:] ia, ja, perm

cdef class _PardisoParams64:
    cdef long_t iparm[64]
    cdef long_t n, mtype, maxfct, mnum, msglvl
    cdef long_t[:] ia, ja, perm

cdef class MKLPardisoSolver:
    cdef _MKL_DSS_HANDLE_t handle[64]
    cdef _PardisoParams  _par
    cdef _PardisoParams64 _par64
    cdef int_t _is_32
    cdef int_t mat_type
    cdef int_t _factored

    cdef void * a

    cdef object _data_type
    cdef object _Adata #a reference to make sure the pointer "a" doesn't get destroyed

    def __init__(self, A, matrix_type=None, factor=True, verbose=False):
        '''ParidsoSolver(A, matrix_type=None, factor=True, verbose=False)
        A simple interface to the intel MKL pardiso sparse matrix solver.

        This is a solver class for a scipy sparse matrix using the Pardiso sparse
        solver in the Intel Math Kernel Library. It is inteded to solve the
        equations:
        \math

        It will factorize the sparse matrix in three steps: a symbolic
        factorization stage, a numerical factorization stage, and a solve stage.

        The purpose is to construct a sparse factorization that can be repeatedly
        called to solve for multiple right-hand sides.

        Note
        ----

        The supported matrix types are: real symmetric positive definite, real
        symmetric indefinite, real structurally symmetric, real nonsymmetric,
        complex hermitian positive definite, complex hermitian indefinite, complex
        symmetric, complex structurally symmetric, and complex nonsymmetric.
        The solver supports both single and double precision matrices.


        Parameters
        ----------
        A : scipy sparse matrix
            A sparse matrix preferably in a CSR format.
        matrix_type : str, int, or None, optional
            A string describing the matrix type, or it's corresponding int code.
            If None, then assumed to be nonsymmetric matrix.
        factor : bool, optional
            Whether to perform the factorization stage upon instantiation of the class.
        verbose : bool, optional
            Enable verbose output from the pardiso solver.
        '''

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

        if self.mat_type in [-2, 2, -4, 4, 6]:
            A = sp.triu(A, format='csr')
        A = _ensure_csr(A)
        A.sort_indices()


        #set integer length
        integer_len = A.indices.itemsize
        self._is_32 = integer_len == sizeof(int_t)
        if self._is_32:
            self._initialize4(A, matrix_type, verbose)
        elif integer_len == 8:
            self._initialize8(A, matrix_type, verbose)
        else:
            raise PardisoError("Unrecognized integer length")

        if(verbose):
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
        A : scipy sparse matrix
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

    def __call__(self, b):
        return self.solve(b)

    def solve(self, b, x=None):
        """solver.solve(b, x=None)
        Solves the equation AX=B using the factored A matrix

        Note
        ----
        The data will be copied if not contiguous in all cases. If multiple rhs
        are given, the input arrays will be copied if not in a contiguous
        Fortran order.

        Parameters
        ----------
        b : numpy array
            array of shape 1D or 2D for the right hand side of the equation
            (of the same data type as A).
        x : numpy array, optional
            A pre-allocated output array (of the same data type as A).
            If None, a new array is constructed.

        Returns
        -------
        numpy array
            array containing the solution (in Fortran ordering)
        """
        in_shape = b.shape
        if(b.dtype!=self._data_type):
            warnings.warn("rhs does not have the same data type as A",
                            PardisoTypeConversionWarning)
            b = b.astype(self._data_type)
        if x is None:
            x = np.empty_like(b)
        if(x.dtype!=self._data_type):
            warnings.warn("output does not have the same data type as A",
                            PardisoTypeConversionWarning)
            x = x.astype(self._data_type)

        #get contiguous F ordering of vectors
        b = np.require(b, requirements='F').reshape(-1, order='F')
        x = np.require(x, requirements='F').reshape(-1, order='F')

        cdef int_t nrhs = 1
        if len(in_shape)>1:
            nrhs = in_shape[1]

        cdef void * bp
        cdef void * xp
        if(self._data_type==np.float32):
            bp = _array_pointer[float](b)
            xp = _array_pointer[float](x)
        elif(self._data_type==np.float64):
            bp = _array_pointer[double](b)
            xp = _array_pointer[double](x)
        elif(self._data_type==np.complex64):
            bp = _array_pointer[floatcomplex](b)
            xp = _array_pointer[floatcomplex](x)
        elif(self._data_type==np.complex128):
            bp = _array_pointer[doublecomplex](b)
            xp = _array_pointer[doublecomplex](x)

        self._solve(bp, xp, nrhs)
        return x.reshape(in_shape, order='F')

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

    cdef _initialize4(self, A, matrix_type, verbose):
        cdef _PardisoParams par = _PardisoParams()

        par.n = A.shape[0]
        par.perm = np.empty(par.n, dtype=np.int32)

        par.maxfct = 1
        par.mnum = 1

        par.mtype = matrix_type

        par.msglvl = verbose

        pardisoinit(self.handle, &par.mtype, par.iparm)

        #par.iparm[1] = 0
        par.iparm[4] = 2 #fill perm with computed permutation vector
        par.iparm[34] = 1 #zero based indexing

        #set precision
        if self._data_type==np.float64 or self._data_type==np.complex128:
            par.iparm[27] = 0
        elif self._data_type==np.float32 or self._data_type==np.complex64:
            par.iparm[27] = 1
        else:
            raise PardisoError("Unsupported data type")

        indices = np.require(A.indices, dtype=np.int32) #these should be satisfied already...
        indptr = np.require(A.indptr, dtype=np.int32) #these should be satisfied already...

        par.ia = indptr
        par.ja = indices

        self._par = par

    cdef _initialize8(self, A, matrix_type, verbose):
        cdef _PardisoParams64 par = _PardisoParams64()

        par.n = A.shape[0]
        par.perm = np.empty(par.n, dtype=np.int64)

        par.maxfct = 1
        par.mnum = 1

        par.mtype = matrix_type

        par.msglvl = verbose

        cdef int_t mtype_temp = matrix_type
        cdef int_t iparm[64]

        pardisoinit(self.handle, &mtype_temp, iparm)

        for i in range(64):
            par.iparm[i] = iparm[i] # copy from iparm32 to iparm64

        par.iparm[4] = 2 # fill perm with computed permutation vector
        par.iparm[34] = 1 # zero based indexing

        #set precision
        if self._data_type==np.float64 or self._data_type==np.complex128:
            par.iparm[27] = 0
        elif self._data_type==np.float32 or self._data_type==np.complex64:
            par.iparm[27] = 1
        else:
            raise PardisoError("Unsupported data type")

        indices = np.require(A.indices, dtype=np.int64) #these should be satisfied already...
        indptr = np.require(A.indptr, dtype=np.int64) #these should be satisfied already...

        par.ia = indptr
        par.ja = indices

        self._par64 = par

    cdef _set_A(self, data):
        data_type = data.dtype
        self._Adata = data
        #storing a reference so it doesn't get garbage collected
        if(data_type==np.float32):
            self.a = _array_pointer[float](data)
        elif(data_type==np.float64):
            self.a = _array_pointer[double](data)
        elif(data_type==np.complex64):
            self.a = _array_pointer[floatcomplex](data)
        elif(data_type==np.complex128):
            self.a = _array_pointer[doublecomplex](data)
        else:
            raise PardisoError("Unsorported data type for A")

    def __dealloc__(self):
        #Need to call paradiso with phase=-1 to release memory
        cdef int_t phase = -1, nrhs=0, error=0
        cdef long_t phase64=-1, nrhs64=0, error64=0

        cdef void *x = NULL
        cdef void *b = NULL

        if self._is_32:
            pardiso(self.handle, &self._par.maxfct, &self._par.mnum, &self._par.mtype,
                    &phase, &self._par.n, self.a, &self._par.ia[0], &self._par.ja[0],
                    &self._par.perm[0], &nrhs, self._par.iparm, &self._par.msglvl, b, x, &error)
        else:
            pardiso_64(self.handle, &self._par64.maxfct, &self._par64.mnum, &self._par64.mtype,
                    &phase64, &self._par64.n, self.a, &self._par64.ia[0], &self._par64.ja[0],
                    &self._par64.perm[0], &nrhs64, self._par64.iparm, &self._par64.msglvl, b, x, &error64)
        err = error or error64
        if err!=0:
            raise PardisoError("Memmory release error "+_err_messages[err])

    cdef _analyze(self):
        #phase = 11
        err = self._run_pardiso(11)
        if err!=0:
            raise PardisoError("Analysis step error, "+_err_messages[err])

    cdef _factor(self):
        #phase = 22
        self._factored = False

        err = self._run_pardiso(22)
        if err!=0:
            raise PardisoError("Factor step error, "+_err_messages[err])
        self._factored = True

    cdef _solve(self, void* b, void* x, int_t nrhs_in):
        #phase = 33
        if(not self._factored):
            raise PardisoError("Cannot solve without a previous factorization.")

        err = self._run_pardiso(33, b, x, nrhs_in)
        if err!=0:
            raise PardisoError("Solve step error, "+_err_messages[err])

    cdef int _run_pardiso(self, int_t phase, void* b=NULL, void* x=NULL, int_t nrhs=0):
        cdef int_t error=0
        cdef long_t error64=0, phase64=phase, nrhs64=nrhs

        if self._is_32:
            pardiso(self.handle, &self._par.maxfct, &self._par.mnum, &self._par.mtype,
                    &phase, &self._par.n, self.a, &self._par.ia[0], &self._par.ja[0],
                    &self._par.perm[0], &nrhs, self._par.iparm, &self._par.msglvl, b, x, &error)
            return error
        else:
            pardiso_64(self.handle, &self._par64.maxfct, &self._par64.mnum, &self._par64.mtype,
                    &phase64, &self._par64.n, self.a, &self._par64.ia[0], &self._par64.ja[0],
                    &self._par64.perm[0], &nrhs64, self._par64.iparm, &self._par64.msglvl, b, x, &error64)
            return error64
