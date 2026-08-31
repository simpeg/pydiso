import enum
import numpy as np
import scipy.sparse as sp
from ._mkl_solver import (
    _PardisoHandle_int_t,
    _PardisoHandle_long_t,
    get_mkl_int_size,
    get_mkl_int64_size,
    get_mkl_max_threads,
    get_mkl_pardiso_max_threads,
    set_mkl_threads,
    set_mkl_pardiso_threads,
    get_mkl_version,
    _err_messages,
    PardisoWarning,
    PardisoError,
)
import warnings


class MatrixType(enum.IntEnum):
    """Matrix type describing the structure/symmetry of the sparse matrix ``A``."""
    REAL_STRUCTURALLY_SYMMETRIC = 1
    REAL_SYMMETRIC_POSITIVE_DEFINITE = 2
    REAL_SYMMETRIC_INDEFINITE = -2
    COMPLEX_STRUCTURALLY_SYMMETRIC = 3
    COMPLEX_HERMITIAN_POSITIVE_DEFINITE = 4
    COMPLEX_HERMITIAN_INDEFINITE = -4
    COMPLEX_SYMMETRIC = 6
    REAL_NONSYMMETRIC = 11
    COMPLEX_NONSYMMETRIC = 13


MATRIX_TYPES = {member.name.lower(): member.value for member in MatrixType}
"""dict : matrix type string keys and corresponding integer value to describe
the different supported matrix types.

Kept for backward compatibility — prefer `MatrixType` for new code.
"""


class FillReducingOrdering(enum.IntEnum):
    """Fill-reducing ordering algorithm used during the analysis phase (``iparm[1]``)."""
    MINIMUM_DEGREE = 0
    NESTED_DISSECTION = 2
    PARALLEL_NESTED_DISSECTION = 3


class OutOfCoreMode(enum.IntEnum):
    """Whether Pardiso factors in-core or spills to disk (``iparm[59]``)."""
    IN_CORE = 0
    OUT_OF_CORE_IF_NEEDED = 1
    OUT_OF_CORE = 2


class PardisoTypeConversionWarning(
        PardisoWarning, sp.SparseEfficiencyWarning):
    pass

class MKLPardisoSolver:

    def __init__(
        self, A, matrix_type=None, factor=True, verbose=False,
        fill_reducing_ordering=FillReducingOrdering.NESTED_DISSECTION,
        max_iterative_refinement_steps=0,
        pivoting_perturbation=None,
        scaling=None,
        weighted_matching=None,
        bunch_kaufman_pivoting=None,
        cnr_threads=0,
        low_rank_update=False,
        report_nnz=True,
        report_mflops=False,
        parallel_factorization=False,
        parallel_solve=False,
        out_of_core_mode=OutOfCoreMode.IN_CORE,
    ):
        '''An interface to the Intel MKL pardiso sparse matrix solver.

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
        matrix_type : MatrixType, str, int, or None, optional
            A `MatrixType` member, its matching int, or (for backward compatibility) the
            lowercase string form of its name (e.g. ``"real_nonsymmetric"``) describing the
            matrix type. If None, then assumed to be nonsymmetric matrix.
        factor : bool, optional
            Whether to perform the factorization stage upon instantiation of the class.
        verbose : bool, optional
            Enable verbose output from the pardiso solver.
        fill_reducing_ordering : FillReducingOrdering, int, or array_like, optional
            Either the fill-reducing ordering algorithm to use during the analysis phase, or a
            user-supplied permutation to use directly instead of computing one.

            - A ``FillReducingOrdering`` member or matching int selects the algorithm
              (``iparm[1]``). Default is ``FillReducingOrdering.NESTED_DISSECTION`` (METIS
              nested dissection).
            - A 1D array of length matching ``A``, containing a permutation of ``0..n-1``, is
              used directly as Pardiso's ordering (``iparm[4] = 1``). Afterward, the `perm` 
              property simply reflects what was supplied, rather than a library-computed
              alternative.
    
        max_iterative_refinement_steps : int, optional
            Maximum number of iterative refinement steps to perform (``iparm[7]``).
            Default is 0 (the solver still performs up to two automatic steps if perturbed
            pivots are encountered during numerical factorization).
        pivoting_perturbation : int, or None, optional
            Exponent (base 10) of the pivoting perturbation threshold (``iparm[9]``).
            If None (default), uses the library default, which depends on ``matrix_type``
            (13 for nonsymmetric matrices, 8 otherwise).
        scaling : bool, or None, optional
            Whether to enable maximum weighted matching-based scaling (``iparm[10]``).
            If None (default), uses the library default, which depends on ``matrix_type``
            (enabled for nonsymmetric matrices, disabled otherwise).
        weighted_matching : bool, or None, optional
            Whether to enable maximum weighted matching for improved accuracy (``iparm[12]``).
            If None (default), uses the library default, which depends on ``matrix_type``
            (enabled for nonsymmetric matrices, disabled otherwise).
        bunch_kaufman_pivoting : bool, or None, optional
            For symmetric indefinite matrices, whether to apply 1x1 and 2x2 Bunch-Kaufman
            pivoting rather than 1x1 diagonal pivoting only (``iparm[20]``). If None (default),
            uses the library default, which depends on ``matrix_type``.
        cnr_threads : int, optional
            Number of OpenMP threads to use for conditional numerical reproducibility (CNR)
            mode (``iparm[33]``). Default is 0, meaning the number of threads is chosen
            automatically (CNR mode itself is only active when explicitly configured
            elsewhere).
        low_rank_update : bool, optional
            Whether to enable low-rank update, which can accelerate refactorization of a
            sequence of matrices that share a sparsity pattern and differ only slightly
            (``iparm[38]``). Default is False.
        report_nnz : bool, optional
            Whether Pardiso computes and reports the number of non-zero elements in the
            factors (``iparm[17]``), which backs the `nnz` property. Default is True. If
            False, `nnz` no longer reflects a real count.
        report_mflops : bool, optional
            Whether Pardiso computes and reports the number of Mflops needed for numerical
            factorization (``iparm[18]``). Default is False (skips the extra computation).
        parallel_factorization : bool, optional
            Whether to use the two-level scheduling algorithm for factorization (``iparm[23]``).
            Default is False (classic algorithm).
        parallel_solve : bool, optional
            Whether to use the parallel algorithm for the solve step (``iparm[24]``).
            Default is False (classic algorithm).
        out_of_core_mode : OutOfCoreMode or int, optional
            Whether Pardiso factors entirely in-core, or spills to disk (``iparm[59]``).
            Default is ``OutOfCoreMode.IN_CORE``. See details about using out of core mode
            here: `Out of core guide <https://www.intel.com/content/www/us/en/developer/articles/training/how-to-use-ooc-pardiso.html>`_

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
        if not sp.issparse(A):
            raise TypeError(f"type(A)={type(A).__name__} must be a sparse array or sparse matrix.")

        if A.ndim != 2:
            raise ValueError(f"A.ndim={A.ndim} must be to 2.")

        n_row, n_col = A.shape
        if n_row != n_col:
            raise ValueError(f"A with shape {A.shape} is not a square matrix.")
        self.shape = n_row, n_col

        data_dtype = A.dtype
        if not(
            np.issubdtype(data_dtype, np.single) or
            np.issubdtype(data_dtype, np.double) or
            np.issubdtype(data_dtype, np.csingle) or
            np.issubdtype(data_dtype, np.cdouble)
        ):
            raise ValueError(
                f"Unrecognized matrix data type, {data_dtype}. Must be single or double precision of real or complex values."
            )
        self._data_dtype = data_dtype

        is_complex = np.issubdtype(data_dtype, np.complexfloating)

        if matrix_type is None:
            matrix_type = MatrixType.COMPLEX_NONSYMMETRIC if is_complex else MatrixType.REAL_NONSYMMETRIC
        elif isinstance(matrix_type, str):
            try:
                matrix_type = MatrixType[matrix_type.upper()]
            except KeyError:
                raise TypeError(f'Unrecognized matrix_type: {matrix_type!r}')
        else:
            try:
                matrix_type = MatrixType(matrix_type)
            except ValueError:
                raise TypeError(f'Unrecognized matrix_type: {matrix_type!r}')

        if matrix_type in [1, 2, -2, 11]:
            if is_complex:
                raise ValueError(
                    f"Complex matrix dtype and matrix_type={matrix_type} are inconsistent, expected a real matrix"
                )
        else:
            if not is_complex:
                raise ValueError(
                    f"Real matrix dtype and matrix_type={matrix_type} are inconsistent, expected a complex matrix"
                )

        self.matrix_type = matrix_type

        A = self._validate_csr_matrix(A)

        max_a_ind_itemsize = max(A.indptr.itemsize, A.indices.itemsize)
        mkl_int_size = get_mkl_int_size()
        mkl_int64_size = get_mkl_int64_size()

        target_int_size = mkl_int_size if max_a_ind_itemsize <= mkl_int_size else mkl_int64_size
        self._ind_dtype = np.dtype(f"i{target_int_size}")

        data, indptr, indices = self._validate_matrix_dtypes(A)
        self._data = data
        self._indptr = indptr
        self._indices = indices

        if target_int_size == mkl_int_size:
            HandleClass = _PardisoHandle_int_t
        else:
            HandleClass = _PardisoHandle_long_t
        self._handle = HandleClass(self._data_dtype, self.shape[0], matrix_type, maxfct=1, mnum=1, msglvl=verbose)

        named_iparms = {}
        if isinstance(fill_reducing_ordering, (int, np.integer)):
            named_iparms[1] = int(FillReducingOrdering(fill_reducing_ordering))
        else:
            perm_arr = np.asarray(fill_reducing_ordering)
            if perm_arr.ndim != 1 or perm_arr.shape[0] != self.shape[0]:
                raise ValueError(
                    "fill_reducing_ordering must be a FillReducingOrdering member/int, or a 1D "
                    f"permutation array of length {self.shape[0]}; got shape {perm_arr.shape}"
                )
            if not np.array_equal(np.sort(perm_arr), np.arange(self.shape[0])):
                raise ValueError("fill_reducing_ordering array must be a permutation of 0..n-1")
            self._handle.perm[:] = np.require(perm_arr, dtype=self._ind_dtype, requirements="C")
            named_iparms[4] = 1

        if not isinstance(max_iterative_refinement_steps, (int, np.integer)):
            raise TypeError(
                "max_iterative_refinement_steps must be an int, got "
                f"{type(max_iterative_refinement_steps).__name__}"
            )
        named_iparms[7] = int(max_iterative_refinement_steps)

        if pivoting_perturbation is not None:
            if not isinstance(pivoting_perturbation, (int, np.integer)) or pivoting_perturbation < 0:
                raise ValueError(
                    f"pivoting_perturbation must be a non-negative int, got {pivoting_perturbation!r}"
                )
            named_iparms[9] = int(pivoting_perturbation)

        # scaling, weighted_matching, and bunch_kaufman_pivoting default to None (rather than
        # a fixed literal) because the library's own default for them depends on matrix_type.
        for name, value, index in (
            ('scaling', scaling, 10),
            ('weighted_matching', weighted_matching, 12),
            ('bunch_kaufman_pivoting', bunch_kaufman_pivoting, 20),
        ):
            if value is not None:
                if not isinstance(value, bool):
                    raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
                named_iparms[index] = int(value)

        for name, value, index in (
            ('parallel_factorization', parallel_factorization, 23),
            ('parallel_solve', parallel_solve, 24),
            ('low_rank_update', low_rank_update, 38),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
            named_iparms[index] = int(value)

        if not isinstance(cnr_threads, (int, np.integer)) or cnr_threads < 0:
            raise ValueError(f"cnr_threads must be a non-negative int, got {cnr_threads!r}")
        named_iparms[33] = int(cnr_threads)

        # report_nnz and report_mflops use -1 (request the report) / 0 (don't bother), rather
        # than the usual bool -> 1/0 mapping used elsewhere.
        for name, value, index in (
            ('report_nnz', report_nnz, 17),
            ('report_mflops', report_mflops, 18),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
            named_iparms[index] = -1 if value else 0

        named_iparms[59] = int(OutOfCoreMode(out_of_core_mode))

        for i, val in named_iparms.items():
            self.set_iparm(i, val)

        self._analyze()
        self._factored = False
        if factor:
            self._factor()

    def refactor(
        self, A,
        pivoting_perturbation=None,
        bunch_kaufman_pivoting=None,
        preconditioned_cgs=None,
        report_mflops=None,
    ):
        """Reuse a symbolic factorization with a new matrix.

        Note
        ----
        Must have the same non-zero pattern as the initial `A` matrix.

        Parameters
        ----------
        A : scipy.sparse.spmatrix
            A sparse matrix preferably in a CSR format.
        pivoting_perturbation : int, or None, optional
            Exponent (base 10) of the pivoting perturbation threshold (``iparm[9]``).
            If None (default), leaves the value from ``__init__`` or a previous call to
            `refactor` unchanged (it is sticky, not reset to a default).
        bunch_kaufman_pivoting : bool, or None, optional
            For symmetric indefinite matrices, whether to apply 1x1 and 2x2 Bunch-Kaufman
            pivoting rather than 1x1 diagonal pivoting only (``iparm[20]``). If None (default),
            leaves the value from ``__init__`` or a previous call to `refactor` unchanged (it
            is sticky, not reset to a default).
        preconditioned_cgs : (int, int), or None, optional
            A ``(L, K)`` pair encoding ``iparm[3]``, which lets Pardiso reuse the previous
            factorization as a preconditioner for an accelerated CGS/CG refactorization instead
            of a full LU/LDL^T refactorization (falling back to full factorization
            automatically if it doesn't converge). ``L`` (1-9) is a stopping-tolerance exponent
            (``10**-L``); ``K`` (1 or 2) selects the CGS/CG variant — per Intel's iparm(4)
            documentation, ``K=1`` for nonsymmetric/structurally-symmetric matrices and ``K=2``
            for symmetric/Hermitian matrices, though that choice is left to the caller rather
            than inferred from ``matrix_type``. If None (default), leaves ``iparm[3]``
            unchanged (it is sticky, not reset to a default). To force a full factorization
            again after enabling this, call ``self.set_iparm(3, 0)`` directly.
        report_mflops : bool, or None, optional
            Whether Pardiso computes and reports the number of Mflops needed for numerical
            factorization (``iparm[18]``). If None (default), leaves the value from
            ``__init__`` or a previous call to `refactor` unchanged (it is sticky, not reset to
            a default).
        """
        #Assumes that the matrix A has the same non-zero pattern and ordering
        #as the initial A matrix

        if not sp.issparse(A):
            raise TypeError("A is not a sparse matrix.")
        if A.shape != self.shape:
            raise ValueError("A is not the same size as the previous matrix.")

        A = self._validate_csr_matrix(A)
        data, indptr, indices = self._validate_matrix_dtypes(A)
        if len(data) != len(self._data):
            raise ValueError("new A matrix does not have the same number of non zeros.")

        self._data = data

        if pivoting_perturbation is not None:
            if not isinstance(pivoting_perturbation, (int, np.integer)) or pivoting_perturbation < 0:
                raise ValueError(
                    f"pivoting_perturbation must be a non-negative int, got {pivoting_perturbation!r}"
                )
            self.set_iparm(9, int(pivoting_perturbation))

        if bunch_kaufman_pivoting is not None:
            if not isinstance(bunch_kaufman_pivoting, bool):
                raise TypeError(
                    f"bunch_kaufman_pivoting must be a bool, got {type(bunch_kaufman_pivoting).__name__}"
                )
            self.set_iparm(20, int(bunch_kaufman_pivoting))

        if preconditioned_cgs is not None:
            l, k = preconditioned_cgs
            if not isinstance(l, (int, np.integer)) or not (1 <= l <= 9):
                raise ValueError(f"preconditioned_cgs[0] (L) must be an int in 1..9, got {l!r}")
            if not isinstance(k, (int, np.integer)) or k not in (1, 2):
                raise ValueError(f"preconditioned_cgs[1] (K) must be 1 or 2, got {k!r}")
            self.set_iparm(3, 10 * int(l) + int(k))

        if report_mflops is not None:
            if not isinstance(report_mflops, bool):
                raise TypeError(f"report_mflops must be a bool, got {type(report_mflops).__name__}")
            self.set_iparm(18, -1 if report_mflops else 0)

        self._factor()

    def __call__(self, b):
        return self.solve(b)

    def solve(self, b, x=None, transpose=False, max_iterative_refinement_steps=None):
        """Solves the equation AX=B using the factored A matrix

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
        max_iterative_refinement_steps : int, or None, optional
            Maximum number of iterative refinement steps to perform (``iparm[7]``). Unlike
            `transpose` (which is set fresh on every call), this is sticky: if None (default),
            it leaves whatever value was last set by ``__init__`` or a previous call to `solve`
            unchanged, rather than resetting to a default.

        Returns
        -------
        numpy.ndarray
            array containing the solution (in Fortran ordering)

        Notes
        -----
        The data will be copied if not contiguous in all cases. If multiple rhs
        are given, the input arrays will be copied if not in a contiguous
        Fortran order.
        """
        if b.dtype != self._data_dtype:
            warnings.warn("rhs does not have the same data type as A",
                            PardisoTypeConversionWarning)
            b = b.astype(self._data_dtype)
        b = np.atleast_1d(b)
        b_was_1d = b.ndim == 1
        if b_was_1d:
            b = b[:, None]
        if b.ndim != 2:
            raise ValueError(f"b.ndim={b.ndim} must be 1 or 2.")
        if b.shape[0] != self.shape[0]:
            raise ValueError(f"incorrect length of b, expected {self.shape[0]}, got {b.shape[0]}")
        b = np.require(b, requirements='F')

        if x is None:
            x = np.empty_like(b)
            x_was_1d = b_was_1d
        else:
            if(x.dtype!=self._data_dtype):
                warnings.warn("output does not have the same data type as A",
                                PardisoTypeConversionWarning)
                x = x.astype(self._data_dtype)
            x = np.atleast_1d(x)
            x_was_1d = x.ndim == 1
            if x_was_1d:
                x = x[:, None]
            if x.ndim != 2:
                raise ValueError(f"x.ndim={x.ndim} must be 1 or 2.")
            if x.shape[0] != self.shape[0]:
                raise ValueError(f"incorrect length of x, expected {self.shape[0]}, got {x.shape[0]}")
            x = np.require(x, requirements='F')

        if b.shape[1] != x.shape[1]:
            raise ValueError(
                f"Inconsistent shapes of right hand side, {b.shape} and output vector, {x.shape}")

        if x is b or (x.base is not None and (x.base is b.base)):
            raise ValueError("x and b cannot point to the same memory")

        if not self._factored:
            self._factor()

        self._handle.set_iparm(11, 2 if transpose else 0)

        if max_iterative_refinement_steps is not None:
            if not isinstance(max_iterative_refinement_steps, (int, np.integer)):
                raise TypeError(
                    "max_iterative_refinement_steps must be an int, got "
                    f"{type(max_iterative_refinement_steps).__name__}"
                )
            self._handle.set_iparm(7, int(max_iterative_refinement_steps))

        phase = 33
        error = self._handle.call_pardiso(phase, self._data, self._indptr, self._indices, b, x)
        if error:
            raise PardisoError("Solve step error, "+_err_messages[error])
        if x_was_1d:
            x = x[:, 0]
        return x

    @property
    def perm(self):
        """ Fill-reducing permutation vector used inside pardiso.
        """
        return np.array(self._handle.perm)

    @property
    def iparm(self):
        """ Parameter options for the pardiso solver.
        """
        return np.array(self._handle.iparm)

    def _validate_csr_matrix(self, mat):
        if self.matrix_type in [-2, 2, -4, 4, 6]:
            # only grab the upper triangle.
            mat = sp.triu(mat, format='csr')

        if mat.format != 'csr':
            warnings.warn(
                "Converting %s matrix to CSR format."% mat.__class__.__name__,
                PardisoTypeConversionWarning,
                stacklevel=3
            )
            mat = mat.tocsr()

        mat.sort_indices()
        mat.sum_duplicates()
        return mat

    def _validate_matrix_dtypes(self, mat):
        data = np.require(mat.data, self._data_dtype, requirements="C")
        indptr = np.require(mat.indptr, self._ind_dtype, requirements="C")
        indices = np.require(mat.indices, self._ind_dtype, requirements="C")
        return data, indptr, indices


    def set_iparm(self, i, val):
        if i > 63 or i < 0:
            raise IndexError(f"index {i} is out of bounds for size 64 array")
        # Deliberately excluded, even though they affect execution:
        # - 5 (write solution on x): setting this without solve() passing a null x pointer
        #   silently corrupts results rather than erroring; needs a call_pardiso change.
        # - 34 (one/zero-based indexing): the whole extension hardcodes zero-based indexing
        #   internally, so changing this would silently break every array we hand to Pardiso.
        # - 30, 35, 36, 42, 55: only meaningful paired with extra arrays/output buffers
        #   (partial solve, Schur complement, diagonal of the inverse) that call_pardiso
        #   doesn't provide yet.
        # - 26 (matrix checker): redundant. `_validate_csr_matrix` always calls
        #   `sort_indices`/`sum_duplicates` first, so A is already in the canonical form
        #   Pardiso's own checker would be verifying.
        if i not in [
            1, 3, 4, 7, 9, 10, 11, 12, 17, 18, 20, 23,
            24, 33, 38, 59
        ]:
            raise ValueError(f"cannot set parameter {i} of the iparm array")

        self._handle.set_iparm(i, val)

    @property
    def nnz(self):
        """ Number of non-zero elements in the factors, if `report_nnz` is enabled.
        """
        return self._handle.iparm[17]

    def _analyze(self):
        phase = 11
        xb_dummy = np.empty([1, 1], dtype=self._data_dtype)
        error = self._handle.call_pardiso(phase, self._data, self._indptr, self._indices, xb_dummy, xb_dummy)
        if error:
            raise PardisoError("Analysis step error, "+_err_messages[error])

    def _factor(self):
        phase = 22
        self._factored = False
        xb_dummy = np.empty([1, 1], dtype=self._data_dtype)
        error = self._handle.call_pardiso(phase, self._data, self._indptr, self._indices, xb_dummy, xb_dummy)

        if error:
            raise PardisoError("Factor step error, "+_err_messages[error])

        self._factored = True