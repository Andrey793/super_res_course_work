import numpy as np
from scipy.sparse import spdiags


def sptoeplitz(col, row=None):
    """
    SPTOEPLITZ Sparse Toeplitz matrix.
    Produces a sparse nonsymmetric Toeplitz matrix having col as its first column
    and row as its first row. Neither col nor row needs to be sparse. No full-size
    dense matrices are formed.

    If only col is provided, a sparse symmetric/Hermitian Toeplitz matrix is created.

    Parameters:
        col (array-like): First column of the Toeplitz matrix.
        row (array-like, optional): First row of the Toeplitz matrix.

    Returns:
        scipy.sparse.csr_matrix: Sparse Toeplitz matrix.
    """
    col = np.asarray(col).flatten()

    if row is None:  # Symmetric case
        row = col.copy()
        col = np.conj(col)
    else:
        row = np.asarray(row).flatten()
        if col[0] != row[0]:
            print("Warning: First element of input column does not match first element "
                  "of input row. Column wins diagonal conflict.")
        row[0] = col[0]

    # Size of the resulting matrix
    m = len(col)
    n = len(row)

    # Locate the nonzero diagonals
    ic = np.nonzero(col)[0]
    sc = col[ic]
    ir = np.nonzero(row)[0]
    sr = row[ir]

    # Diagonal indices
    if len(ic) and ic[0] == 0:
        ic = ic[1:]
        sc = sc[1:]
    d = np.concatenate((ir, -ic))
    data = np.concatenate((sr, sc))

    # Values for the diagonals
    diags = np.repeat(data, np.minimum(m, n))
    diags = diags.reshape(len(d), np.minimum(m, n))

    # Construct the sparse Toeplitz matrix
    T = spdiags(diags, d, m, n ,format='csr')

    return T
