import numpy as np

def np_cholesky_diag_block(A, block_size=4):
    """
    For the top-left block (0,0) of A, compute its Cholesky factor.
    
    Mathematically, if A₁₁ is the top-left block then:
         L₁₁ = chol(A₁₁)
    All elements above the diagonal in this block are set to zero.
    """
    L = A.copy()
    L11 = np.linalg.cholesky(A[:block_size, :block_size])
    # Ensure L11 is strictly lower triangular (zero out above-diagonals)
    for i in range(block_size):
        for j in range(i+1, block_size):
            L11[i, j] = 0.0
    L[:block_size, :block_size] = L11
    # Also zero out block (0,1) since it is not part of the lower-triangular factor
    L[:block_size, block_size:] = 0.0
    return L

def np_offdiag_update(A, L_in, block_size=4):
    """
    For the off-diagonal block (block (1,0)) of A:

    For i in [block_size, 2*block_size) and j in [0, block_size),
    perform forward substitution:

         L[i,j] = (A[i,j] - \\sum_{{t=0}}^{{j-1}} L[i,t] * L_in[j,t]) / L_in[j,j]

    The rest of the matrix remains unchanged.
    """
    L = L_in.copy()
    for i in range(block_size, 2 * block_size):
        for j in range(0, block_size):
            s = 0.0
            for t in range(j):
                s += L[i, t] * L_in[j, t]
            L[i, j] = (A[i, j] - s) / L_in[j, j]
    return L

def np_trailing_update(A, L_in, block_size=4):
    """
    For the trailing block (block (1,1)) of A:
    
    1. Compute the Schur complement update:
           A' = A₂₂ - L₂₁ (L₂₁)ᵀ,
    2. Then compute its Cholesky factor:
           L₂₂ = chol(A')
    3. Set the upper triangular part of L₂₂ to zero.
    4. Also, explicitly zero out block (0,1) so that the full matrix is lower–triangular.
    """
    L = L_in.copy()
    A22 = A[block_size:2 * block_size, block_size:2 * block_size]
    L21 = L_in[block_size:2 * block_size, :block_size]
    A22_updated = A22 - L21 @ L21.T
    L22 = np.linalg.cholesky(A22_updated)
    # Zero out upper triangular entries in L22
    for i in range(block_size):
        for j in range(i+1, block_size):
            L22[i, j] = 0.0
    L[block_size:2 * block_size, block_size:2 * block_size] = L22
    # Also clear block (0,1) to form a proper lower triangular factor.
    L[:block_size, block_size:2 * block_size] = 0.0
    return L

def np_trailing_schur_update(A, L_in, block_size=4):
    """
    Compute only the Schur complement update for block (1,1):
         A' = A₂₂ - L₂₁ (L₂₁)ᵀ.
    
    Returns the updated trailing block (a block_size×block_size matrix).
    """
    A22 = A[block_size:2 * block_size, block_size:2 * block_size]
    L21 = L_in[block_size:2 * block_size, :block_size]
    return A22 - L21 @ L21.T

if __name__ == '__main__':
    # Simple test for CPU reference functions.
    N = 8
    block_size = 4
    R = np.random.randn(N, N).astype(np.float32)
    A = R @ R.T + N * np.eye(N, dtype=np.float32)
    L_diag = np_cholesky_diag_block(A, block_size)
    L_off = np_offdiag_update(A, L_diag, block_size)
    L_full = np_trailing_update(A, L_off, block_size)
    print("CPU Reference Blocked Cholesky result:")
    print(L_full)
    # Compare with the gold-standard Cholesky factor
    L_np = np.linalg.cholesky(A)
    print("NumPy Cholesky factor:")
    print(L_np)
    diff = np.linalg.norm(L_full - L_np)
    print("Difference:", diff)
