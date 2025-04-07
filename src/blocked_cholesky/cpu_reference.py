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

         L[i,j] = (A[i,j] - \sum_{{t=0}}^{{j-1}} L[i,t] * L_in[j,t]) / L_in[j,j]

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


# --- General CPU Blocked Cholesky Functions ---

def np_blocked_cholesky(A, block_size):
    """
    General CPU implementation of blocked Cholesky.

    Partition A into B×B blocks (B = N/block_size). For each block index k:

      1. Diagonal update:
         Compute the residual diagonal block by subtracting contributions
         from previously computed blocks:
            A_kk^res = A[k,k] - sum_{p=0}^{k-1} L[k,p] @ L[k,p]^T,
         then compute its Cholesky factorization:
            L[k,k] = chol(A_kk^res)

      2. Off-diagonal update:
         For each i > k, compute the residual block:
            A_ik^res = A[i,k] - sum_{p=0}^{k-1} L[i,p] @ L[k,p]^T,
         and solve for:
            L[i,k] = solve(L[k,k], A_ik^res).

      3. (Trailing update is implicitly done by subtracting contributions in the next iterations.)

    Finally, L is lower-triangular and satisfies A = L L^T.
    """
    A = A.copy()
    N = A.shape[0]
    B = N // block_size
    L = np.zeros_like(A)
    for k in range(B):
        # Diagonal update.
        start = k * block_size
        end = (k+1) * block_size
        A_kk = A[start:end, start:end].copy()
        for p in range(k):
            pp = p * block_size
            A_kk -= L[start:end, pp:pp+block_size] @ L[start:end, pp:pp+block_size].T
        L_kk = np.linalg.cholesky(A_kk)
        L[start:end, start:end] = L_kk

        # Off-diagonal update.
        for i in range(k+1, B):
            i_start = i * block_size
            i_end = (i+1) * block_size
            A_ik = A[i_start:i_end, start:end].copy()
            for p in range(k):
                pp = p * block_size
                A_ik -= L[i_start:i_end, pp:pp+block_size] @ L[start:end, pp:pp+block_size].T
            # Solve L_kk * X = A_ik^T, then take transpose.
            X = np.linalg.solve(L_kk, A_ik.T).T
            L[i_start:i_end, start:end] = X
    return L

def cpu_cholesky_diag_block_general(A, k, block_size):
    """
    Compute the Cholesky factorization of block (k,k) of A.
    """
    bs = block_size
    block = A[k*bs:(k+1)*bs, k*bs:(k+1)*bs].copy()
    L = np.linalg.cholesky(block)
    for i in range(bs):
        for j in range(i+1, bs):
            L[i, j] = 0.0
    return L

def cpu_offdiag_update_general(A, L, k, i, block_size):
    """
    Compute the off-diagonal update for block (i,k).
    For k == 0 (no previous contributions), simply solve:
         L[i,k] = solve(L[0:bs,0:bs], A[i*bs:(i+1)*bs,0:bs])
    """
    bs = block_size
    if k == 0:
        A_ik = A[i*bs:(i+1)*bs, 0:bs].copy()
        X = np.linalg.solve(L[0:bs, 0:bs], A_ik.T).T
        return X
    else:
        # For k > 0, use the original loop-based approach.
        block = np.empty((bs, bs), dtype=A.dtype)
        for r in range(bs):
            for c in range(bs):
                s = 0.0
                for t in range(c):
                    s += L[i*bs + r, k*bs + t] * L[k*bs + c, k*bs + t]
                block[r, c] = (A[i*bs + r, k*bs + c] - s) / L[k*bs + c, k*bs + c]
        return block

def cpu_trailing_update_general(A, L, k, i, j, block_size):
    """
    Compute the trailing update for block (i,j) given the updates in iteration k.
    First compute:
         updated = A[i*bs:(i+1)*bs, j*bs:(j+1)*bs] - L_i @ (L_j)^T,
    where L_i = L[i*bs:(i+1)*bs, k*bs:(k+1)*bs] and L_j = L[j*bs:(j+1)*bs, k*bs:(k+1)*bs].
    If i == j, then perform the Cholesky factorization on the updated block.
    """
    bs = block_size
    A_block = A[i*bs:(i+1)*bs, j*bs:(j+1)*bs].copy()
    L_i = L[i*bs:(i+1)*bs, k*bs:(k+1)*bs]
    L_j = L[j*bs:(j+1)*bs, k*bs:(k+1)*bs]
    updated = A_block - L_i @ L_j.T
    if i == j:
        L_updated = np.linalg.cholesky(updated)
        for r in range(bs):
            for c in range(r+1, bs):
                L_updated[r, c] = 0.0
        return L_updated
    else:
        return updated

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
