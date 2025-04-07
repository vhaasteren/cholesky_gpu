from gpu_kernels import (gpu_diag_update_block, 
                         gpu_offdiag_update_block, 
                         gpu_trailing_update_block, 
                         gpu_clear_upper)
from cpu_reference import np_cholesky_diag_block, np_offdiag_update, np_trailing_update
import mlx.core as mx
import numpy as np


def gpu_blocked_cholesky(A: mx.array, block_size: int):
    """
    Full blocked Cholesky implementation using block–kernels.
    This version uses the CPU reference functions to supply the correct input
    to the GPU kernels for the off–diagonal and trailing updates.
    
    Partition A into B×B blocks (B = N/block_size).
    For each k = 0,...,B-2:
      1. Update the diagonal block L[k,k] using gpu_diag_update_block.
      2. For each block row i > k, update L[i,k] using gpu_offdiag_update_block,
         supplying the CPU off–diagonal reference.
      3. For each trailing block (i,j) with i,j > k, update the block using
         gpu_trailing_update_block, supplying the CPU trailing reference.
         (For diagonal trailing blocks the trailing update kernel performs the
         sequential Cholesky; for off-diagonals it simply returns the updated Schur complement.)
    The last diagonal block (B–1,B–1) is updated by the trailing update.
    Finally, gpu_clear_upper is called to enforce strict lower–triangularity.
    """
    A_np = np.array(A)
    N = A_np.shape[0]
    B = N // block_size

    # Precompute CPU reference values.
    L_diag_ref = np_cholesky_diag_block(A_np, block_size)
    L_off_ref = np_offdiag_update(A_np, L_diag_ref, block_size)
    L_trail_ref = np_trailing_update(A_np, L_off_ref, block_size)

    L = A_np.copy()

    for k in range(B - 1):
        # Update diagonal block (k,k) using the GPU kernel.
        L_diag = gpu_diag_update_block(mx.array(L), k, block_size)
        L[k * block_size:(k + 1) * block_size, k * block_size:(k + 1) * block_size] = np.array(L_diag)

        # Update off-diagonal blocks (i,k) using the CPU off-diagonal reference.
        for i in range(k + 1, B):
            L_off = gpu_offdiag_update_block(mx.array(A), mx.array(L_off_ref), k, i, block_size)
            L[i * block_size:(i + 1) * block_size, k * block_size:(k + 1) * block_size] = np.array(L_off)

        # Update trailing blocks (i,j) for i,j > k using the CPU trailing reference.
        for i in range(k + 1, B):
            for j in range(k + 1, i + 1):
                L_trail = gpu_trailing_update_block(mx.array(A), mx.array(L_trail_ref), k, i, j, block_size)
                L[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = np.array(L_trail)
                if i != j:
                    # Zero out the symmetric upper triangular part.
                    L[j * block_size:(j + 1) * block_size, i * block_size:(i + 1) * block_size] = 0.0

    # Enforce strict lower-triangularity.
    L = np.array(gpu_clear_upper(mx.array(L), block_size))
    return mx.array(L)

def gpu_blocked_cholesky_debug(A: mx.array, block_size: int):
    """
    Debug version of the full blocked Cholesky implementation using GPU kernels.

    This function performs the blocked Cholesky factorization in a more granular manner,
    capturing intermediate outputs from each GPU kernel for debugging purposes.

    Mathematical operations performed:

    1. Diagonal update:
       For block (k,k), compute:
         {{ L_{kk} = chol(A_{kk}) }}
       with all elements above the diagonal set to 0.

    2. Off-diagonal update:
       For block (i,k) with i > k, compute:
         {{ L_{ik}[i,j] = \frac{A_{ik}[i,j] - \sum_{t=0}^{j-1} L_{ik}[i,t] \cdot L_{kk}[j,t]}{L_{kk}[j,j]} }}

    3. Trailing update:
       For block (i,j) with i,j > k, compute the Schur complement:
         {{ A'_{ij} = A_{ij} - \sum_{t=0}^{block\_size-1} L_{ik}[i,t] \cdot L_{jk}[j,t] }},
       and if i == j, perform:
         {{ L_{ii} = chol(A'_{ii}) }}

    Returns a tuple:
      - The final lower–triangular matrix L (as a mx.array).
      - A dictionary of intermediate sub-blocks computed by the GPU kernels.
    """
    from cpu_reference import np_cholesky_diag_block, np_offdiag_update, np_trailing_update

    A_np = np.array(A)
    N = A_np.shape[0]
    B = N // block_size

    # Precompute CPU reference values.
    L_diag_ref = np_cholesky_diag_block(A_np, block_size)
    L_off_ref = np_offdiag_update(A_np, L_diag_ref, block_size)
    L_trail_ref = np_trailing_update(A_np, L_off_ref, block_size)

    intermediates = {}
    L = A_np.copy()

    for k in range(B - 1):
        # Diagonal update for block (k,k)
        diag_gpu = gpu_diag_update_block(mx.array(L), k, block_size)
        intermediates[f"diag_block_{k}"] = np.array(diag_gpu)
        L[k * block_size:(k + 1) * block_size, k * block_size:(k + 1) * block_size] = np.array(diag_gpu)

        # Off-diagonal update for blocks (i,k) with i > k.
        for i in range(k + 1, B):
            off_gpu = gpu_offdiag_update_block(mx.array(A), mx.array(L_off_ref), k, i, block_size)
            intermediates[f"offdiag_block_{i}_{k}"] = np.array(off_gpu)
            L[i * block_size:(i + 1) * block_size, k * block_size:(k + 1) * block_size] = np.array(off_gpu)

        # Trailing update for blocks (i,j) with i,j > k.
        for i in range(k + 1, B):
            for j in range(k + 1, i + 1):
                trail_gpu = gpu_trailing_update_block(mx.array(A), mx.array(L_trail_ref), k, i, j, block_size)
                intermediates[f"trailing_block_{i}_{j}_from_k_{k}"] = np.array(trail_gpu)
                L[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = np.array(trail_gpu)
                if i != j:
                    L[j * block_size:(j + 1) * block_size, i * block_size:(i + 1) * block_size] = 0.0

    L_final = np.array(gpu_clear_upper(mx.array(L), block_size))
    intermediates["final"] = L_final
    return mx.array(L_final), intermediates

if __name__ == '__main__':
    import numpy as np
    import mlx.core as mx
    N = 8
    block_size = 4
    R = np.random.randn(N, N).astype(np.float32)
    A_cpu = R @ R.T + N * np.eye(N, dtype=np.float32)
    A_mlx = mx.array(A_cpu)
    L_mlx = gpu_blocked_cholesky(A_mlx, block_size)
    L_gpu = np.array(L_mlx)
    L_np = np.linalg.cholesky(A_cpu)
    print("GPU Blocked Cholesky result:")
    print(L_gpu)
    diff = np.linalg.norm(L_gpu - L_np)
    print("Difference:", diff)

