import unittest
import numpy as np
import mlx.core as mx
from cpu_reference import (np_cholesky_diag_block, np_offdiag_update,
                           np_trailing_update, np_trailing_schur_update,
                           np_blocked_cholesky,
                           cpu_cholesky_diag_block_general, cpu_offdiag_update_general,
                           cpu_trailing_update_general)
from gpu_kernels import (gpu_diag_update_block, gpu_offdiag_update_block,
                         gpu_trailing_update_block, gpu_clear_upper)
from gpu_blocked_cholesky import gpu_blocked_cholesky, gpu_blocked_cholesky_debug

def build_pd_matrix(N: int):
    """
    Builds a positive definite matrix A_np of size N×N from a randomly generated
    lower-triangular matrix L_np with integer entries between 1 and 10.

    Steps:
      1. Set the random seed to N for reproducibility.
      2. Compute the number of lower-triangular elements: N*(N+1)//2.
      3. Generate a 1D array of random integers in the range [1, 10] of that length.
      4. Create an NxN matrix of zeros (dtype np.float32) and fill its lower-triangular
         part using numpy's advanced indexing with np.tril_indices.
      5. Compute A_np = L_np @ L_np.T, which is guaranteed to be symmetric and
         positive definite.

    Parameters:
      N (int): The size of the matrix.

    Returns:
      L_np (np.ndarray): The generated lower-triangular matrix (of type np.float32).
      A_np (np.ndarray): The positive definite matrix, computed as L_np @ L_np.T.
    """
    np.random.seed(N)
    n_elements = N * (N + 1) // 2
    lower_values = np.random.randint(1, 11, size=n_elements)
    
    L_np = np.zeros((N, N), dtype=np.float32)
    tril_indices = np.tril_indices(N)
    L_np[tril_indices] = lower_values
    
    A_np = L_np @ L_np.T
    return L_np, A_np

# --------------------- Existing 4x4 GPU and CPU Tests ---------------------

class TestBlockedCholesky4x4(unittest.TestCase):
    def setUp(self):
        # Define a fixed 4×4 lower-triangular matrix.
        self.L_true = np.array([
            [2, 0, 0, 0],
            [6, 1, 0, 0],
            [4, 0, 3, 0],
            [8, 0, 5, 2]
        ], dtype=np.float32)
        self.A = self.L_true @ self.L_true.T
        self.block_size = 2

    def test_cpu_reference_4x4(self):
        """Test CPU reference functions on 4×4 matrix."""
        L_diag = np_cholesky_diag_block(self.A, self.block_size)
        L_off = np_offdiag_update(self.A, L_diag, self.block_size)
        L_full = np_trailing_update(self.A, L_off, self.block_size)
        L_np = np.linalg.cholesky(self.A)
        diff = np.linalg.norm(L_full - L_np)
        print("CPU reference 4x4 difference:", diff)
        self.assertAlmostEqual(diff, 0, places=5)

    def test_gpu_diag_update_block(self):
        """Test GPU diagonal update block for block (0,0) on 4×4 matrix."""
        A_mlx = mx.array(self.A)
        L_diag_gpu = gpu_diag_update_block(A_mlx, 0, self.block_size)
        L_diag_cpu = np_cholesky_diag_block(self.A, self.block_size)[:self.block_size, :self.block_size]
        diff = np.linalg.norm(np.array(L_diag_gpu) - L_diag_cpu)
        print("GPU diagonal block difference:", diff)
        self.assertAlmostEqual(diff, 0, places=5)

    def test_gpu_offdiag_update_block(self):
        """Test GPU off-diagonal update block for block (1,0) on 4×4 matrix."""
        L_diag_cpu = np_cholesky_diag_block(self.A, self.block_size)
        L_off_cpu = np_offdiag_update(self.A, L_diag_cpu, self.block_size)
        cpu_off_block = L_off_cpu[self.block_size:2*self.block_size, :self.block_size]
        A_mlx = mx.array(self.A)
        gpu_off_block = np.array(gpu_offdiag_update_block(A_mlx, mx.array(L_off_cpu), 0, 1, self.block_size))
        print("CPU offdiag block:")
        print(cpu_off_block)
        print("GPU offdiag block:")
        print(gpu_off_block)
        diff = np.linalg.norm(gpu_off_block - cpu_off_block)
        print("GPU off-diagonal block difference:", diff)
        self.assertAlmostEqual(diff, 0, places=5)

    def test_gpu_trailing_update_block(self):
        """Test GPU trailing update block for block (1,1) on 4×4 matrix."""
        L_diag_cpu = np_cholesky_diag_block(self.A, self.block_size)
        L_off_cpu = np_offdiag_update(self.A, L_diag_cpu, self.block_size)
        L_full_cpu = np_trailing_update(self.A, L_off_cpu, self.block_size)
        cpu_trailing_block = L_full_cpu[self.block_size:2*self.block_size, self.block_size:2*self.block_size]
        A_mlx = mx.array(self.A)
        L_full_cpu_mlx = mx.array(L_full_cpu)
        gpu_trailing_block = np.array(gpu_trailing_update_block(A_mlx, L_full_cpu_mlx, 0, 1, 1, self.block_size))
        print("CPU trailing block:")
        print(cpu_trailing_block)
        print("GPU trailing block:")
        print(gpu_trailing_block)
        diff = np.linalg.norm(gpu_trailing_block - cpu_trailing_block)
        print("GPU trailing block difference:", diff)
        self.assertAlmostEqual(diff, 0, places=5)

    def test_gpu_full(self):
        """Test full blocked Cholesky host function on 4×4 matrix."""
        L_gpu = gpu_blocked_cholesky(mx.array(self.A), self.block_size)
        L_gpu_arr = np.array(L_gpu)
        L_np = np.linalg.cholesky(self.A)
        diff = np.linalg.norm(L_gpu_arr - L_np)
        if diff > 1e-4:
            print("GPU full:")
            print(L_gpu_arr)
            print("CPU full:")
            print(L_np)
            print("Elementwise difference:")
            print(L_gpu_arr - L_np)
        self.assertAlmostEqual(diff, 0, places=4)

    def test_gpu_debug_subcomponents(self):
        """Test the debug version of GPU blocked Cholesky subcomponents on 4×4 matrix."""
        L_debug, intermediates = gpu_blocked_cholesky_debug(mx.array(self.A), self.block_size)
        diag_cpu = np_cholesky_diag_block(self.A, self.block_size)[:self.block_size, :self.block_size]
        diag_gpu = intermediates.get("diag_block_0", None)
        diff_diag = np.linalg.norm(diag_cpu - diag_gpu)
        print("Debug Diagonal Block Difference:", diff_diag)
        self.assertAlmostEqual(diff_diag, 0, places=5)
        off_cpu_full = np_offdiag_update(self.A, np_cholesky_diag_block(self.A, self.block_size), self.block_size)
        off_cpu = off_cpu_full[self.block_size:2*self.block_size, :self.block_size]
        off_gpu = intermediates.get("offdiag_block_1_0", None)
        diff_off = np.linalg.norm(off_cpu - off_gpu)
        print("Debug Off-Diagonal Block Difference:", diff_off)
        self.assertAlmostEqual(diff_off, 0, places=5)
        trailing_cpu_full = np_trailing_update(self.A, np_offdiag_update(self.A, np_cholesky_diag_block(self.A, self.block_size), self.block_size), self.block_size)
        trailing_cpu = trailing_cpu_full[self.block_size:2*self.block_size, self.block_size:2*self.block_size]
        trailing_gpu = intermediates.get("trailing_block_1_1_from_k_0", None)
        diff_trailing = np.linalg.norm(trailing_cpu - trailing_gpu)
        print("Debug Trailing Block Difference:", diff_trailing)
        self.assertAlmostEqual(diff_trailing, 0, places=5)

# --------------------- New CPU Blocked Cholesky General Tests ---------------------

class TestNPBlockedCholeskyGeneral(unittest.TestCase):
    def test_general_blocked_cholesky(self):
        """
        Test the general CPU blocked Cholesky implementation on various matrix sizes.
        We use 4x4 blocks for matrices of size 8x8, 16x16, and 32x32.
        """
        for N in [8, 16, 32]:
            block_size = 4  # Use 4x4 blocks
            _, A = build_pd_matrix(N)
            L_blocked = np_blocked_cholesky(A, block_size)
            L_np = np.linalg.cholesky(A)
            diff = np.linalg.norm(L_blocked - L_np)
            print(f"NP Blocked Cholesky for {N}x{N} with block size {block_size}: diff = {diff}")
            self.assertAlmostEqual(diff, 0, places=4)

    def test_cpu_cholesky_diag_block_general(self):
        """Test the general CPU diagonal block function on various blocks."""
        for N in [8, 16]:
            block_size = 4
            _, A = build_pd_matrix(N)
            B = A.shape[0] // block_size
            for k in range(B):
                L_diag = cpu_cholesky_diag_block_general(A, k, block_size)
                # Compare with direct Cholesky of the k-th block
                start = k * block_size
                end = (k+1) * block_size
                L_direct = np.linalg.cholesky(A[start:end, start:end])
                # Zero out the upper triangle in L_direct
                for i in range(block_size):
                    for j in range(i+1, block_size):
                        L_direct[i, j] = 0.0
                diff = np.linalg.norm(L_diag - L_direct)
                print(f"CPU diag block {k} difference for {N}x{N} matrix: {diff}")
                self.assertAlmostEqual(diff, 0, places=4)

    def test_cpu_offdiag_update_general(self):
        """Test the general CPU off-diagonal update function on various blocks (k=0 only)."""
        # For k=0, expected off-diagonal blocks equal the corresponding blocks in the true factor.
        for N in [8, 16]:
            block_size = 4
            L_true, A = build_pd_matrix(N)
            B = A.shape[0] // block_size
            k = 0
            for i in range(1, B):
                expected = L_true[i*block_size:(i+1)*block_size, 0:block_size]
                L_dummy = A.copy()
                L_dummy[0:block_size, 0:block_size] = L_true[0:block_size, 0:block_size]
                L_off = cpu_offdiag_update_general(A, L_dummy, k, i, block_size)
                diff = np.linalg.norm(L_off - expected)
                print(f"CPU offdiag block ({i},{k}) difference: {diff}")
                self.assertAlmostEqual(diff, 0, places=4)

    def test_cpu_trailing_update_general(self):
        """Test the general CPU trailing update function on various blocks (k=0 only)."""
        # For k=0, the expected trailing block is computed as the residual:
        # expected = A[i,j] - L_true[i,0] @ L_true[j,0]^T.
        for N in [8, 16]:
            block_size = 4
            L_true, A = build_pd_matrix(N)
            B = A.shape[0] // block_size
            k = 0
            # Initialize L_dummy for k=0: compute the first block column from the true factor.
            L_dummy = A.copy()
            L_dummy[0:block_size, 0:block_size] = L_true[0:block_size, 0:block_size]
            for i in range(1, B):
                L_dummy[i*block_size:(i+1)*block_size, 0:block_size] = L_true[i*block_size:(i+1)*block_size, 0:block_size]
            for i in range(1, B):
                for j in range(1, i+1):
                    # Expected trailing block: A[i,j] - L_true[i,0] @ L_true[j,0]^T.
                    start_i = i * block_size
                    end_i = (i+1) * block_size
                    start_j = j * block_size
                    end_j = (j+1) * block_size
                    expected = A[start_i:end_i, start_j:end_j] - \
                               L_true[start_i:end_i, 0:block_size] @ L_true[start_j:end_j, 0:block_size].T
                    if i == j:
                        expected = np.linalg.cholesky(expected)
                        for r in range(block_size):
                            for c in range(r+1, block_size):
                                expected[r, c] = 0.0
                    L_trail = cpu_trailing_update_general(A, L_dummy, k, i, j, block_size)
                    diff = np.linalg.norm(L_trail - expected)
                    print(f"CPU trailing block ({i},{j}) from k={k} difference: {diff}")
                    self.assertAlmostEqual(diff, 0, places=4)

if __name__ == '__main__':
    unittest.main()
