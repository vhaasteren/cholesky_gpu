import unittest
import numpy as np
import mlx.core as mx
from cpu_reference import (np_cholesky_diag_block, np_offdiag_update,
                           np_trailing_update, np_trailing_schur_update)
from gpu_kernels import (gpu_diag_update_block, gpu_offdiag_update_block,
                         gpu_trailing_update_block, gpu_clear_upper)
from gpu_blocked_cholesky import gpu_blocked_cholesky, gpu_blocked_cholesky_debug



class TestBlockedCholesky4x4(unittest.TestCase):
    def setUp(self):
        # Define a fixed 4x4 lower-triangular matrix.
        # L_true is used to generate A = L_true * L_true^T.
        self.L_true = np.array([
            [2, 0, 0, 0],
            [6, 1, 0, 0],
            [4, 0, 3, 0],
            [8, 0, 5, 2]
        ], dtype=np.float32)
        self.A = self.L_true @ self.L_true.T
        self.block_size = 2

    def test_cpu_reference_4x4(self):
        """Test CPU reference: full blocked Cholesky using CPU functions."""
        L_diag = np_cholesky_diag_block(self.A, self.block_size)
        L_off = np_offdiag_update(self.A, L_diag, self.block_size)
        L_full = np_trailing_update(self.A, L_off, self.block_size)
        L_np = np.linalg.cholesky(self.A)
        diff = np.linalg.norm(L_full - L_np)
        print("CPU reference 4x4 difference:", diff)
        self.assertAlmostEqual(diff, 0, places=5)

    def test_gpu_diag_update_block(self):
        """Test GPU diagonal update block for block (0,0)."""
        A_mlx = mx.array(self.A)
        L_diag_gpu = gpu_diag_update_block(A_mlx, 0, self.block_size)
        L_diag_cpu = np_cholesky_diag_block(self.A, self.block_size)[:self.block_size, :self.block_size]
        diff = np.linalg.norm(np.array(L_diag_gpu) - L_diag_cpu)
        print("GPU diagonal block difference:", diff)
        self.assertAlmostEqual(diff, 0, places=5)

    def test_gpu_offdiag_update_block(self):
        """Test GPU off-diagonal update block for block (1,0)."""
        L_diag_cpu = np_cholesky_diag_block(self.A, self.block_size)
        L_off_cpu = np_offdiag_update(self.A, L_diag_cpu, self.block_size)
        # CPU reference off-diagonal block for (1,0)
        cpu_off_block = L_off_cpu[self.block_size:2*self.block_size, :self.block_size]
        A_mlx = mx.array(self.A)
        # The GPU offdiag update kernel returns a (block_size, block_size) block directly.
        gpu_off_block = np.array(gpu_offdiag_update_block(A_mlx, mx.array(L_off_cpu), 0, 1, self.block_size))
        print("CPU offdiag block:")
        print(cpu_off_block)
        print("GPU offdiag block:")
        print(gpu_off_block)
        diff = np.linalg.norm(gpu_off_block - cpu_off_block)
        print("GPU off-diagonal block difference:", diff)
        self.assertAlmostEqual(diff, 0, places=5)

    def test_gpu_trailing_update_block(self):
        """Test GPU trailing update block for block (1,1)."""
        L_diag_cpu = np_cholesky_diag_block(self.A, self.block_size)
        L_off_cpu = np_offdiag_update(self.A, L_diag_cpu, self.block_size)
        L_full_cpu = np_trailing_update(self.A, L_off_cpu, self.block_size)
        # CPU reference trailing block for (1,1)
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
        """Test full blocked Cholesky host function."""
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
        """Test the debug version of GPU blocked Cholesky subcomponents.
        This test compares intermediate outputs from the debug version against CPU reference functions."""
        L_debug, intermediates = gpu_blocked_cholesky_debug(mx.array(self.A), self.block_size)
        
        # Check diagonal block for block 0.
        diag_cpu = np_cholesky_diag_block(self.A, self.block_size)[:self.block_size, :self.block_size]
        diag_gpu = intermediates.get("diag_block_0", None)
        diff_diag = np.linalg.norm(diag_cpu - diag_gpu)
        print("Debug Diagonal Block Difference:", diff_diag)
        self.assertAlmostEqual(diff_diag, 0, places=5)

        # Check off-diagonal block for block (1,0).
        off_cpu_full = np_offdiag_update(self.A, np_cholesky_diag_block(self.A, self.block_size), self.block_size)
        off_cpu = off_cpu_full[self.block_size:2*self.block_size, :self.block_size]
        off_gpu = intermediates.get("offdiag_block_1_0", None)
        diff_off = np.linalg.norm(off_cpu - off_gpu)
        print("Debug Off-Diagonal Block Difference:", diff_off)
        self.assertAlmostEqual(diff_off, 0, places=5)

        # Check trailing block for block (1,1).
        trailing_cpu_full = np_trailing_update(self.A, np_offdiag_update(self.A, np_cholesky_diag_block(self.A, self.block_size), self.block_size), self.block_size)
        trailing_cpu = trailing_cpu_full[self.block_size:2*self.block_size, self.block_size:2*self.block_size]
        trailing_gpu = intermediates.get("trailing_block_1_1_from_k_0", None)
        diff_trailing = np.linalg.norm(trailing_cpu - trailing_gpu)
        print("Debug Trailing Block Difference:", diff_trailing)
        self.assertAlmostEqual(diff_trailing, 0, places=5)

if __name__ == '__main__':
    unittest.main()
