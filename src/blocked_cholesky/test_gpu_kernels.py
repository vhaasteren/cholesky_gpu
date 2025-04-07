import unittest
import numpy as np
import mlx.core as mx
from cpu_reference import np_cholesky_diag_block, np_offdiag_update, np_trailing_update, np_trailing_schur_update
from gpu_kernels import gpu_diag_update_block, gpu_offdiag_update_block, gpu_trailing_update_block, gpu_clear_upper

class TestGPUKernels4x4(unittest.TestCase):
    def setUp(self):
        # Define a fixed 4x4 matrix built from a known lower-triangular matrix.
        # This way, we know what the "true" Cholesky should be.
        L_true = np.array([
            [2, 0, 0, 0],
            [6, 1, 0, 0],
            [4, 0, 3, 0],
            [8, 0, 5, 2]
        ], dtype=np.float32)
        self.A = L_true @ L_true.T
        self.block_size = 2  # For a 4x4 matrix.

    def test_gpu_diag_update(self):
        """Test the GPU diagonal update kernel on the 4x4 matrix."""
        A_mlx = mx.array(self.A)
        L_diag_gpu = gpu_diag_update_block(A_mlx, 0, self.block_size)
        L_diag_cpu = np_cholesky_diag_block(self.A, self.block_size)[:self.block_size, :self.block_size]
        diff = np.linalg.norm(np.array(L_diag_gpu) - L_diag_cpu)
        print("GPU diag block:\n", np.array(L_diag_gpu))
        print("CPU diag block:\n", L_diag_cpu)
        print("Diag elementwise diff:\n", np.array(L_diag_gpu) - L_diag_cpu)
        self.assertAlmostEqual(diff, 0, places=5)

    def test_gpu_offdiag_update(self):
        """Test the GPU off-diagonal update kernel for block (1,0) on the 4x4 matrix."""
        L_diag_cpu = np_cholesky_diag_block(self.A, self.block_size)
        L_off_cpu = np_offdiag_update(self.A, L_diag_cpu, self.block_size)
        # CPU reference off-diagonal block for (1,0)
        cpu_off_block = L_off_cpu[self.block_size:2*self.block_size, :self.block_size]
        A_mlx = mx.array(self.A)
        # The GPU offdiag kernel returns a (block_size, block_size) array.
        gpu_off_block = np.array(gpu_offdiag_update_block(A_mlx, mx.array(L_off_cpu), 0, 1, self.block_size))
        print("CPU offdiag block:")
        print(cpu_off_block)
        print("GPU offdiag block:")
        print(gpu_off_block)
        diff = np.linalg.norm(gpu_off_block - cpu_off_block)
        print("GPU off-diagonal block difference:", diff)
        self.assertAlmostEqual(diff, 0, places=5)

    def test_gpu_trailing_update(self):
        """Test the GPU trailing update (Schur complement and Cholesky) for block (1,1) on the 4x4 matrix."""
        L_diag_cpu = np_cholesky_diag_block(self.A, self.block_size)
        L_off_cpu = np_offdiag_update(self.A, L_diag_cpu, self.block_size)
        L_full_cpu = np_trailing_update(self.A, L_off_cpu, self.block_size)
        # CPU reference trailing block for (1,1)
        cpu_trailing_block = L_full_cpu[self.block_size:2*self.block_size, self.block_size:2*self.block_size]
        A_mlx = mx.array(self.A)
        L_full_cpu_mlx = mx.array(L_full_cpu)
        # The GPU trailing kernel returns a (block_size, block_size) array.
        gpu_trailing_block = np.array(gpu_trailing_update_block(A_mlx, L_full_cpu_mlx, 0, 1, 1, self.block_size))
        print("CPU trailing block:")
        print(cpu_trailing_block)
        print("GPU trailing block:")
        print(gpu_trailing_block)
        diff = np.linalg.norm(gpu_trailing_block - cpu_trailing_block)
        print("GPU trailing block difference:", diff)
        self.assertAlmostEqual(diff, 0, places=5)

if __name__ == '__main__':
    unittest.main()
