import mlx.core as mx
import mlx.core.fast

def gpu_diag_update_block(L_in: mx.array, k: int, block_size: int):
    """
    GPU Kernel: Diagonal update for block (k,k).

    Let 
      {{ OFFSET = k * block_size }}.
    For the submatrix 
      L_in[OFFSET:OFFSET+block_size, OFFSET:OFFSET+block_size],
    compute its Cholesky factorization:
      {{ L_{kk} = chol(L_in[OFFSET:OFFSET+block_size, OFFSET:OFFSET+block_size]) }},
    and set all elements above the diagonal to 0.

    The kernel is launched on a grid of size (block_size, block_size, 1)
    and returns a block of shape (block_size, block_size).

    All LaTeX formulas have their curly braces escaped.
    """
    offset = k * block_size
    source = f"""
    int N = L_in_shape[0];
    #define BLOCK_SIZE {block_size}
    #define OFFSET {offset}
    
    // Each thread computes one element of the block.
    uint local_row = thread_position_in_grid.y;
    uint local_col = thread_position_in_grid.x;
    uint idx = local_row * BLOCK_SIZE + local_col;
    
    // Global indices.
    uint global_row = OFFSET + local_row;
    uint global_col = OFFSET + local_col;
    
    // Load element into shared memory.
    threadgroup float L_block[BLOCK_SIZE * BLOCK_SIZE];
    L_block[idx] = L_in[global_row * N + global_col];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Sequential Cholesky on the block.
    for (uint p = 0; p < BLOCK_SIZE; p++) {{
        if (local_row == p && local_col == p) {{
            float sum = 0.0f;
            for (uint t = 0; t < p; t++) {{
                sum += L_block[p * BLOCK_SIZE + t] * L_block[p * BLOCK_SIZE + t];
            }}
            float diag = L_block[p * BLOCK_SIZE + p] - sum;
            diag = diag > 1e-8f ? diag : 1e-8f;
            L_block[p * BLOCK_SIZE + p] = sqrt(diag);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (local_row > p && local_col == p) {{
            float sum = 0.0f;
            for (uint t = 0; t < p; t++) {{
                sum += L_block[local_row * BLOCK_SIZE + t] * L_block[p * BLOCK_SIZE + t];
            }}
            L_block[local_row * BLOCK_SIZE + p] = (L_block[local_row * BLOCK_SIZE + p] - sum) / L_block[p * BLOCK_SIZE + p];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    
    // Enforce lower-triangularity.
    if (local_row < local_col) {{
        L_block[idx] = 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    L_out[idx] = L_block[idx];
    """
    kernel = mx.fast.metal_kernel(
        name=f"gpu_diag_update_block_k{k}",
        input_names=["L_in"],
        output_names=["L_out"],
        source=source
    )
    L_block_out = kernel(
        inputs=[L_in],
        grid=(block_size, block_size, 1),
        threadgroup=(block_size, block_size, 1),
        output_shapes=[(block_size, block_size)],
        output_dtypes=[L_in.dtype],
        verbose=False
    )[0]
    return L_block_out

def gpu_offdiag_update_block(A: mx.array, L_in: mx.array, k: int, i_block: int, block_size: int):
    """
    GPU Kernel: Off-diagonal update for block (i,k) with i > k.

    Let 
      {{ OFFSET_K = k * block_size }} and {{ OFFSET_I = i_block * block_size }}.
    For each element in the block with local indices (local_row, local_col),
      where global_row = OFFSET_I + local_row and global_col = OFFSET_K + local_col,
    compute forward substitution:
      {{ L_{ik}[global_row, global_col] = 
         \frac{A[global_row, global_col] - \sum_{t=0}^{local_col-1}(L_in[global_row, OFFSET_K+t] \cdot L_in[(OFFSET_K+local_col), (OFFSET_K+t])}{L_in[(OFFSET_K+local_col), (OFFSET_K+local_col)] }}.

    The kernel is launched on a grid of size (block_size, block_size, 1)
    and returns a block of shape (block_size, block_size).

    We use the default thread coordinates:
      local_row = thread_position_in_grid.y and local_col = thread_position_in_grid.x.
    """
    offset_k = k * block_size
    offset_i = i_block * block_size
    source = f"""
    int N = A_shape[0];
    #define BLOCK_SIZE {block_size}
    #define OFFSET_K {offset_k}
    #define OFFSET_I {offset_i}
    
    // Use default thread coordinates.
    uint local_row = thread_position_in_grid.y;
    uint local_col = thread_position_in_grid.x;
    uint global_row = OFFSET_I + local_row;
    uint global_col = OFFSET_K + local_col;
    uint idx = local_row * BLOCK_SIZE + local_col;
    
    float sum = 0.0f;
    for (uint t = 0; t < local_col; t++) {{
         sum += L_in[global_row * N + (OFFSET_K + t)] *
                L_in[(OFFSET_K + local_col) * N + (OFFSET_K + t)];
    }}
    float updated = (A[global_row * N + (OFFSET_K + local_col)] - sum)
                      / L_in[(OFFSET_K + local_col) * N + (OFFSET_K + local_col)];
    L_out[idx] = updated;
    """
    kernel = mx.fast.metal_kernel(
        name=f"gpu_offdiag_update_block_k{k}_i{i_block}",
        input_names=["A", "L_in"],
        output_names=["L_out"],
        source=source
    )
    L_block_out = kernel(
         inputs=[A, L_in],
         grid=(block_size, block_size, 1),
         threadgroup=(block_size, block_size, 1),
         output_shapes=[(block_size, block_size)],
         output_dtypes=[A.dtype],
         verbose=False
    )[0]
    return L_block_out


def gpu_trailing_update_block(A: mx.array, L_in: mx.array, k: int, i_block: int, j_block: int, block_size: int):
    """
    GPU Kernel: Trailing update for block (i,j) with i,j > k.

    Let 
      {{ OFFSET_K = k * block_size }},
      {{ OFFSET_I = i_block * block_size }}, and
      {{ OFFSET_J = j_block * block_size }}.
    For each element in the block with local indices (local_row, local_col),
      where global_row = OFFSET_I + local_row and global_col = OFFSET_J + local_col,
    compute the Schur complement update:
      {{ B[local_row, local_col] = A[global_row, global_col] - \sum_{{t=0}}^{{BLOCK_SIZE-1}} (L_in[global_row, OFFSET_K+t] \cdot L_in[global_col, OFFSET_K+t]) }}.
    If the block is diagonal (i_block == j_block), perform sequential Cholesky on B:
      {{ L_{ii} = chol(B) }},
    enforcing lower-triangularity. For off-diagonal blocks, simply return B.

    The kernel is launched on a grid of size (block_size, block_size, 1)
    and returns a block of shape (block_size, block_size).


    """
    offset_k = k * block_size
    offset_i = i_block * block_size
    offset_j = j_block * block_size
    source = f"""
    int N = A_shape[0];
    #define BLOCK_SIZE {block_size}
    #define OFFSET_K {offset_k}
    #define OFFSET_I {offset_i}
    #define OFFSET_J {offset_j}

    // Use default thread coordinates.
    uint local_row = thread_position_in_grid.y;
    uint local_col = thread_position_in_grid.x;
    uint global_row = OFFSET_I + local_row;
    uint global_col = OFFSET_J + local_col;
    uint idx = local_row * BLOCK_SIZE + local_col;

    // Compute Schur complement update.
    float sum = 0.0f;
    for (uint t = 0; t < BLOCK_SIZE; t++) {{
        float a_val = L_in[global_row * N + (OFFSET_K + t)];
        float b_val = L_in[global_col * N + (OFFSET_K + t)];
        sum += a_val * b_val;
    }}
    float updated = A[global_row * N + global_col] - sum;

    // Allocate shared memory for block B.
    threadgroup float B_block[BLOCK_SIZE * BLOCK_SIZE];
    B_block[idx] = updated;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // If this is a diagonal trailing block, perform sequential Cholesky.
    if (OFFSET_I == OFFSET_J) {{
        // Diagonal block: perform sequential Cholesky on B_block.
        for (uint p = 0; p < BLOCK_SIZE; p++) {{
            if (local_row == p && local_col == p) {{
                float s = 0.0f;
                for (uint t = 0; t < p; t++) {{
                    s += B_block[p * BLOCK_SIZE + t] * B_block[p * BLOCK_SIZE + t];
                }}
                float d = B_block[p * BLOCK_SIZE + p] - s;
                d = d > 1e-8f ? d : 1e-8f;
                B_block[p * BLOCK_SIZE + p] = sqrt(d);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (local_row > p && local_col == p) {{
                float s = 0.0f;
                for (uint t = 0; t < p; t++) {{
                    s += B_block[local_row * BLOCK_SIZE + t] * B_block[p * BLOCK_SIZE + t];
                }}
                B_block[local_row * BLOCK_SIZE + p] = (B_block[local_row * BLOCK_SIZE + p] - s) / B_block[p * BLOCK_SIZE + p];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        // Zero out the upper triangular part.
        if (local_row < local_col) {{
            B_block[idx] = 0.0f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        L_out[idx] = B_block[idx];
    }} else {{
        // Off-diagonal block: simply return the updated value.
        L_out[idx] = updated;
    }}
    """
    kernel = mx.fast.metal_kernel(
         name=f"gpu_trailing_update_block_k{k}_i{i_block}_j{j_block}",
         input_names=["A", "L_in"],
         output_names=["L_out"],
         source=source
    )
    L_block_out = kernel(
         inputs=[A, L_in],
         grid=(block_size, block_size, 1),
         threadgroup=(block_size, block_size, 1),
         output_shapes=[(block_size, block_size)],
         output_dtypes=[A.dtype],
         verbose=False
    )[0]
    return L_block_out


def gpu_clear_upper(L_in: mx.array, block_size: int):
    """
    GPU Kernel: Clear the upper-triangular part of a full matrix.

    For each element (i,j) in L_in, if i < j then set L[i,j] = 0.
    The kernel is launched on a grid of size (N, N, 1), where N = L_in.shape[0].
    """
    N = L_in.shape[0]
    source = f"""
    int N = L_in_shape[0];
    uint i = thread_position_in_grid.y;
    uint j = thread_position_in_grid.x;
    uint idx = i * N + j;
    if (i < j) {{
        L_out[idx] = 0.0f;
    }} else {{
        L_out[idx] = L_in[idx];
    }}
    """
    kernel = mx.fast.metal_kernel(
        name="gpu_clear_upper",
        input_names=["L_in"],
        output_names=["L_out"],
        source=source
    )
    L_out = kernel(
        inputs=[L_in],
        grid=(N, N, 1),
        threadgroup=(block_size, block_size, 1),
        output_shapes=[(N, N)],
        output_dtypes=[L_in.dtype],
        verbose=False
    )[0]
    return L_out
