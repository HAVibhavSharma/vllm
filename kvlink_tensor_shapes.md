# KVLink path — tensor shapes for `XFormersImpl.forward`
# File:   vllm/v1/attention/backends/xformers.py
# Branch: releases/v0.17.0
# Model:  Qwen/Qwen3.5-9B  (Qwen3-Next hybrid: full-attn + linear-attn layers)
# Run:    benchmarks_ours/evals/e2e/main.py --dataset hotpotqa --approach kvlink-4

# ----------------------------------------------------------------------------
# Symbols
# ----------------------------------------------------------------------------
# T_new      = num_actual_tokens                 = 11833   (full prompt length)
# T_padded   = padded buffer length              >= T_new  (caller-allocated)
# S_old      = key_old.shape[0]                            (sum of len(token_ids[i])
#                                                           collected so far)
# H_q        = num_heads                         = 16      (Qwen3-Next 9B full-attn)
# H_kv       = num_kv_heads                      = 2
# G          = num_queries_per_kv = H_q / H_kv   = 8
# D          = head_size                         = 256
# K          = imp_indices.numel()                         (kvlink-4 selection size)
# B          = block_size                        = 528     (paged-cache block)

# ----------------------------------------------------------------------------
# Inputs to XFormersImpl.forward
# ----------------------------------------------------------------------------
query                                : [T_padded, H_q,  D]   = [>=11833, 16, 256]
key                                  : [T_padded, H_kv, D]   = [>=11833,  2, 256]
value                                : [T_padded, H_kv, D]   = [>=11833,  2, 256]
output                               : [T_padded, H_q,  D]   = [>=11833, 16, 256]
kv_cache                             : [2, num_blocks, B, H_kv, D]
attn_metadata.num_actual_tokens      : scalar                 = 11833
attn_metadata.max_query_len          : scalar                 = 11833
attn_metadata.slot_mapping           : [T_padded]             (int)
attn_metadata.block_table            : [num_seqs, max_blocks]
attn_metadata.query_start_loc        : [num_seqs + 1]
attn_metadata.seq_lens               : [num_seqs]

# ----------------------------------------------------------------------------
# After slicing to actual tokens (lines 303–305)
# ----------------------------------------------------------------------------
q  = query[:T_new]                   : [T_new,  H_q,  D]    = [11833, 16, 256]
k  = key[:T_new]                     : [T_new,  H_kv, D]    = [11833,  2, 256]
v  = value[:T_new]                   : [T_new,  H_kv, D]    = [11833,  2, 256]

# ----------------------------------------------------------------------------
# old_kv (installed by set_old_kvs in main.py)
# ----------------------------------------------------------------------------
# CORRECT (after fix): set_old_kvs runs once after the collect loop completes.
old_kv[0]                            : [T_new,  H_kv, D]    = [11833,  2, 256]
old_kv[1]                            : [T_new,  H_kv, D]    = [11833,  2, 256]
key_old   = old_kv[0].view(-1, H_kv, D) : [T_new,  H_kv, D] = [11833,  2, 256]
value_old = old_kv[1].view(-1, H_kv, D) : [T_new,  H_kv, D] = [11833,  2, 256]

# BUGGY (before fix): set_old_kvs ran inside the per-chunk loop, so on i=0
# only chunk 0 had been collected:
# key_old.shape   = [S_old, H_kv, D]   with S_old = len(token_ids[0])  (<< 11833)
# value_old.shape = [S_old, H_kv, D]
# This is what triggered the IndexKernel.cu:111 out-of-bounds asserts.

# ----------------------------------------------------------------------------
# imp_indices  ( cache_fuse_metadata["kvlink"] )
# ----------------------------------------------------------------------------
# Built in main.py from start_offset = [0, len(tok[0]), len(tok[0])+len(tok[1]), …, T_new]
# For kvlink-N (N=4): N tokens after each chunk boundary, plus the entire last chunk.
imp_indices (raw, int64)             : [K]
  - dtype                            : torch.long
  - device                           : cuda:0
  - min                              : start_offset[1] = len(token_ids[0])
  - max                              : start_offset[-1] - 1 = T_new - 1 = 11832
  - K (kvlink-4, hotpotqa, 4 chunks) ≈ 4*(num_chunks-2) + len(last_chunk)

imp_indices after `< T_new` filter   : [K]                   (no-op: all were < T_new)

# ----------------------------------------------------------------------------
# In-place merges of fresh K/V into the cached K/V (lines 380–381)
# ----------------------------------------------------------------------------
key_old[imp_indices]   = k[imp_indices]
  - LHS shape          : [K, H_kv, D]
  - RHS shape          : [K, H_kv, D]
  - LHS index range    : needs all imp_indices < key_old.shape[0]
                         FIXED: T_new (=11833)        →  in bounds
                         BUG  : S_old (<< 11833)      →  OUT OF BOUNDS

value_old[imp_indices] = v[imp_indices]
  - same shapes / same index requirement as above

# ----------------------------------------------------------------------------
# Query subselect (line 393)
# ----------------------------------------------------------------------------
q = q[imp_indices]                   : [K, H_q, D]          = [K, 16, 256]

# ----------------------------------------------------------------------------
# cache_fuse_metadata fields written here
# ----------------------------------------------------------------------------
cache_fuse_metadata["kvlink"]        : []                    (consumed)
cache_fuse_metadata["imp_indices"]   : [K]                   (= imp_indices)
cache_fuse_metadata["org_seq_len"]   : scalar = key_old.shape[0]
                                       FIXED: T_new (=11833)
                                       BUG  : S_old
cache_fuse_metadata["kv_cache_dtype"]: torch.bfloat16

# ----------------------------------------------------------------------------
# k_full / v_full passed to xformers
# ----------------------------------------------------------------------------
k_full = key_old                     : [T_new, H_kv, D]     = [11833, 2, 256]
v_full = value_old                   : [T_new, H_kv, D]     = [11833, 2, 256]

# ----------------------------------------------------------------------------
# _make_partial_bias_gqa internals (lines 637–657)
# ----------------------------------------------------------------------------
seq_len     = cache_fuse_metadata["org_seq_len"]            = 11833
padded_len  = ((seq_len + 7) // 8) * 8                       = 11840
torch.ones(padded_len, padded_len)   : [11840, 11840]       (bf16 on CUDA)
torch.triu(..., diagonal=1)          : [11840, 11840]
attn_mask after view                 : [1, 1, 1, 11840, 11840]
attn_mask[:, :, :, imp_indices]      : [1, 1, 1, K, 11840]
attn_mask.expand(1, H_kv, G, -1, -1) : [1, 2, 8, K, 11840]
attn_mask[..., :seq_len]             : [1, 2, 8, K, 11833]   ← returned attn_bias

# ----------------------------------------------------------------------------
# _xformers_custom_bias_attention reshape (GQA) — lines 580–598
# ----------------------------------------------------------------------------
query.view(K, H_kv, G, D)            : [K, 2, 8, 256]
key  [:, :, None, :].expand(...)     : [T_new, 2, 8, 256]   = [11833, 2, 8, 256]
value[:, :, None, :].expand(...)     : [T_new, 2, 8, 256]   = [11833, 2, 8, 256]
query.unsqueeze(0)                   : [1, K, 2, 8, 256]
key.unsqueeze(0)                     : [1, 11833, 2, 8, 256]
value.unsqueeze(0)                   : [1, 11833, 2, 8, 256]
attn_bias                            : [1, 2, 8, K, 11833]
xops.memory_efficient_attention_forward(...) out
                                     : [1, K, 2, 8, 256]
out.view_as(original_query)          : [K, H_q, D]          = [K, 16, 256]

# ----------------------------------------------------------------------------
# Output write-back (line 465)
# ----------------------------------------------------------------------------
output[: K]                          : [K, H_q, D]
out.view(K, H_q, D)                  : [K, H_q, D]
                                       (caller slices residual/hidden_states the same way)

# ----------------------------------------------------------------------------
# Root-cause one-liner
# ----------------------------------------------------------------------------
# expected:  key_old.shape[0] == num_actual_tokens          (= T_new = 11833)
# observed:  key_old.shape[0] == len(token_ids[0])          (= S_old, just chunk 0)
# trigger :  key_old[imp_indices] = k[imp_indices]
#            with max(imp_indices) = 11832 vs key_old.shape[0] = S_old
#            → IndexKernel.cu:111  "index out of bounds"
