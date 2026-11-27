# AITER gfx950 DSA kernels (vendored)

Verbatim copies of HIP kernels from [ROCm/aiter](https://github.com/ROCm/aiter)
(MIT, see `LICENSE`), built in-tree by `vllm/models/glm5next/amd/ops/hip`
against the headers of the AITER package that is installed (`aiter_meta`), so
they do not need an AITER release that contains them.

| Files | Source |
|---|---|
| `csrc/kernels/mla/sparse_mla_fp8.cu`, `csrc/kernels/mla/fp8/*` | ROCm/aiter#6037, main `deca29ee4` |
| `csrc/kernels/fp8_mqa_logits.cu`, `csrc/include/fp8_mqa_logits.h` | ROCm/aiter#7183 relanded as ROCm/aiter#8507, main `deca29ee4` |
| `csrc/kernels/kpool_compress.cu`, `csrc/include/kpool_compress.h` | ROCm/aiter#7374, main `deca29ee4` |
| `csrc/kernels/fp8_mqa_topk.cu`, `csrc/kernels/fp8_mqa_topk/*`, `csrc/include/fp8_mqa_topk.h` | ROCm/aiter#8010, head `8122e5686` |
| `csrc/kernels/fp8_paged_mqa_logits.cu`, `csrc/include/fp8_paged_mqa_logits.h` | ROCm/aiter#7184, head `eb335c0d5` |

ROCm/aiter#8780 only changes the Python arch gate of `fp8_mqa_logits`; the
wrappers in `ops/hip` check the live device the same way.

To update, copy the files from the new AITER commit and rerun
`tests/kernels/models/amd/test_glm5next_gfx950_dsa_kernels.py` on gfx950.
