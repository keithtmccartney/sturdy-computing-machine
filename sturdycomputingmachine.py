from llama_cpp import Llama

llm = Llama(model_path="C:/Users/Keith/Downloads/llama-2-7b-chat.Q2_K.gguf")

response = llm("Who directed The Dark Knight?")

print(response['choices'][0]['text'])

# --------------------

# .\run_app.bat

# C:\Users\Keith\source\repos\sturdy-computing-machine>pipenv run streamlit run sturdycomputingmachine.py 

#   You can now view your Streamlit app in your browser.

#   Local URL: http://localhost:8501
#   Network URL: http://192.168.0.129:8501

# llama_model_loader: loaded meta data with 19 key-value pairs and 291 tensors from ./models/llama-2-7b-chat.Q2_K.gguf (version GGUF V2)
# llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
# llama_model_loader: - kv   0:                       general.architecture str              = llama
# llama_model_loader: - kv   1:                               general.name str              = LLaMA v2
# llama_model_loader: - kv   2:                       llama.context_length u32              = 4096
# llama_model_loader: - kv   3:                     llama.embedding_length u32              = 4096
# llama_model_loader: - kv   4:                          llama.block_count u32              = 32
# llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 11008
# llama_model_loader: - kv   6:                 llama.rope.dimension_count u32              = 128
# llama_model_loader: - kv   7:                 llama.attention.head_count u32              = 32
# llama_model_loader: - kv   8:              llama.attention.head_count_kv u32              = 32
# llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000001
# llama_model_loader: - kv  10:                          general.file_type u32              = 10
# llama_model_loader: - kv  11:                       tokenizer.ggml.model str              = llama
# llama_model_loader: - kv  12:                      tokenizer.ggml.tokens arr[str,32000]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
# llama_model_loader: - kv  13:                      tokenizer.ggml.scores arr[f32,32000]   = [0.000000, 0.000000, 0.000000, 0.0000...
# llama_model_loader: - kv  14:                  tokenizer.ggml.token_type arr[i32,32000]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
# llama_model_loader: - kv  15:                tokenizer.ggml.bos_token_id u32              = 1
# llama_model_loader: - kv  16:                tokenizer.ggml.eos_token_id u32              = 2
# llama_model_loader: - kv  17:            tokenizer.ggml.unknown_token_id u32              = 0
# llama_model_loader: - kv  18:               general.quantization_version u32              = 2
# llama_model_loader: - type  f32:   65 tensors
# llama_model_loader: - type q2_K:   65 tensors
# llama_model_loader: - type q3_K:  160 tensors
# llama_model_loader: - type q6_K:    1 tensors
# llm_load_vocab: special tokens definition check successful ( 259/32000 ).
# llm_load_print_meta: format           = GGUF V2
# llm_load_print_meta: arch             = llama
# llm_load_print_meta: vocab type       = SPM
# llm_load_print_meta: n_vocab          = 32000
# llm_load_print_meta: n_merges         = 0
# llm_load_print_meta: n_ctx_train      = 4096
# llm_load_print_meta: n_embd           = 4096
# llm_load_print_meta: n_head           = 32
# llm_load_print_meta: n_head_kv        = 32
# llm_load_print_meta: n_layer          = 32
# llm_load_print_meta: n_rot            = 128
# llm_load_print_meta: n_gqa            = 1
# llm_load_print_meta: f_norm_eps       = 0.0e+00
# llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
# llm_load_print_meta: f_clamp_kqv      = 0.0e+00
# llm_load_print_meta: f_max_alibi_bias = 0.0e+00
# llm_load_print_meta: n_ff             = 11008
# llm_load_print_meta: n_expert         = 0
# llm_load_print_meta: n_expert_used    = 0
# llm_load_print_meta: rope scaling     = linear
# llm_load_print_meta: freq_base_train  = 10000.0
# llm_load_print_meta: freq_scale_train = 1
# llm_load_print_meta: n_yarn_orig_ctx  = 4096
# llm_load_print_meta: rope_finetuned   = unknown
# llm_load_print_meta: model type       = 7B
# llm_load_print_meta: model ftype      = Q2_K
# llm_load_print_meta: model params     = 6.74 B
# llm_load_print_meta: model size       = 2.63 GiB (3.35 BPW)
# llm_load_print_meta: general.name     = LLaMA v2
# llm_load_print_meta: BOS token        = 1 '<s>'
# llm_load_print_meta: EOS token        = 2 '</s>'
# llm_load_print_meta: UNK token        = 0 '<unk>'
# llm_load_print_meta: LF token         = 13 '<0x0A>'
# llm_load_tensors: ggml ctx size       =    0.11 MiB
# llm_load_tensors: system memory used  = 2694.43 MiB
# .................................................................................................
# llama_new_context_with_model: n_ctx      = 512
# llama_new_context_with_model: freq_base  = 10000.0
# llama_new_context_with_model: freq_scale = 1
# llama_new_context_with_model: KV self size  =  256.00 MiB, K (f16):  128.00 MiB, V (f16):  128.00 MiB
# llama_build_graph: non-view tensors processed: 676/676
# llama_new_context_with_model: compute buffer total size = 73.69 MiB
# AVX = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 0 | VSX = 0 |

# llama_print_timings:        load time =    7691.08 ms
# llama_print_timings:      sample time =      10.20 ms /    16 runs   (    0.64 ms per token,  1568.63 tokens per second)
# llama_print_timings: prompt eval time =    7690.67 ms /     7 tokens ( 1098.67 ms per token,     0.91 tokens per second)
# llama_print_timings:        eval time =   14235.45 ms /    15 runs   (  949.03 ms per token,     1.05 tokens per second)
# llama_print_timings:       total time =   22052.81 ms

#  nobody in particular. The Dark Knight is a 2008 super

# --------------------

# .\run_app.bat

# C:\Users\Keith\source\repos\sturdy-computing-machine>pipenv run streamlit run sturdycomputingmachine.py

#   You can now view your Streamlit app in your browser.

#   Local URL: http://localhost:8501
#   Network URL: http://192.168.0.129:8501

# llama_model_loader: loaded meta data with 19 key-value pairs and 291 tensors from C:/Users/Keith/Downloads/llama-2-7b-chat.Q2_K.gguf (version GGUF V2)
# llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
# llama_model_loader: - kv   0:                       general.architecture str              = llama
# llama_model_loader: - kv   1:                               general.name str              = LLaMA v2
# llama_model_loader: - kv   2:                       llama.context_length u32              = 4096
# llama_model_loader: - kv   3:                     llama.embedding_length u32              = 4096
# llama_model_loader: - kv   4:                          llama.block_count u32              = 32
# llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 11008
# llama_model_loader: - kv   6:                 llama.rope.dimension_count u32              = 128
# llama_model_loader: - kv   7:                 llama.attention.head_count u32              = 32
# llama_model_loader: - kv   8:              llama.attention.head_count_kv u32              = 32
# llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000001
# llama_model_loader: - kv  10:                          general.file_type u32              = 10
# llama_model_loader: - kv  11:                       tokenizer.ggml.model str              = llama
# llama_model_loader: - kv  12:                      tokenizer.ggml.tokens arr[str,32000]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
# llama_model_loader: - kv  13:                      tokenizer.ggml.scores arr[f32,32000]   = [0.000000, 0.000000, 0.000000, 0.0000...
# llama_model_loader: - kv  14:                  tokenizer.ggml.token_type arr[i32,32000]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
# llama_model_loader: - kv  15:                tokenizer.ggml.bos_token_id u32              = 1
# llama_model_loader: - kv  16:                tokenizer.ggml.eos_token_id u32              = 2
# llama_model_loader: - kv  17:            tokenizer.ggml.unknown_token_id u32              = 0
# llama_model_loader: - kv  18:               general.quantization_version u32              = 2
# llama_model_loader: - type  f32:   65 tensors
# llama_model_loader: - type q2_K:   65 tensors
# llama_model_loader: - type q3_K:  160 tensors
# llama_model_loader: - type q6_K:    1 tensors
# llm_load_vocab: special tokens definition check successful ( 259/32000 ).
# llm_load_print_meta: format           = GGUF V2
# llm_load_print_meta: arch             = llama
# llm_load_print_meta: vocab type       = SPM
# llm_load_print_meta: n_vocab          = 32000
# llm_load_print_meta: n_merges         = 0
# llm_load_print_meta: n_ctx_train      = 4096
# llm_load_print_meta: n_embd           = 4096
# llm_load_print_meta: n_head           = 32
# llm_load_print_meta: n_head_kv        = 32
# llm_load_print_meta: n_layer          = 32
# llm_load_print_meta: n_rot            = 128
# llm_load_print_meta: n_gqa            = 1
# llm_load_print_meta: f_norm_eps       = 0.0e+00
# llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
# llm_load_print_meta: f_clamp_kqv      = 0.0e+00
# llm_load_print_meta: f_max_alibi_bias = 0.0e+00
# llm_load_print_meta: n_ff             = 11008
# llm_load_print_meta: n_expert         = 0
# llm_load_print_meta: n_expert_used    = 0
# llm_load_print_meta: rope scaling     = linear
# llm_load_print_meta: freq_base_train  = 10000.0
# llm_load_print_meta: freq_scale_train = 1
# llm_load_print_meta: n_yarn_orig_ctx  = 4096
# llm_load_print_meta: rope_finetuned   = unknown
# llm_load_print_meta: model type       = 7B
# llm_load_print_meta: model ftype      = Q2_K
# llm_load_print_meta: model params     = 6.74 B
# llm_load_print_meta: model size       = 2.63 GiB (3.35 BPW)
# llm_load_print_meta: general.name     = LLaMA v2
# llm_load_print_meta: BOS token        = 1 '<s>'
# llm_load_print_meta: EOS token        = 2 '</s>'
# llm_load_print_meta: UNK token        = 0 '<unk>'
# llm_load_print_meta: LF token         = 13 '<0x0A>'
# llm_load_tensors: ggml ctx size       =    0.11 MiB
# llm_load_tensors: system memory used  = 2694.43 MiB
# .................................................................................................
# llama_new_context_with_model: n_ctx      = 512
# llama_new_context_with_model: freq_base  = 10000.0
# llama_new_context_with_model: freq_scale = 1
# llama_new_context_with_model: KV self size  =  256.00 MiB, K (f16):  128.00 MiB, V (f16):  128.00 MiB
# llama_build_graph: non-view tensors processed: 676/676
# llama_new_context_with_model: compute buffer total size = 73.69 MiB
# AVX = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 0 | VSX = 0 |

# llama_print_timings:        load time =    5174.79 ms
# llama_print_timings:      sample time =       4.90 ms /    16 runs   (    0.31 ms per token,  3264.64 tokens per second)
# llama_print_timings: prompt eval time =    5174.46 ms /     7 tokens (  739.21 ms per token,     1.35 tokens per second)
# llama_print_timings:        eval time =   10004.15 ms /    15 runs   (  666.94 ms per token,     1.50 tokens per second)
# llama_print_timings:       total time =   15244.41 ms

#  Unterscheidung zwischen "The Dark Knight" und "The Dark Knight Rises"

# --------------------