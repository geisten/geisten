# Test-Migration-Plan (Phase E-2 Triage)

Triage-Pass durch alle 34 Test/Bench-Files. Kategorisierung steuert Phase E-3
(Streichen), E-4 (Migration mit Suffix), E-5 (Legacy-Bucket).

Stand: 2026-05-12 (Phase E-2 Output).

## Kategorisierungs-Regeln

| Kategorie | Bedeutung |
|---|---|
| **HALTEN** `_unit` | Aktiver Wert. Kernel-level, sub-second, kein GGUF (oder kleine Test-Daten). Phase-E-4 Rename. |
| **HALTEN** `_int`  | Aktiver Wert. Mehrere Module zusammen, kann GGUF brauchen, seconds. Phase-E-4 Rename. |
| **STREICHEN**      | Redundant oder superseded. Phase-E-3 löscht via `git rm`. |
| **DEFER**          | Schwere E2E mit komplexem Setup. Phase-E-5 markiert mit `_legacy` Suffix. Revisit nach C/B. |
| **BENCH**          | Nicht Pass/Fail — Timing-Tool. Phase-E-5 zieht in eigene `make bench` Target. |

## Triage-Tabelle

### HALTEN — `_unit` (11)

| Datei | Neuer Name | Begründung |
|---|---|---|
| test_3plane.c              | test_3plane_unit.c              | Self-contained kernel test, validates linear_ptqtp_decode_3plane vs scalar reference. |
| test_gguf_dequant.c        | test_gguf_dequant_unit.c        | Utility: GGUF tensor → FP32 dump for Python validation; <1s, no assertions. |
| test_gguf_reader.c         | test_gguf_reader_unit.c         | Smoke test: opens GGUF, prints metadata; passes without assertions. |
| test_iq_dequant.c          | test_iq_dequant_unit.c          | Validates IQ2_S/IQ3_S row dequant via cosine similarity. |
| test_mel_pipeline.c        | test_mel_pipeline_unit.c        | Mel spectrogram from WAV; self-contained, <1s. |
| test_safetensors.c         | test_safetensors_unit.c         | Bit-exact fingerprint diff vs Python oracle. |
| test_step1_embedding.c     | test_step1_embedding_unit.c     | Token embedding lookup + scaling. |
| test_step2_input_layernorm.c | test_step2_input_layernorm_unit.c | Embedding → input_layernorm. |
| test_step3_q_proj.c        | test_step3_q_proj_unit.c        | LayerNorm → q_proj. |
| test_step4_qkv_norm.c      | test_step4_qkv_norm_unit.c      | Q/K/V proj + per-head RMSNorm. |
| test_tokenizer.c           | test_tokenizer_unit.c           | sp_bpe_tokenizer encoding diff vs Python oracle. |

### HALTEN — `_int` (10)

| Datei | Neuer Name | Begründung |
|---|---|---|
| test_full_logits.c         | test_full_logits_int.c          | Multi-layer forward (35 layers) from safetensors (BF16 reference). |
| test_full_logits_gguf.c    | test_full_logits_gguf_int.c     | Same math but weights from quantized GGUF — confirms quant quality. |
| test_layers0to4.c          | test_layers0to4_int.c           | Layers 0-4 chain (small/fast) — sliding-attn validation. |
| test_layers0to14.c         | test_layers0to14_int.c          | Layers 0-14 chain (larger/full) — adds 2× full-attention layers. |
| test_prefill_q3k.c         | test_prefill_q3k_int.c          | Sweeps Q3_K tensors from GGUF, validates linear_q3k_w3a8_prefill vs CBLAS. |
| test_ptqtp.c               | test_ptqtp_int.c                | PTQTP linear_decode_2plane cosine-sim ≥0.99999 vs FP32. |
| test_q4k_kernel.c          | test_q4k_kernel_int.c           | Q4_K decode validation vs FP32 dequant+sgemv. |
| test_step12to14_ple.c      | test_step12to14_ple_int.c       | Steps 12-14: PLE pre-compute + per-layer merge. |
| test_step5_attn_oproj.c    | test_step5_attn_oproj_int.c     | Steps 5-7: RoPE → attention → o_proj. |
| test_step8to11_mlp_norms.c | test_step8to11_mlp_norms_int.c  | Steps 8-11: post_attn_norm → MLP → post_ff_norm. |

### STREICHEN (0)

Triage-Korrektur 2026-05-12: die initial als STREICHEN markierten Files
sind tatsächlich NICHT redundant. Diff zeigt:

- `test_full_logits_gguf.c` lädt Weights aus GGUF (mit Quant-Dequant), während
  `test_full_logits.c` aus Safetensors (BF16 reference) lädt. Beide getrennt
  wertvoll — der Cross-Check beider beweist Quantisierungs-Qualität.
- `test_layers0to14.c` testet 15 Layer (inkl. 2× full-attention), während
  `test_layers0to4.c` nur 5 Layer (alle sliding) testet. Ist eine Erweiterung,
  kein Duplikat.

Phase E-3 hat damit nichts zu löschen.

### DEFER (7) — wird `_legacy` markiert

| Datei | Begründung |
|---|---|
| test_audio_subsample.c     | Phase 6/7 audio pipeline, braucht WAV + mel-constants + safetensors; timing-sensitive. |
| test_chat_audio.c          | E2E mixed text+audio prefill, ~10s+. |
| test_chat_audio_repl.c     | REPL server-mode mit per-query timing, 30s+. |
| test_chat_audio_stream.c   | Threading + async streaming audio encode; komplex multi-phase. |
| test_greedy_kv_gguf.c      | Full LM forward 35 layers + greedy decode aus GGUF; minutes. |
| test_greedy_quant.c        | Thin lm.c driver: prefill + greedy decode; GGUF-bound, ~10s+. |
| test_greedy_repl.c         | Server-mode REPL für multi-job greedy; warm-state timing, minutes. |

Diese 7 werden **nach Phase C/B** revisited — viele werden dann durch
saubere Engine-Layer-Tests ersetzt oder können ihre Komplexität reduzieren.

### BENCH (6) — eigenes `make bench` Target (Phase E-5)

| Datei | Was es misst |
|---|---|
| bench_5trit_probe.c   | 4-bpw joint vs 5-trit-per-byte encoding throughput für PTQTP. |
| bench_dequant.c       | Q3_K / Q4_K row dequant in GB/s auf Gemma 4 E2B tensor sizes. |
| bench_ptqtp.c         | PTQTP 2-plane decoder ms/call + GB/s. |
| bench_q4k_kernel.c    | Q4_K W4A8 decode ms/call + GB/s; perf-stat-friendly. |
| bench_sgemv.c         | Apple BLAS sgemv throughput → bandwidth ceiling. |
| bench_stream.c        | STREAM Triad sustained DRAM bandwidth. |

Benches sind nicht Pass/Fail — sie messen. Werden separates `make bench`
Target bekommen, NICHT in `make test` aufgenommen.

## Gesamt-Statistik

| Kategorie | Count | Phase |
|---|---:|---|
| HALTEN _unit | 11 | E-4 Migration |
| HALTEN _int  | 10 | E-4 Migration |
| STREICHEN    |  0 | E-3 (no-op) |
| DEFER (→ `_legacy`) |  7 | E-5 Mark |
| BENCH (→ `make bench`) |  6 | E-5 Separate target |
| **Total** | **34** | |

## Erwartete `make test` Performance nach Migration

- **`make test-unit`**: 11 tests, ~5-10s total (sub-second je test, einige laden GGUF-Headers)
- **`make test-int`**: 10 tests, ~40-90s total (mehrere GGUF-bound, full-forward Tests)
- **`make test`** (default = unit + int): ~50-100s total

Diese Performance ist akzeptabel für daily-refactor-iteration. Wenn `_int`
zu langsam wird, splittet weiter zu `_int_fast` und `_int_slow`.

## Pre-Migration-Risiko

- Test-Tests werden Setup brauchen (GGUF-Path-Detection, Audio-Files):
  Phase E-4 fügt `GEIST_REQUIRE_GGUF` und ähnliche Skip-Logic ein.
- Some test files print "OK" without proper exit codes — Phase E-4
  ergänzt `return GEIST_TEST_PASS;` und Cleanup.
- Einige _int tests könnten heute fehlschlagen weil sie spezifische
  test-data brauchen (z.B. dumps/, *.npy files); Phase E-4 macht das
  explicit via Skip-Logic statt FAIL.
