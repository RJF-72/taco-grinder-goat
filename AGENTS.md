# TITAN Architecture Document
## The Titan System: 680B → 6B Distillation with Externalized Intelligence

### Philosophy
TITAN is not a "shrunken LLM." It is a re-architected intelligence system that separates language, reasoning, memory, and control into independent, optimizable components. The student model (1.5B–6B) only needs to *speak* and *pattern-match*; the heavy lifting (planning, memory, reward, tool orchestration) lives outside the model.

This lets a 6B model behave like a 680B model inside a specialized domain (Rust + project tooling).

---

## Four Pillars

### 1. The Driver (Control Engine)
- **Perfect = deterministic, schema-driven, mode-aware, tier-aware, memory-aware, tool-strict.**
- Responsibilities:
  - Task decomposition
  - Mode selection (REQUIREMENT, SPEC, CODE, EXPLAIN, VISION)
  - Context selection (top-K relevant files, summaries, never whole repo)
  - Tool orchestration (strict JSON schema)
  - Loop control (stop conditions, rollback triggers, step limits)
- File: `crates/lariat_driver/src/lib.rs`

### 2. The Inference Model (Language + Pattern Engine)
- **Job:** Generate Rust code, emit tool calls, summarize, explain, interpret vision inputs.
- Sizes:
  - ULTRA → 6B (Q5/FP16)
  - STANDARD → 3B equivalent (Q4)
  - LITE → 1.5B (Q3)
- Trained multi-task on:
  - 40% problem solving + code
  - 20% tools
  - 15% NLU
  - 15% NLP
  - 10% vision
- Distilled from a 680B general teacher by behavior alignment, not weight copying.
- File: `crates/lariat_inference/src/lib.rs`

### 3. The Physical Model (Memory + Skill Engine)
- **On-disk structured data:** trajectories, diffs, errors, fixes, rewards.
- Never held in RAM; streamed, summarized, versioned.
- Layout:
  - `.lariat/training/trajectories/<run_id>/step_*.json`
  - `.lariat/training/diffs/`
  - `.lariat/training/rewards/`
  - `.lariat/models/`
- File: `crates/lariat_physical_model/src/lib.rs`

### 4. MMIE (Reasoning + Improvement Engine)
- **Job:** Score every step, keep only top trajectories, retrain the model periodically.
- Reward components:
  - compile pass/fail
  - test pass/fail
  - step efficiency
  - diff size
  - tool correctness
  - requirement alignment
  - explanation quality
  - memory/latency penalties
- Tuned per tier (ULTRA vs STANDARD vs LITE).
- File: `crates/lariat_mmie/src/lib.rs`

---

## Capability Profile (The Machine Fingerprint)
At startup, TITAN probes the host:
- RAM / VRAM totals and free
- CPU cores / speed
- Disk type / free space / speed
- Network
- Power mode

From this, it derives:
- **Tier:** ULTRA / STANDARD / LITE
- **Limits:** max model memory, max context tokens, max KV cache, max parallel tasks
- **Features:** vision enabled/disabled, high precision allowed/disallowed

The driver uses this to choose model variant, quantization, context length, and aggressiveness.

---

## Memory-First Design Rules
1. Always quantize by default.
2. Hard-cap KV cache per tier.
3. Never load full repo into context.
4. Summarize logs aggressively.
5. Stream files; keep only current-task data in RAM.
6. Store trajectories on disk; only current session in memory.
7. Penalize memory abuse in MMIE rewards.

---

## Distillation Pipeline (680B → 6B)
1. **Teacher Runs:** 680B agent under final driver → record trajectories.
2. **Dataset Build:** Extract SFT pairs (input prompt → teacher output).
3. **Weighted SFT:** High loss weight on tool JSON + code; lower on explanations.
4. **Gap Filling:** Evaluate 6B; where weak, generate more teacher data; re-distill.
5. **MMIE Self-Improvement:** Turn off teacher. Let 6B run. Score with MMIE. Periodically retrain on top trajectories.

---

## Tokenizer Strategy
- Rust-first vocabulary
- Single-token keywords, common generics, crate names
- Smart identifier splitting
- Trained on Rust-only corpora; shared between teacher and student
- Reduces tokens-per-file, which directly reduces memory pressure

---

## Mode Prompt Contracts
- REQUIREMENT → structured plan from vague request
- SPEC → extract constraints from docs
- CODE → minimal patch + tool calls
- EXPLAIN → summary + impact + rollback note
- VISION → extract structure from images (tier-gated)

Prompt templates live in: `training/prompts.md`

---

## File Map
- `crates/lariat_capability` — hardware profiler + tier logic
- `crates/lariat_driver` — deterministic agent conductor
- `crates/lariat_physical_model` — on-disk memory schemas + I/O
- `crates/lariat_mmie` — reward scoring + trajectory curation
- `crates/lariat_inference` — model facade + generation
- `crates/lariat_common` — shared types
- `training/distill.py` — dataset builder for distillation
- `training/tokenizer/design.md` — tokenizer spec
- `training/prompts.md` — mode prompt templates

---

## Next Steps (To Implement)
- [ ] Real OS/GPU probing in `CapabilityProfile::probe`
- [ ] Model backend integration (llama.cpp / candle / custom)
- [ ] Tool JSON schema validation + executor
- [ ] Embedding retrieval for context ranking
- [ ] MMIE offline retraining scheduler
- [ ] Vision encoder integration (tier-gated)
- [ ] Config predictor small-ML model for profile tuning
- [ ] Full end-to-end integration test on a Rust repo
