# TITAN Tokenizer Design
## Rust-first, domain-dense, memory-light

### Goals
- Fewer tokens per file than general code tokenizers
- Single-token keywords and common patterns
- Smart identifier splitting (snake_case / CamelCase)

### Single-token keywords (kept whole)
fn, impl, trait, struct, enum, match, async, await, move, unsafe, dyn, ref, mut, pub, crate, super, Self, self, let, if, else, loop, while, for, return, break, continue, mod, use, as, where, type, const, static, extern, macro_rules

### Single-token common generics / patterns
Result<T, E>, Option<T>, Arc<Mutex<T>>, Box<dyn, Pin<Box<dyn>, Vec<T>, HashMap<K, V>, RwLock<T>, Mutex<T>, Option<Result<T, E>>

### Single-token common crate names
tokio, serde, serde_json, tracing, anyhow, thiserror, axum, hyper, reqwest, sqlx, chrono, rand, itertools, rayon, crossbeam, futures, async_trait

### Smart identifier splitting
- snake_case: user_service → user, _, service
- CamelCase: UserService → User, Service

### Lifetime shorthand
'a, 'static, 'b, 'c, 'de as single tokens where vocabulary allows.

### Macro invocations
Treat entire macro name as single token where frequency justifies:
println!, format!, vec!, assert!, cfg!, derive, tokio::main!, etc.

### Process for building
1. Train BPE or SentencePiece on ~100GB of Rust-only corpora (crates.io, GH Rust projects, Rust book summaries, RFCs summarized)
2. Post-process vocabulary to force single-token mapping for keywords/patterns/crates above.
3. Use this tokenizer for both teacher and student during distillation to ensure alignment.
