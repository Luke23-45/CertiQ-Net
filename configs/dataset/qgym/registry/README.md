# Dataset Registry Configuration Reference

## Architecture Overview

The registry consists of **one file**: `datasets.yaml`.  It defines **every QGym dataset** in a compact structure with a shared `defaults` section and per-dataset overrides.  The parser deep-merges each dataset entry onto the defaults so that only diverging parameters need to be written.

The file spans **two separate systems** with different parameter ownership:

```
Registry YAML  (this file)
    │
    ├── env: ──────────────────────────────► QGym Environment YAML
    │                                             │
    │                                   ┌─────────┼──────────┐
    │                                   ▼         ▼          ▼
    │                              network.npy  mu.npy   lam.npy  delta.npy
    │                              (s×q)        (s×q)    (q,)     (2q×q)
    │
    ├── collection:* ─────────────────────► CertiQ-Net DatasetCollectionManager
    │                                             │
    │                                             ▼
    │                                       QGymAdapter._sample_online()
    │                                        (steps QGym env, records states)
    │
    └── training_defaults:* ───────────────► CertiQ-Net QGymDataModule
                                              │
                                              ▼
                                         _build_train()
                                          (mixes QGym states + synthetic)
```

**Two config files are required per topology:**

| File | Location | Purpose |
|------|----------|---------|
| **Registry YAML** | `configs/dataset/qgym/registry/datasets.yaml` | All datasets: collection + training params |
| **Env YAML** | `extern/QGym/configs/env/<name>.yaml` | Queueing topology (links to .npy data) |

The `.npy` data files live in `extern/QGym/configs/env_data/<env_type>/`.

---

## File Format: Multi-Dataset YAML with Inheritance

```yaml
defaults:
  output_dir: final_dataset/qgym
  collection:
    n_steps: 500000
    n_valid: 10000
    ...

datasets:
  reentrant_2: &re2
    env: extern/QGym/configs/env/reentrant_2.yaml
    env_N: 6
    ...

  reentrant_2_debug:
    <<: *re2
    collection: {n_steps: 10000, n_valid: 1000, ...}
    training_defaults: {n_samples: 256}
```

**Inheritance model** (two-layer):

| Layer | Source | Mechanism |
|-------|--------|-----------|
| 1. Global defaults | `defaults:` section | Python `_deep_merge()` — recursive dict merge |
| 2. Topology anchors | YAML `&anchor` / `<<:` | Standard YAML 1.1 merge key (shallow) |

The `defaults` section sets shared parameters once. Each `datasets` entry inherits from `defaults` via deep merge — nested `collection` and `training_defaults` dicts are merged recursively, not replaced wholesale.  YAML anchors (`&re2` / `<<: *re2`) share topology-specific fields (`env`, `env_N`, `env_mu_fixed`, `env_lam`) across production and debug variants.

### Adding a new topology

```yaml
new_topology: &nt
  env: extern/QGym/configs/env/new_topology.yaml
  env_N: 12
  env_lam: 0.05

new_topology_debug:
  <<: *nt
  collection: {n_steps: 10000, n_valid: 1000, n_test: 1000, shard_size: 5000}
  training_defaults: {n_samples: 256}
```

### Adding a variant to an existing topology

```yaml
reentrant_2_large:
  <<: *re2
  collection:
    n_steps: 2000000
    n_valid: 50000
    n_test: 50000
```

---

## Section 1: Registry YAML Parameters

Every parameter in the registry YAML, exhaustively documented.

### File Structure

The file has two top-level keys:

| Key | Required | Type | Purpose |
|-----|----------|------|---------|
| `defaults` | No | `dict` | Shared parameters deep-merged into every dataset |
| `datasets` | Yes | `dict[str, dict]` | Per-dataset overrides; each key is the dataset name |

`name` is **not** a top-level key — it is inferred from the dataset key in `datasets`.

### 1.1 Identity & Topology

---

#### `name`
| | |
|---|---|
| **Type** | `str` |
| **Required** | Yes |
| **Owner** | CertiQ-Net |
| **Scope** | Common (every dataset) |

**Controls**: The unique registry key. Used by `DatasetRegistry` as the dict key for spec lookup. Referenced by:
- `collect_qgym.py` — `collect reentrant_2`
- `QGymDataModule` — `dataset_name: reentrant_2`
- `DatasetRegistry.get("reentrant_2")` — spec resolution
- `DatasetRegistry.resolve_path("reentrant_2")` — `final_dataset/qgym/reentrant_2/`

**Constraint**: Must be unique across all `registry/*.yaml` files. Duplicates silently overwrite.

---

#### `env`
| | |
|---|---|
| **Type** | `str` (file path) |
| **Required** | Yes |
| **Owner** | Bridge (CertiQ-Net → QGym) |
| **Scope** | Common (every dataset) |

**Controls**: Relative path to the QGym environment YAML. Resolved by `resolve_env_config_path()` which searches 8 candidate locations in order:
1. `configs/qgym/env/<name>.yaml` (CWD-relative)
2. `extern/QGym/configs/env/<name>.yaml` (CWD-relative)
3. Path as-is
4. With `.yaml` stripped if it has an extension
5–8. Same 4 paths relative to project root

The resolved YAML defines the queueing topology. The numeric network/mu matrices come from `.npy` files, not directly from this YAML (see env YAML section).

**Example**: `extern/QGym/configs/env/reentrant_2.yaml`

---

#### `env_N`
| | |
|---|---|
| **Type** | `int \| None` |
| **Default** | `null` (not set) |
| **Owner** | CertiQ-Net |
| **Scope** | Topology-specific |
| **Status** | **Derived convenience** — can be obtained from `len(h)` or `.npy` matrix dims |

**Controls**: Cache of the number of queues `N`. Used as fast-path fallback in:
- `train/runner.py:165` — model architecture dimension
- `eval/audit.py:60` — state bank dimension
- `eval/baselines.py:58` — model dimension
- `experiments/runner/engine.py:67` — run name tag

When `None`, all four consumers fall back to computing from the env config.

**Derived from**: The env YAML's `h` field length (e.g., `h: [1,1,1,1,1,1]` → N=6), or the second dimension of the `mu` / `network` `.npy` matrices.

**Recommendation**: Include in topology anchor for performance (avoids loading `.npy` at parse time). Omission is safe — consumers compute on demand.

---

#### `env_mu_fixed`
| | |
|---|---|
| **Type** | `list[float] \| None` |
| **Default** | `null` (not set) |
| **Owner** | CertiQ-Net |
| **Scope** | Topology-specific |
| **Status** | **Derived convenience** — `(network × mu).sum(dim=0)` from .npy files |

**Controls**: Cache of the effective per-queue service rate vector. Used as fast-path fallback in:
- `train/runner.py:166-169` — mu tensor for QGymDataModule
- `eval/audit.py:61` — state bank generator
- `eval/baselines.py:59` — baseline comparison

When `None`, all three consumers fall back to `build_mu(cfg)` which loads from the experiment config (non-QGym fallback path).

**Derived from**: `(network_npy × mu_npy).sum(axis=0)` — the elementwise product of the `(s,q)` network and mu matrices summed over servers. For `reentrant_2`:
```
network:  [[1,1,1,0,0,0],          mu:  [[0.125,0.5,0.25,0,0,0],
          [0,0,0,1,1,1]]                [0,0,0,0.167,0.143,1.0]]
→ effective_mu = [0.125, 0.5, 0.25, 0.167, 0.143, 1.0]
```

**Recommendation**: Include for topologies where it's known (like `reentrant_2`). Omit for others — the fallback path handles it.

---

#### `env_lam`
| | |
|---|---|
| **Type** | `float \| None` |
| **Default** | `null` (not set) |
| **Owner** | CertiQ-Net |
| **Scope** | Topology-specific |
| **Status** | **Derived convenience** — `sum(lam_npy)` from .npy file |

**Controls**: Cache of the sum of per-queue arrival rates. Used as fast-path fallback in:
- `train/runner.py:171-174` — lam scalar for cost computation
- `eval/audit.py:62` — state bank generator
- `eval/baselines.py:60` — baseline comparison
- `experiments/runner/engine.py:68` — run name tag

When `None`, all four consumers fall back to `build_mu(cfg)`.

**Derived from**: `sum(lam_vector)` where lam is loaded from `env_type_lam.npy`. This is NOT the throughput — it is the sum of per-queue independent Poisson arrival rates. Different topologies distribute arrivals differently:

| Topology | Entry queues | Per-entry rate | env_lam (= sum) |
|----------|-------------|----------------|-----------------|
| reentrant_2 | 0, 2 | 0.064286 | **0.128575** |
| reentrant_3 | 0, 2 | 0.064286 | **0.128578** |
| re-reentrant_3 | 0 | 0.064286 | **0.064294** |

All three have the **same bottleneck utilization ρ = 0.514** at the bottleneck queues (μ=0.125, entry rate=0.064286). The env_lam differs only because of different numbers of entry points.

**Recommendation**: Include in topology anchor for performance. Omission is safe — consumers compute on demand.

---

#### `output_dir`
| | |
|---|---|
| **Type** | `str` |
| **Default** | `"final_dataset/qgym"` |
| **Owner** | CertiQ-Net |
| **Scope** | Common |

**Controls**: Parent directory (relative to project root) where collected dataset is stored. The full path becomes `<project_root>/<output_dir>/<name>/`.

**Override behavior**: If path is relative, it's resolved relative to the project root (3 levels up from `collection_manager.py`). If absolute, used as-is.

**Recommendation**: Keep the default unless you need a custom data root for disk space or organizational reasons.

---

### 1.2 Collection Control

All parameters under the `collection:` key are consumed by `DatasetCollectionManager.collect()` and forwarded to `QGymAdapter.sample_batch()`.

---

#### `collection.n_steps`
| | |
|---|---|
| **Type** | `int` |
| **Default** | `500000` |
| **Owner** | CertiQ-Net |
| **Scope** | Common |

**Controls**: Number of training states to collect. These are individual queue-length snapshots (not simulation steps — the CTMC runs for far more steps). Total collected = `n_steps + n_valid + n_test`.

**Effect on data**:
- More states → better coverage of the state space → better training
- 500K is adequate for N=6, marginal for N=9
- Scale roughly as `(max_queue)^N` for full coverage (impossible above N≈4)

**Typical values**: 500000 (production), 10000 (debug/CI).

---

#### `collection.n_valid`
| | |
|---|---|
| **Type** | `int` |
| **Default** | `10000` |
| **Owner** | CertiQ-Net |
| **Scope** | Common |

**Controls**: Number of validation states. Sampled from the same shuffled pool as training states.

**Typical values**: 10000 (production), 1000 (debug).

---

#### `collection.n_test`
| | |
|---|---|
| **Type** | `int` |
| **Default** | `10000` |
| **Owner** | CertiQ-Net |
| **Scope** | Common |

**Controls**: Number of test states. Same split logic as validation.

**Typical values**: 10000 (production), 1000 (debug).

---

#### `collection.shard_size`
| | |
|---|---|
| **Type** | `int` |
| **Default** | `50000` |
| **Owner** | CertiQ-Net |
| **Scope** | Common |

**Controls**: Maximum number of states per `.pt` shard file. Controls memory usage:
- Load time: `QGymDataset` loads entire shards into RAM
- Training: shards are accessed randomly via `__getitem__` which requires them in memory

**Effect**: Smaller shard_size = more files = lower memory per load, more frequent disk I/O. Larger shard_size = fewer files = higher memory per load, less I/O.

**Typical values**: 50000 (production), 5000 (debug).

---

#### `collection.seed`
| | |
|---|---|
| **Type** | `int` |
| **Default** | `42` |
| **Owner** | Both (CertiQ-Net + QGym) |
| **Scope** | Common |

**Controls**: Seeds the following in order:
1. `torch.Generator().manual_seed(seed)` — for the global shuffle permutation
2. `QGymAdapter._sample_online()` — seeds `np.random.seed(seed)` from the torch generator
3. `load_rl_p_env(seed=seed)` → `RL_Wrapper_P_DiffDiscreteEventSystem(seed=seed)` → `DiffDiscreteEventSystem(seed=seed)` → `np.random.RandomState(seed)` — for all CTMC randomness: service times, arrival times, policy sampling

**Effect**: Deterministic reproduction of the CTMC trajectory. Same seed → same sequence of states.

---

#### `collection.policy`
| | |
|---|---|
| **Type** | `str` — one of `"random"`, `"sed"`, `"qmd"`, `"softmax"`, `"mixed"` |
| **Default** | `"mixed"` |
| **Owner** | CertiQ-Net |
| **Scope** | Common |

**Controls**: The dispatching policy used to generate actions during data collection. This is the single most important parameter affecting the **state distribution**. It is NOT a QGym parameter — it is implemented entirely in `QGymAdapter._resolve_collect_policy()` as NumPy closures.

| Policy | Algorithm | State Distribution |
|--------|-----------|-------------------|
| `random` | `uniform(0,5) × network` | Broad coverage, unrealistic large queues |
| `sed` | `argmin((Q+1)/μ_eff)` | Short queues, greedy delay minimization |
| `qmd` | `argmin((2Q+1)/μ_eff)` | Balanced queues, MaxWeight-like |
| `softmax` | Boltzmann over SED scores | Stochastic version of SED |
| `mixed` | Weighted random choice among above 4 | Balanced coverage (default) |

**Important**: The policy actions are **stored in shards as optional metadata but never loaded during training**. They are irrelevant to the model — the QGym states are used only as **starting states for RL rollouts**, not as action demonstrations (see training loop).

---

#### `collection.policy_weights`
| | |
|---|---|
| **Type** | `dict[str, float]` |
| **Default** | `{random: 0.25, sed: 0.25, qmd: 0.25, softmax: 0.25}` |
| **Owner** | CertiQ-Net |
| **Scope** | Relevant only when `policy: "mixed"` |

**Controls**: Per-policy sampling weights for the mixed policy. At each environment step, a sub-policy is chosen with probability proportional to its weight. Must sum to ~1.0 ± 0.01.

**Effect**: Higher weight = more states generated under that policy's action distribution. Equal weights (0.25 each) gives the most diverse coverage.

---

#### `collection.batch_size_env` ❌ REMOVED
| | |
|---|---|
| **Status** | **Removed** from the new YAML format; hardcoded to `1` in `CollectionConfig` |

**Controls**: Number of parallel QGym environment instances. When > 1, multiple CTMC simulations run in parallel via batched tensor operations in `DiffDiscreteEventSystem`.

**History**: Always 1 in every existing config, never overridden. Parallel environments would speed up collection but are unused. If needed in the future, add as a per-dataset override in `collection:`.

---

### 1.3 Training Defaults

All parameters under `training_defaults:` are passed as keyword arguments to `QGymDataModule.__init__()`. They are overridden by explicit experiment config values (experiment config wins).

---

#### `training_defaults.mode`
| | |
|---|---|
| **Type** | `str` — `"static"` or `"online"` |
| **Default** | `"static"` |
| **Owner** | CertiQ-Net |
| **Scope** | Common |

**Controls**: Whether the data module loads pre-collected `.pt` shards (`static`) or steps a live QGym environment during training (`online`).

| Mode | Data Source | Pros | Cons |
|------|-------------|------|------|
| `static` | Pre-collected shards | Reproducible, fast, no QGym dependency | Fixed state distribution |
| `online` | Live QGym env | Infinite fresh data, adaptive distribution | Slow, non-reproducible, QGym dependency |

All current production datasets use `static` mode.

---

#### `training_defaults.batch_size`
| | |
|---|---|
| **Type** | `int` |
| **Default** | `64` |
| **Owner** | CertiQ-Net |
| **Scope** | Common |

**Controls**: Batch size for PyTorch `DataLoader`. Each gradient update processes this many states.

---

#### `training_defaults.n_samples`
| | |
|---|---|
| **Type** | `int` |
| **Default** | `512` |
| **Owner** | CertiQ-Net |
| **Scope** | Common |

**Controls**: Number of training states assembled per epoch by `_build_train()`. This is the TOTAL before mixing — after mixing fractions are applied, the actual number of distinct states is larger (adversarial is additive).

---

#### `training_defaults.max_queue`
| | |
|---|---|
| **Type** | `int` |
| **Default** | `15` |
| **Owner** | CertiQ-Net |
| **Scope** | Common |

**Controls**: Maximum queue length for the **fallback synthetic generator** only. Used only when:
1. No QGym source is available (no static dataset, no online adapter)
2. Replay buffers are empty
3. Bootstrapping fails

In that case, states are drawn from `Uniform(0, max_queue)`. This path is essentially never hit in production.

---

#### `training_defaults.resample_every_epoch`
| | |
|---|---|
| **Type** | `bool` |
| **Default** | `true` |
| **Owner** | CertiQ-Net |
| **Scope** | Common |

**Controls**: Whether `_build_train()` is called at the start of each epoch to produce a fresh mixing of all data sources. When `true`, each epoch sees a different random subset of QGym states and synthetic states. When `false`, the same training set is reused.

**Effect**: Resampling prevents overfitting to a fixed set of states and exposes the model to more diverse data over the course of training.

---

#### `training_defaults.policy_buffer_max`
| | |
|---|---|
| **Type** | `int` |
| **Default** | `4096` |
| **Owner** | CertiQ-Net |
| **Scope** | Common |

**Controls**: Maximum capacity of the policy and teacher replay buffers (deques with maxlen). When full, oldest states are evicted. States in these buffers are sampled as part of the training set (controlled by `policy_mix_fraction` and `teacher_mix_fraction`).

---

#### `training_defaults.teacher_mix_fraction`
| | |
|---|---|
| **Type** | `float` (0.0–1.0) |
| **Default** | `0.25` |
| **Owner** | CertiQ-Net |
| **Scope** | Common |

**Controls**: Fraction of `n_samples` drawn from the **teacher replay buffer** — states visited by the SED/QMD heuristic during evaluation rollouts. 0.25 means 128 out of 512 training states come from teacher replay.

---

#### `training_defaults.policy_mix_fraction`
| | |
|---|---|
| **Type** | `float` (0.0–1.0) |
| **Default** | `0.25` |
| **Owner** | CertiQ-Net |
| **Scope** | Common |

**Controls**: Fraction of `n_samples` drawn from the **policy replay buffer** — states visited by the model's own policy during training rollouts. Creates a self-supervised loop: the model learns from states it previously visited.

---

#### `training_defaults.hard_state_fraction`
| | |
|---|---|
| **Type** | `float` (0.0–1.0) |
| **Default** | `0.75` |
| **Owner** | CertiQ-Net |
| **Scope** | Common |

**Controls**: Fraction of the QGym-allocated portion replaced by synthetic high-backlog states from the state bank generator. Applied within the already-reduced QGym fraction:

```
qgym_fraction = 1 - teacher_mix - policy_mix = 0.50
hard_count = int(qgym_count * hard_state_fraction)     = 75% of QGym portion
easy_qgym_count = qgym_count - hard_count               = 25% of QGym portion
```

With defaults: hard = 0.50 × 0.75 = 0.375 of total = 192 states. Pure QGym = 0.50 × 0.25 = 0.125 = 64 states.

**Effect**: Forces the model to learn from high-backlog regions that are rarely visited by the collection policy but critical for certificate safety.

---

#### `training_defaults.adversarial_fraction`
| | |
|---|---|
| **Type** | `float` (0.0–1.0) |
| **Default** | `0.25` |
| **Owner** | CertiQ-Net |
| **Scope** | Common |

**Controls**: **Additive** fraction of `n_samples` drawn from the adversarial state bank — states optimized via gradient ascent on `-certificate_slack` using the model's own gradients. This is NOT subtracted from the QGym fraction.

With defaults: adversarial = 0.25 × 512 = 128 additional states per epoch.

**Effect**: Trains the model to handle worst-case certificate-violating states, improving robustness.

---

## Section 2: Env YAML Parameters (QGym Topology)

Located in `extern/QGym/configs/env/<name>.yaml`. These define the queueing network topology and are consumed by both QGym's `DiffDiscreteEventSystem` and CertiQ-Net's `env_loader.py`.

### Parameters Passed to QGym

---

#### `name`
| | |
|---|---|
| **Type** | `str` |
| **Required** | Yes |
| **Owner** | QGym |

**Controls**: Environment name tag. Used by `env_loader.py` to locate `.npy` data files when `env_type` is not set. The `.npy` lookup path is `configs/env_data/<env_type or name>/`.

---

#### `env_type` (optional)
| | |
|---|---|
| **Type** | `str \| None` |
| **Default** | Falls back to `name` |
| **Owner** | CertiQ-Net (stripped before QGym) |

**Controls**: Overrides the subdirectory name for `.npy` data files. Allows a registry dataset name to differ from its `.npy` directory name. For example, `reentrant_3_hyper.yaml` sets `env_type: reentrant_3` so it loads data from `configs/env_data/reentrant_3/`.

**Stripped**: Removed from the config dict before passing to `load_rl_p_env()` (line 141 of env_loader.py). Only used by CertiQ-Net's env_loader for file resolution.

---

#### `service_type`
| | |
|---|---|
| **Type** | `str` — `"exponential"` or `"hyper"` |
| **Default** | `"exponential"` |
| **Owner** | QGym (patched by CertiQ-Net) |

**Controls**: Service time distribution at each queue:
- `"exponential"`: standard `Exponential(1)` per service — multiplied by the service rate (the mu matrix handles rate scaling, this is the base distribution)
- `"hyper"`: hyperexponential — `Bernoulli(0.5)` mix of `Exponential(1+scale)` and `Exponential(1-scale)`

The `"hyper"` mode was added by CertiQ-Net's `qgym_patch.py`. QGym natively only supports exponential.

**Effect on data**: Hyperexponential service times have higher variance (heavy-tailed). Some service completions happen very fast, others very slow. This creates more queue buildup and harder scheduling problems.

---

#### `lam_type`
| | |
|---|---|
| **Type** | `str` — `"constant"`, `"step"`, `"hyper"` |
| **Default** | `"constant"` |
| **Owner** | QGym (patched by CertiQ-Net) |

**Controls**: Arrival rate dynamics:
- `"constant"`: fixed per-queue arrival rates from `.npy` file (or `lam_params.val`)
- `"step"`: step change at `t_step` from `val1` to `val2` (not used in current datasets)
- `"hyper"`: Bernoulli(0.5) switching between `λ/(1+scale)` and `λ/(1-scale)` — added by `qgym_patch.py`

---

#### `lam_params`
| | |
|---|---|
| **Type** | `dict` |
| **Default** | `{val: null}` |
| **Owner** | QGym |

**Controls**: Arrival rate parameters:
- `val`: per-queue arrival rates. When `null`, loaded from `env_type_lam.npy`.
- `scale`: (used when `lam_type: "hyper"`) controls the switching amplitude. Default 0.8.
- `t_step`, `val1`, `val2`: (used when `lam_type: "step"`) step change parameters.

---

#### `network`
| | |
|---|---|
| **Type** | `list[list[float]] \| None` |
| **Default** | `null` |
| **Owner** | QGym |

**Controls**: Server-to-queue connectivity matrix of shape `(s, q)`. Element `network[s, q] = 1` if server `s` can serve queue `q`. When `null`, loaded from `env_type_network.npy`.

---

#### `mu`
| | |
|---|---|
| **Type** | `list[list[float]] \| None` |
| **Default** | `null` |
| **Owner** | QGym |

**Controls**: Service-rate matrix of shape `(s, q)`. Each element `mu[s, q]` is the service rate of server `s` when working on queue `q`. When `null`, loaded from `env_type_mu.npy`.

**These values are always static** — they do not change with `service_type`. The hyperexponential service type affects the random service time draws, not the underlying rate matrix.

---

#### `h`
| | |
|---|---|
| **Type** | `list[float]` |
| **Required** | Yes |
| **Owner** | Both |

**Controls**: Per-queue holding-cost vector of length `N`. Each element `h[i]` is the cost per unit time per job in queue `i`. Used by:
- QGym: `cost = event_time × Q · h` (per-step cost during simulation)
- CertiQ-Net: `compute_queue_holding_cost(Q, h) = (Q × h).sum(-1)` (training cost)

**N is determined by `len(h)`.** The queue count in the network/mu matrices must match this length, or validation fails.

**Current datasets**: All use `h = [1,1,...]` (uniform holding cost).

---

#### `init_queues`
| | |
|---|---|
| **Type** | `list[float]` |
| **Default** | All zeros |
| **Owner** | QGym |

**Controls**: Initial queue lengths at `env.reset()`. Always zero for current datasets — the CTMC warms up from empty after each reset.

---

#### `queue_event_options`
| | |
|---|---|
| **Type** | `list[list[float]] \| str \| None` |
| **Default** | `null` |
| **Owner** | QGym |

**Controls**: When `"custom"`, loaded from `env_type_delta.npy`. This matrix defines all possible event types (arrivals, departures, routing) as delta vectors applied to the queue state. Each row is one event type with shape `(N,)` where a `+1` indicates an arrival to that queue and a `-1` indicates a departure.

The delta matrix ENCODES THE JOB ROUTING TOPOLOGY — which queues feed into which others after service. This is the real "network topology" definition, separate from the server-queue bipartite graph.

---

#### `train_T`
| | |
|---|---|
| **Type** | `int` |
| **Default** | `20000` |
| **Owner** | QGym |

**Controls**: Maximum number of simulation steps per training episode. The CTMC is truncated at this many events (not wall-clock time). When `env.reset()` is triggered during collection, the environment runs for up to train_T steps before being truncated and reset again.

**Current values**: 20000 for reentrant_2, 30000 for hyper variants.

---

#### `test_T`
| | |
|---|---|
| **Type** | `int` |
| **Default** | `10000` |
| **Owner** | QGym |

**Controls**: Same as `train_T` but for evaluation episodes. Not used during data collection — only during validation/evaluation rollouts if the QGym environment is used directly.

**Current values**: 10000 for reentrant_2, 80000 for hyper variants.

### Env YAML Parameters Stripped Before QGym

These are consumed by CertiQ-Net's `env_loader.py` and removed from the config dict before being passed to QGym (line 141-142 of `env_loader.py`):

| Parameter | Purpose | Where Used |
|-----------|---------|------------|
| `env_type` | Override .npy subdirectory name | `env_loader.py:111` — file resolution |
| `num_pool` | Number of server pools | `env_loader.py:141` — stripped, unused |
| `server_pool_size` | Servers per pool | `env_loader.py:141` — stripped, unused |

---

## Section 3: Parameter Dependency Graph

```
datasets.yaml
│
├── defaults: ──────── shared, deep-merged into every dataset entry
│
└── datasets.<name>:
    ├── env ──────────────────── independent (points to env YAML)
    ├── env_N ────────────────── DERIVED from len(h) or .npy dims (cached)
    ├── env_mu_fixed ─────────── DERIVED from (network × mu).sum(axis=0) (cached)
    ├── env_lam ──────────────── DERIVED from sum(lam) (cached)
    ├── output_dir ───────────── independent
    │
    ├── collection
    │   ├── n_steps ──────────── independent
    │   ├── n_valid ──────────── independent
    │   ├── n_test ───────────── independent
    │   ├── shard_size ───────── independent
    │   ├── seed ─────────────── independent
    │   ├── policy ───────────── independent
    │   └── policy_weights ───── required only when policy="mixed"
    │
    └── training_defaults
        ├── mode ─────────────── independent ("static" or "online")
        ├── batch_size ───────── independent
        ├── n_samples ────────── independent
        ├── max_queue ────────── independent (fallback only)
        ├── resample_every_epoch ─ independent
        ├── policy_buffer_max ── independent
        ├── teacher_mix_fraction ─ independent
        ├── policy_mix_fraction ── independent
        ├── hard_state_fraction ── independent
        └── adversarial_fraction ─ independent
```

### Env YAML → QGym flow:

```
Env YAML ──► QGymEnvConfig.from_yaml() ──► dataclasses.asdict()
  │                                              │
  │  env_type, num_pool,                          │
  │  server_pool_size ────► stripped              │
  │                                              ▼
  │  network, mu, queue_event_options,       raw_dict
  │  lam_params.val                                  │
  │     │                                            │
  │     ▼                                            ▼
  │  If null: loaded from .npy files       load_rl_p_env(raw_dict)
  │     │                                            │
  │     └── env_type/name ───► data_dir             ▼
  │                            └── *_network.npy  DiffDiscreteEventSystem
  │                            └── *_mu.npy         (s,q,t) → (Q', cost)
  │                            └── *_lam.npy
  │                            └── *_delta.npy
```

---

## Section 4: Minimal Multi-Dataset Template

```yaml
# configs/dataset/qgym/registry/datasets.yaml

defaults:
  output_dir: final_dataset/qgym

  collection:
    n_steps: 500000
    n_valid: 10000
    n_test: 10000
    shard_size: 50000
    policy: mixed
    policy_weights:
      random: 0.25
      sed: 0.25
      qmd: 0.25
      softmax: 0.25

  training_defaults:
    mode: static
    batch_size: 64
    n_samples: 512
    max_queue: 15
    resample_every_epoch: true
    policy_buffer_max: 4096
    teacher_mix_fraction: 0.25
    policy_mix_fraction: 0.25
    hard_state_fraction: 0.75
    adversarial_fraction: 0.25

datasets:

  my_topology: &my_topo
    env: extern/QGym/configs/env/my_topology.yaml
    env_N: 6
    env_mu_fixed: [0.5, 0.3, 0.2]
    env_lam: 0.15

  my_topology_debug:
    <<: *my_topo
    collection: {n_steps: 10000, n_valid: 1000, n_test: 1000, shard_size: 5000}
    training_defaults: {n_samples: 256}
```

---

## Section 5: Removed Legacy Parameters

The following parameters existed in the old per-file YAML format and have been **removed** from the new single-file design:

| Parameter | Reason | Notes |
|-----------|--------|-------|
| `batch_size_env` | Always 1; never overridden in any dataset | Hardcoded in `CollectionConfig(batch_size_env=1)` |

The derivation fields `env_N`, `env_mu_fixed`, `env_lam` are **retained** as optional topology-anchor convenience fields — they serve as fast-path caches consumed by `train/runner.py`, `eval/audit.py`, `eval/baselines.py`, and `experiments/runner/engine.py`. When omitted, those consumers compute the values from the env config at runtime.
