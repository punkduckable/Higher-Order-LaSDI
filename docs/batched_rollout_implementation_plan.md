# Padded Batched Rollout Implementation Plan

## Goal

Replace the per-window Python/RK4 execution path with padded, batched simulation, beginning with the profiled second-order SOME rollout loss. Retain the existing scalar RK4 solvers as readable CPU/NumPy references while developing a tensor-only batched implementation and, if profiling warrants it, a specialized GPU implementation. The objective is to reduce CPU/autograd dispatch overhead and tiny CUDA launches without changing RK4, time normalization, gradients, or loss-reduction semantics. Extend the batched contract to first-order rollout models after validating the second-order path; consider `torch.compile` only after the data flow is GPU-friendly.

## Motivation and baseline

The profile from

```text
results/Telegraphers_11_02_17_40_42_1600329/
HLaSDI_train.1793670398081175596.pt.trace.json
```

indicates a launch-bound workload rather than a compute- or memory-bound workload:

- Backward takes about 5.103 s (55.6% of the profiled epoch).
- Rollout simulation takes about 2.912 s (31.7%).
- Together these account for about 87.3% of the profiled epoch.
- The trace contains 414,519 CUDA kernel events and 381,559 `cudaLaunchKernel` calls.
- Summed kernel execution is only about 603 ms, with an average kernel duration of about 1.46 microseconds.
- About 98.8% of kernels use `grid=[1,1,1]`.
- There are 96,732 `aten::mm` calls; about 99.24% are tiny rollout-related matrix multiplications:
  - 64,000 calls with `[1, 20] @ [20, 20]`
  - 32,000 calls with `[20, 1] @ [1, 20]`
- Peak allocation is about 368 MB on an 80 GB H100, so memory capacity is not the limiting resource.

The current rollout path serializes independent trajectories in Python and unrolls RK4 separately for each trajectory. Batching changes the dominant expert operations from many `[1, n_z]` matrix multiplications into fewer `[B, n_z]` operations and builds one larger autograd graph per time step instead of many nearly identical scalar-batch graphs. The 603 ms figure is summed kernel execution, not an unprofiled wall-time lower bound; profiler overhead and host-side autograd work also matter.

The profiled SOME configuration already has a time/parameter-only gate. Its scalar torch path already evaluates gate weights and effective affine coefficients at RK stage times before integrating *each window*. Removing `use_z_in_gate` does not by itself accelerate that workload. The additional opportunity is to form those coefficients in larger batches, reuse time/parameter evaluations across windows where appropriate, and remove the float-keyed RHS closure from the future GPU hot path.

## Assumptions

- Different parameter values may use different time-step sizes and rollout durations.
- Sampled rollouts will usually have similar frame counts, so padding overhead should be moderate.
- Time grids may be nonuniform and must be represented per rollout.
- SOME and SOME_weak gates depend only on reference-normalized time and parameters. CABLE's separate `use_z_in_gate` option remains unchanged.
- Stage-time gate weights and effective coefficients retain autograd connections to the current gate and expert parameters. They must be recomputed after each optimizer update, not cached across training steps.
- The existing scalar first- and second-order RK4 solvers and `simulate()` paths remain reference behavior for CPU/NumPy inference and equivalence tests.

## Scope

### Included

1. Remove `use_z_in_gate` from SOME/SOME_weak schemas and implementation; retain the existing differentiable stage-time precomputation for tensor rollouts.
2. Define a shared batch/time-grid contract for first- and second-order RK4; implement and validate the second-order tensor path first.
3. Batch SOME stage-time gate weights and effective affine coefficients for selected rollout windows while preserving gradients.
4. Add a `simulate_batched()` interface and integrate sampled second-order rollout losses in both strong and weak trainers.
5. Validate trajectories, gradients, loss weighting, and unprofiled CPU/GPU performance against the scalar reference.
6. Extend the same contract to first-order solvers/models in a separately reviewable phase.

### Initially excluded

- Batching the initial-condition rollout loss.
- Replacing RK4 with a different integrator.
- CUDA graphs, `torch.compile`, or custom/fused CUDA kernels in the initial patches.
- Removing CABLE's independent state-dependent gate option.
- Replacing or deleting the existing scalar/reference RK4 and simulation APIs.

These can be considered after the padded batched path is correct and profiled.

## Proposed tensor contract

Let `B` be the number of sampled rollout windows in the epoch batch, `T_max` the longest window, `n_z` the latent dimension, and `n_p` the parameter dimension.

The first-order solver uses one state tensor `[B, n_z]`; the second-order solver uses displacement and velocity tensors of that shape. Both use the same `[B, T_max]` grid and `[B]` lengths. A generic batched RHS must evaluate all `B` states at once. The SOME-specific fast path may instead receive precomputed stage coefficients.

| Value | Shape | Meaning |
|---|---:|---|
| `initial_displacement` | `[B, n_z]` | Latent displacement at the first valid frame |
| `initial_velocity` | `[B, n_z]` | Latent velocity at the first valid frame |
| `t_grid` | `[B, T_max]` | Per-rollout padded time grids |
| `lengths` | `[B]` | Number of valid frames in each rollout |
| `params` | `[B, n_p]` | Parameter row associated with each rollout |
| `reference_t0` | `[B]` | Beginning of each trajectory's full/reference time domain |
| `reference_t_span` | `[B]` | Span of each trajectory's full/reference time domain |
| displacement output | `[B, T_max, n_z]` | Predicted displacement, frozen after each valid length |
| velocity output | `[B, T_max, n_z]` | Predicted velocity, frozen after each valid length |
| valid-frame mask | `[B, T_max]` | `frame_index < lengths[:, None]` |
| parameter index | `[B]` | Maps each rollout back to its training parameter |

For one second-order interval, RK4 evaluates the affine RHS at `t_n`, `t_n + h_n/2` (twice, at two different intermediate states), and `t_n + h_n`. Thus the time-only gate needs **three distinct stage-time coefficient sets per interval**. For each selected interval, the precomputed effective matrices and bias are `K_start/K_mid/K_end`, `C_start/C_mid/C_end`, and `b_start/b_mid/b_end`; the two midpoint RHS calls share coefficients but not state. The initial batched solver is affine-specific and consumes these coefficient tensors directly, rather than accepting an arbitrary RHS callable.

Time grids are padded by repeating the final valid timestamp for each rollout. Targets are padded to `T_max`, but padded target values are never included in a loss.

## Detailed implementation

### 1. Remove the SOME/SOME_weak state-dependent gate option

**Files:** `src/HLaSDI/Schemas.py`, `src/HLaSDI/LatentDynamics/SOME.py`, SOME example YAML files, and focused tests.

- Remove `use_z_in_gate` from `SOMELatentDynamicsSettings` (shared by SOME and SOME_weak) and from the SOME implementation. Do not change CABLE.
- Make the gate input unconditionally `[reference_normalized_time, parameters]`. Remove latent-state dummy arguments from the gate-weight helper.
- Keep the existing torch scalar-rollout precomputation of gate weights and effective `K`, `C`, and `b`; make this path unconditional. This is simplification and a prerequisite for batched precomputation, not an expected standalone speedup on the profiled configuration.
- Remove the obsolete key from SOME example YAML files. The schema forbids unknown keys, so existing SOME YAML/checkpoint configs containing even `use_z_in_gate: false` require that key to be removed before validation. Checkpoints with a state-dependent gate have an incompatible first gate-layer shape and cannot be loaded as the new model without an explicit conversion strategy.
- Test schema rejection of the removed SOME key, CABLE's unchanged support, and time-dependent SOME forward/gradient behavior.

### 2. Add a batched second-order RK4 solver

**File:** `src/HLaSDI/Utilities/SecondOrderSolvers.py`

Add a tensor-only affine-specific `RK4_affine_batched`. Keep the existing `RK4` unchanged as the independent, readable scalar/NumPy reference and leave CPU training on that path for now. The new tensor solver can run on CPU for equivalence tests, but is designed for batched CUDA execution. Add a GPU-only fused implementation only when measured performance justifies its complexity.

Coefficient-aware contract:

```python
def RK4_affine_batched(
    y0: torch.Tensor,
    Dy0: torch.Tensor,
    t_grid: torch.Tensor,
    lengths: torch.Tensor,
    K: torch.Tensor,  # [B, T-1, 3, n_z, n_z]
    C: torch.Tensor,  # [B, T-1, 3, n_z, n_z]
    b: torch.Tensor,  # [B, T-1, 3, n_z]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate a padded batch of affine second-order systems with classical RK4."""
```

Requirements:

- Validate that:
  - `y0` and `Dy0` have shape `[B, n_z]` and matching dtype/device.
  - `t_grid` has shape `[B, T_max]`.
  - `lengths` has shape `[B]`, is integral, and satisfies `1 <= lengths <= T_max`.
  - Valid portions of each time grid are strictly increasing. Expensive validation may be limited to tests/debug assertions if it would synchronize the GPU in normal training.
- Use one Python loop over `range(T_max - 1)`, not a Python loop over `B`.
- At step `n`, use vectors:

  ```python
  t_n = t_grid[:, n]                         # [B]
  h_n = t_grid[:, n + 1] - t_grid[:, n]     # [B]
  h = h_n[:, None]                           # [B, 1]
  active = (n + 1 < lengths)[:, None]        # [B, 1]
  ```

- Evaluate all four RK stages on `[B, n_z]` state tensors. Stages 2 and 3 reuse the midpoint `K`, `C`, and `b` but must compute separate accelerations from their different intermediate states.
- Compute the candidate next displacement and velocity using the same second-order RK4 equations as the scalar implementation.
- Freeze completed trajectories without host synchronization. Mask all interval step sizes once before the loop so inactive intervals have `h_n == 0`; with finite coefficients, the RK4 update is then an exact no-op. An additional per-step state mask is unnecessary:

  ```python
  active = torch.arange(T_max - 1, device=t_grid.device)[None, :] < lengths[:, None] - 1
  h_all = torch.where(active, t_grid[:, 1:] - t_grid[:, :-1], 0)
  ```

- Do not use `if active.any():`, `.item()`, or another data-dependent CPU/GPU synchronization in the time loop.
- Stack the per-step tensors at the end and transpose/stack into `[B, T_max, n_z]`.
- Preserve autograd through every valid stage and the precomputed coefficient tensors.

Keep the completed-state invariant in tests. Avoid synchronizing validation (`.item()`, `active.any()`) inside the time loop.

### 3. Batch SOME gate and effective-coefficient precomputation

**File:** `src/HLaSDI/LatentDynamics/SOME.py`

Keep the existing scalar path as the oracle. For the new path, evaluate gate weights at all selected windows' RK stage times in one batched tensor call. From those weights, form effective `K`, `C`, and optional `b` for each stage time. Pass those tensors directly to `RK4_affine_batched`; do not perform a gate-network call at every RK stage in the time loop.

The batched evaluation must accept:

- `t`: scalar, `[N]`, or `[B, T_max - 1, 3]` stage times (flattened to `[N]` for a gate-network call) as appropriate for the caller.
- parameters:
  - a shared row `[n_p]`, expanded across samples, or
  - one row per sample `[N, n_p]` / `[B, n_p]`.
- reference-domain values `t0` and `t_span` as scalars or one value per sample.

For a batched RK stage, time normalization is elementwise:

```python
tau = (t - reference_t0) / reference_t_span       # [B] per stage, then flatten for the gate
```

The gate input is always:

```text
[tau[:, None], params]
```

Important invariants:

- Normalize using the full trajectory's reference domain, not the sampled/cropped rollout window. Otherwise the same physical time receives different gate inputs depending on the sampled window.
- Preserve the exact RK4 stage-time arithmetic on nonuniform grids. Stages 2 and 3 share midpoint coefficients, not intermediate states.
- Do not detach, convert through NumPy, or copy tensors to CPU in the batched training path. Recompute differentiable coefficients after each optimizer update; never reuse a stale autograd graph across epochs.
- Keep expert coefficients shared across the batch while gate weights vary by rollout and RK stage.
- Use batched operations (`einsum`, `matmul`, or the repository's clearest existing pattern) so effective `K`, `C`, and optional `b` contributions are computed without a loop over rollout rows.
- Continue to apply coefficient masks and optional biases exactly as in the scalar RHS.

The general form is:

```text
acceleration[b] = sum_m alpha[b, m] * (K[m] @ z[b] + b[m])
                + sum_m beta[b, m]  * (C[m] @ velocity[b])
```

Choose the contraction form based first on clarity and exact gradient behavior. Use profiling, rather than intuition, to decide whether precomputed effective matrices or directly weighted expert outputs are faster on the target H100 workload. Reuse repeated stage-time/parameter evaluations across overlapping windows only if doing so preserves exact time values and gradients.

### 4. Add an explicit batched simulation interface

**File:** `src/HLaSDI/LatentDynamics/SOME.py`

Add a batched-simulation capability to the latent-dynamics interface rather than coupling a trainer to `isinstance(latent_dynamics, SOME)`.

Suggested SOME API:

```python
def simulate_batched(
    self,
    initial_displacement: torch.Tensor,
    initial_velocity: torch.Tensor,
    t_grid: torch.Tensor,
    lengths: torch.Tensor,
    params: torch.Tensor,
    reference_t0: torch.Tensor,
    reference_t_span: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate a padded batch of second-order latent trajectories."""
```

Design notes:

- Keep this API tensor-only. NumPy/list adaptation remains the responsibility of the existing scalar `simulate()` API.
- Return two tensors with shape `[B, T_max, n_z]`.
- Form stage-time coefficients from `params`, `reference_t0`, and `reference_t_span` before integration; the batched RK4 core consumes them through its explicit affine interface. A general batched-RHS solver for other models is not part of this phase.
- `SOME_weak` should inherit this implementation unless it truly changes rollout dynamics.
- Add the base-interface capability/failure contract when integrating the trainer, so unsupported latent-dynamics models fail clearly without a concrete-type check.
- Do not rewrite scalar `simulate()` in the first patch. Keeping independent scalar and batched paths makes equivalence testing meaningful and limits the regression surface.

### 5. Pack all sampled rollout windows before simulation

**File:** `src/HLaSDI/Trainer/Second_Order_Rollout.py`

Replace the nested “sample one window, simulate it, decode it, accumulate its loss” path for the sampled rollout loss with these stages:

1. Sample rollout windows using the existing curriculum and selection logic.
2. Store metadata for every sampled window:
   - training-parameter index,
   - start and stop frame indices,
   - length,
   - per-window time grid,
   - initial latent displacement/velocity,
   - latent targets,
   - full-order targets,
   - parameter row,
   - full/reference-domain `t0` and `t_span`.
3. Compute `T_max` and allocate/construct padded tensors.
4. Repeat each final valid timestamp through the padded portion of its time grid.
5. Stack initial states, parameter rows, lengths, and reference-domain values.
6. Call `latent_dynamics.simulate_batched(...)` once.
7. Decode predicted states in one encoder-decoder call. Benchmark a padded flatten against gathering only valid frames first; the latter avoids running a potentially expensive full-order decoder on padding:

   ```python
   flat_D = predicted_D.reshape(B * T_max, n_z)
   flat_V = predicted_V.reshape(B * T_max, n_z)
   decoded_D, decoded_V = encoder_decoder.Decode(flat_D, flat_V)
   decoded_D = decoded_D.reshape(B, T_max, *frame_shape)
   decoded_V = decoded_V.reshape(B, T_max, *frame_shape)
   ```

   This snippet shows the simple padded variant. Adapt the exact call and output handling to the existing `EncoderDecoder` interface, and retain the faster measured choice.
8. Compute masked latent-space and full-order losses.

Keep window selection behavior unchanged in this first optimization. This isolates batching from curriculum/random-sampling changes and makes fixed-seed scalar-vs-batched comparison possible.

If directly padding high-dimensional full-order targets creates excessive copies, first attempt to stack slices into preallocated `[B, T_max, ...]` tensors. Peak memory from the profile leaves substantial headroom, but target packing should still be measured rather than assumed free.

### 6. Preserve the current loss reduction exactly

The current sampled rollout loss has three reduction levels:

1. Mean over valid frames and feature/spatial dimensions for each rollout.
2. Mean over sampled rollouts belonging to each training parameter.
3. Sum over training parameters.

A single global masked mean is not equivalent when rollout lengths or per-parameter rollout counts differ. Preserve the existing behavior for each of the four rollout terms:

- latent/ROM displacement,
- latent/ROM velocity,
- decoded/FOM displacement,
- decoded/FOM velocity.

For each term:

```python
# element_error has shape [B, T_max, ...]
# valid has shape [B, T_max, 1, ...] after adding feature dimensions.
masked_sum = (element_error * valid).sum(dim=all_non_batch_dims)
valid_count = valid.expand_as(element_error).sum(dim=all_non_batch_dims)
per_rollout = masked_sum / valid_count

per_parameter_sum = torch.zeros(n_train, ...).index_add(
    0, parameter_index, per_rollout
)
rollouts_per_parameter = torch.zeros(n_train, ...).index_add(
    0, parameter_index, torch.ones_like(per_rollout)
)
loss = (per_parameter_sum / rollouts_per_parameter).sum()
```

Use absolute error for MAE and squared error for MSE. Construct counts in a dtype/device-compatible way and assert through packing logic that every included training parameter has at least one sampled rollout before division.

### 7. Apply the same sampled-rollout path to the weak trainer

**File:** `src/HLaSDI/Trainer/Second_Order_Weak.py`

Determine whether the weak trainer inherits the relevant rollout implementation directly or duplicates part of the epoch loop. Reuse a small, explicit packing/loss helper if both trainers need identical logic; do not duplicate the padded simulation implementation.

The helper boundary should be based on a coherent operation, for example:

```text
collect sampled windows -> padded rollout batch
batched predictions + padded targets -> semantics-preserving rollout losses
```

Avoid a broad trainer refactor. Only extract code required to share the new behavior between the two second-order trainers.

## Instrumentation

Add profiler regions around the new phases so a follow-up trace distinguishes packing overhead from GPU work:

```text
HLaSDI/rollout_loss/pack
HLaSDI/rollout_loss/simulate_batched
HLaSDI/rollout_loss/decode_batched
HLaSDI/rollout_loss/masked_loss
```

Log/cache at least:

- rollout batch size `B`,
- minimum, mean, and maximum valid frame count,
- `T_max`,
- padding efficiency:

  ```text
  sum(lengths) / (B * T_max)
  ```

These metrics should be detached and should not force synchronization inside the RK loop. Logging once per epoch or at the repository's existing metric cadence is sufficient.

## Test plan

### A. Batched RK4 numerical equivalence

**File:** `Test/test_latent_dynamics_rhs.py` or a focused solver test file if that better matches repository organization.

Compare each valid batched trajectory against an independent call to scalar `RK4` for:

- equal-length uniform grids,
- unequal lengths,
- different step sizes,
- nonuniform grids,
- batch size one,
- a trajectory with only one frame.

Verify:

- displacement and velocity agree within explicit dtype-appropriate tolerances,
- padded output frames exactly repeat/freeze the last valid state,
- output shape, dtype, and device are correct.

Use deterministic analytic or linear RHS functions so failures identify the integrator rather than the gate network.

### B. SOME scalar-vs-batched forward equivalence

For fixed model weights and rollout windows, compare batched valid frames against the existing scalar `SOME.simulate()` path for:

- time/parameter-varying gate weights,
- biases enabled and disabled,
- masking enabled where practical,
- heterogeneous parameter rows and reference time domains.

Include a cropped window whose first timestamp is not the beginning of the reference domain. This catches accidental normalization by the rollout window.

### C. Gradient equivalence

Compute the same scalar objective from scalar and batched simulations, backpropagate separately, and compare gradients for:

- initial displacement,
- initial velocity,
- `K`,
- `C`,
- `b` when enabled,
- all gate-network parameters.

Verify gradients through precomputed gate weights and effective coefficients. In particular, neither precomputation nor reuse across overlapping windows may detach the gate network or expert tensors.

Use float64 for the strictest small deterministic tests when supported, and document `rtol`/`atol`. Also retain a float32 smoke case representative of training.

### D. Loss-reduction invariance

**File:** `Test/test_second_order_training_invariants.py`

Construct synthetic padded predictions/targets with:

- unequal rollout lengths,
- unequal numbers of rollouts per training parameter,
- deliberately large padded target values that must not affect the result.

Compare the batched masked result to an explicit nested-loop reference for both MAE and MSE and all four rollout terms. This test must fail if the implementation accidentally uses a global masked mean.

### E. Trainer smoke tests

Run short deterministic steps for:

- `Second_Order_Rollout`,
- `Second_Order_Weak`.

Verify that:

- forward and backward complete,
- relevant model parameters receive finite gradients,
- losses are finite,
- padding does not change the result relative to the scalar reference on a tiny fixture,
- profiler-disabled training remains valid.

## Validation and acceptance criteria

### Correctness gate

Do not evaluate speedup until all of the following hold:

- scalar and batched valid trajectories agree within declared tolerances,
- gradients agree within declared tolerances,
- full-domain gate normalization is covered by a regression test,
- padded frames do not contribute to any loss,
- per-rollout/per-parameter loss reduction matches existing behavior,
- existing unit tests remain green.

### Performance comparison

Profile the same short training configuration before and after the change, with the same seed, curriculum stage, number of sampled rollouts, and profiler schedule. Also measure unprofiled mature-epoch wall time because `with_stack=True`, shape recording, memory profiling, and FLOP estimation add substantial profiler overhead.

Initial targets:

- at least 5x fewer CUDA kernel launches in the sampled rollout path,
- at least 5x fewer tiny matrix calls,
- at least 2x lower unprofiled mature-epoch wall time,
- a practical target of 3--6x speedup if sampled rollout/backward remains dominant,
- padding efficiency preferably above 80% for the intended workloads.

The exact end-to-end speedup will be lower than the reduction in rollout launch count because the encoder/decoder, non-rollout losses, optimizer, and unavoidable time-step dependency remain serial components.

## Risks and mitigations

### Padding becomes inefficient

**Risk:** A small number of long windows inflate `T_max`, causing excessive inactive work.

**Mitigation:** Record padding efficiency first. If it is consistently low, bucket rollouts by length and make one batched call per bucket. Do not add bucketing before measurements show it is necessary.

### Batched path changes time normalization

**Risk:** Normalizing by each cropped rollout window changes gate behavior.

**Mitigation:** Pass `reference_t0` and `reference_t_span` explicitly and add the cropped-window regression test.

### Stage coefficients are stale or misaligned

**Risk:** Reusing coefficients across optimizer steps, or mixing up start/midpoint/end coefficients, silently changes the numerical model or breaks gradients.

**Mitigation:** Compute coefficients anew in each training forward pass, preserve exact stage-time arithmetic, and require scalar-vs-batched trajectory and gradient equivalence on nonuniform/cropped windows.

### Loss weighting silently changes

**Risk:** A global masked mean reweights long rollouts or parameters with more sampled windows.

**Mitigation:** Retain the three-level reduction and test it against an explicit loop reference with unequal lengths/counts.

### Packing or decoding becomes the new bottleneck

**Risk:** Python packing and padded full-order tensors consume the simulation savings.

**Mitigation:** Add separate profiler ranges. Flatten and decode once. Optimize packing only if the new profile identifies it as material.

### Autograd memory increases

**Risk:** Larger batched graphs increase peak live memory even though launch overhead falls.

**Mitigation:** Measure peak memory in the follow-up profile. Current usage is far below H100 capacity. If necessary, split into a small number of length buckets or microbatches while retaining vectorization within each group.

## Implementation sequence

### Patch 1: Remove the SOME state-dependent gate option

Expected files:

```text
src/HLaSDI/Schemas.py
src/HLaSDI/LatentDynamics/SOME.py
examples/KleinGordon.yml
examples/Telegraphers.yml
examples/Telegraphers_Profile.yml
examples/WaveEquation.yml
Test/test_latent_dynamics_rhs.py
Test/test_config_schemas.py
```

Work:

1. Remove `use_z_in_gate` from the shared SOME settings and implementation, but leave CABLE unchanged.
2. Make time/parameter-only gate evaluation and existing scalar-torch stage precomputation unconditional.
3. Remove the obsolete field from SOME example YAML files and document the saved-config migration.
4. Verify gate/RHS/simulation behavior and gradients for SOME and SOME_weak.

This patch does **not** claim a GPU speedup: the profiled run already uses the precomputed scalar-torch path.

### Patch 2: Batched numerical foundation

Expected files:

```text
src/HLaSDI/Utilities/SecondOrderSolvers.py
src/HLaSDI/LatentDynamics/SOME.py
Test/test_affine_batched_rk4.py
```

Work:

1. Define the batch contract and add the second-order tensor `RK4_affine_batched` without replacing scalar `RK4`.
2. Add batched SOME stage-time gate/effective-coefficient precomputation and `simulate_batched()`.
3. Add solver, scalar-vs-batched, time-normalization, and gradient tests on CPU.

Stop after this patch and review numerical/gradient equivalence before changing trainer behavior.

### Patch 3: Second-order trainer integration and profiling

Expected files:

```text
src/HLaSDI/Trainer/Second_Order_Rollout.py
src/HLaSDI/Trainer/Second_Order_Weak.py
Test/test_second_order_training_invariants.py
```

Work:

1. Gather and pad all sampled rollout windows.
2. Run one batched simulation and one flattened batched decode.
3. Add masked loss aggregation that preserves current semantics.
4. Reuse the path from the weak trainer.
5. Add profiler ranges and batch/padding metrics.
6. Add loss-invariance and trainer smoke tests.
7. Produce an equivalent before/after profile and unprofiled timing comparison.

### Patch 4: First-order extension and backend optimization

Add a first-order batched RK4 implementation and migrate first-order rollout models/trainers using the same batch contract, while retaining the scalar reference. First-order CABLE still supports a state-dependent gate, so its batched RHS must evaluate the gate on each intermediate RK stage state; it cannot use SOME's time-only coefficient precomputation. Benchmark CPU scalar versus batched execution as well as GPU execution; batching may reduce CPU dispatch but padding may lose for small/heterogeneous workloads. Select a GPU-only fused RK4 implementation by configured device only if profiling demonstrates a worthwhile speedup and it passes the same trajectory/gradient tests. Keep a pure tensor batched solver usable on both devices. Any less-transparent GPU-specialized kernel/step must have detailed, human-readable comments explaining its state layout, each RK stage, synchronization assumptions, gradient path, and padding behavior.

## Follow-up optimizations (only after re-profiling)

1. **Length bucketing:** Use if padding efficiency is poor.
2. **More coefficient reuse:** Evaluate shared stage times/parameters only once across overlapping windows when this is measurably useful and numerically equivalent.
3. **`torch.compile`:** Consider compiling a fixed-shape batched RK4 step after graph breaks and shape stability are understood; keep CPU eager or benchmark a separate CPU-compiled path.
4. **IC rollout batching:** Apply the same padded interface if the new profile shows IC rollouts are material.
5. **Expert contraction tuning:** Benchmark `einsum`, effective-matrix construction, and direct expert-output weighting on representative `B`, `n_z`, and expert counts.

The immediate optimization should remain focused: remove the obsolete SOME gate option, then remove the rollout-level Python loop and expose useful batch dimensions to PyTorch/CUDA without changing the RK4 method or loss semantics.
