# Number transformation with a fixed step budget

The `numbers` environment presents a start and a target integer, such as
`63 -> 4`. Each move selects one of `+,-,multiply,divide_exact` and an operand
from 1 through 9. There are **36 policy actions**, with no STOP action.

Episodes last exactly `max_steps` moves. Reaching the target early gives no
reward and does not terminate. Only the final value is checked: reward is 1
if it equals the target at the deadline, otherwise 0. `multiply 1` can preserve
an early solution; leaving and returning is also allowed. For example, with
budget four: `63 /9 -> 7 -3 -> 4 *1 -> 4 *1 -> 4` succeeds. Finishing at 5 fails
even if 4 was visited earlier.

## Arithmetic and observations

The domain is a bounded interval of integers (default 0 through 999). Division
requires exact divisibility. Division is never floating-point or floor division.
An out-of-range result or invalid division leaves the current value unchanged
and consumes a move, with a visible invalid-action marker. No action masking,
step penalty, intermediate reward, or extra KL rule is added.

Action encoding is `operation*9 + digit-1`, with operation order add, subtract,
multiply, divide. Thus action 18 is multiply-by-one, and 35 is divide-by-nine.

Observations use **53 token IDs**: decimal digits, sign and separator markers,
36 action IDs, and an invalid-action marker. Numbers are sign plus fixed-width
zero-padded decimal digits, not separate embeddings for each integer. The prompt
contains start, target and budget; each history record contains the action,
result and remaining budget. There is no pair ID, shortest distance, solution,
reference value, or correctness flag in model inputs. Environment arithmetic is
exact; the policy learns arithmetic planning rather than generating intermediate
arithmetic results itself.

## Search and large problem pools

At environment construction, CPU graph search computes shortest distances for
every bounded integer pair. Search uses the actual 36 actions and the same
bounds/division rules as rollouts. This is an optimal-distance classification,
**not the value function of the current Transformer policy**.

The default distance range is 2 through 5, giving:

| Shortest moves | Training pairs | Held-out pairs |
|---:|---:|---:|
| 2 | 102,317 | 25,397 |
| 3 | 339,839 | 85,067 |
| 4 | 325,174 | 81,160 |
| 5 | 14,865 | 3,657 |
| Total | 782,195 | 195,281 |

Only twelve pairs in this domain require six moves, so distance six is excluded
by default. Search and selection are deterministic and cached in memory across
matching environment instances. Reset samples a populated distance bucket
uniformly, then a pair uniformly within that bucket. New pairs continue to be
sampled throughout training; `--num_problems` does not limit the pool.

A stable hash partitions start/target pairs, independent of seed, into roughly
80% training and 20% test. The default `number_split_mode=pair` holds out initial
prompts, not numbers or every intermediate state: a training trajectory may visit
a current/target combination that is a held-out initial pair. Use
`number_split_mode=target` to hold entire target numbers out of training. This
is stronger goal generalization, but still uses the same numerical domain.
Larger-number and longer-horizon extrapolation are separate experiments, not
implied by success on either split.

The search solution policy is used only for optional BC warmup and the fixed-policy
critic benchmark. PPO trunks and branches use the learned actor. By default the
reference policy mixes shortest-path actions with uniform exploration (teacher
weight .75); at the target its teacher action is multiply-by-one.

## Cluster comparison

```sh
git pull origin main
DRY_RUN=1 bash scripts/submit_number_comparison.sh results/numbers_preview
bash scripts/submit_number_comparison.sh results/numbers_compare
```

This submits MC, TD(.8), FBI(.8,K=2), Vine-target K=2/K=4 and E_min. Actor settings
are shared: LR=3e-5, clip=.2, entropy=0, four epochs, GAE(1), no KL penalty or
stopping rule. Default horizon is eight, with 100 PPO updates, 100 BC warmup
updates, five seeds and 64 rollout environments. No performance claim is made
before cluster calibration.

Useful overrides:

| Environment variable | Default |
|---|---:|
| `NUMBER_MIN_VALUE` | 0 |
| `NUMBER_MAX_VALUE` | 999 |
| `NUMBER_MIN_DISTANCE` | 2 |
| `NUMBER_MAX_DISTANCE` | 5 |
| `NUMBER_TEACHER_PROBABILITY` | .75 |
| `NUMBER_SPLIT_MODE` | pair |
| `MAX_STEPS` | 8 in the number launcher |

Constraints: `1 <= min_distance <= max_distance <= max_steps <=99`, at most 4096
integer states, and bounds within -9999 through 9999. Search classification changes
if arithmetic bounds change. Some requested hard buckets may be empty; no pairs
are invented to fill them. Domain size controls the quadratic search-table cost.

For example:

```sh
NUMBER_SPLIT_MODE=target bash scripts/submit_number_comparison.sh results/numbers_new_targets
NUMBER_MIN_DISTANCE=4 NUMBER_MAX_DISTANCE=5 MAX_STEPS=6 \
  bash scripts/submit_number_comparison.sh results/numbers_harder
```

Outputs include config and environment metadata, classified train/test pools as
compressed NPZ files, per-seed metrics, sampled initial/final training and held-out
success, and success/counts by original shortest distance. `invalid_action_fraction`
measures wasted moves separately from terminal success.

```sh
python3 plot_ppo_comparison.py --results_dir results/numbers_compare
```

Alongside the standard main-algorithm plots and AUC/final-return table, this writes
`heldout_evaluation` and `success_by_distance` figures and a per-distance CSV.
Means/SEM are across seeds; the CSV includes evaluation episode counts per bucket.

Existing actor sweeps and critic-only/K sweeps also accept `numbers`:

```sh
MAX_STEPS=8 bash scripts/submit_vine_actor_sweep.sh numbers results/numbers_actor_sweep
MAX_STEPS=8 bash scripts/submit_benchmark.sh numbers 100 5 results/numbers_value
MAX_STEPS=8 bash scripts/submit_k_sweep.sh numbers 100 5 0.8 results/numbers_k
```

Critic evaluation uses a capped, distance-balanced subset of pairs and independent
MC continuations under the frozen reference policy, with uncertainty recorded.
It does not enumerate neural-network predictions for the entire pair pool or use
optimal search distances as critic training labels.
