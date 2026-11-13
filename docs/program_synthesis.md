# Restricted arithmetic/polynomial program synthesis

Select `--env_type program` in `train_dag_ppo.py` or `run_benchmark.py`. This is a
new functional environment inspired by the compute-program formulation of
[Palermo, Ye and Singh](https://arxiv.org/abs/2107.07373), not a port or reproduction
of their environment. It has no SymPy, subprocess, or symbolic solver in rollouts.
No dataset download or training starts automatically.

## Task and observations

A problem gives an arithmetic or univariate polynomial expression to evaluate or
expand. The initial workspace contains its distinct integer constants and `x`,
represented as coefficient vectors in ascending powers of x. For example,
`(x+2)*(x-3)` has exact target coefficients `[-6,-1,1,0]` at degree bound 3.

The actor constructs a program using available registers. Every successful
operation appends a reusable register: `(x+2)*(x+2)` can compute `x+2` once and
reuse that intermediate twice. The prompt encodes the expression and initial
registers; the history encodes selected operations and their public results.
There is no problem-ID token, target coefficient vector, reference program,
correctness label, or privileged value in observations.

Each binary action is `(operation, left_register, right_register)`, encoded as
`operation*R*R + left_register*R + right_register`. Operators are addition,
subtraction, multiplication, and exact division by a nonzero constant polynomial.
Division must divide every coefficient exactly. Arithmetic is integer-only;
fractions and multiple variables are outside this version's supported domain.

There are also `SUBMIT(register)`, `UNDO`, and `RESET` actions. Undo removes the
last computed register, which cannot have surviving descendants; reset restores
initial registers. Both consume time and remain visible in history. Domain
failures, unavailable-register references, workspace exhaustion, or out-of-bound
results are visible invalid operations that consume time without changing the
workspace. No validity/action mask is supplied to PPO.

Only submitting a register whose exact value equals the answer earns reward 1.
Any other submission terminates with reward 0. Horizon exhaustion without a
correct submission earns 0, even if an unsubmitted correct answer is available.
Earlier irrelevant or abandoned work does not invalidate a correct submission.
PPO records `invalid_action_fraction` to diagnose wasted steps separately from
terminal success.
Terminal states are absorbing. Time always advances, so even undo/reset preserve
the time-layered DAG property.

This version studies correct tool execution with potentially wrong program
choices. It does not simulate an agent asserting false arithmetic, natural-language
proofs, or general equation solving.

## Bounds and difficulty

Default environment controls:

| CLI flag | Default | Meaning |
|---|---:|---|
| `--program_min_ops` | 2 | Minimum compiled reference-program operations |
| `--program_max_ops` | 4 | Maximum compiled reference-program operations |
| `--program_degree` | 3 | Maximum polynomial degree |
| `--program_max_inputs` | 6 | Maximum distinct initial inputs |
| `--program_registers` | 10 | Total workspace capacity, independent of action horizon |
| `--program_coefficient_limit` | 1024 | Bound on every input/result coefficient |
| `--program_teacher_probability` | .75 | Reference-policy mixture weight, not transition difficulty |
| `--program_problem_file` | none | Optional validated external JSONL |

R registers give `4*R*R + R + 2` actions (412 for R=10). `--num_actions` is derived
for this family. Capacity must cover `program_max_inputs + program_max_ops`, and
the horizon must allow at least `program_max_ops + 1` steps including submission.
Current implementation bounds degree to 6 and coefficients to 4096 to keep all
int32 products and sums exact without overflow.

Reference-program length is a generation constraint, not a proven shortest
solution length: equivalent algebra or incidental numerical identities may yield
shorter valid programs. Immediate-submit tasks are excluded. To isolate depth,
keep register capacity, input bounds, coefficient bounds and horizon fixed while
varying operation counts; capacity changes also change the action space.

## Data provenance and train/test split

Without a problem file, tasks come from a new synthetic expression grammar. They
must not be reported as DeepMind benchmark examples. Task generation is seeded,
canonical expressions are deduplicated, and a stable hash assigns 80% of expression
identities to training and 20% to test. Training and test banks have identical
model shapes and disjoint exact expression identities. This is a held-out-instance
split, not a guarantee of algebraic-equivalence or compositional-distribution
separation.

PPO evaluates both fresh rollouts on its training problem bank and a distinct test
bank. Initial/final test success is recorded as `initial_heldout_success` and
`final_heldout_success`; these are not training rewards. Saved JSONL problem banks,
source tags, bank hashes and environment metadata make each run reproducible.

For an existing DeepMind arithmetic/polynomial alternating Q/A text file:

```sh
python3 scripts/prepare_program_tasks.py \
  --input /path/to/arithmetic__mixed.txt --output data/program_tasks.jsonl \
  --degree 3 --max_ops 4 --max_inputs 6 --coefficient_limit 1024
```

The importer supports expression-only `Calculate`, `Evaluate`, `Expand`, and
`Simplify` question templates. A single variable name is normalized to `x`.
It compiles the expression and independently checks the supplied answer with
exact arithmetic. Unsupported language, rational coefficients, excessive bounds,
duplicates and incorrect supplied answers are rejected in an accompanying report.
Use compatible compilation bounds in the importer and environment. The file must
contain enough compatible examples in each hash partition for the requested
`num_problems`; it never silently fills gaps with synthetic examples.

## Reference policy and evaluations

The fixed reference policy mixes a solution demonstrator with exploration over
publicly available register references. The demonstrator may use the compiled
solution; it is used only for optional behavior-cloning initialization and the
fixed-policy critic benchmark. PPO rollouts and every learned-policy branch use
the frozen actor, not the demonstrator. No teacher correctness label enters the
networks.

The previous small DAG dynamic-programming tables do not apply to this state.
Critic-only evaluation instead samples fixed prefixes and policy actions, then
forms independent MC continuation references under the same fixed reference
policy. Controls are `--program_eval_k` (128), `--program_eval_actions` (4), and
`--program_eval_prefixes` (2 per problem). Reference standard errors and action IDs
are saved. They never enter critic training targets.

Plots explicitly label MSE to an MC reference and sampled-reference advantage
cosine; these are not exact MSVE or exact advantages. Reference MSE includes a
sampling-noise floor, reported in `evaluation_reference.json`. The legacy pickle
key `eval_true_values` contains estimates for this family; always read
`evaluation_reference_is_exact=False` and its accompanying description.

## Cluster commands

Six-method PPO comparison, with the shared tuned actor configuration:

```sh
DRY_RUN=1 bash scripts/submit_program_comparison.sh results/program_preview
bash scripts/submit_program_comparison.sh results/program_compare
```

Defaults: 24-step horizon, 100 PPO updates, 100 BC updates, five seeds, 64 trunk
episodes/update, 64-wide two-layer model. Actor LR=3e-5, clip=.2, entropy=0,
4 epochs, GAE(1); no actor KL penalty or stopping. E_min still uses one independent
probe and its shared unbiased bilinear gradient estimator.

Existing data instead of the synthetic distribution:

```sh
PROGRAM_PROBLEM_FILE=data/program_tasks.jsonl \
  bash scripts/submit_program_comparison.sh results/program_external
```

Depth comparison while holding capacity fixed:

```sh
PROGRAM_MIN_OPS=2 PROGRAM_MAX_OPS=2 PROGRAM_REGISTERS=12 MAX_STEPS=32 \
  bash scripts/submit_program_comparison.sh results/program_depth2
PROGRAM_MIN_OPS=5 PROGRAM_MAX_OPS=6 PROGRAM_REGISTERS=12 MAX_STEPS=32 \
  bash scripts/submit_program_comparison.sh results/program_depth6
```

Plot all algorithms, final training-bank evaluations and separate held-out results:

```sh
python3 plot_ppo_comparison.py --results_dir results/program_compare
```

Pure MC VinePPO K=128 actor sweep and critic-only comparisons also support this
family (activate the cluster GPU environment first):

```sh
bash scripts/submit_vine_actor_sweep.sh program results/program_actor_sweep
MAX_STEPS=24 bash scripts/submit_benchmark.sh program 100 5 results/program_value
MAX_STEPS=24 bash scripts/submit_k_sweep.sh program 100 5 0.8 results/program_k
```

`VINE_BATCH_SIZE` controls bounded PPO branch inference. The policy network head
is larger than the original eight-action DAG; sampling/learning performance has
not been calibrated. Unit tests validate exact arithmetic, reference solutions,
branching, importer checks, split isolation, terminal behavior, and abstract
trainer wiring without executing policy optimization.
