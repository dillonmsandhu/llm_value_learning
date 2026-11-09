# dispatcher for the CLI, redirects to either evaluate.py or sweep.py.
import os
from core.evaluate import evaluate
from core.sweep import tune
import jax

def run_experiment_main(make_train, SAVE_DIR):
    import argparse
    import datetime
    import traceback
    import core.helpers as helpers
    import core.config as config
    
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--run-suffix', type=str, default=run_timestamp)
    parser.add_argument('--n-seeds', type=int, default=0)
    parser.add_argument('--save-checkpoint', action='store_true')
    parser.add_argument('--save-metrics', action='store_true')
    parser.add_argument('--save-video', action='store_true', help='Save a video gif of the trained policy')
    parser.add_argument('--env-ids', nargs='+', default=[])
    parser.add_argument('--sweep', nargs='?', const='default', default=None, help='Path to sweep config JSON or "default" for LR sweep')

    args = parser.parse_args()

    config = config.config
    env_list = [config.get('ENV_NAME')]

    # 3. Environment Priority (CLI takes precedence)
    if args.env_ids:
        env_list = args.env_ids

    for i, env_name in enumerate(env_list):
        if env_name is None: continue
            
        # Create a clean copy for this specific environment run
        run_config = config.copy()
        run_config['ENV_NAME'] = env_name
        
        # Apply command-line JSON overrides if they exist
        if args.config:
            from core.utils import parse_config_override
            run_config.update(parse_config_override(args.config))
            
        if args.n_seeds > 0:
            run_config['N_SEEDS'] = args.n_seeds

        print(f"\n{'='*50}")
        print(f"RUNNING ENV {i+1}/{len(env_list)}: {env_name}")
        print(f"Config: {args.config}")
        # print(f"Network: {run_config.get('NETWORK_TYPE')}")
        print(f"{'='*50}")
        
        rng = jax.random.PRNGKey(run_config.get('SEED', 42))
        run_dir = os.path.join(f"results/{SAVE_DIR}/{args.run_suffix}")
        
        try:
            if args.sweep:
                if args.sweep == 'default':
                    param_grid = {"LR": [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]}
                else:
                    import json
                    if os.path.exists(args.sweep):
                        with open(args.sweep, 'r') as f:
                            param_grid = json.load(f)
                    else:
                        param_grid = json.loads(args.sweep)
                
                # If run_suffix was default timestamp, save directly under results/{SAVE_DIR}/tuning
                save_base = f"results/{SAVE_DIR}/tuning" if args.run_suffix == run_timestamp else f"{run_dir}/tuning"
                tune(
                    make_train,
                    run_config,
                    param_grid,
                    save_dir=save_base,
                    save_checkpoint=args.save_checkpoint,
                    save_metrics=args.save_metrics or True,
                )
            else:
                # Note: make_train and evaluate should be defined in your scope
                evaluate(run_config, make_train, run_dir, args, rng)
        except Exception as e:
            print(f"!!! CRITICAL ERROR running {env_name} !!!")
            traceback.print_exc()
            print("Continuing to next environment...")
