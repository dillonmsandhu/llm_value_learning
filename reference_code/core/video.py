# core/video.py
import os
import argparse
import gym
import gymnasium
import gymnax
import jax
import jax.numpy as jnp
import gymnax.visualize.vis_gym as vis_gym
from gymnax.visualize import Visualizer
import core.helpers as helpers
from core.utils import load_run_data_from_path

# Bypasses the strict old gym version assert inside Gymnax Visualizer
gym.__version__ = "0.19.0"

# Monkeypatch vis_gym to support modern Gymnasium rendering
def patched_init_gym(ax, env, state, params):
    if env.name == "Pendulum-v1":
        gym_env = gymnasium.make("Pendulum-v1", render_mode="rgb_array")
    else:
        gym_env = gymnasium.make(env.name, render_mode="rgb_array")
    gym_env.reset()
    vis_gym.set_gym_params(gym_env, env.name, params)
    gym_state = vis_gym.get_gym_state(state, env.name)
    if env.name == "Pendulum-v1":
        gym_env.unwrapped.last_u = gym_state[-1]
    gym_env.unwrapped.state = gym_state
    rgb_array = gym_env.render()
    ax.set_xticks([])
    ax.set_yticks([])
    gym_env.close()
    return ax.imshow(rgb_array)

def patched_update_gym(im, env, state):
    if env.name == "Pendulum-v1":
        gym_env = gymnasium.make("Pendulum-v1", render_mode="rgb_array")
    else:
        gym_env = gymnasium.make(env.name, render_mode="rgb_array")
    gym_env.reset()  # Reset to satisfy OrderEnforcer
    gym_state = vis_gym.get_gym_state(state, env.name)
    if env.name == "Pendulum-v1":
        gym_env.unwrapped.last_u = gym_state[-1]
    gym_env.unwrapped.state = gym_state
    rgb_array = gym_env.render()
    im.set_data(rgb_array)
    gym_env.close()
    return im

def unwrap_state(state):
    """Recursively unwraps any wrapper state objects (like LogEnvState) to get the raw state."""
    while hasattr(state, "env_state"):
        state = state.env_state
    return state

vis_gym.init_gym = patched_init_gym
vis_gym.update_gym = patched_update_gym


def generate_policy_video(run_config, train_state, env_dir, seed=0, save_name="policy.gif"):
    """
    Rolls out a trained policy in the environment and generates a gif.
    """
    try:
        env, env_params = helpers.make_env(run_config)
    except Exception as e:
        print(f"Error creating environment for video: {e}")
        return

    # Check if Gymnax's Visualizer supports this environment
    supported_envs = [
        "Acrobot-v1", "CartPole-v1", "Pendulum-v1", "MountainCar-v0", "MountainCarContinuous-v0",
        "Catch-bsuite", "Asterix-MinAtar", "Breakout-MinAtar", "Freeway-MinAtar",
        "Seaquest-MinAtar", "SpaceInvaders-MinAtar", "Pong-misc", "PointRobot-misc",
        "MetaMaze-misc", "FourRooms-misc"
    ]
    
    # We use env.unwrapped to get the raw Gymnax environment properties (like name or occupied_map)
    raw_env = env.unwrapped if hasattr(env, "unwrapped") else env
    
    if raw_env.name not in supported_envs:
        print(f"Environment '{raw_env.name}' is not supported by Gymnax's Visualizer. Skipping video generation.")
        return

    print(f"Generating video for environment: {raw_env.name}...")

    key = jax.random.PRNGKey(seed)
    key, key_reset = jax.random.split(key)
    obs, env_state = env.reset(key_reset, env_params)

    state_seq, reward_seq = [], []
    t_counter = 0
    max_steps = env_params.max_steps_in_episode

    while True:
        state_seq.append(env_state)
        key, key_act, key_step = jax.random.split(key, 3)

        # Retrieve action from policy distribution
        pi, _ = train_state.apply_fn(train_state.params, obs)
        action = pi.sample(seed=key_act)

        # If the policy returned a batched action (e.g. shape (1,) or (1, D)),
        # squeeze the batch dimension so we pass a clean unbatched action to env.step.
        if action.ndim > 0 and action.shape[0] == 1:
            action = action[0]

        next_obs, next_env_state, reward, done, info = env.step(
            key_step, env_state, action, env_params
        )
        reward_seq.append(reward)
        t_counter += 1

        if bool(done) or t_counter >= max_steps:
            break
        else:
            obs = next_obs
            env_state = next_env_state

    try:
        cum_rewards = jnp.cumsum(jnp.array(reward_seq))
        unwrapped_state_seq = [unwrap_state(s) for s in state_seq]
        
        vis = Visualizer(raw_env, env_params, unwrapped_state_seq, cum_rewards)
        os.makedirs(env_dir, exist_ok=True)
        gif_path = os.path.join(env_dir, save_name)
        vis.animate(gif_path)
        print(f"Successfully saved animation to: {gif_path}")
    except Exception as e:
        print(f"Error during animation/saving: {e}")


def main():
    parser = argparse.ArgumentParser(description="Create a video of a saved policy in action.")
    parser.add_argument("--checkpoint-path", type=str, required=True,
                        help="Path to the directory containing config.json and out.pkl")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for evaluation (default: 0)")
    parser.add_argument("--save-name", type=str, default="policy.gif",
                        help="Filename for the generated video (default: policy.gif)")
    args = parser.parse_args()

    print(f"Loading run data from {args.checkpoint_path}...")
    try:
        run_config, results = load_run_data_from_path(args.checkpoint_path)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    if "runner_state" not in results:
        print("Error: 'runner_state' not found in loaded results. Did you save checkpoints during training?")
        return

    # Extract the trained state for the selected seed index
    runner_state = results["runner_state"]
    try:
        policy_train_state = runner_state[0]
        n_seeds = run_config.get("N_SEEDS", 1)
        target_seed_idx = args.seed % n_seeds
        
        # Slices parameters to extract single seed
        policy_params = jax.tree_util.tree_map(lambda x: x[target_seed_idx] if hasattr(x, "shape") and len(x.shape) > 0 else x, policy_train_state.params)
        train_state_seed = policy_train_state.replace(params=policy_params)
    except Exception as e:
        print(f"Error extracting policy from runner_state: {e}")
        return

    generate_policy_video(run_config, train_state_seed, args.checkpoint_path, seed=args.seed, save_name=args.save_name)


if __name__ == "__main__":
    main()