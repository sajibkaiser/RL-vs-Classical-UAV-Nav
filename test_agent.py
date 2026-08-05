import argparse
from stable_baselines3 import PPO
from drone_env import DroneDynamicsEnv, run_episode
from shields import naive_shield, lookahead_shield, momentum_aware_shield, no_shield

SHIELDS = {"none": no_shield, "naive": naive_shield, "lookahead": lookahead_shield, "momentum": momentum_aware_shield}

parser = argparse.ArgumentParser(description="Fly the trained PPO pilot in the drone nav environment.")
parser.add_argument("--watch", action="store_true",
                     help="Watch flights live with rendering on, looping forever (default: run a headless batch tally).")
parser.add_argument("--episodes", type=int, default=50,
                     help="Number of episodes to run in batch mode (default: 50).")
parser.add_argument("--wind", type=float, default=0.3,
                     help="Wind strength (default: 0.3).")
parser.add_argument("--model", type=str, default="ppo_drone_model",
                     help="Model file to load, without .zip (default: ppo_drone_model).")
parser.add_argument("--shield", choices=["none", "naive", "lookahead", "momentum"], default="none",
                     help="Safety shield to veto unsafe actions (default: none).")
parser.add_argument("--noise", type=float, default=0.0,
                     help="Sensor noise strength - std dev of Gaussian noise on the pilot's position/velocity readings (default: 0.0).")
args = parser.parse_args()

# Load the Environment and the trained AI brain
env = DroneDynamicsEnv()
env.wind_strength = args.wind
env.noise_strength = args.noise
shield = SHIELDS[args.shield]

print(f"Loading the trained PPO brain from {args.model}.zip...")
model = PPO.load(args.model)


def ppo_pilot(obs):
    action, _states = model.predict(obs, deterministic=True)
    return action


if args.watch:
    # --- Live visual flight, looping forever (the original behaviour) ---
    import pygame
    pygame.init()

    obs, info = env.reset()
    print("Starting simulation. Press Ctrl+C in the terminal to stop.")
    interventions = 0

    for i in range(1000):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        noisy_obs = env.get_noisy_obs(obs)
        proposed_action = ppo_pilot(noisy_obs)
        action, overridden = shield(env, noisy_obs, proposed_action)
        if overridden:
            interventions += 1

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            print(f"Episode ended. Reward: {reward}. Shield interventions: {interventions}. Resetting maze...")
            obs, info = env.reset()
            interventions = 0

    pygame.quit()
else:
    # --- Headless batch run with a tally at the end ---
    print(f"Running {args.episodes} episodes with the PPO pilot (rendering off, wind = {env.wind_strength}, shield = {args.shield}, noise = {env.noise_strength})...")

    tally = {"REACHED": 0, "CRASHED": 0, "STUCK": 0, "TIMED_OUT": 0}
    reached_steps = []
    interventions_per_episode = []

    for ep in range(args.episodes):
        outcome, steps, interventions = run_episode(env, ppo_pilot, render=False, shield=shield)
        tally[outcome] += 1
        if outcome == "REACHED":
            reached_steps.append(steps)
        interventions_per_episode.append(interventions)

    avg_steps = (sum(reached_steps) / len(reached_steps)) if reached_steps else None
    avg_interventions = sum(interventions_per_episode) / len(interventions_per_episode)

    print(f"\n--- Tally over {args.episodes} episodes ---")
    print(f"REACHED:   {tally['REACHED']}")
    print(f"CRASHED:   {tally['CRASHED']}")
    print(f"STUCK:     {tally['STUCK']}")
    print(f"TIMED OUT: {tally['TIMED_OUT']}")
    if avg_steps is not None:
        print(f"Average steps for successful runs: {avg_steps:.1f}")
    else:
        print("Average steps for successful runs: N/A (no successful runs)")
    print(f"Average shield interventions per episode: {avg_interventions:.2f}")

    env.close()
