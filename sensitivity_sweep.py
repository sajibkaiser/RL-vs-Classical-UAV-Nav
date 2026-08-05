import numpy as np
from drone_env import DroneDynamicsEnv, run_episode, apf_action
from shields import momentum_aware_shield

HORIZONS = [3, 5, 8]
WIND_MARGINS = [0.2, 0.4, 0.6]
WIND = 0.6
EPISODES = 250
REPEATS = 3

print(f"Sweeping momentum_aware_shield over horizon x wind_margin at wind={WIND}, "
      f"{REPEATS} repeats x {EPISODES} episodes each...\n")

print(f"{'horizon':<9} {'wind_margin':<12} {'run':<4} {'REACHED':>8} {'CRASHED':>8} {'STUCK':>6} {'TIMED_OUT':>10} {'interventions':>14}")

results = {}  # (horizon, wind_margin) -> list of per-run dicts

for horizon in HORIZONS:
    for wind_margin in WIND_MARGINS:
        runs = []
        for run in range(1, REPEATS + 1):
            env = DroneDynamicsEnv()
            env.wind_strength = WIND
            apf_pilot = lambda obs: apf_action(obs, env.gap_height, env.wall_width)
            shield = lambda env, obs, action, h=horizon, wm=wind_margin: momentum_aware_shield(env, obs, action, horizon=h, wind_margin=wm)

            tally = {"REACHED": 0, "CRASHED": 0, "STUCK": 0, "TIMED_OUT": 0}
            interventions_total = 0

            for ep in range(EPISODES):
                outcome, steps, interventions = run_episode(env, apf_pilot, render=False, shield=shield)
                tally[outcome] += 1
                interventions_total += interventions

            avg_interventions = interventions_total / EPISODES
            runs.append({**tally, "avg_interventions": avg_interventions})

            print(f"{horizon:<9} {wind_margin:<12} {run:<4} {tally['REACHED']:>8} {tally['CRASHED']:>8} {tally['STUCK']:>6} {tally['TIMED_OUT']:>10} {avg_interventions:>14.2f}")

        results[(horizon, wind_margin)] = runs

print(f"\n--- Mean +/- std over {REPEATS} repeats ---")
print(f"{'horizon':<9} {'wind_margin':<12} {'reached %':>18} {'crashed %':>18} {'interventions':>18}")

for horizon in HORIZONS:
    for wind_margin in WIND_MARGINS:
        runs = results[(horizon, wind_margin)]
        reached_pct = np.array([r["REACHED"] / EPISODES * 100 for r in runs])
        crashed_pct = np.array([r["CRASHED"] / EPISODES * 100 for r in runs])
        interventions = np.array([r["avg_interventions"] for r in runs])

        reached_str = f"{reached_pct.mean():.1f} +/- {reached_pct.std():.1f}"
        crashed_str = f"{crashed_pct.mean():.1f} +/- {crashed_pct.std():.1f}"
        interventions_str = f"{interventions.mean():.2f} +/- {interventions.std():.2f}"

        print(f"{horizon:<9} {wind_margin:<12} {reached_str:>18} {crashed_str:>18} {interventions_str:>18}")
