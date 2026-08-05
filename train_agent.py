import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from drone_env import DroneDynamicsEnv


class RandomWindWrapper(gym.Wrapper):
    """Picks a fresh wind_strength between wind_low and wind_high at the start of every episode."""

    def __init__(self, env, wind_low=0.0, wind_high=0.6):
        super().__init__(env)
        self.wind_low = wind_low
        self.wind_high = wind_high

    def reset(self, **kwargs):
        self.env.wind_strength = np.random.uniform(self.wind_low, self.wind_high)
        return self.env.reset(**kwargs)


# 1. Build the obstacle course from your blueprint, with wind randomized every episode
env = RandomWindWrapper(DroneDynamicsEnv())

# 2. Initialize the AI Pilot (PPO)
print("Initializing the PPO Neural Network...")
model = PPO("MlpPolicy", env, verbose=1)

# 3. Train the AI (Let it crash and learn)
print("Starting 300,000 timesteps of training. This will take a moment...")
model.learn(total_timesteps=1000000)

# 4. Save the trained "brain" to your computer
model.save("ppo_drone_model_windy_v3")
print("Training Complete! The AI brain has been saved as ppo_drone_model_windy_v2.zip")
