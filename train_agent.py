from stable_baselines3 import PPO
from drone_env import DroneDynamicsEnv

# 1. Build the obstacle course from your blueprint
env = DroneDynamicsEnv()

# 2. Initialize the AI Pilot (PPO)
print("Initializing the PPO Neural Network...")
model = PPO("MlpPolicy", env, verbose=1)

# 3. Train the AI (Let it crash and learn)
print("Starting 50,000 timesteps of training. This will take a moment...")
model.learn(total_timesteps=50000)

# 4. Save the trained "brain" to your computer
model.save("ppo_drone_model")
print("Training Complete! The AI brain has been saved.")