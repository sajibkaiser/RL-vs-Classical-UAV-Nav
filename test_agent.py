import pygame
from stable_baselines3 import PPO
from drone_env import DroneDynamicsEnv

# Initialize Pygame explicitly before doing anything else
pygame.init()

# 1. Load the Environment
env = DroneDynamicsEnv()

# 2. Load the Trained AI Brain
print("Loading the trained PPO brain...")
model = PPO.load("ppo_drone_model")

# 3. Reset the maze to a random layout to start
obs, info = env.reset()

print("Starting simulation. Press Ctrl+C in the terminal to stop.")

# 4. The Infinite Flight Loop
for i in range(1000):
    # Let Pygame process window events (so it doesn't freeze)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
            
    # The AI looks at the observation and decides on an action
    action, _states = model.predict(obs, deterministic=True)
    
    # The environment processes the physics of that action
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Paint the screen
    env.render()
    
    # If the drone crashes or wins, reset the maze to a new random layout
    if terminated or truncated:
        print(f"Episode ended. Reward: {reward}. Resetting maze...")
        obs, info = env.reset()

pygame.quit()
