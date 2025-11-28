from stable_baselines3 import PPO
from drone_env import DroneEnv
import pygame

# 1. Load Env (Human Mode = Visuals ON)
env = DroneEnv(render_mode="human")

# 2. Load the trained brain
# Make sure the path matches your saved file
model_path = "models/PPO/drone_physics.zip" 

print("--- LOADING MODEL FOR PLAYBACK ---")
try:
    model = PPO.load(model_path, env=env, device='cpu')
except:
    print("Error: Could not load model. Make sure 'models/PPO/drone_physics.zip' exists.")
    exit()

# 3. Play Loop
obs, _ = env.reset()
running = True

while running:
    # Ask the brain what to do
    action, _ = model.predict(obs)
    
    # Do it
    obs, reward, terminated, truncated, info = env.step(action)
    
    # If episode ends, reset
    if terminated or truncated:
        obs, _ = env.reset()
        
    # Allow quitting via Window X button (Handled inside env.render, but good to have here too)