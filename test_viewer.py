from stable_baselines3 import PPO
from drone_env import DroneEnv
import pygame

# 1. Create Env
env = DroneEnv(render_mode="human")

# 2. Load Brain
model_path = "models/PPO7/drone_physics" 
print(f"--- LOADING BRAIN: {model_path} ---")

try:
    model = PPO.load(model_path, env=env, device='cpu')
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 3. Play Loop
print("--- SIMULATION STARTED ---")
obs, _ = env.reset()

while True:
    # --- CHANGE: Removed pygame.event.get() from here ---
    # The environment handles events inside env.step() -> env.render()
    
    # AI Decision
    action, _ = model.predict(obs, deterministic=True)
    
    # Execution (Render happens here)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, _ = env.reset()