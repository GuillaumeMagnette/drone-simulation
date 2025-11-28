import gymnasium as gym
from stable_baselines3 import PPO
from drone_env import DroneEnv
import os

# Create log dir
models_dir = "models/PPO"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 1. Create Environment
# Set render_mode="human" to WATCH it learn (Slower)
# Set render_mode=None to train FAST (Background)
env = DroneEnv(render_mode="human") 

# 2. Define Model
# We reload the old model if it exists, or create a new one
model_path = f"{models_dir}/drone_physics.zip"

if os.path.exists(model_path):
    print("--- LOADING EXISTING BRAIN ---")
    model = PPO.load(model_path, env=env, device='cpu')
else:
    print("--- CREATING NEW BRAIN ---")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device='cpu')

# 3. Continuous Training Loop
TIMESTEPS = 10000
iters = 0

print("--- STARTING TRAINING LOOP (Press Ctrl+C to stop) ---")
while True:
    iters += 1
    
    # Train for a chunk of time
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    
    # Save the model
    model.save(f"{models_dir}/drone_physics")
    print(f"--- SAVED ITERATION {iters} ---")

env.close()