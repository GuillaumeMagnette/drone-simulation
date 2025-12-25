import os
import glob
import pygame
from stable_baselines3 import PPO
from drone_env_commander import DroneEnvCommander

# Base directory where training saves models
BASE_DIR = "models/PPO"

def get_latest_model_path():
    """
    Finds the most recent experiment folder (PPO_X) 
    and the most recent checkpoint inside it.
    """
    # 1. Find all PPO_X folders
    experiments = glob.glob(f"{BASE_DIR}*")
    if not experiments:
        print("No training folders found!")
        return None
    
    # Sort by modification time (newest first)
    experiments.sort(key=os.path.getmtime, reverse=True)
    latest_exp = experiments[0]
    print(f"Checking folder: {latest_exp}")
    
    # 2. Find all .zip files in that folder
    models = glob.glob(f"{latest_exp}/*.zip")
    if not models:
        print(f"No .zip models found in {latest_exp}")
        return None
        
    # Sort by step count (commander_12500_steps.zip)
    # We extract the number from the filename to sort correctly
    def get_step_count(filename):
        try:
            # simple parsing: commander_12500_steps.zip -> 12500
            parts = filename.split('_')
            for p in parts:
                if p.isdigit():
                    return int(p)
            return 0
        except:
            return 0

    models.sort(key=get_step_count, reverse=True)
    latest_model = models[0]
    return latest_model

def main():
    model_path = get_latest_model_path()
    if not model_path:
        return

    print(f"\n--- LOADING: {model_path} ---")
    
    # 1. Load Environment
    # n_agents MUST match training (2)
    env = DroneEnvCommander(render_mode="human", n_agents=2) 
    
    # 2. Load Model
    model = PPO.load(model_path)

    obs, info = env.reset()
    
    # Force window creation
    env.render()
    
    print("Running simulation... (Press ESC to quit)")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        # Predict
        action, _ = model.predict(obs, deterministic=True)
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode Reward: {reward:.2f}")
            obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()