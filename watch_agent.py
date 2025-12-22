import time
import pygame
from stable_baselines3 import PPO
from drone_env_commander import DroneEnvCommander

# Path to your trained model
MODEL_PATH = "models/PPO/commander_final" 

def main():
    # 1. Load Environment
    env = DroneEnvCommander(render_mode="human", n_agents=1)
    
    # 2. Load Model
    try:
        model = PPO.load(MODEL_PATH, device='cpu')
        print(f"Loaded model: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Model not found at {MODEL_PATH}. Run train_commander.py first!")
        return

    obs, info = env.reset()
    
    # --- FIX: Create Window Immediately ---
    # We must render once so the window exists before checking events
    env.render()
    
    print("\n--- WATCHING AI COMMANDER ---")
    print("Green Circle = Target")
    print("Blue X = AI Command Waypoint")
    print("Yellow Line = Navigator Path")
    
    running = True
    while running:
        # Listen for Quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Predict Action
        action, _states = model.predict(obs, deterministic=True)
        
        # Step Env
        # (Render is called internally inside step if render_mode="human")
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
            print("Resetting environment...")

    env.close()

if __name__ == "__main__":
    main()