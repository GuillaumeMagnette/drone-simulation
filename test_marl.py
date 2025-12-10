"""
TACTICAL VIEWER: SIM 2.0 (MARL 2.5D)
====================================

Visual validation tool for the Hive Mind.
Supports switching between Flight School and Combat modes.
"""

import pygame
import time
import numpy as np
from stable_baselines3 import PPO
from drone_env_marl_25d import DroneEnvMARL25D

MODEL_PATH = "models/PPO_MARL_25D/hive_mind"

# CURRICULUM SCENARIOS
SCENARIOS = {
    # Key: (Name, Active Threats)
    1: {"name": "5.0 Flight School", "threats": False}, # No Missiles, learn to fly/hover
    2: {"name": "5.1 Hive Combat",   "threats": True},  # Missiles Active, learn NOE
}

def print_help():
    print("\n" + "="*60)
    print("  HIVE MIND VIEWER - 2.5D")
    print("="*60 + "\n")
    print("  SCENARIOS:")
    print("    [1] 5.0 Flight School (No Threats, Learn Hover/Nav)")
    print("    [2] 5.1 Hive Combat   (Threats Active, Learn Evasion)")
    print("\n  CONTROLS:")
    print("    [V]     Toggle Visual (OFF = turbo speed)")
    print("    [+/-]   Sim Speed Control")
    print("    [R]     Reset Episode")
    print("    [ESC]   Quit")
    print("\n  VISUAL KEY:")
    print("    GREEN Drone: Low Altitude (Safe from Radar)")
    print("    CYAN Drone:  High Altitude (VISIBLE to Radar!)")
    print("    RED Dot:     Missile (Low)")
    print("    PINK Dot:    Missile (High)")
    print("="*60 + "\n")

class StatsTracker:
    def __init__(self): 
        self.reset()
    def reset(self):
        self.episodes = 0
        self.total_rewards = []
    def record(self, reward):
        self.episodes += 1
        self.total_rewards.append(reward)
    def print_stats(self, scenario_name):
        if not self.total_rewards: return
        avg = sum(self.total_rewards[-10:]) / min(len(self.total_rewards), 10)
        print(f"[{scenario_name}] Ep: {self.episodes} | Last 10 Avg Reward: {avg:.1f}")

def main():
    pygame.init()
    print_help()
    
    # Setup
    env = DroneEnvMARL25D(render_mode="human")
    visual_mode = True
    steps_per_frame = 1
    current_sc_key = 2 # Default to Combat
    
    # Load Model
    try:
        model = PPO.load(MODEL_PATH, env=env, device="cpu")
        print("✓ Hive Mind Loaded.")
        use_model = True
    except:
        print("! NO MODEL FOUND. Running Random Actions.")
        use_model = False

    stats = StatsTracker()
    
    def reset_env(key):
        sc = SCENARIOS[key]
        print(f"\n>>> SWITCH: {sc['name']}")
        return env.reset(options={'active_threats': sc['threats']})

    obs, _ = reset_env(current_sc_key)
    
    running = True
    clock = pygame.time.Clock()
    episode_reward = 0
    
    while running:
        # --- INPUT HANDLING ---
        if visual_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: running = False
                    
                    # Switch Scenarios
                    elif event.key == pygame.K_1: 
                        current_sc_key = 1
                        obs, _ = reset_env(1)
                        episode_reward = 0
                    elif event.key == pygame.K_2: 
                        current_sc_key = 2
                        obs, _ = reset_env(2)
                        episode_reward = 0
                        
                    # Speed
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        steps_per_frame = min(steps_per_frame + 1, 10)
                        print(f"Speed: {steps_per_frame}x")
                    elif event.key == pygame.K_MINUS:
                        steps_per_frame = max(steps_per_frame - 1, 1)
                        print(f"Speed: {steps_per_frame}x")
                    
                    # Reset
                    elif event.key == pygame.K_r:
                        obs, _ = reset_env(current_sc_key)
                        episode_reward = 0
                        print("Reset.")
                        
                    # Toggle Visuals
                    elif event.key == pygame.K_v:
                        visual_mode = not visual_mode
                        if visual_mode:
                            env.render_mode = "human"
                            env.screen = None # Force re-init
                        else:
                            env.render_mode = None
                            if env.screen:
                                pygame.display.quit()
                                env.screen = None
                        print(f"Visuals: {visual_mode}")

        # --- PHYSICS LOOP ---
        for _ in range(steps_per_frame):
            if use_model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                stats.record(episode_reward)
                stats.print_stats(SCENARIOS[current_sc_key]['name'])
                episode_reward = 0
                obs, _ = reset_env(current_sc_key)
                break
        
        # --- RENDER ---
        if visual_mode:
            env.render()
            clock.tick(60)

    env.close()

if __name__ == "__main__":
    main()