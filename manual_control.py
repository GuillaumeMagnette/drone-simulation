"""
manual_control.py
Be the Commander.
Test the translation from High-Level Commands -> Navigator -> Physics.
"""

import pygame
import numpy as np
from drone_env_commander import DroneEnvCommander

def get_human_action(n_agents):
    """
    Reads keyboard input and converts it to Commander Action Space.
    Action = [Distance (-1..1), Angle (-1..1), Alt (-1..1)]
    """
    keys = pygame.key.get_pressed()
    
    # 1. Angle (Left/Right Arrows)
    angle_cmd = 0.0
    if keys[pygame.K_LEFT]:  angle_cmd = -0.5  # 90 deg Left
    if keys[pygame.K_RIGHT]: angle_cmd = 0.5   # 90 deg Right
    if keys[pygame.K_UP]:    angle_cmd = 0.0   # Straight at target
    if keys[pygame.K_DOWN]:  angle_cmd = 1.0   # Retreat
    
    # 2. Distance (W/S Keys)
    dist_cmd = 0.0 # Default ~200m
    if keys[pygame.K_w]: dist_cmd = 0.8  # Far
    if keys[pygame.K_s]: dist_cmd = -0.8 # Close
    
    # 3. Altitude (Spacebar)
    alt_cmd = -1.0
    if keys[pygame.K_SPACE]: alt_cmd = 1.0
    
    # Create action for Agent 0
    action = np.zeros(n_agents * 3, dtype=np.float32)
    action[0] = dist_cmd
    action[1] = angle_cmd
    action[2] = alt_cmd
    
    return action

def main():
    # --- FIX 1: Initialize Pygame explicitly ---
    pygame.init()
    
    env = DroneEnvCommander(render_mode="human", n_agents=1)
    obs, info = env.reset()
    
    # --- FIX 2: Create the window immediately ---
    # We must render once so the window exists to capture keystrokes
    env.render()
    
    print("\n--- MANUAL COMMANDER MODE ---")
    print("ARROWS:  Issue Direction Command (Relative to Green Target)")
    print("W / S:   Issue Distance Command (Far / Close)")
    print("SPACE:   Hold for HIGH ALTITUDE (Cyan)")
    print("         Release for LOW ALTITUDE (Orange - Pathfinding)")
    print("-----------------------------")
    
    running = True
    try:
        while running:
            # --- FIX 3: Process Window Events ---
            # This pumps the event loop so pygame.key.get_pressed() updates
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Get Action from Keyboard
            action = get_human_action(env.n_agents)
            
            # Step Environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print(f"Episode Done. Reward: {reward:.1f}")
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("User cancelled.")
    finally:
        env.close()
        pygame.quit()

if __name__ == "__main__":
    main()