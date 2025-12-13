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
    # Map to -1 (Left) to 1 (Right)
    angle_cmd = 0.0
    if keys[pygame.K_LEFT]:  angle_cmd = -0.5  # 90 deg Left
    if keys[pygame.K_RIGHT]: angle_cmd = 0.5   # 90 deg Right
    if keys[pygame.K_UP]:    angle_cmd = 0.0   # Straight at target
    if keys[pygame.K_DOWN]:  angle_cmd = 1.0   # Retreat
    
    # 2. Distance (W/S Keys)
    # Map to -1 (0m) to 1 (400m)
    dist_cmd = 0.0 # Default ~200m
    if keys[pygame.K_w]: dist_cmd = 0.8  # Far
    if keys[pygame.K_s]: dist_cmd = -0.8 # Close
    
    # 3. Altitude (Spacebar)
    # High (>0) or Low (<0)
    alt_cmd = -1.0
    if keys[pygame.K_SPACE]: alt_cmd = 1.0
    
    # Create action for Agent 0
    # For now, we only control the first agent, others get zeros
    action = np.zeros(n_agents * 3, dtype=np.float32)
    action[0] = dist_cmd
    action[1] = angle_cmd
    action[2] = alt_cmd
    
    return action

def main():
    env = DroneEnvCommander(render_mode="human", n_agents=1)
    obs, info = env.reset()
    
    print("\n--- MANUAL COMMANDER MODE ---")
    print("ARROWS:  Issue Direction Command (Relative to Green Target)")
    print("W / S:   Issue Distance Command (Far / Close)")
    print("SPACE:   Hold for HIGH ALTITUDE (Cyan)")
    print("         Release for LOW ALTITUDE (Orange - Pathfinding)")
    print("-----------------------------")
    
    running = True
    while running:
        # Get Action from Keyboard
        action = get_human_action(env.n_agents)
        
        # Step Environment
        # Note: This runs 30 physics ticks (0.5s). 
        # Control feel will be "Strategy Game" pace, not "Flight Sim" pace.
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check for window close inside the loop
        if terminated or truncated:
            obs, info = env.reset()
            
        # Optional: Print debug info
        # agent_pos = env.agents[0].position
        # print(f"Rew: {reward:.1f} | Alt: {agent_pos[2]:.1f}")

    env.close()

if __name__ == "__main__":
    main()