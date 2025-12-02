"""
TACTICAL DRONE TEST VIEWER
==========================

Visual validation tool for the trained evasion brain.

CONTROLS:
---------
[1] Nav Basics      - Sparse map, no threats
[2] Evasion Arena   - Empty arena, 1 FAST interceptor
[3] Terrain Evasion - Sparse map, 1 interceptor
[4] Multi-Threat    - Sparse map, 2 interceptors
[5] Urban Nav       - Urban map, no threats
[6] Urban Evasion   - Urban map, 2 interceptors

[R]   Reset current scenario
[ESC] Quit

WHAT TO LOOK FOR:
-----------------
- Stage 2: Does the agent panic and dodge the interceptor?
- Stage 3: Does it use obstacles to break pursuit?
- Stage 6: Can it navigate AND evade simultaneously?
"""

from stable_baselines3 import PPO
from drone_env import DroneEnv
import pygame
import numpy as np


MODEL_PATH = "models/PPO_Tactical3/drone_tactical"

SCENARIOS = {
    # Match curriculum stages
    1: {"name": "Nav Basics",     "map": "sparse", "interceptors": 0},
    2: {"name": "Evasion Arena",  "map": "arena",  "interceptors": 1},
    3: {"name": "Terrain Evasion","map": "sparse", "interceptors": 1},
    4: {"name": "Multi-Threat",   "map": "sparse", "interceptors": 2},
    5: {"name": "Urban Nav",      "map": "urban",  "interceptors": 0},
    6: {"name": "Urban Evasion",  "map": "urban",  "interceptors": 2},
}


def print_help():
    print()
    print("=" * 50)
    print("  TACTICAL DRONE VIEWER")
    print("=" * 50)
    print()
    for key, sc in SCENARIOS.items():
        print(f"  [{key}] {sc['name']:<16} ({sc['map']}, {sc['interceptors']} threats)")
    print()
    print("  [R]   Reset scenario")
    print("  [ESC] Quit")
    print("=" * 50)
    print()


def main():
    # Initialize pygame FIRST before any event handling
    pygame.init()
    
    print_help()
    
    # Create environment
    env = DroneEnv(render_mode="human")
    
    # Load model
    print(f"Loading: {MODEL_PATH}")
    try:
        model = PPO.load(MODEL_PATH, env=env, device='cpu')
        print("Model loaded!")
        use_model = True
    except Exception as e:
        print(f"Could not load model: {e}")
        print("Using random actions for testing.")
        use_model = False
    
    # Initial scenario
    current = 2  # Start with Arena Basic
    sc = SCENARIOS[current]
    
    print(f"\nStarting: {sc['name']}")
    obs, _ = env.reset(options={
        'map_type': sc['map'],
        'num_interceptors': sc['interceptors']
    })
    
    # Stats
    episodes = 0
    successes = 0
    
    # Main loop
    running = True
    while running:
        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                # Scenario switch
                if event.key in [pygame.K_1, pygame.K_2, pygame.K_3,
                                pygame.K_4, pygame.K_5, pygame.K_6]:
                    current = event.key - pygame.K_0
                    sc = SCENARIOS[current]
                    print(f"\n>>> {sc['name']} ({sc['map']}, {sc['interceptors']} threats)")
                    episodes = 0
                    successes = 0
                    obs, _ = env.reset(options={
                        'map_type': sc['map'],
                        'num_interceptors': sc['interceptors']
                    })
                
                # Reset
                elif event.key == pygame.K_r:
                    sc = SCENARIOS[current]
                    obs, _ = env.reset(options={
                        'map_type': sc['map'],
                        'num_interceptors': sc['interceptors']
                    })
                    print("Reset!")
                
                # Quit
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        if not running:
            break
        
        # Action
        if use_model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.random.uniform(-1, 1, size=(2,)).astype(np.float32)
        
        # Step
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # Episode end
        if terminated or truncated:
            episodes += 1
            reason = env.termination_reason
            
            if reason == "Success":
                successes += 1
            
            rate = (successes / episodes * 100) if episodes > 0 else 0
            print(f"[{SCENARIOS[current]['name']}] Ep {episodes}: {reason} | Success: {rate:.0f}%")
            
            sc = SCENARIOS[current]
            obs, _ = env.reset(options={
                'map_type': sc['map'],
                'num_interceptors': sc['interceptors']
            })
    
    env.close()
    print("\nViewer closed.")


if __name__ == "__main__":
    main()
