"""
TACTICAL DRONE TEST VIEWER - PHASE 3 (SCAVENGER READY)
======================================================
"""

from stable_baselines3 import PPO
from drone_env_patched import DroneEnv
import pygame
import numpy as np
import time

MODEL_PATH = "models/PPO_Tactical_Enhanced_Optimized_v2/drone_tactical"

# ALIGNED WITH TRAIN_FAST.PY
SCENARIOS = {
    # Phase 1: Basics
    1: {"name": "Nav Basics",      "map": "sparse", "threats": 0, "bait": False, "respawn": False, "dynamic": False},
    2: {"name": "Evasion Arena",   "map": "arena",  "threats": 1, "bait": False, "respawn": True,  "dynamic": True},
    
    # Phase 2: Scavenger Hunt
    3: {"name": "Scavenger Hunt",  "map": "sparse", "threats": 1, "bait": False, "respawn": True,  "dynamic": True}, # <-- STAGE 2.0
    4: {"name": "Terrain Mission", "map": "sparse", "threats": 1, "bait": False, "respawn": True,  "dynamic": False}, # <-- STAGE 2.1
    
    # Phase 3: Urban
    5: {"name": "Urban Scavenger", "map": "urban",  "threats": 2, "bait": False, "respawn": True,  "dynamic": True},
    6: {"name": "Urban Mission",   "map": "urban",  "threats": 2, "bait": False, "respawn": True,  "dynamic": False},
}

def print_help():
    print("\n" + "="*70)
    print("  TACTICAL DRONE VIEWER - SCAVENGER EDITION")
    print("="*70 + "\n")
    print("  SCENARIOS:")
    for key, sc in SCENARIOS.items():
        mode = "SCAVENGER" if sc['dynamic'] else "MISSION"
        print(f"    [{key}] {sc['name']:<18} ({sc['map']}, {sc['threats']} threats, {mode})")
    print("\n  CONTROLS: [1-6] Scenario, [V] Visual, [+/-] Speed, [R] Reset\n" + "="*70)

class StatsTracker:
    def __init__(self): self.reset()
    def reset(self):
        self.episodes = 0
        self.successes = 0
        self.failures = {"Crash": 0, "Intercepted": 0, "Timeout": 0}
    def record_episode(self, reason):
        self.episodes += 1
        if reason == "Success": self.successes += 1
        self.failures[reason] = self.failures.get(reason, 0) + 1
    def print_stats(self, name):
        if self.episodes == 0: return
        sr = (self.successes / self.episodes * 100)
        print(f"[{name}] Eps: {self.episodes:3d} | Win: {sr:5.1f}% | "
              f"Crash: {self.failures.get('Crash',0)} | Caught: {self.failures.get('Intercepted',0)} | Timeout: {self.failures.get('Timeout',0)}")

def main():
    pygame.init()
    print_help()
    
    visual_mode = True
    steps_per_frame = 1
    env = DroneEnv(render_mode="human")
    
    try:
        model = PPO.load(MODEL_PATH, env=env, device='cpu')
        use_model = True
        print("✓ Model loaded.")
    except:
        use_model = False
        print("✗ No model found. Random actions.")
    
    current = 3 # Default to Scavenger Hunt
    sc = SCENARIOS[current]
    stats = StatsTracker()
    
    def reset_env(scenario):
        return env.reset(options={
            'map_type': scenario['map'], 'num_interceptors': scenario['threats'],
            'respawn': scenario['respawn'], 'use_bait': scenario['bait'],
            'dynamic_target': scenario['dynamic'] # <--- NEW FLAG
        })

    obs, _ = reset_env(sc)
    running = True
    clock = pygame.time.Clock()
    
    while running:
        if visual_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6]:
                        current = event.key - pygame.K_0
                        sc = SCENARIOS[current]
                        stats.reset()
                        print(f"\n>>> SWITCH: {sc['name']}")
                        obs, _ = reset_env(sc)
                    elif event.key == pygame.K_v:
                        visual_mode = not visual_mode
                        env.close()
                        env = DroneEnv(render_mode="human" if visual_mode else None)
                        if use_model: model.set_env(env)
                        obs, _ = reset_env(sc)
                    elif event.key == pygame.K_r:
                        stats.reset()
                        obs, _ = reset_env(sc)
                        print("Reset.")
                    elif event.key == pygame.K_PLUS: steps_per_frame = min(steps_per_frame+1, 10)
                    elif event.key == pygame.K_MINUS: steps_per_frame = max(steps_per_frame-1, 1)

        for _ in range(steps_per_frame):
            if use_model: action, _ = model.predict(obs, deterministic=True)
            else: action = env.action_space.sample()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            
            if terminated or truncated:
                # In Scavenger mode, Timeout is effectively "Game Over" (ran out of fuel/time)
                # Success is meaningless per se, but high reward accumulation is key.
                # For stats, we track "Success" as hitting at least one target? 
                # Actually, the env only terminates on Crash/Catch/Timeout in Scavenger mode.
                stats.record_episode(env.termination_reason)
                stats.print_stats(sc['name'])
                obs, _ = reset_env(sc)
                break
        
        if visual_mode: clock.tick(60)

    env.close()

if __name__ == "__main__":
    main()