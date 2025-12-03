"""
TACTICAL DRONE TEST VIEWER - ENHANCED
======================================

Visual validation tool with performance testing mode.

CONTROLS:
---------
[1-6] Select scenario (see list below)
[V]   Toggle visual rendering (OFF = 10x faster testing)
[R]   Reset current scenario
[+/-] Speed control (steps per frame when visual ON)
[ESC] Quit

SCENARIOS:
----------
[1] Nav Basics      - Sparse map, no threats
[2] Evasion Arena   - Empty arena, 2 FAST interceptors
[3] Terrain Evasion - Sparse map, 1 interceptor
[4] Multi-Threat    - Sparse map, 2 interceptors
[5] Urban Nav       - Urban map, no threats
[6] Urban Evasion   - Urban map, 2 interceptors

TESTING MODES:
--------------
VISUAL ON:  Watch agent behavior, useful for debugging
VISUAL OFF: Rapid performance testing, see stats only

WHAT TO LOOK FOR:
-----------------
- Stage 2: Does the agent panic and dodge interceptors?
- Stage 3: Does it use obstacles to break pursuit?
- Stage 6: Can it navigate AND evade simultaneously?
"""

from stable_baselines3 import PPO
from drone_env_patched import DroneEnv
import pygame
import numpy as np
import time


MODEL_PATH = "models/PPO_Tactical4/drone_tactical"

SCENARIOS = {
    # Match curriculum stages
    1: {"name": "Nav Basics",     "map": "sparse", "interceptors": 0},
    2: {"name": "Evasion Arena",  "map": "arena",  "interceptors": 2},
    3: {"name": "Terrain Evasion","map": "sparse", "interceptors": 1},
    4: {"name": "Multi-Threat",   "map": "sparse", "interceptors": 2},
    5: {"name": "Urban Nav",      "map": "urban",  "interceptors": 0},
    6: {"name": "Urban Evasion",  "map": "urban",  "interceptors": 2},
}


def print_help():
    print()
    print("=" * 60)
    print("  TACTICAL DRONE VIEWER - ENHANCED")
    print("=" * 60)
    print()
    print("  SCENARIOS:")
    for key, sc in SCENARIOS.items():
        print(f"    [{key}] {sc['name']:<18} ({sc['map']}, {sc['interceptors']} threats)")
    print()
    print("  CONTROLS:")
    print("    [V]     Toggle Visual (OFF = faster testing)")
    print("    [+/-]   Speed control (steps per frame)")
    print("    [R]     Reset scenario")
    print("    [ESC]   Quit (when visual ON)")
    print("    [Ctrl+C] Quit (when visual OFF)")
    print()
    print("  NOTE: In headless mode (visual OFF), use Ctrl+C to stop")
    print("=" * 60)
    print()


class StatsTracker:
    """Track and display performance statistics."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.episodes = 0
        self.successes = 0
        self.failures = {"Crash": 0, "Intercepted": 0, "Timeout": 0}
        self.total_rewards = []
        self.episode_lengths = []
        self.start_time = time.time()
        
    def record_episode(self, reason, reward, length):
        self.episodes += 1
        self.total_rewards.append(reward)
        self.episode_lengths.append(length)
        
        if reason == "Success":
            self.successes += 1
        else:
            self.failures[reason] = self.failures.get(reason, 0) + 1
    
    def get_stats(self):
        if self.episodes == 0:
            return None
            
        success_rate = (self.successes / self.episodes * 100)
        avg_reward = np.mean(self.total_rewards[-100:])
        avg_length = np.mean(self.episode_lengths[-100:])
        elapsed = time.time() - self.start_time
        eps_per_min = (self.episodes / elapsed) * 60 if elapsed > 0 else 0
        
        return {
            'episodes': self.episodes,
            'success_rate': success_rate,
            'successes': self.successes,
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'eps_per_min': eps_per_min,
            'crashes': self.failures.get('Crash', 0),
            'intercepted': self.failures.get('Intercepted', 0),
            'timeouts': self.failures.get('Timeout', 0)
        }
    
    def print_stats(self, scenario_name):
        stats = self.get_stats()
        if stats is None:
            return
            
        print(f"\n[{scenario_name}] Episode {stats['episodes']:3d} | "
              f"Success: {stats['success_rate']:5.1f}% ({stats['successes']}/{stats['episodes']}) | "
              f"Avg Reward: {stats['avg_reward']:6.1f} | "
              f"Avg Length: {stats['avg_length']:5.1f} | "
              f"Rate: {stats['eps_per_min']:4.1f} eps/min")
        
        if stats['episodes'] % 10 == 0:
            print(f"         Failures - Crash: {stats['crashes']}, "
                  f"Intercepted: {stats['intercepted']}, "
                  f"Timeout: {stats['timeouts']}")


def main():
    # Initialize pygame FIRST
    pygame.init()
    
    print_help()
    
    # Create environment with visual mode
    visual_mode = True
    steps_per_frame = 1
    env = DroneEnv(render_mode="human")
    
    # Load model
    print(f"Loading: {MODEL_PATH}")
    try:
        model = PPO.load(MODEL_PATH, env=env, device='cpu')
        print("✓ Model loaded!")
        use_model = True
    except Exception as e:
        print(f"✗ Could not load model: {e}")
        print("Using random actions for testing.")
        use_model = False
    
    # Initial scenario
    current = 2  # Start with Evasion Arena
    sc = SCENARIOS[current]
    stats = StatsTracker()
    
    print(f"\n>>> Starting: {sc['name']} (VISUAL: ON)")
    obs, _ = env.reset(options={
        'map_type': sc['map'],
        'num_interceptors': sc['interceptors']
    })
    
    # Episode tracking
    episode_reward = 0
    episode_length = 0
    
    # Main loop
    running = True
    clock = pygame.time.Clock()
    
    while running:
        # Handle events (only if visual mode is ON)
        if visual_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.KEYDOWN:
                    # Scenario switch
                    if event.key in [pygame.K_1, pygame.K_2, pygame.K_3,
                                    pygame.K_4, pygame.K_5, pygame.K_6]:
                        current = event.key - pygame.K_0
                        sc = SCENARIOS[current]
                        stats.reset()
                        episode_reward = 0
                        episode_length = 0
                        print(f"\n>>> {sc['name']} (VISUAL: {'ON' if visual_mode else 'OFF'})")
                        obs, _ = env.reset(options={
                            'map_type': sc['map'],
                            'num_interceptors': sc['interceptors']
                        })
                    
                    # Visual toggle
                    elif event.key == pygame.K_v:
                        visual_mode = not visual_mode
                        
                        # Recreate environment with correct mode
                        env.close()
                        render_mode = "human" if visual_mode else None
                        env = DroneEnv(render_mode=render_mode)
                        
                        # Reload model into new env
                        if use_model:
                            model.set_env(env)
                        
                        # Reset with current scenario
                        obs, _ = env.reset(options={
                            'map_type': sc['map'],
                            'num_interceptors': sc['interceptors']
                        })
                        
                        mode_str = "ON (slower, visual)" if visual_mode else "OFF (FAST, stats only)"
                        print(f"\n>>> VISUAL: {mode_str}")
                        episode_reward = 0
                        episode_length = 0
                    
                    # Speed control
                    elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                        steps_per_frame = min(steps_per_frame + 1, 10)
                        print(f"Speed: {steps_per_frame}x")
                    
                    elif event.key == pygame.K_MINUS:
                        steps_per_frame = max(steps_per_frame - 1, 1)
                        print(f"Speed: {steps_per_frame}x")
                    
                    # Reset
                    elif event.key == pygame.K_r:
                        stats.reset()
                        episode_reward = 0
                        episode_length = 0
                        obs, _ = env.reset(options={
                            'map_type': sc['map'],
                            'num_interceptors': sc['interceptors']
                        })
                        print("\n>>> Reset!")
                    
                    # Quit
                    elif event.key == pygame.K_ESCAPE:
                        running = False
            
            if not running:
                break
        else:
            # Headless mode - check for Ctrl+C or run until manual interrupt
            # In headless mode, user must use Ctrl+C to stop
            try:
                pass  # Just continue running
            except KeyboardInterrupt:
                running = False
                break
        
        # Run simulation steps
        for _ in range(steps_per_frame):
            # Action
            if use_model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = np.random.uniform(-1, 1, size=(2,)).astype(np.float32)
            
            # Step
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Episode end
            if terminated or truncated:
                reason = env.termination_reason
                stats.record_episode(reason, episode_reward, episode_length)
                stats.print_stats(sc['name'])
                
                # Reset for next episode
                episode_reward = 0
                episode_length = 0
                obs, _ = env.reset(options={
                    'map_type': sc['map'],
                    'num_interceptors': sc['interceptors']
                })
                
                break  # Don't run more steps this frame if episode ended
        
        # Frame rate control (only matters in visual mode)
        if visual_mode:
            clock.tick(60)  # 60 FPS max
    
    env.close()
    
    # Final stats
    print("\n" + "=" * 60)
    print("  FINAL STATISTICS")
    print("=" * 60)
    final_stats = stats.get_stats()
    if final_stats:
        print(f"  Total Episodes:  {final_stats['episodes']}")
        print(f"  Success Rate:    {final_stats['success_rate']:.1f}%")
        print(f"  Avg Reward:      {final_stats['avg_reward']:.1f}")
        print(f"  Avg Length:      {final_stats['avg_length']:.1f}")
        print(f"  Crashes:         {final_stats['crashes']}")
        print(f"  Intercepted:     {final_stats['intercepted']}")
        print(f"  Timeouts:        {final_stats['timeouts']}")
    print("=" * 60)
    print("\nViewer closed.")


if __name__ == "__main__":
    main()
