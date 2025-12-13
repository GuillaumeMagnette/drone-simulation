import time
import numpy as np
from drone_env_commander import DroneEnvCommander

# Optional: Import SB3's checker if you have it installed
try:
    from stable_baselines3.common.env_checker import check_env
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("Stable Baselines 3 not found. Skipping API check.")

def main():
    # 1. Initialize the Environment
    # We use render_mode='human' to see the window
    env = DroneEnvCommander(render_mode="human", n_agents=1)
    
    # 2. Run API Check (Strict Validation)
    if HAS_SB3:
        print("Checking Environment Compliance...")
        # We assume render_mode=None for the checker to prevent window spam
        check_env(DroneEnvCommander(n_agents=1), warn=True)
        print("[OK] Environment API is valid!")

    # 3. Visual Loop
    print("\nRunning Simulation Loop...")
    print("Visual Legend:")
    print("  - GREEN CIRCLE: Target Zone")
    print("  - CYAN DOT:     High Altitude Agent (Fast, ignores walls)")
    print("  - ORANGE DOT:   Low Altitude Agent (Stealth, uses A* pathing)")
    print("  - RED DOT:      Interceptor Missile")
    print("  - WHITE LINE:   Velocity Vector")
    
    obs, info = env.reset()
    
    try:
        for episode in range(5):
            print(f"--- Episode {episode + 1} ---")
            terminated = False
            truncated = False
            step = 0
            
            while not (terminated or truncated):
                # 4. Sample Random Actions
                # The RL agent would output this.
                # [Distance, Angle, Altitude_Mode]
                action = env.action_space.sample()
                
                # OPTIONAL: Force specific behaviors to test
                # Force High Altitude (Test Direct Flight)
                # action[2] = 1.0 
                # Force Low Altitude (Test A* Pathfinding)
                # action[2] = -1.0 
                
                # 5. Step the Environment
                # This runs 30 physics ticks per 1 RL step!
                obs, reward, terminated, truncated, info = env.step(action)
                
                step += 1
            
            # Auto-reset happens at top of loop, but let's reset explicitly
            obs, info = env.reset()
            
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()