"""
TACTICAL DRONE TRAINER - Interceptor Evasion Curriculum
========================================================

Training a reconnaissance drone to reach objectives while evading pursuit.

CURRICULUM PHILOSOPHY:
----------------------
Phase 1: Learn pure evasion (no obstacles to crash into)
Phase 2: Learn terrain exploitation (use cover to break pursuit)
Phase 3: Full operational capability (navigate + evade)

STAGES:
-------
0 - Flight School:    Urban map, 0 threats (navigation baseline)
1 - Evasion Basics:   Arena, 1 interceptor (learn "red = danger")
2 - Evasion Pressure: Arena, 2 interceptors (multi-threat awareness)
3 - Evasion Mastery:  Arena, 3 interceptors (intense pressure)
4 - Terrain Intro:    Sparse, 1 interceptor (use cover)
5 - Terrain Tactics:  Sparse, 2 interceptors (juke into obstacles)
6 - Urban Ops Light:  Urban, 1 interceptor (combine skills)
7 - Urban Ops Full:   Urban, 2 interceptors (mission ready)

HOW TO USE:
-----------
1. Train Stage 0 first (if not done) - navigation baseline
2. Jump to Stage 1 - this is where evasion learning begins
3. Progress when success rate > 60-70%
4. Use test_viewer.py to visually validate

The key insight: Pure evasion (Stage 1-3) teaches reflexes.
Terrain (Stage 4-5) teaches "juke into wall" tactic.
Urban (Stage 6-7) combines everything.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["IN_MPI"] = "1"

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from drone_env import DroneEnv


# ============================================
# CURRICULUM - CHANGE THIS TO PROGRESS
# ============================================
CURRICULUM_STAGE = 0  # <-- Change this as you progress

STAGES = {
    # ===========================================
    # PHASE 1: ISOLATED SKILLS
    # ===========================================
    
    0: {
        "name": "Navigation Basics",
        "map_type": "sparse",        # Easier than urban
        "num_interceptors": 0,
        "desc": "Learn: walls hurt, navigate around obstacles"
    },
    
    1: {
        "name": "Evasion Basics", 
        "map_type": "arena",         # No walls to worry about
        "num_interceptors": 1,       # One SCARY fast interceptor
        "desc": "Learn: red thing WILL catch you, must dodge"
    },
    
    # ===========================================
    # PHASE 2: SKILL INTEGRATION  
    # ===========================================
    
    2: {
        "name": "Terrain Evasion",
        "map_type": "sparse",
        "num_interceptors": 1,
        "desc": "Combine: use obstacles to break pursuit"
    },
    
    3: {
        "name": "Multi-Threat Terrain",
        "map_type": "sparse",
        "num_interceptors": 2,
        "desc": "Handle multiple pursuers with terrain"
    },
    
    # ===========================================
    # PHASE 3: URBAN OPERATIONS
    # ===========================================
    
    4: {
        "name": "Urban Navigation",
        "map_type": "urban",
        "num_interceptors": 0,
        "desc": "Complex navigation through city"
    },
    
    5: {
        "name": "Urban Evasion",
        "map_type": "urban",
        "num_interceptors": 1,
        "desc": "Navigate city while evading pursuit"
    },
    
    6: {
        "name": "Urban Warfare",
        "map_type": "urban", 
        "num_interceptors": 2,
        "desc": "Full operational capability"
    },
    
    # ===========================================
    # PHASE 4: MULTI-AGENT (Future)
    # ===========================================
    
    7: {
        "name": "Swarm Basics",
        "map_type": "arena",
        "num_interceptors": 1,
        "desc": "Multi-agent coordination (Stage M)"
    },
}


def make_env():
    """Factory for curriculum-configured environment."""
    def _init():
        env = DroneEnv(render_mode=None)
        stage = STAGES.get(CURRICULUM_STAGE, STAGES[1])
        env.default_map_type = stage["map_type"]
        env.default_num_interceptors = stage["num_interceptors"]
        return env
    return _init


if __name__ == "__main__":
    # Directories
    models_dir = "models/PPO_Tactical3"
    log_dir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Display info
    stage = STAGES.get(CURRICULUM_STAGE, STAGES[1])
    print()
    print("=" * 60)
    print(f"  STAGE {CURRICULUM_STAGE}: {stage['name']}")
    print("=" * 60)
    print(f"  Map:          {stage['map_type']}")
    print(f"  Interceptors: {stage['num_interceptors']}")
    print(f"  Goal:         {stage['desc']}")
    print("=" * 60)
    print()
    
    # Parallel environments
    NUM_CPU = 20
    env = make_vec_env(make_env(), n_envs=NUM_CPU, vec_env_cls=SubprocVecEnv)
    
    # --- HYPERPARAMETERS ---
    # Tuned for faster learning on navigation tasks
    LEARNING_RATE = 0.0005      # 3x default (0.0003) - faster learning
    N_STEPS = 2048             # Steps per update (default)
    BATCH_SIZE = 64            # Minibatch size (default)
    N_EPOCHS = 10              # Passes over data (default)
    GAMMA = 0.99               # Discount factor (default)
    
    # Load or create model
    model_path = f"{models_dir}/drone_tactical.zip"
    
    if os.path.exists(model_path):
        print("Loading existing model...")
        try:
            model = PPO.load(model_path, env=env, device='cpu')
            # Update learning rate on loaded model
            model.learning_rate = LEARNING_RATE
            print(f"Model loaded! LR set to {LEARNING_RATE}")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Creating new model...")
            model = PPO("MlpPolicy", env, 
                       learning_rate=LEARNING_RATE,
                       n_steps=N_STEPS,
                       batch_size=BATCH_SIZE,
                       n_epochs=N_EPOCHS,
                       gamma=GAMMA,
                       verbose=1, 
                       tensorboard_log=log_dir, 
                       device='cpu')
    else:
        print(f"Creating new model with LR={LEARNING_RATE}...")
        model = PPO("MlpPolicy", env, 
                   learning_rate=LEARNING_RATE,
                   n_steps=N_STEPS,
                   batch_size=BATCH_SIZE,
                   n_epochs=N_EPOCHS,
                   gamma=GAMMA,
                   verbose=1, 
                   tensorboard_log=log_dir, 
                   device='cpu')
    
    # Training loop
    TIMESTEPS = 50000
    iters = 0
    
    print(f"\nTraining on {NUM_CPU} cores. Press Ctrl+C to stop.\n")
    
    try:
        while True:
            iters += 1
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
            model.save(f"{models_dir}/drone_tactical")
            print(f"[Stage {CURRICULUM_STAGE}] Iter {iters} | Steps: {iters * TIMESTEPS:,}")
    
    except KeyboardInterrupt:
        print("\n\nTraining stopped.")
        model.save(f"{models_dir}/drone_tactical")
        print("Model saved.")
        print()
        print("NEXT STEPS:")
        print("  1. Run: python test_viewer.py")
        print("  2. Check success rate")
        print(f"  3. If > 60%, advance to Stage {CURRICULUM_STAGE + 1}")
    
    env.close()
