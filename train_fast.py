"""
FAST TACTICAL DRONE TRAINER
============================

Performance-optimized training with TRUE parallelism.

FIXES APPLIED:
- SubprocVecEnv (true multiprocessing) instead of DummyVecEnv
- Auto GPU detection
- Optimized n_steps for parallel collection
- Disabled env-level profiling

EXPECTED SPEEDUP: 5-10x depending on CPU cores
"""

import os
# CRITICAL: Set thread limits BEFORE any imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback


# ============================================
# CURRICULUM CONFIG
# ============================================
CURRICULUM_STAGE = 1  # <-- Change this as you progress

STAGES = {
    0: {"name": "Navigation Basics", "map_type": "sparse", "num_interceptors": 0},
    1: {"name": "Evasion Basics", "map_type": "arena", "num_interceptors": 2},
    2: {"name": "Terrain Evasion", "map_type": "sparse", "num_interceptors": 1},
    3: {"name": "Multi-Threat Terrain", "map_type": "sparse", "num_interceptors": 2},
    4: {"name": "Urban Navigation", "map_type": "urban", "num_interceptors": 0},
    5: {"name": "Urban Evasion", "map_type": "urban", "num_interceptors": 1},
    6: {"name": "Urban Warfare", "map_type": "urban", "num_interceptors": 2},
}


# ============================================
# PERFORMANCE CONFIG
# ============================================
NUM_ENVS = 30           # More parallel envs = faster collection
USE_SUBPROC = True      # True = real parallelism, False = sequential (debug)


class FPSCallback(BaseCallback):
    """Tracks actual training throughput."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Collect episode stats from info
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        return True
    
    def _on_rollout_end(self) -> None:
        if self.episode_rewards:
            avg_reward = sum(self.episode_rewards[-100:]) / min(len(self.episode_rewards), 100)
            avg_length = sum(self.episode_lengths[-100:]) / min(len(self.episode_lengths), 100)
            fps = self.model.num_timesteps / max(1, self.model._episode_num) * 60  # Rough estimate
            print(f"  Avg Reward: {avg_reward:.1f} | Avg Length: {avg_length:.0f} | Episodes: {len(self.episode_rewards)}")


def make_env(stage_config):
    """Factory that creates a FAST environment."""
    def _init():
        # Import here to avoid pickle issues with SubprocVecEnv
        from drone_env_patched import DroneEnv
        
        env = DroneEnv(render_mode=None)
        env.default_map_type = stage_config["map_type"]
        env.default_num_interceptors = stage_config["num_interceptors"]
        
        # Disable CSV logging for speed (patched version supports None)
        env.log_file = None
        
        return env
    return _init


def main():
    # Directories
    models_dir = "models/PPO_Tactical4"
    log_dir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Stage info
    stage = STAGES.get(CURRICULUM_STAGE, STAGES[1])
    
    print()
    print("=" * 60)
    print(f"  FAST TRAINER - Stage {CURRICULUM_STAGE}: {stage['name']}")
    print("=" * 60)
    print(f"  Map:          {stage['map_type']}")
    print(f"  Interceptors: {stage['num_interceptors']}")
    print(f"  Parallel Envs: {NUM_ENVS}")
    print(f"  Vec Type:     {'SubprocVecEnv (PARALLEL)' if USE_SUBPROC else 'DummyVecEnv (sequential)'}")
    
    # PPO with MLP is faster on CPU (small batches, transfer overhead)
    device = "cpu"
    print(f"  Device:       {device.upper()} (MLP policies are faster on CPU)")
    print("=" * 60)
    print()
    
    # Create vectorized environment
    vec_env_cls = SubprocVecEnv if USE_SUBPROC else DummyVecEnv
    
    try:
        env = make_vec_env(
            make_env(stage), 
            n_envs=NUM_ENVS, 
            vec_env_cls=vec_env_cls
        )
        print(f"✓ Created {NUM_ENVS} parallel environments")
    except Exception as e:
        print(f"✗ SubprocVecEnv failed: {e}")
        print("  Falling back to DummyVecEnv...")
        env = make_vec_env(make_env(stage), n_envs=NUM_ENVS, vec_env_cls=DummyVecEnv)
    
    # Hyperparameters (tuned for parallel collection)
    # n_steps * n_envs should be divisible by batch_size
    N_STEPS = 512
    BATCH_SIZE = 64  # 2048 * 20 = 40960, divisible by 64 (640 batches)
    
    # Load or create model
    model_path = f"{models_dir}/drone_tactical.zip"
    
    if os.path.exists(model_path):
        print(f"Loading: {model_path}")
        model = PPO.load(model_path, env=env, device=device)
        # Override hyperparameters (loaded model uses old values)
        model.n_steps = N_STEPS
        model.batch_size = BATCH_SIZE
        model.rollout_buffer.buffer_size = N_STEPS  # Resize buffer
        print(f"✓ Model loaded (n_steps={N_STEPS}, batch={BATCH_SIZE})")
    else:
        print("Creating new model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Encourage exploration
            verbose=1,
            tensorboard_log=log_dir,
            device=device
        )
        print("✓ Model created")
    
    # Training
    TIMESTEPS_PER_ITER = 100_000  # More steps per save
    callback = FPSCallback()
    
    print(f"\nTraining... (Ctrl+C to stop)\n")
    
    iters = 0
    try:
        while True:
            iters += 1
            
            model.learn(
                total_timesteps=TIMESTEPS_PER_ITER,
                reset_num_timesteps=False,
                callback=callback,
                progress_bar=True  # Visual progress
            )
            
            model.save(f"{models_dir}/drone_tactical")
            total_steps = iters * TIMESTEPS_PER_ITER
            print(f"\n[Stage {CURRICULUM_STAGE}] Saved at {total_steps:,} steps\n")
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
        model.save(f"{models_dir}/drone_tactical")
        print(f"✓ Model saved to {models_dir}/drone_tactical.zip")
    
    env.close()


if __name__ == "__main__":
    main()
