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
CURRICULUM_STAGE = 3.0

STAGES = {
    # Phase 1: Basics
    0:   {"name": "Nav Basics",      "map": "sparse", "threats": 0, "bait": False, "respawn": False, "dynamic": False},
    1:   {"name": "Evasion Arena",   "map": "arena",  "threats": 1, "bait": False, "respawn": True,  "dynamic": True}, # Moving target in Arena prevents edge camping!
    
    # Phase 2: Scavenger Hunt (The Fix)
    # 2.0: Scavenger Hunt. Run around collecting points while dodging.
    2.0: {"name": "Scavenger Hunt",  "map": "sparse", "threats": 1, "bait": False, "respawn": True,  "dynamic": True},
    
    # 2.1: Final Exam. One fixed target.
    2.1: {"name": "Terrain Mission", "map": "sparse", "threats": 1, "bait": False, "respawn": True,  "dynamic": False},
    
    # Phase 3: Urban
    3:   {"name": "Urban Scavenger", "map": "urban",  "threats": 1, "bait": False, "respawn": True,  "dynamic": True},
    4:   {"name": "Urban Mission",   "map": "urban",  "threats": 2, "bait": False, "respawn": True,  "dynamic": False},
}


# ============================================
# PERFORMANCE CONFIG
# ============================================
NUM_ENVS = 24           # 24 Envs is usually stable for desktop CPUs
USE_SUBPROC = True      

class FPSCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        return True
    
    def _on_rollout_end(self) -> None:
        if self.episode_rewards:
            avg_reward = sum(self.episode_rewards[-100:]) / min(len(self.episode_rewards), 100)
            avg_length = sum(self.episode_lengths[-100:]) / min(len(self.episode_lengths), 100)
            print(f"  Avg Reward: {avg_reward:.1f} | Avg Length: {avg_length:.0f} | Stage {CURRICULUM_STAGE}")

def make_env(stage_config):
    def _init():
        from drone_env_patched import DroneEnv
        env = DroneEnv(render_mode=None)
        
        env.default_map_type = stage_config["map"]
        env.default_num_interceptors = stage_config["threats"]
        
        env.use_bait = stage_config["bait"] 
        env.respawn_threats = stage_config["respawn"]
        # NEW
        env.dynamic_target = stage_config["dynamic"]
        
        env.log_file = None 
        return env
    return _init

def main():
    models_dir = "models/PPO_Tactical_Enhanced_Optimized_v2"
    log_dir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    stage = STAGES.get(CURRICULUM_STAGE, STAGES[2])
    
    print(f"\n=== FAST TRAINER: STAGE {CURRICULUM_STAGE} ===")
    print(f"Scenario: {stage['name']}")
    print(f"Map: {stage['map']} | Threats: {stage['threats']} | Bait Drone: {stage['bait']}")
    
    vec_env_cls = SubprocVecEnv if USE_SUBPROC else DummyVecEnv
    
    # Wrap in try-except to handle potential Subproc failures gracefully
    try:
        env = make_vec_env(make_env(stage), n_envs=NUM_ENVS, vec_env_cls=vec_env_cls)
    except Exception as e:
        print(f"Warning: Multiprocessing error ({e}). Falling back to sequential.")
        env = make_vec_env(make_env(stage), n_envs=NUM_ENVS, vec_env_cls=DummyVecEnv)
    
    # Tuned for Continuous Control with high FPS
    # Larger batch size stabilizes gradients for complex wall evasion
    N_STEPS = 1024 
    BATCH_SIZE = 128
    
    model_path = f"{models_dir}/drone_tactical.zip"
    device = "cpu" # MLP is faster on CPU for this observation size

    # Define custom policy kwargs for a bigger brain
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                         net_arch=dict(pi=[128, 128], vf=[128, 128]))
    
    if os.path.exists(model_path):
        print(f"Loading existing model: {model_path}")
        model = PPO.load(model_path, env=env, device=device)
        model.n_steps = N_STEPS
        model.batch_size = BATCH_SIZE
        model.rollout_buffer.buffer_size = N_STEPS
    else:
        print("Initializing NEW model...")
        model = PPO("MlpPolicy", env, 
                   policy_kwargs=policy_kwargs, # <--- ADD THIS
                   learning_rate=3e-4, 
                   n_steps=N_STEPS, 
                   batch_size=BATCH_SIZE, 
                   verbose=1, 
                   tensorboard_log=log_dir, 
                   device=device)
    
    TIMESTEPS_PER_ITER = 100_000
    callback = FPSCallback()
    
    print("Starting training loop...")
    try:
        while True:
            model.learn(total_timesteps=TIMESTEPS_PER_ITER, reset_num_timesteps=False, callback=callback)
            model.save(f"{models_dir}/drone_tactical")
            print(f"Saved model. Total Steps: {model.num_timesteps}")
    except KeyboardInterrupt:
        model.save(f"{models_dir}/drone_tactical")
        env.close()
        print("Training stopped and model saved.")

if __name__ == "__main__":
    main()