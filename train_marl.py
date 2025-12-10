import os
# CRITICAL: Thread locking for SubprocVecEnv stability
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from drone_env_marl_25d import DroneEnvMARL25D

# ============================================
# CURRICULUM CONFIGURATION
# ============================================
# START HERE -> 5.0 (Flight School)
# ONCE AVG REWARD > 100 -> Switch to 5.1 (Combat)
CURRICULUM_STAGE = 5.0

STAGES = {
    # Phase 5: The Hive Mind (MARL 2.5D)
    
    # 5.0: Flight School & Energy Management
    # Goal: Learn to Hover (Fight Gravity) and Navigate the Urban Maze.
    # No Missiles. Just +20 Reward for targets and -50 for Crashing.
    5.0: {"name": "Hive Flight School", "threats": False},

    # 5.1: The Gauntlet (SEAD)
    # Goal: Use Altitude Masking (NOE) to survive the SAM.
    # Missiles Active.
    5.1: {"name": "Hive Combat", "threats": True},
}

# ============================================
# TRAINING SETUP
# ============================================
NUM_ENVS = 16           
TOTAL_STEPS = 10_000_000 

class TacticalStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True
    
    def _on_rollout_end(self) -> None:
        if self.episode_rewards:
            avg_rew = sum(self.episode_rewards[-100:]) / min(len(self.episode_rewards), 100)
            stage_name = STAGES[CURRICULUM_STAGE]['name']
            print(f"  >> [{stage_name}] Avg Team Reward: {avg_rew:.1f} | Steps: {self.num_timesteps}")

# Factory function that injects curriculum settings
def make_env(stage_config):
    def _init():
        env = DroneEnvMARL25D(render_mode=None)
        # Inject Stage Config
        env.active_threats = stage_config["threats"]
        return env
    return _init

def main():
    models_dir = "models/PPO_MARL_25D"
    log_dir = "logs_marl"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Load Stage
    stage_conf = STAGES[CURRICULUM_STAGE]
    print(f"\n=== STARTING HIVE MIND TRAINING: STAGE {CURRICULUM_STAGE} ===")
    print(f"Scenario: {stage_conf['name']}")
    print(f"Active Threats: {stage_conf['threats']}")
    
    # Try parallel processing
    try:
        env = make_vec_env(make_env(stage_conf), n_envs=NUM_ENVS, vec_env_cls=SubprocVecEnv)
    except Exception as e:
        print(f"Warning: Multiprocessing error ({e}). Using sequential.")
        env = make_vec_env(make_env(stage_conf), n_envs=NUM_ENVS, vec_env_cls=DummyVecEnv)
    
    # NETWORK ARCHITECTURE
    # We use a larger network because 3 agents * 45 inputs is complex state
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
    
    model_path = f"{models_dir}/hive_mind.zip"
    
    if os.path.exists(model_path):
        print(f"Loading existing Hive Mind: {model_path}")
        model = PPO.load(model_path, env=env)
    else:
        print("Initializing NEW Hive Mind...")
        model = PPO(
            "MlpPolicy", 
            env, 
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4, 
            n_steps=2048, 
            batch_size=512, 
            ent_coef=0.01, # Encourage exploration
            gamma=0.99,
            verbose=1, 
            tensorboard_log=log_dir, 
            device="cpu"
        )
    
    callback = TacticalStatsCallback()
    
    print("Training loop started...")
    try:
        while True:
            model.learn(total_timesteps=100_000, reset_num_timesteps=False, callback=callback)
            model.save(f"{models_dir}/hive_mind")
            print("Model Saved.")
    except KeyboardInterrupt:
        model.save(f"{models_dir}/hive_mind")
        env.close()
        print("Training stopped manually.")

if __name__ == "__main__":
    main()