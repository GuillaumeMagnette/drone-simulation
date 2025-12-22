import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from drone_env_commander import DroneEnvCommander

# --- CONFIGURATION ---
TOTAL_TIMESTEPS = 1_000_000 
NUM_ENVS = 4               
LEARNING_RATE = 3e-4
ENTROPY_COEF = 0.02        # Increased slightly to prevent getting stuck in local minima

# --- AUTO-VERSIONING FUNCTION ---
def get_experiment_dir(base_dir):
    """Finds the next available directory name (e.g., PPO_1, PPO_2)."""
    if not os.path.exists(base_dir):
        return base_dir
    
    counter = 1
    while True:
        new_dir = f"{base_dir}_{counter}"
        if not os.path.exists(new_dir):
            return new_dir
        counter += 1

def main():
    # 1. Setup Directories
    base_model_dir = "models/PPO"
    experiment_dir = get_experiment_dir(base_model_dir)
    log_dir = "logs"
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"--- STARTING TRAINING ---")
    print(f"Saving to: {experiment_dir}")

    # 2. Create Vectorized Env
    env = make_vec_env(
        lambda: DroneEnvCommander(render_mode=None, n_agents=1), 
        n_envs=NUM_ENVS,
        vec_env_cls=SubprocVecEnv
    )

    # 3. Define Model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=LEARNING_RATE,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ENTROPY_COEF,
        tensorboard_log=log_dir,
        device="cpu"
    )

    # 4. Checkpoint Callback (Math Fix)
    # If we want to save every 50k REAL steps, we divide by NUM_ENVS
    real_save_freq = 50000 // NUM_ENVS
    
    checkpoint_callback = CheckpointCallback(
        save_freq=real_save_freq,
        save_path=experiment_dir,
        name_prefix="commander"
    )

    # 5. Train
    print(f"Training for {TOTAL_TIMESTEPS} steps...")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=checkpoint_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # 6. Save Final
    model.save(f"{experiment_dir}/commander_final")
    env.close()

if __name__ == "__main__":
    main()