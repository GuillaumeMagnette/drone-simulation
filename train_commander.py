import os
import glob
import time
import re
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from drone_env_commander import DroneEnvCommander

# --- CONFIGURATION ---
LOAD_FROM_LATEST = True    # Set TRUE to resume, FALSE to start new
TOTAL_TIMESTEPS = 2_000_000 
NUM_ENVS = 4               
LEARNING_RATE = 3e-4
ENTROPY_COEF = 0.001       # Low entropy for precision

BASE_DIR = "models/PPO"
LOG_DIR = "logs"

def get_latest_experiment_folder(base_dir):
    """Finds the PPO_X folder with the highest number."""
    if not os.path.exists(base_dir): return None
    folders = glob.glob(f"{base_dir}_*")
    if not folders: return None
    
    # Sort by number suffix
    def get_num(fmt): 
        return int(fmt.split('_')[-1]) if fmt.split('_')[-1].isdigit() else 0
    
    folders.sort(key=get_num, reverse=True)
    return folders[0]

def get_latest_checkpoint(folder):
    """Finds the .zip file with the highest step count in the folder."""
    files = glob.glob(f"{folder}/*.zip")
    if not files: return None
    
    def get_steps(filename):
        # Extract number from 'commander_120000_steps.zip'
        match = re.search(r'_(\d+)_steps', filename)
        return int(match.group(1)) if match else 0
        
    files.sort(key=get_steps, reverse=True)
    return files[0]

def create_new_experiment_folder(base_dir):
    counter = 1
    while True:
        new_dir = f"{base_dir}_{counter}"
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            return new_dir
        counter += 1

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 1. Setup Environment
    # Can use DummyVecEnv to avoid multiprocessing issues with Pygame state
    env = make_vec_env(
        lambda: DroneEnvCommander(render_mode=None, n_agents=2), 
        n_envs=NUM_ENVS,
        vec_env_cls=SubprocVecEnv 
    )

    model = None
    experiment_dir = ""
    reset_timesteps = True

    # 2. Logic to Load or Create
    if LOAD_FROM_LATEST:
        latest_folder = get_latest_experiment_folder(BASE_DIR)
        if latest_folder:
            latest_checkpoint = get_latest_checkpoint(latest_folder)
            if latest_checkpoint:
                print(f"--- RESUMING TRAINING ---")
                print(f"Loading: {latest_checkpoint}")
                
                # Load the model
                model = PPO.load(latest_checkpoint, env=env, device="cpu")
                
                # Continue saving in the SAME folder
                experiment_dir = latest_folder
                reset_timesteps = False 
            else:
                print("Found folder but no checkpoints. Starting fresh.")
        else:
            print("No previous experiments found. Starting fresh.")

    # 3. Create New if not loaded
    if model is None:
        print("--- STARTING NEW TRAINING ---")
        experiment_dir = create_new_experiment_folder(BASE_DIR)
        print(f"Created: {experiment_dir}")
        
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
            tensorboard_log=LOG_DIR,
            device="cpu"
        )

    # 4. Setup Callbacks
    # Save freq needs to be divided by num_envs
    real_save_freq = 50000 // NUM_ENVS
    
    checkpoint_callback = CheckpointCallback(
        save_freq=real_save_freq,
        save_path=experiment_dir,
        name_prefix="commander"
    )

    # 5. Train
    print(f"Targeting {TOTAL_TIMESTEPS} steps...")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=checkpoint_callback,
            progress_bar=True,
            reset_num_timesteps=reset_timesteps # Crucial for resuming correct logs
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving safe exit...")
        model.save(f"{experiment_dir}/commander_interrupted")

    # 6. Save Final
    model.save(f"{experiment_dir}/commander_final")
    env.close()
    print("Done.")

if __name__ == "__main__":
    main()