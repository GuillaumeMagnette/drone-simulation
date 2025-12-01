import os
# Force numpy and torch to use one thread per process
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["IN_MPI"] = "1" 


import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from drone_env import DroneEnv
import os

if __name__ == '__main__':
    # Log Dirs
    models_dir = "models/PPO7"
    log_dir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # --- TURBO CONFIG ---
    NUM_CPU = 8  # Number of parallel simulations
    
    # render_mode=None ensures HEADLESS mode (Fast)
    env_kwargs = {'render_mode': None}

    # Create the Vector Environment
    env = make_vec_env(DroneEnv, n_envs=NUM_CPU, env_kwargs=env_kwargs, vec_env_cls=SubprocVecEnv)

    # Load or New
    model_path = f"{models_dir}/drone_physics.zip"
    if os.path.exists(model_path):
        print("--- LOADING EXISTING BRAIN ---")
        model = PPO.load(model_path, env=env, device='cpu')
    else:
        print("--- CREATING NEW 28-INPUT BRAIN ---")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device='cpu')

    # Train
    TIMESTEPS = 50000
    iters = 0
    print(f"--- STARTING TRAINING ON {NUM_CPU} CORES ---")
    
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{models_dir}/drone_physics")
        print(f"Saved Iteration {iters} (Total Steps: {iters * TIMESTEPS})")

    env.close()