from stable_baselines3 import PPO
from multi_agent_env import MultiAgentSwarm

# 1. Initialize the Multi-Agent Arena
sim = MultiAgentSwarm(num_drones=8) # 8 Drones!

# 2. Load the Brain
model_path = "models/PPO3/drone_physics"
print("Loading Brain...")
# We need a dummy env just to load the model architecture if using .load
# BUT, PPO.load usually works without env if just predicting.
model = PPO.load(model_path, device='cpu')

print("--- SWARM ACTIVATED ---")
print("Press 'R' to Reset Scenario")

sim.reset()

while True:
    # The 'step' function inside MultiAgentSwarm handles the loop
    # for all agents using the provided 'model'
    is_running = sim.step(model)
    
    if not is_running:
        break