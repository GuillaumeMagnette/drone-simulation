"""
EMERGENT SWARM TRAINING v4 - Progressive Curriculum with Frozen Partners
=========================================================================

Key Innovation:
- Stage 1: Train 1 agent (obs=50)
- Stage 2: Train 1 agent, but 2nd agent uses FROZEN Stage 1 policy
- Stage 3: Train 1 agent, but agents 2&3 use FROZEN Stage 2 policy
- Later stages: All agents share the same (training) policy

This allows:
1. Progressive curriculum (1 -> 2 -> 3 agents)
2. Weight transfer (the "learner" architecture stays consistent at 50 obs)
3. Emergent coordination (learner must adapt to frozen partners)

The trick: We keep the learner's observation space at 50 (single agent view),
but the ENVIRONMENT has multiple agents. The frozen partners act autonomously.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import numpy as np
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium import spaces

# Import base environment
from drone_env_marl_v3 import DroneEnvMARL_V3

# ============================================
# CURRICULUM CONFIGURATION
# ============================================
# 
# NEW DESIGN PHILOSOPHY:
# 1. Navigation first (no threats)
# 2. Evasion second (solo + threats, gradually harder)
# 3. Coordination last (multi-agent + threats)
#
# Key insight: Agents must learn to SURVIVE before learning to COORDINATE.
# The old curriculum skipped evasion entirely, so agents learned
# "go straight to target" which becomes "go straight to death" when
# missiles are added.
#

CURRICULUM = {
    # ─────────────────────────────────────────────────────────
    # PHASE 1: NAVIGATION
    # ─────────────────────────────────────────────────────────
    1.0: {
        "name": "Solo Navigation",
        "n_agents": 1,
        "n_frozen": 0,
        "threats": False,
        "max_missiles": 0,
        # SAM params (not used but included for consistency)
        "sam_rotation_speed": 1.0,
        "sam_lock_time": 0.5,
        "missile_speed": 550,
        "advance_reward": 40.0,
    },
    
    # ─────────────────────────────────────────────────────────
    # PHASE 2: EVASION (the critical missing piece!)
    # ─────────────────────────────────────────────────────────
    2.0: {
        "name": "Solo Evasion (Easy)",
        "n_agents": 1,
        "n_frozen": 0,
        "threats": True,  # MISSILES NOW!
        "max_missiles": 1,
        # EASY SAM: Slow rotation, long lock time, slow missiles
        "sam_rotation_speed": 1.0,   # Half speed rotation
        "sam_lock_time": 1.0,        # Double lock time (more time to react)
        "missile_speed": 350,        # Slower missiles (can outrun)
        "advance_reward": 20.0,      # Lower threshold (deaths expected)
    },
    3.0: {
        "name": "Solo Evasion (Hard)",
        "n_agents": 1,
        "n_frozen": 0,
        "threats": True,
        "max_missiles": 1,
        # NORMAL SAM: faster missile
        "sam_rotation_speed": 1.0,
        "sam_lock_time": 0.5,
        "missile_speed": 550,
        "advance_reward": 25.0,
    },
    
    # ─────────────────────────────────────────────────────────
    # PHASE 3: COORDINATION (now with evasion skills!)
    # ─────────────────────────────────────────────────────────
    4.0: {
        "name": "Duo Combat",
        "n_agents": 2,
        "n_frozen": 0,  # Shared policy
        "threats": True,
        "max_missiles": 2,
        "sam_rotation_speed": 1.0,
        "sam_lock_time": 0.3,
        "missile_speed": 550,
        "advance_reward": 35.0,
    },
    5.0: {
        "name": "Squad Combat (Full)",
        "n_agents": 3,
        "n_frozen": 0,
        "threats": True,
        "max_missiles": 2,
        "sam_rotation_speed": 2.0,
        "sam_lock_time": 0.3,
        "missile_speed": 550,
        "advance_reward": None,  # Final stage
    },
}

CURRENT_STAGE = 1.0
NUM_ENVS = 24
CHECKPOINT_INTERVAL = 100_000
AUTO_ADVANCE = False

# Live Reload Configuration
# When enabled, frozen partners reload the latest checkpoint periodically.
# This creates a "self-play" dynamic where partners improve alongside the learner.
#
# IMPORTANT: The reload counter is per-environment, but checkpoints are saved
# based on TOTAL timesteps (across all NUM_ENVS). So we divide by NUM_ENVS.
#
# Example: CHECKPOINT_INTERVAL=100k with NUM_ENVS=16
#   → Each env does ~6,250 steps between checkpoints
#   → LIVE_RELOAD_INTERVAL should be ~6,250 to reload once per checkpoint
#
LIVE_RELOAD_ENABLED = True
LIVE_RELOAD_INTERVAL = CHECKPOINT_INTERVAL // NUM_ENVS  # Reload once per checkpoint


class HybridSwarmEnv(gym.Env):
    """
    Wrapper that presents a SINGLE-AGENT observation space to the learner,
    while running a multi-agent environment under the hood.
    
    Frozen partners are controlled by a separate (frozen) policy.
    
    Key Feature: Live-reload support!
    The frozen policy can be reloaded during training to always use
    the latest checkpoint. This creates a self-play dynamic.
    """
    
    def __init__(self, n_agents, n_frozen, frozen_model_path, threats, max_missiles,
                 sam_rotation_speed=2.0, sam_lock_time=0.5, missile_speed=550,
                 live_reload_path=None):
        super().__init__()
        
        self.n_agents = n_agents
        self.n_frozen = n_frozen
        self.n_learners = n_agents - n_frozen  # Should be 1 for stages 1-3
        
        # Create the underlying multi-agent environment
        self.env = DroneEnvMARL_V3(render_mode=None, n_agents=n_agents)
        self.env.active_threats = threats
        self.env.max_missiles = max_missiles
        
        # SAM difficulty parameters (curriculum-controlled)
        self.env.sam_rotation_speed = sam_rotation_speed
        self.env.sam_lock_required = sam_lock_time
        self.env.missile_speed = missile_speed
        # Note: target is ALWAYS the SAM position now (handled in env)
        
        # Single-agent observation/action space (for the learner)
        self.obs_dim = 50  # Single agent observation
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )
        
        # Frozen policy management
        self.frozen_policy = None
        self.frozen_model_path = frozen_model_path
        self.live_reload_path = live_reload_path  # Path to reload from (current training checkpoint)
        self.reload_counter = 0
        self.reload_interval = LIVE_RELOAD_INTERVAL  # Use config value
        
        # Initial load
        self._load_frozen_policy()
    
    def _load_frozen_policy(self):
        """Load or reload the frozen policy."""
        # Priority: live_reload_path > frozen_model_path
        path_to_load = None
        
        if self.live_reload_path and os.path.exists(self.live_reload_path):
            path_to_load = self.live_reload_path
        elif self.frozen_model_path and os.path.exists(self.frozen_model_path):
            path_to_load = self.frozen_model_path
        
        if path_to_load and self.n_frozen > 0:
            try:
                self.frozen_policy = PPO.load(path_to_load, device="cpu")
            except Exception as e:
                # Silently fail on reload errors (file might be mid-write)
                if self.frozen_policy is None:
                    print(f"    WARNING: Could not load frozen policy: {e}")
    
    def reset(self, seed=None, options=None):
        # Reset underlying environment
        full_obs, info = self.env.reset(seed=seed, options=options)
        
        # Return only the learner's observation (agent 0)
        learner_obs = full_obs[:self.obs_dim]
        return learner_obs, info
    
    def step(self, learner_action):
        # Periodic reload of frozen policy (self-play dynamic)
        if self.n_frozen > 0:
            self.reload_counter += 1
            if self.live_reload_path and self.reload_counter >= self.reload_interval:
                self.reload_counter = 0
                self._load_frozen_policy()
        
        # Build full action array for all agents
        full_action = np.zeros(self.n_agents * 3, dtype=np.float32)
        
        # Agent 0 = Learner (always uses the provided action)
        full_action[0:3] = learner_action
        
        if self.n_frozen > 0:
            # FROZEN MODE: Agents 1+ use frozen policy
            for i in range(1, self.n_agents):
                frozen_obs = self.env._get_single_agent_obs(i)
                
                if self.frozen_policy is not None:
                    frozen_action, _ = self.frozen_policy.predict(
                        frozen_obs, deterministic=True
                    )
                else:
                    frozen_action = np.random.uniform(-1, 1, 3).astype(np.float32)
                
                full_action[i*3:(i+1)*3] = frozen_action
        else:
            # SHARED POLICY MODE: Agents 1+ use the SAME policy as agent 0
            # We need to query the current learner policy for each agent's obs
            # But we don't have access to it here... 
            # 
            # SOLUTION: Store actions for all agents from the last policy call.
            # This requires a different architecture - see SharedPolicyWrapper below.
            #
            # For now, we use a simpler approach: all agents mirror agent 0's action
            # scaled by their own observation similarity. This is a placeholder.
            #
            # TODO: Implement proper shared policy in VecEnv wrapper
            #
            # TEMPORARY: Use the frozen policy (which should be latest checkpoint)
            # This effectively makes all agents use the same trained policy
            if self.frozen_policy is not None:
                for i in range(1, self.n_agents):
                    agent_obs = self.env._get_single_agent_obs(i)
                    agent_action, _ = self.frozen_policy.predict(
                        agent_obs, deterministic=False  # Allow exploration
                    )
                    full_action[i*3:(i+1)*3] = agent_action
            else:
                # No policy available - random actions as fallback
                for i in range(1, self.n_agents):
                    full_action[i*3:(i+1)*3] = np.random.uniform(-1, 1, 3).astype(np.float32)
        
        # Step the environment with all actions
        full_obs, reward, terminated, truncated, info = self.env.step(full_action)
        
        # Return only learner's observation
        learner_obs = full_obs[:self.obs_dim]
        
        return learner_obs, reward, terminated, truncated, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()


class SharedPolicySwarmEnv(gym.Env):
    """
    For later stages: All agents share the SAME policy.
    
    This is standard MARL parameter sharing - we present the full
    observation space but the policy is applied to each agent independently.
    """
    
    def __init__(self, n_agents, threats, max_missiles):
        super().__init__()
        
        self.n_agents = n_agents
        self.obs_dim = 50  # Per-agent observation
        
        # Create underlying environment
        self.env = DroneEnvMARL_V3(render_mode=None, n_agents=n_agents)
        self.env.active_threats = threats
        self.env.max_missiles = max_missiles
        
        # Observation: Stack of all agent observations
        # But we process them ONE AT A TIME through the same policy
        # So the learner still sees (50,) obs
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )
        
        # Track which agent we're "being" this step
        self.current_agent_idx = 0
        self.cached_obs = None
        self.pending_actions = []
    
    def reset(self, seed=None, options=None):
        full_obs, info = self.env.reset(seed=seed, options=options)
        self.cached_obs = full_obs.reshape(self.n_agents, self.obs_dim)
        self.current_agent_idx = 0
        self.pending_actions = []
        
        # Return first agent's observation
        return self.cached_obs[0].copy(), info
    
    def step(self, action):
        # Collect action for current agent
        self.pending_actions.append(action.copy())
        self.current_agent_idx += 1
        
        # If we haven't collected all actions yet, return dummy values
        # The SB3 VecEnv will call step() n_agents times per "real" step
        if self.current_agent_idx < self.n_agents:
            # Return next agent's observation, no reward yet
            return (
                self.cached_obs[self.current_agent_idx].copy(),
                0.0,
                False,
                False,
                {}
            )
        
        # All actions collected - execute the real step
        full_action = np.concatenate(self.pending_actions)
        full_obs, reward, terminated, truncated, info = self.env.step(full_action)
        
        # Reset for next round
        self.cached_obs = full_obs.reshape(self.n_agents, self.obs_dim)
        self.current_agent_idx = 0
        self.pending_actions = []
        
        # Return first agent's new observation
        return self.cached_obs[0].copy(), reward, terminated, truncated, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()


class CurriculumCallback(BaseCallback):
    """Tracks training progress and handles curriculum advancement."""
    
    def __init__(self, stage_config, verbose=1):
        super().__init__(verbose)
        self.stage_config = stage_config
        self.episode_rewards = deque(maxlen=100)
        self.episode_targets = deque(maxlen=100)
        self.episode_saturations = deque(maxlen=100)
        self.episode_deaths = deque(maxlen=100)
        self.total_episodes = 0
        self.should_advance = False
        
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.total_episodes += 1
                
                # Get detailed stats if available
                if "episode_stats" in info:
                    stats = info["episode_stats"]
                    self.episode_targets.append(stats.get("targets_hit", 0))
                    self.episode_saturations.append(stats.get("saturation_hits", 0))
                    self.episode_deaths.append(stats.get("deaths", 0))
        return True
    
    def _on_rollout_end(self) -> None:
        if len(self.episode_rewards) < 10:
            return
            
        avg_reward = np.mean(self.episode_rewards)
        avg_targets = np.mean(self.episode_targets) if self.episode_targets else 0
        avg_saturations = np.mean(self.episode_saturations) if self.episode_saturations else 0
        avg_deaths = np.mean(self.episode_deaths) if self.episode_deaths else 0
        
        stage_name = self.stage_config["name"]
        
        print(f"\n  [{stage_name}] Steps: {self.num_timesteps:,}")
        print(f"    Avg Reward (last 100): {avg_reward:.1f}")
        print(f"    Avg Targets/Ep: {avg_targets:.1f}")
        print(f"    Avg Saturation Hits: {avg_saturations:.2f}")  # Key metric for coordination!
        print(f"    Avg Deaths/Ep: {avg_deaths:.1f}")
        print(f"    Episodes: {self.total_episodes}")
        
        # Check advancement
        if AUTO_ADVANCE and self.stage_config.get("advance_reward"):
            if avg_reward >= self.stage_config["advance_reward"]:
                if len(self.episode_rewards) >= 50:
                    print(f"\n  *** ADVANCEMENT CRITERIA MET! ***")
                    self.should_advance = True


def make_hybrid_env(stage_config, frozen_model_path, live_reload_path=None):
    """Factory for HybridSwarmEnv with optional live reload."""
    def _init():
        # For shared policy mode (n_frozen=0), we STILL need the frozen policy
        # to be loaded so all agents can use it. The "frozen" policy in this case
        # is actually the latest checkpoint of the learning policy.
        effective_frozen_path = frozen_model_path
        effective_reload_path = live_reload_path
        
        # In shared policy mode, make sure we have a policy to use
        if stage_config["n_frozen"] == 0 and live_reload_path:
            effective_frozen_path = live_reload_path
            effective_reload_path = live_reload_path
        
        return HybridSwarmEnv(
            n_agents=stage_config["n_agents"],
            n_frozen=stage_config["n_frozen"],
            frozen_model_path=effective_frozen_path,
            threats=stage_config["threats"],
            max_missiles=stage_config["max_missiles"],
            sam_rotation_speed=stage_config.get("sam_rotation_speed", 2.0),
            sam_lock_time=stage_config.get("sam_lock_time", 0.5),
            missile_speed=stage_config.get("missile_speed", 550),
            live_reload_path=effective_reload_path
        )
    return _init


def get_model_path(models_dir, stage):
    return f"{models_dir}/swarm_v4_stage_{stage:.1f}.zip"


def main():
    models_dir = "models/SwarmV4"
    log_dir = "logs_swarm_v4"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    current_stage = CURRENT_STAGE
    
    print("\n" + "="*60)
    print("  EMERGENT SWARM TRAINING v4")
    print("  Progressive Curriculum with Frozen Partners")
    print("="*60)
    
    while True:
        stage_config = CURRICULUM[current_stage]
        
        print(f"\n>>> STAGE {current_stage}: {stage_config['name']}")
        print(f"    Agents: {stage_config['n_agents']} (Frozen: {stage_config['n_frozen']})")
        print(f"    Threats: {stage_config['threats']} (Max missiles: {stage_config['max_missiles']})")
        if stage_config['threats']:
            print(f"    SAM Difficulty: rotation={stage_config.get('sam_rotation_speed', 2.0)} rad/s, "
                  f"lock={stage_config.get('sam_lock_time', 0.5)}s, "
                  f"missile_speed={stage_config.get('missile_speed', 550)}")
        
        # Determine frozen model path (from previous stage)
        frozen_model_path = None
        if current_stage > 1.0:
            prev_stage = current_stage - 1.0
            frozen_model_path = get_model_path(models_dir, prev_stage)
        
        # LIVE RELOAD: Frozen partners will reload from current checkpoint!
        # This creates a self-play dynamic where the frozen policy improves
        # alongside the learner.
        model_path = get_model_path(models_dir, current_stage)
        
        if LIVE_RELOAD_ENABLED and stage_config["n_frozen"] > 0:
            live_reload_path = model_path
            print(f"    Mode: 1 Learner + {stage_config['n_frozen']} Frozen (reloads every {LIVE_RELOAD_INTERVAL} steps)")
        elif stage_config["n_frozen"] == 0 and stage_config["n_agents"] > 1:
            live_reload_path = model_path
            print(f"    Mode: SHARED POLICY - All {stage_config['n_agents']} agents use same policy")
            print(f"    (Agents 1+ use latest checkpoint, Agent 0 uses live training policy)")
        else:
            live_reload_path = None
            print(f"    Mode: Single agent")
        
        # Create environments
        def make_env():
            return make_hybrid_env(stage_config, frozen_model_path, live_reload_path)()
        
        try:
            from stable_baselines3.common.env_util import make_vec_env
            env = make_vec_env(make_env, n_envs=NUM_ENVS, vec_env_cls=SubprocVecEnv)
            print("    Using: SubprocVecEnv (parallel)")
        except Exception as e:
            print(f"    Warning: {e}")
            env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])
            print("    Using: DummyVecEnv (sequential)")
        
        # Policy architecture (consistent across all stages!)
        policy_kwargs = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])
        )
        
        # Load or create model
        model_path = get_model_path(models_dir, current_stage)
        prev_model_path = get_model_path(models_dir, current_stage - 1.0) if current_stage > 1.0 else None
        
        if os.path.exists(model_path):
            print(f"    Resuming from: {model_path}")
            model = PPO.load(model_path, env=env, device="cpu")
        elif prev_model_path and os.path.exists(prev_model_path):
            print(f"    Bootstrapping from Stage {current_stage - 1.0}: {prev_model_path}")
            model = PPO.load(prev_model_path, env=env, device="cpu")
        else:
            print("    Initializing new model...")
            model = PPO(
                "MlpPolicy",
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=512,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=0,
                tensorboard_log=log_dir,
                device="cpu"
            )
        
        callback = CurriculumCallback(stage_config, verbose=1)
        
        print(f"\n    Training started (obs_space: {env.observation_space.shape})...")
        
        try:
            while True:
                model.learn(
                    total_timesteps=CHECKPOINT_INTERVAL,
                    reset_num_timesteps=False,
                    callback=callback
                )
                
                model.save(model_path)
                print(f"\n    Checkpoint saved: {model_path}")
                
                if callback.should_advance:
                    next_stage = current_stage + 1.0
                    if next_stage in CURRICULUM:
                        print(f"\n>>> ADVANCING TO STAGE {next_stage}!")
                        final_path = f"{models_dir}/swarm_v4_stage_{current_stage:.1f}_final.zip"
                        model.save(final_path)
                        current_stage = next_stage
                        env.close()
                        break
                    else:
                        print("\n>>> FINAL STAGE COMPLETE!")
                        callback.should_advance = False
                        
        except KeyboardInterrupt:
            print("\n\n>>> Training interrupted")
            model.save(model_path)
            print(f"    Model saved: {model_path}")
            env.close()
            
            resp = input("\n    Continue to next stage? (y/n): ")
            if resp.lower() == 'y':
                next_stage = current_stage + 1.0
                if next_stage in CURRICULUM:
                    current_stage = next_stage
                    continue
            break
    
    print("\n>>> Training session ended")


if __name__ == "__main__":
    main()
