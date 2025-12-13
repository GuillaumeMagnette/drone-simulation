"""
EMERGENT SWARM VIEWER v4 - With Frozen Partner Visualization
=============================================================

Visual validation tool that shows:
- Learner agent (BLUE) 
- Frozen partner agents (GREEN/CYAN)
- How they coordinate (or don't!)

Controls:
    [1-5]   Switch curriculum stage
    [V]     Toggle visuals (off = turbo speed)
    [+/-]   Simulation speed
    [R]     Reset episode
    [D]     Toggle deterministic mode
    [F]     Toggle frozen partners (all learner vs mixed)
    [ESC]   Quit
"""

import pygame
import numpy as np
import os

# Check for model loading capability
try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("Warning: stable_baselines3 not found, using random actions")

from drone_env_marl_v3 import DroneEnvMARL_V3, BUILDING_HEIGHT

# Model paths
MODELS_DIR = "models/SwarmV4"

# Stage configurations matching train_swarm_v4.py (NEW CURRICULUM)
# Key change: Stages 2-3 now have threats (evasion training)
STAGES = {
    1.0: {
        "name": "Solo Navigation (No Threats)",
        "n_agents": 1, "n_frozen": 0,
        "threats": False, "max_missiles": 0,
        "sam_rotation_speed": 2.0, "sam_lock_time": 0.5, "missile_speed": 550,
    },
    2.0: {
        "name": "Solo Evasion (Easy)",
        "n_agents": 1, "n_frozen": 0,
        "threats": True, "max_missiles": 1,
        "sam_rotation_speed": 1.0, "sam_lock_time": 1.0, "missile_speed": 350,
    },
    3.0: {
        "name": "Solo Evasion (Hard)",
        "n_agents": 1, "n_frozen": 0,
        "threats": True, "max_missiles": 1,
        "sam_rotation_speed": 2.0, "sam_lock_time": 0.5, "missile_speed": 550,
    },
    4.0: {
        "name": "Duo Combat",
        "n_agents": 2, "n_frozen": 0,
        "threats": True, "max_missiles": 1,
        "sam_rotation_speed": 2.0, "sam_lock_time": 0.5, "missile_speed": 550,
    },
    5.0: {
        "name": "Squad Combat (Full)",
        "n_agents": 3, "n_frozen": 0,
        "threats": True, "max_missiles": 2,
        "sam_rotation_speed": 2.0, "sam_lock_time": 0.5, "missile_speed": 550,
    },
}


class SwarmViewer:
    """
    Viewer that can show both learner and frozen agents with different colors.
    """
    
    def __init__(self):
        pygame.init()
        
        # Auto-detect latest available stage
        self.current_stage = self._detect_latest_stage()
        self.stage_config = STAGES[self.current_stage]
        
        print(f"Auto-detected stage: {self.current_stage}")
        
        # Create environment
        self.env = DroneEnvMARL_V3(
            render_mode=None,  # We'll do custom rendering
            n_agents=self.stage_config["n_agents"]
        )
        self.env.active_threats = self.stage_config["threats"]
        self.env.max_missiles = self.stage_config["max_missiles"]
        
        # Display
        self.screen_size = 1200
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Swarm Viewer v4 - Frozen Partners")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.small_font = pygame.font.SysFont(None, 18)
        
        # Models
        self.learner_model = None
        self.frozen_model = None
        self.load_models()
        
        # State
        self.visual_mode = True
        self.deterministic = True
        self.use_frozen = True  # Toggle between all-learner and mixed mode
        self.steps_per_frame = 1
        self.episode_reward = 0
        self.episode_count = 0
        self.recent_rewards = []
        
        # Reset
        self.obs, _ = self.env.reset()
    
    def _detect_latest_stage(self):
        """Auto-detect the latest stage with a saved model."""
        latest_stage = 1.0  # Default
        
        for stage in sorted(STAGES.keys(), reverse=True):
            path = f"{MODELS_DIR}/swarm_v4_stage_{stage:.1f}.zip"
            if os.path.exists(path):
                latest_stage = stage
                break
        
        return latest_stage
    
    def load_models(self):
        """Load learner and frozen models for current stage."""
        if not HAS_SB3:
            return
        
        stage = self.current_stage
        
        # Learner model (current stage)
        learner_path = f"{MODELS_DIR}/swarm_v4_stage_{stage:.1f}.zip"
        if os.path.exists(learner_path):
            try:
                self.learner_model = PPO.load(learner_path, device="cpu")
                print(f"✓ Loaded learner: {learner_path}")
            except Exception as e:
                print(f"✗ Failed to load learner: {e}")
                self.learner_model = None
        else:
            # Try previous stages
            for s in sorted(STAGES.keys(), reverse=True):
                if s <= stage:
                    path = f"{MODELS_DIR}/swarm_v4_stage_{s:.1f}.zip"
                    if os.path.exists(path):
                        try:
                            self.learner_model = PPO.load(path, device="cpu")
                            print(f"✓ Loaded learner from stage {s}: {path}")
                            break
                        except:
                            continue
            else:
                print("✗ No learner model found, using random")
                self.learner_model = None
        
        # Frozen model (only if stage uses frozen partners)
        n_frozen = self.stage_config.get("n_frozen", 0)
        if n_frozen > 0 and stage > 1.0:
            frozen_path = f"{MODELS_DIR}/swarm_v4_stage_{stage - 1.0:.1f}.zip"
            if os.path.exists(frozen_path):
                try:
                    self.frozen_model = PPO.load(frozen_path, device="cpu")
                    print(f"✓ Loaded frozen: {frozen_path}")
                except Exception as e:
                    print(f"✗ Failed to load frozen: {e}")
                    self.frozen_model = self.learner_model  # Fallback
            else:
                self.frozen_model = self.learner_model  # Use same model
        elif n_frozen == 0 and self.stage_config["n_agents"] > 1:
            # Shared policy mode - all agents use learner model
            self.frozen_model = self.learner_model
            print(f"  Shared policy mode: All {self.stage_config['n_agents']} agents use learner model")
        else:
            self.frozen_model = None
    
    def switch_stage(self, new_stage):
        """Switch to a different curriculum stage."""
        if new_stage not in STAGES:
            return
        
        self.current_stage = new_stage
        self.stage_config = STAGES[new_stage]
        
        print(f"\n>>> Stage {new_stage}: {self.stage_config['name']}")
        
        # Recreate environment if agent count changed
        if self.env.n_agents != self.stage_config["n_agents"]:
            self.env.close()
            self.env = DroneEnvMARL_V3(
                render_mode=None,
                n_agents=self.stage_config["n_agents"]
            )
        
        # Apply curriculum parameters
        self.env.active_threats = self.stage_config["threats"]
        self.env.max_missiles = self.stage_config["max_missiles"]
        
        # SAM difficulty parameters
        self.env.sam_rotation_speed = self.stage_config.get("sam_rotation_speed", 2.0)
        self.env.sam_lock_required = self.stage_config.get("sam_lock_time", 0.5)
        self.env.missile_speed = self.stage_config.get("missile_speed", 550)
        
        if self.stage_config["threats"]:
            print(f"    SAM: rotation={self.env.sam_rotation_speed} rad/s, "
                  f"lock={self.env.sam_lock_required}s, "
                  f"missile_speed={self.env.missile_speed}")
        
        # Reload models
        self.load_models()
        
        # Reset
        self.obs, _ = self.env.reset()
        self.episode_reward = 0
    
    def get_action(self, agent_idx, obs):
        """Get action for an agent (learner or frozen)."""
        n_frozen = self.stage_config["n_frozen"] if self.use_frozen else 0
        
        # Agent 0 is always the learner (or all are learners if use_frozen=False)
        if agent_idx == 0 or n_frozen == 0:
            if self.learner_model is not None:
                action, _ = self.learner_model.predict(obs, deterministic=self.deterministic)
                return action
        else:
            # Frozen partner
            if self.frozen_model is not None:
                action, _ = self.frozen_model.predict(obs, deterministic=True)
                return action
        
        # Fallback to random
        return np.random.uniform(-1, 1, 3).astype(np.float32)
    
    def step(self):
        """Execute one environment step."""
        n_agents = self.stage_config["n_agents"]
        
        # Build actions for all agents
        actions = []
        for i in range(n_agents):
            agent_obs = self.obs[i * 50:(i + 1) * 50]
            action = self.get_action(i, agent_obs)
            actions.append(action)
        
        full_action = np.concatenate(actions)
        
        # Step environment
        self.obs, reward, terminated, truncated, info = self.env.step(full_action)
        self.episode_reward += reward
        
        if terminated or truncated:
            self.recent_rewards.append(self.episode_reward)
            if len(self.recent_rewards) > 20:
                self.recent_rewards.pop(0)
            
            self.episode_count += 1
            avg = np.mean(self.recent_rewards) if self.recent_rewards else 0
            print(f"[{self.stage_config['name']}] Ep {self.episode_count}: {self.episode_reward:.0f} (avg: {avg:.0f})")
            
            self.episode_reward = 0
            self.obs, _ = self.env.reset()
    
    def render(self):
        """Custom rendering with learner/frozen distinction."""
        self.screen.fill((30, 30, 30))
        
        env = self.env
        
        # Buildings
        for w in env.urban_walls:
            pygame.draw.rect(self.screen, (50, 50, 60), w)
            pygame.draw.rect(self.screen, (70, 70, 80), w.inflate(-10, -10))
        
        # SAM site
        pygame.draw.circle(self.screen, (200, 0, 0), env.hazard_source, 15)
        pygame.draw.circle(self.screen, (150, 0, 0), env.hazard_source, 25, 2)
        
        # SAM aim direction (visual feedback for tracking)
        if env.active_threats:
            aim_length = 80 + (env.sam_lock_time / env.sam_lock_required) * 120
            aim_end = (
                int(env.hazard_source[0] + env.sam_aim_direction[0] * aim_length),
                int(env.hazard_source[1] + env.sam_aim_direction[1] * aim_length)
            )
            
            # Color based on lock status
            if env.sam_lock_time >= env.sam_lock_required:
                aim_color = (255, 0, 0)  # RED = LOCKED
                line_width = 4
            elif env.sam_lock_time > 0:
                lock_ratio = env.sam_lock_time / env.sam_lock_required
                aim_color = (255, int(255 * (1 - lock_ratio)), 0)
                line_width = 2
            else:
                aim_color = (100, 100, 0)  # Dim yellow = searching
                line_width = 1
            
            pygame.draw.line(self.screen, aim_color, env.hazard_source, aim_end, line_width)
            
            # Detection cone
            cone_length = 150
            cone_angle = 0.52
            import math
            left_dir = (
                env.sam_aim_direction[0] * math.cos(cone_angle) - env.sam_aim_direction[1] * math.sin(cone_angle),
                env.sam_aim_direction[0] * math.sin(cone_angle) + env.sam_aim_direction[1] * math.cos(cone_angle)
            )
            right_dir = (
                env.sam_aim_direction[0] * math.cos(-cone_angle) - env.sam_aim_direction[1] * math.sin(-cone_angle),
                env.sam_aim_direction[0] * math.sin(-cone_angle) + env.sam_aim_direction[1] * math.cos(-cone_angle)
            )
            left_end = (int(env.hazard_source[0] + left_dir[0] * cone_length),
                       int(env.hazard_source[1] + left_dir[1] * cone_length))
            right_end = (int(env.hazard_source[0] + right_dir[0] * cone_length),
                        int(env.hazard_source[1] + right_dir[1] * cone_length))
            pygame.draw.line(self.screen, (80, 80, 0), env.hazard_source, left_end, 1)
            pygame.draw.line(self.screen, (80, 80, 0), env.hazard_source, right_end, 1)
        
        # Target
        pygame.draw.circle(self.screen, (0, 255, 0), env.target.center, 20)
        pygame.draw.circle(self.screen, (0, 150, 0), env.target.center, 30, 2)
        
        # Agents with role-based coloring
        n_frozen = self.stage_config["n_frozen"] if self.use_frozen else 0
        
        for i, ag in enumerate(env.agents):
            if not ag.active:
                continue
            
            cx, cy = int(ag.position[0]), int(ag.position[1])
            is_high = ag.position[2] > BUILDING_HEIGHT
            scale = 10 + (ag.position[2] / 12.0)
            
            # Color based on role
            if i == 0:
                # LEARNER - Blue shades
                color = (100, 150, 255) if not is_high else (150, 200, 255)
                role = "L"
            elif i < n_frozen + 1:
                # FROZEN - Green shades  
                color = (0, 200, 100) if not is_high else (100, 255, 200)
                role = "F"
            else:
                # Additional learners (shared policy mode)
                color = (100, 150, 255) if not is_high else (150, 200, 255)
                role = "L"
            
            # Draw agent
            pygame.draw.circle(self.screen, color, (cx, cy), int(scale))
            pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), int(scale), 1)
            
            # Height bar
            h_bar = int(ag.position[2] / 3)
            pygame.draw.rect(self.screen, (255, 255, 0), (cx + 15, cy - h_bar, 4, h_bar))
            
            # Role label
            label = self.small_font.render(f"{role}{i}", True, (255, 255, 255))
            self.screen.blit(label, (cx - 8, cy - 25))
        
        # Missiles
        for m in env.interceptors:
            cx, cy = int(m.position[0]), int(m.position[1])
            color = (255, 50, 50) if m.position[2] < BUILDING_HEIGHT else (255, 150, 150)
            pygame.draw.circle(self.screen, color, (cx, cy), 6)
            
            # Trail
            if np.linalg.norm(m.velocity) > 0:
                vel_norm = m.velocity[:2] / (np.linalg.norm(m.velocity[:2]) + 0.1)
                trail_end = (cx - int(vel_norm[0] * 20), cy - int(vel_norm[1] * 20))
                pygame.draw.line(self.screen, (255, 100, 0), (cx, cy), trail_end, 2)
        
        # HUD
        self._draw_hud()
        
        pygame.display.flip()
    
    def _draw_hud(self):
        """Draw heads-up display."""
        y = 10
        
        # Stage info
        text = f"Stage {self.current_stage}: {self.stage_config['name']}"
        label = self.font.render(text, True, (255, 255, 255))
        self.screen.blit(label, (10, y))
        y += 25
        
        # Mode info
        n_frozen = self.stage_config["n_frozen"] if self.use_frozen else 0
        mode = f"Agents: {self.stage_config['n_agents']} (Learner: 1, Frozen: {n_frozen})"
        label = self.font.render(mode, True, (200, 200, 200))
        self.screen.blit(label, (10, y))
        y += 25
        
        # Episode reward
        text = f"Episode Reward: {self.episode_reward:.0f}"
        label = self.font.render(text, True, (255, 255, 100))
        self.screen.blit(label, (10, y))
        y += 25
        
        # Average reward
        if self.recent_rewards:
            avg = np.mean(self.recent_rewards)
            text = f"Avg (last 20): {avg:.0f}"
            label = self.font.render(text, True, (100, 255, 100))
            self.screen.blit(label, (10, y))
        y += 25
        
        # Controls reminder
        y = self.screen_size - 100
        controls = [
            "[1-5] Stage  [R] Reset  [D] Deterministic",
            "[F] Toggle Frozen  [V] Visuals  [+/-] Speed",
            "[ESC] Quit"
        ]
        for ctrl in controls:
            label = self.small_font.render(ctrl, True, (150, 150, 150))
            self.screen.blit(label, (10, y))
            y += 18
        
        # Legend
        y = 10
        x = self.screen_size - 150
        
        pygame.draw.circle(self.screen, (100, 150, 255), (x, y + 8), 8)
        label = self.small_font.render("Learner", True, (200, 200, 200))
        self.screen.blit(label, (x + 15, y))
        y += 20
        
        pygame.draw.circle(self.screen, (0, 200, 100), (x, y + 8), 8)
        label = self.small_font.render("Frozen", True, (200, 200, 200))
        self.screen.blit(label, (x + 15, y))
    
    def run(self):
        """Main loop."""
        running = True
        
        while running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    
                    # Stage switching
                    elif event.key == pygame.K_1:
                        self.switch_stage(1.0)
                    elif event.key == pygame.K_2:
                        self.switch_stage(2.0)
                    elif event.key == pygame.K_3:
                        self.switch_stage(3.0)
                    elif event.key == pygame.K_4:
                        self.switch_stage(4.0)
                    elif event.key == pygame.K_5:
                        self.switch_stage(5.0)
                    
                    # Reset
                    elif event.key == pygame.K_r:
                        self.obs, _ = self.env.reset()
                        self.episode_reward = 0
                        print("Reset.")
                    
                    # Deterministic toggle
                    elif event.key == pygame.K_d:
                        self.deterministic = not self.deterministic
                        print(f"Deterministic: {self.deterministic}")
                    
                    # Frozen toggle
                    elif event.key == pygame.K_f:
                        self.use_frozen = not self.use_frozen
                        mode = "Mixed (Learner + Frozen)" if self.use_frozen else "All Learner"
                        print(f"Mode: {mode}")
                    
                    # Visual toggle
                    elif event.key == pygame.K_v:
                        self.visual_mode = not self.visual_mode
                        print(f"Visuals: {self.visual_mode}")
                    
                    # Speed
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        self.steps_per_frame = min(self.steps_per_frame + 1, 20)
                        print(f"Speed: {self.steps_per_frame}x")
                    elif event.key == pygame.K_MINUS:
                        self.steps_per_frame = max(self.steps_per_frame - 1, 1)
                        print(f"Speed: {self.steps_per_frame}x")
            
            # Simulation
            for _ in range(self.steps_per_frame):
                self.step()
            
            # Render
            if self.visual_mode:
                self.render()
                self.clock.tick(60)
        
        self.env.close()
        pygame.quit()


def print_help():
    print("\n" + "="*60)
    print("  SWARM VIEWER v4 - Frozen Partners")
    print("="*60)
    print("\n  STAGES:")
    for s, cfg in STAGES.items():
        frozen = f"({cfg['n_frozen']} frozen)" if cfg['n_frozen'] > 0 else "(all learning)"
        print(f"    [{int(s)}] {cfg['name']} - {cfg['n_agents']} agents {frozen}")
    print("\n  VISUAL KEY:")
    print("    BLUE  = Learner agent (training)")
    print("    GREEN = Frozen agent (fixed policy)")
    print("    Bright = High altitude (radar visible)")
    print("    Dark = Low altitude (NOE)")
    print("="*60 + "\n")


if __name__ == "__main__":
    print_help()
    viewer = SwarmViewer()
    viewer.run()
