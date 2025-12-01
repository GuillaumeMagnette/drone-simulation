import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import os
from grid import Grid
from agent import Agent
from projectile import Projectile
from turret import Turret
from algorithm import a_star_algorithm

class DroneEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None):
        super(DroneEnv, self).__init__()
        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 800
        self.PIXEL_SIZE = 10
        self.ROWS = int(self.SCREEN_WIDTH / self.PIXEL_SIZE)
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # --- SCALE ---
        agent_size = 14 
        self.agent = Agent(200, 200, agent_size)
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        # --- UPDATE: 28 INPUTS ---
        # 8 Lidar + 2 Vel + 2 GPS + 8 Bullets + 8 Neighbors = 28
        self.observation_space = spaces.Box(low=-1, high=1, shape=(28,), dtype=np.float32)

        self.grid = Grid(self.ROWS, self.SCREEN_WIDTH)
        self.target = pygame.Rect(600, 600, 40, 40)
        self.projectiles = []
        self.turrets = []
        self.walls = []
        
        # Config
        self.repath_interval = 20
        self.max_steps = 2000
        
        # Logging
        self.cum_reward_dist = 0
        self.cum_reward_time = 0
        self.cum_penalty_collision = 0
        self.cum_reward_win = 0
        self.episode_count = 0
        self.termination_reason = "Timeout"
        
        self.log_file = "training_log.csv"
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write("Episode,DistReward,TimeReward,CollReward,TotalReward,Reason\n")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Log Logic
        if self.episode_count > 0:
            total = self.cum_reward_dist + self.cum_reward_time + self.cum_penalty_collision + self.cum_reward_win
            with open(self.log_file, "a") as f:
                f.write(f"{self.episode_count},{self.cum_reward_dist:.2f},{self.cum_reward_time:.2f},{self.cum_penalty_collision},{total:.2f},{self.termination_reason}\n")

        self.cum_reward_dist = 0
        self.cum_reward_time = 0
        self.cum_penalty_collision = 0
        self.cum_reward_win = 0
        self.episode_count += 1
        self.termination_reason = "Timeout"
        
        self.current_step = 0
        self.repath_steps = 0
        
        # Reset World
        self.grid = Grid(self.ROWS, self.SCREEN_WIDTH)
        self._randomize_walls()
        self.walls = self.grid.get_obstacle_rects()
        
        self.projectiles = []
        self.turrets = []
        
        # SPAWN 0 - 2 - 8 TURRETS (Training Curriculum)
        for _ in range(8): 
            self._spawn_turret()

        self.agent.velocity[:] = 0
        self.agent.acceleration[:] = 0
        self.agent.path = []
        
        self._spawn_entities()
        self._recalc_path()
        
        self.last_dist = np.linalg.norm(np.array(self.target.center) - self.agent.position)
        
        return self._get_obs(), {}

    def step(self, action):
        # 1. Increment Counters
        self.current_step += 1
        self.repath_steps += 1
        
        # 2. Apply Physics
        force = action * self.agent.max_force
        self.agent.apply_force(force)
        self.agent.update_physics(0.016)
        
        # Sync hitbox
        self.agent.rect.center = self.agent.position
        
        # --- 3. HARDCORE LETHAL WALLS (No Sliding) ---
        hit_wall = False
        
        # A. Screen Boundaries
        if (self.agent.rect.left < 0 or self.agent.rect.right > self.SCREEN_WIDTH or
            self.agent.rect.top < 0 or self.agent.rect.bottom > self.SCREEN_HEIGHT):
            hit_wall = True
            
        # B. Obstacles (Buildings & Turrets)
        if not hit_wall: 
            if self.agent.rect.collidelist(self.walls) != -1:
                hit_wall = True

        # NOTE: We REMOVED self.agent._handle_collisions()
        # The penalty block below handles the result.

        # 4. Combat & Projectiles Logic
        for t in self.turrets:
            t.update(0.016, [self.agent], self.projectiles, self.walls)
            
        hit_bullet = False
        for p in self.projectiles[:]:
            p.update(0.016, self.walls)
            if not p.active: 
                self.projectiles.remove(p)
                continue
            
            if self.agent.rect.colliderect(p.rect):
                hit_bullet = True
                self.projectiles.remove(p)

        # 5. Pathing Logic
        if self.repath_steps >= self.repath_interval:
            self.repath_steps = 0
            self._recalc_path()

        if self.agent.path:
            t_node = self.agent.path[0]
            t_pos = np.array([t_node.x+7, t_node.y+7])
            if np.linalg.norm(t_pos - self.agent.position) < 20:
                self.agent.path.pop(0)
        if not self.agent.path: self._recalc_path()

        # 6. Generate Observation
        observation = self._get_obs()

        # 7. Rewards Calculation
        step_reward = 0
        terminated = False
        truncated = False
        
        # A. Progress Reward
        cur_dist = np.linalg.norm(np.array(self.target.center) - self.agent.position)
        progress = self.last_dist - cur_dist
        self.last_dist = cur_dist
        
        # Slightly reduced progress reward to emphasize survival
        step_reward += progress * 0.1 
        self.cum_reward_dist += progress * 0.1
        
        # B. Time Penalty
        step_reward -= 0.05
        self.cum_reward_time -= 0.05
        
        # C. Trajectory Pain (INCREASED WEIGHT)
        danger_penalty = 0
        for p in self.projectiles:
            # Visual Check
            if not self.agent._has_line_of_sight(p.position, self.walls):
                continue

            vec_to_agent = self.agent.position - p.position
            dist_raw = np.linalg.norm(vec_to_agent)
            
            if dist_raw < 150.0:
                forward_dist = np.dot(vec_to_agent, p.direction)
                if forward_dist > 0:
                    sq_term = (dist_raw**2) - (forward_dist**2)
                    miss_dist = np.sqrt(max(0, sq_term))
                    safety_radius = 25.0
                    
                    if miss_dist < safety_radius:
                        intensity = 1.0 - (miss_dist / safety_radius)
                        proximity_multiplier = 1.0 - (dist_raw / 150.0)
                        danger_penalty -= (intensity * proximity_multiplier)
                        
        # WEIGHT INCREASED: Was 0.5, Now 1.0 (Same importance as movement)
        step_reward += (danger_penalty * 1.0)

        # D. Stall Penalty (Strict)
        speed = np.linalg.norm(self.agent.velocity)
        # If moving slow AND not at target AND not dead yet
        if speed < 30.0 and cur_dist > 50.0 and not hit_wall and not hit_bullet:
            step_reward -= 0.5 
        
        # E. Wall Hugging Penalty (Optional Polish)
        # 0-7 are Lidar rays. If < 0.15 (very close), apply stress.
        if np.min(observation[0:8]) < 0.15:
            step_reward -= 0.1

        # --- TERMINAL STATES ---

        # 1. Wall Death (The New Reality)
        if hit_wall:
            step_reward -= 50
            self.cum_penalty_collision -= 50
            terminated = True
            self.termination_reason = "Crashed" # Wall Death
            # print("   -> CRASHED (Wall)")

        # 2. Bullet Death
        elif hit_bullet:
            step_reward -= 50
            self.cum_penalty_collision -= 50
            terminated = True
            self.termination_reason = "Died" # Shot
            # print("   -> DIED (Shot)")
            
        # 3. Victory
        elif self.agent.rect.colliderect(self.target):
            step_reward += 100
            self.cum_reward_win += 100
            terminated = True
            self.termination_reason = "Success"
            # print("   -> SUCCESS")

        # 4. Timeout
        if not terminated and self.current_step >= self.max_steps:
            truncated = True
            self.termination_reason = "Timeout"

        if self.render_mode == "human": self.render()
        return observation, step_reward, terminated, truncated, {}

    def _get_obs(self):
        # 1. Lidar (8)
        lidar = self.agent.cast_rays(self.walls, num_rays=8, max_dist=200)
        
        # 2. Velocity (2)
        vel = self.agent.velocity / self.agent.max_speed
        
        # 3. GPS (2)
        if self.agent.path:
            node = self.agent.path[0]
            wp = np.array([node.x + 7, node.y + 7])
        else: wp = np.array(self.target.center)
        vec_wp = wp - self.agent.position
        d_wp = np.linalg.norm(vec_wp)
        if d_wp > 0: vec_wp /= d_wp
        
        # --- UPDATE: 8 SECTORS ---
        # 4. Bullet Sectors (8)
        vis_bullets = [p for p in self.projectiles if self.agent._has_line_of_sight(p.position, self.walls)]
        bul_sec = self.agent.get_sector_readings(vis_bullets, radius=300.0, num_sectors=8)
        
        # 5. Neighbor Sectors (8)
        # (Empty for single training, but must match shape)
        nei_sec = np.ones(8, dtype=np.float32) 
        
        return np.concatenate([lidar, vel, vec_wp, bul_sec, nei_sec]).astype(np.float32)

    def _randomize_walls(self):
        rows = self.grid.rows
        block_size = 12
        street_width = 6
        for x in range(2, rows - block_size, block_size + street_width):
            for y in range(2, rows - block_size, block_size + street_width):
                b_w = np.random.randint(block_size - 4, block_size)
                b_h = np.random.randint(block_size - 4, block_size)
                for i in range(b_w):
                    for j in range(b_h):
                        if x+i < rows and y+j < rows:
                            self.grid.grid[x+i][y+j].make_obstacle()

    def _spawn_entities(self):
        self.walls = self.grid.get_obstacle_rects()
        # Add turrets to walls for collision
        for t in self.turrets: self.walls.append(t.rect)
        
        while True:
            rx = np.random.randint(50, self.SCREEN_WIDTH - 50)
            ry = np.random.randint(50, self.SCREEN_HEIGHT - 50)
            self.agent.position = np.array([float(rx), float(ry)])
            self.agent.rect.center = self.agent.position
            if self.agent.rect.collidelist(self.walls) == -1: break
            
        while True:
            tx = np.random.randint(50, self.SCREEN_WIDTH - 50)
            ty = np.random.randint(50, self.SCREEN_HEIGHT - 50)
            self.target.topleft = (tx, ty)
            if self.target.collidelist(self.walls) == -1 and not self.target.colliderect(self.agent.rect): break

    def _spawn_turret(self):
        while True:
            # Random position
            x = np.random.randint(50, 750)
            y = np.random.randint(50, 750)
            
            # Snap to grid
            gx = (x // 10) * 10
            gy = (y // 10) * 10
            
            # --- FIX: DISTANCE CHECK ---
            # Create a temporary vector for the proposed turret location
            turret_pos = np.array([gx + 10, gy + 10]) # Center of turret
            dist_to_agent = np.linalg.norm(turret_pos - self.agent.position)
            
            # If too close to start (e.g. 300px), try again
            if dist_to_agent < 300.0:
                continue
            # ---------------------------

            # Check overlap with walls
            test_rect = pygame.Rect(gx, gy, 20, 20)
            if test_rect.collidelist(self.walls) == -1:
                t = Turret(gx, gy, 20)
                self.turrets.append(t)
                self.walls.append(t.rect)
                break

    def _recalc_path(self):
        start = self.grid.get_node_from_pos(self.agent.position)
        end = self.grid.get_node_from_pos(self.target.center)
        if start and end and not start.is_obstacle:
            path = a_star_algorithm(None, self.grid, start, end)
            if path: self.agent.path = path

    def render(self):
        if self.render_mode != "human": return
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: quit()

        self.screen.fill((255, 255, 255))
        self.grid.draw(self.screen)
        pygame.draw.rect(self.screen, (0, 255, 0), self.target)
        for t in self.turrets: t.draw(self.screen)
        for p in self.projectiles: p.draw(self.screen)
        self.agent.draw_lidar(self.screen, self.walls)
        self.agent.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(60)
        
    def close(self):
        if self.screen: pygame.quit()