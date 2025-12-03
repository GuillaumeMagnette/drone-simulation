"""
DRONE ENVIRONMENT - Tactical Reconnaissance Trainer
====================================================

A Gymnasium environment for training evasive drone navigation.

Mission: Reach target while evading enemy interceptors.

Observation Space (36 inputs):
------------------------------
[0-7]   Lidar       - 8-sector wall distances
[8-9]   Velocity    - Current velocity (normalized)
[10-11] GPS         - Direction to waypoint
[12-19] Threats     - 8-sector interceptor distances
[20-27] Projectiles - 8-sector bullet distances (reserved)
[28-35] Neighbors   - 8-sector friendly drone distances (MARL ready)

Map Types:
----------
- 'arena':  Empty field - pure evasion training
- 'sparse': Scattered cover - terrain exploitation
- 'urban':  Operational urban - realistic mission environment
"""

import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import os
from grid import Grid
from agent import Agent
from interceptor import Interceptor
from algorithm import a_star_algorithm

# --- NEW: Simple Dummy Drone for Baiting ---
class DummyDrone:
    def __init__(self, x, y):
        self.position = np.array([float(x), float(y)])
        self.velocity = np.array([0.0, 0.0])
        self.size = 14
        self.rect = pygame.Rect(x, y, 14, 14)
        self.active = False
        self.target_pos = None
        self.speed = 100.0 # Slow and steady

    def update(self, dt):
        if not self.active or self.target_pos is None:
            return
        
        # Simple fly-to behavior
        direction = self.target_pos - self.position
        dist = np.linalg.norm(direction)
        if dist > 5:
            direction /= dist
            self.velocity = direction * self.speed
            self.position += self.velocity * dt
            self.rect.center = self.position

class DroneEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None):
        super(DroneEnv, self).__init__()
        
        # --- SCREEN ---
        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 800
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # --- AGENT ---
        self.agent = Agent(200, 200, size=14)
        self.dummy = DummyDrone(0, 0) # Initialize inactive dummy
        
        # --- SPACES ---
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        # 36 inputs: Lidar(8), Vel(2), GPS(2), Threats(8), Proj(8), Neighbors(8)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(36,), dtype=np.float32)
        
        # --- WORLD ---
        self.ROWS = self.SCREEN_WIDTH // 10
        self.grid = Grid(self.ROWS, self.SCREEN_WIDTH)
        self.target = pygame.Rect(600, 600, 40, 40)
        self.hazard_source = (600, 600)
        self.walls = []
        self.interceptors = []
        
        # --- CONFIG ---
        self.missile_timer = 0.0
        self.missile_interval = 1.0
        self.max_interceptors = 5
        self.use_bait = False # Toggle for bait drone
        
        # Curriculum Defaults
        self.default_map_type = 'arena'
        self.default_num_interceptors = 1
        self.current_map_type = 'arena'
        
        self.repath_interval = 20
        self.max_steps = 2000
        
        # Logging
        self.episode_count = 0
        self.log_file = "training_log.csv"
        self._init_log()

    def _init_log(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write("Episode,Map,Interceptors,DistRew,TimeRew,CollPen,WinRew,Total,Reason\n")

    def _spawn_hazard(self):
        """Spawns the static Missile Battery (SAM Site)."""
        # 1. Arena Mode: Center
        if self.current_map_type == 'arena':
            self.hazard_source = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
            return

        # 2. Terrain/Urban Mode: Random valid location
        for _ in range(100):
            x = np.random.randint(50, self.SCREEN_WIDTH - 50)
            y = np.random.randint(50, self.SCREEN_HEIGHT - 50)
            test_rect = pygame.Rect(x - 20, y - 20, 40, 40)
            
            # Check A: Not inside a wall
            if test_rect.collidelist(self.walls) != -1: continue
            
            # Check B: Not too close to Agent
            dist_to_agent = np.linalg.norm(np.array([x, y]) - self.agent.position)
            if dist_to_agent > 300:
                self.hazard_source = (x, y)
                return
        
        self.hazard_source = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Options defaults
        map_type = self.default_map_type
        num_interceptors = self.default_num_interceptors
        
        # New Feature Flags (Default to Mission Mode settings)
        self.use_bait = False
        self.respawn_threats = True
        # NEW: Replaces "Survival Mode"
        self.dynamic_target = False 
        
        if options:
            map_type = options.get('map_type', self.default_map_type)
            num_interceptors = options.get('num_interceptors', self.default_num_interceptors)
            
            self.use_bait = options.get('use_bait', False)
            self.respawn_threats = options.get('respawn', True)
            # Grab the new flag
            self.dynamic_target = options.get('dynamic_target', False)
        
        self.current_map_type = map_type
        
        # Logging Setup (Same as before)
        if self.episode_count > 0 and self.log_file:
            total = self.cum_reward_dist + self.cum_reward_time + self.cum_penalty_collision + self.cum_reward_win
            with open(self.log_file, "a") as f:
                f.write(f"{self.episode_count},{self.current_map_type},{len(self.interceptors)},"
                       f"{self.cum_reward_dist:.2f},{self.cum_reward_time:.2f},"
                       f"{self.cum_penalty_collision:.2f},{self.cum_reward_win:.2f},"
                       f"{total:.2f},{self.termination_reason}\n")
        
        # Reset Counters
        self.cum_reward_dist = 0
        self.cum_reward_time = 0
        self.cum_penalty_collision = 0
        self.cum_reward_win = 0
        self.episode_count += 1
        self.termination_reason = "Timeout"
        self.current_step = 0
        self.repath_steps = 0
        self.total_missiles_spawned = 0
        
        # Generate Map
        self.grid.clear() 
        if map_type == 'arena': self._generate_arena()
        elif map_type == 'sparse': self._generate_sparse()
        else: self._generate_urban()

        self.grid.apply_costmap()
        self.walls = self.grid.get_obstacle_rects()
        
        # Reset Entities
        self.interceptors = []
        self.missile_timer = 0.0
        self.max_interceptors = num_interceptors
        
        self._spawn_agent_and_target()

        # [ADD/REPLACE THIS BLOCK]
        # Logic: Where do we put the Missile Battery?
        if self.dynamic_target:
            # Scavenger Mode: Spawn hazard in a fixed, valid location
            self._spawn_hazard()
        else:
            # Mission Mode: The Target IS the Hazard (Defended Base)
            self.hazard_source = self.target.center
            
        # [ADD THIS]
        # If we are in Scavenger Mode, move the target immediately 
        # so it doesn't start on top of the battery.
        if self.dynamic_target:
            self._relocate_target()
        
        # Bait/Dummy Setup
        if self.use_bait:
            self.dummy.active = True
            self.dummy.position = self.agent.position + np.array([40.0, 0.0])
            self.dummy.target_pos = np.array(self.target.center)
            self.dummy.rect.center = self.dummy.position
        else:
            self.dummy.active = False
            self.dummy.position = np.array([-100.0, -100.0])
        
        # Agent Init
        self.agent.velocity[:] = 0
        self.agent.acceleration[:] = 0
        self.agent.path = []
        
        # Calculate initial path
        if map_type != 'arena':
            self._recalc_path()

        self.last_dist = np.linalg.norm(np.array(self.target.center) - self.agent.position)
        
        return self._get_obs(), {}
    
    def _relocate_target(self):
        """Moves the target to a new random location away from the agent."""

        # Try to find a spot far from the agent (forces movement)
        for _ in range(20):
            x = np.random.randint(50, self.SCREEN_WIDTH - 50)
            y = np.random.randint(50, self.SCREEN_HEIGHT - 50)
            
            candidate_rect = pygame.Rect(x, y, 40, 40)
            # Check walls
            if candidate_rect.collidelist(self.walls) != -1: continue
            
            dist = np.linalg.norm(np.array([x,y]) - self.agent.position)
            
            # We want a target that is at least 300px away
            if dist > 300:
                self.target.topleft = (x, y)
                return
                
        # Fallback if hard check fails
        x = np.random.randint(50, self.SCREEN_WIDTH - 50)
        y = np.random.randint(50, self.SCREEN_HEIGHT - 50)
        self.target.topleft = (x, y)

    def _get_obs(self):
        # 1. Lidar (8)
        lidar = self.agent.cast_rays(self.walls, num_rays=8, max_dist=200)
        
        # 2. Velocity (2)
        vel = self.agent.velocity / self.agent.max_speed
        
        # 3. GPS (2) - ALWAYS ACTIVE NOW
        # In Scavenger Hunt, we need to see the vector to the new target.
        if self.current_map_type == 'arena':
            # Optimization: Direct vector in empty space
            waypoint = np.array(self.target.center, dtype=float)
            vec_to_wp = waypoint - self.agent.position
            dist_to_wp = np.linalg.norm(vec_to_wp)
            if dist_to_wp > 0: vec_to_wp /= dist_to_wp
            
        elif self.agent.path:
            # Follow A* nodes in complex maps
            node = self.agent.path[0]
            waypoint = np.array([node.x + 7, node.y + 7])
            vec_to_wp = waypoint - self.agent.position
            dist_to_wp = np.linalg.norm(vec_to_wp)
            if dist_to_wp > 0: vec_to_wp /= dist_to_wp
            
        else:
            # Fallback if path failing (direct vector)
            waypoint = np.array(self.target.center, dtype=float)
            vec_to_wp = waypoint - self.agent.position
            dist_to_wp = np.linalg.norm(vec_to_wp)
            if dist_to_wp > 0: vec_to_wp /= dist_to_wp
        
        # 4. Threats (8)
        visible_threats = []
        for interceptor in self.interceptors:
            if not interceptor.alive: continue
            if self.current_map_type == 'arena':
                visible_threats.append(interceptor)
            elif self.agent._has_line_of_sight(interceptor.position, self.walls):
                visible_threats.append(interceptor)
        threat_sectors = self.agent.get_sector_readings(visible_threats, radius=400.0, num_sectors=8)
        
        # 5. Projectiles (8)
        projectile_sectors = np.ones(8, dtype=np.float32)
        
        # 6. Neighbors (8)
        visible_neighbors = []
        if self.dummy.active:
            visible_neighbors.append(self.dummy)
        neighbor_sectors = self.agent.get_sector_readings(visible_neighbors, radius=300.0, num_sectors=8)
        
        return np.concatenate([
            lidar, vel, vec_to_wp, threat_sectors, projectile_sectors, neighbor_sectors
        ]).astype(np.float32)
    

    def step(self, action):
        self.current_step += 1
        self.repath_steps += 1
        
        # --- 1. PHYSICS ---
        force = action * self.agent.max_force
        self.agent.apply_force(force)
        self.agent.update_physics(0.016)
        self.agent.rect.center = self.agent.position
        if self.dummy.active: self.dummy.update(0.016)
        
        hit_wall = False
        if (self.agent.rect.left < 0 or self.agent.rect.right > self.SCREEN_WIDTH or
            self.agent.rect.top < 0 or self.agent.rect.bottom > self.SCREEN_HEIGHT): hit_wall = True   
        if not hit_wall and self.agent.rect.collidelist(self.walls) != -1: hit_wall = True

        # Update Dummy Target if dynamic
        if self.dummy.active and self.dynamic_target:
             self.dummy.target_pos = np.array(self.target.center)

        # --- 2. MISSILE SPAWNING (Standard) ---
        self.missile_timer -= 0.016
        if self.missile_timer <= 0:
            self.missile_timer = self.missile_interval + np.random.uniform(-0.2, 0.3)
            alive_interceptors = sum(1 for i in self.interceptors if i.alive)
            should_spawn = False
            if self.respawn_threats:
                if alive_interceptors < self.max_interceptors: should_spawn = True
            else:
                if self.total_missiles_spawned < self.max_interceptors: should_spawn = True
            if should_spawn:
                # [CHANGE THIS] Check distance to HAZARD SOURCE
                dist_to_hazard = np.linalg.norm(self.agent.position - np.array(self.hazard_source))
                too_close = dist_to_hazard < 75
                can_fire = not too_close
                if can_fire and self.current_map_type != 'arena':
                    # [CHANGE THIS] Check LOS from HAZARD SOURCE
                    can_fire = self.agent._has_line_of_sight(self.hazard_source, self.walls)
                if can_fire:
                    self._spawn_interceptor()
                    self.total_missiles_spawned += 1

        # --- 3. ENTITY UPDATES ---
        hit_interceptor = False
        potential_targets = [self.agent]
        if self.dummy.active: potential_targets.append(self.dummy)
        for interceptor in self.interceptors:
            if not interceptor.alive: continue
            interceptor.update(0.016, potential_targets, self.walls)
            if interceptor.alive and self.agent.rect.colliderect(interceptor.rect):
                hit_interceptor = True
                interceptor._die()
            if self.dummy.active and interceptor.alive and self.dummy.rect.colliderect(interceptor.rect):
                interceptor._die()
                if self.render_mode == "human" and hasattr(self, 'explosion_manager'):
                    self.explosion_manager.add(self.dummy.position[0], self.dummy.position[1])

        # --- 4. PATHFINDING ---
        if self.current_map_type != 'arena':
            if self.repath_steps >= self.repath_interval:
                self.repath_steps = 0
                self._recalc_path()
            if self.agent.path:
                node = self.agent.path[0]
                node_pos = np.array([node.x + 7, node.y + 7])
                if np.linalg.norm(node_pos - self.agent.position) < 20: self.agent.path.pop(0)
            if not self.agent.path: self._recalc_path()

        # ============================================================
        # 5. THE DEFCON SYSTEM (Dynamic Context-Aware Rewards)
        # ============================================================
        
        obs = self._get_obs()
        reward = 0
        terminated = False
        truncated = False
        
        # 1. Distance Reward (Standard)
        cur_dist = np.linalg.norm(np.array(self.target.center) - self.agent.position)
        progress = self.last_dist - cur_dist
        self.last_dist = cur_dist
        reward += progress * 0.15
        self.cum_reward_dist += progress * 0.15
        
        # 2. Time Penalty (Always apply, discourages camping)
        reward -= 0.1 
        self.cum_reward_time -= 0.1

        # 3. Detect Threat Level
        # Check if any live missile is within close range (Combat Range)
        closest_threat_dist = 9999.0
        prediction_time = 0.3
        future_agent_pos = self.agent.position + (self.agent.velocity * prediction_time)
        
        threat_detected = False

        for interceptor in self.interceptors:
            if not interceptor.alive: continue
            dist = np.linalg.norm(self.agent.position - interceptor.position)
            closest_threat_dist = min(closest_threat_dist, dist)
            if dist < 350.0: # Detection Radius
                if self.current_map_type == 'arena' or self.agent._has_line_of_sight(interceptor.position, self.walls):
                    threat_detected = True

        # Trajectory Pain (The "Dodge" Signal)
            vec_to_future = future_agent_pos - interceptor.position
            dist_raw = np.linalg.norm(vec_to_future)
            if dist_raw < 200.0:
                vel_mag = np.linalg.norm(interceptor.velocity)
                if vel_mag > 0:
                    int_dir = interceptor.velocity / vel_mag
                    forward_dist = np.dot(vec_to_future, int_dir)
                    if forward_dist > 0:
                        sq_term = (dist_raw**2) - (forward_dist**2)
                        miss_dist = np.sqrt(max(0, sq_term))
                        safety_radius = 40.0
                        if miss_dist < safety_radius:
                            intensity = 1.0 - (miss_dist / safety_radius)
                            reward -= intensity * 2.0 

        # 4. Penalties (Relaxed when threatened)
        # We can remove the "Arena Ring of Fire" hack now, 
        # because the Dynamic Target forces them to leave the edges!
        
        is_safe = not threat_detected
        
        # Stall Penalty (Only if safe)
        speed = np.linalg.norm(self.agent.velocity)
        if is_safe and speed < 5 and cur_dist > 50.0:
             reward -= 0.5 if self.current_map_type == 'arena' else 1.0

        # Scrape Penalty (Only if safe)
        if is_safe and np.min(obs[0:8]) < 0.15: 
            reward -= 0.2
        
        # Predictive Crash (Don't hit walls at high speed)
        lookahead_vec = self.agent.velocity * 0.2 
        future_pos = self.agent.position + lookahead_vec
        future_rect = self.agent.rect.copy()
        future_rect.center = future_pos
        future_hit = False
        if (future_rect.left < 0 or future_rect.right > self.SCREEN_WIDTH or
            future_rect.top < 0 or future_rect.bottom > self.SCREEN_HEIGHT): future_hit = True
        elif future_rect.collidelist(self.walls) != -1: future_hit = True
        
        if future_hit:
            base_crash_pen = 2.0 * (speed / self.agent.max_speed)
            # If threatened, reduce crash penalty (Panic braking is better than death)
            if threat_detected: reward -= base_crash_pen * 0.5
            else: reward -= base_crash_pen

        # --- TERMINATION ---
        if hit_wall:
            reward -= 50 
            self.cum_penalty_collision -= 50
            terminated = True
            self.termination_reason = "Crashed"
            if self.render_mode == "human" and hasattr(self, 'explosion_manager'):
                self.explosion_manager.add(self.agent.position[0], self.agent.position[1])

        elif hit_interceptor:
            reward -= 50
            self.cum_penalty_collision -= 50
            terminated = True
            self.termination_reason = "Caught"
            if self.render_mode == "human" and hasattr(self, 'explosion_manager'):
                self.explosion_manager.add(self.agent.position[0], self.agent.position[1])
        
        elif self.agent.rect.colliderect(self.target):
            # HIT TARGET
            if self.dynamic_target:
                # SCAVENGER MODE: Reward + Respawn Target
                reward += 20 # Big burst reward
                self.cum_reward_win += 20
                self._relocate_target()
                self._recalc_path() # Path to NEW target
                self.last_dist = np.linalg.norm(np.array(self.target.center) - self.agent.position)
                # DO NOT TERMINATE
            else:
                # CLASSIC MISSION MODE
                reward += 100
                self.cum_reward_win += 100
                terminated = True
                self.termination_reason = "Success"
        
        elif self.current_step >= self.max_steps:
            truncated = True
            self.termination_reason = "Timeout"
        
        if self.render_mode == "human": self.render()
        
        return obs, reward, terminated, truncated, {}

    # ==========================================
    # MAP GENERATORS
    # ==========================================
    
    def _generate_arena(self):
        """Empty arena - pure evasion training."""
        pass  # Grid starts empty
    
    def _generate_sparse(self):
        """Scattered cover - learn terrain exploitation."""
        rows = self.ROWS
        
        # 10-15 random obstacles
        for _ in range(np.random.randint(10, 16)):
            w = np.random.randint(3, 8)
            h = np.random.randint(3, 8)
            x = np.random.randint(5, rows - w - 5)
            y = np.random.randint(5, rows - h - 5)
            
            for i in range(w):
                for j in range(h):
                    if 0 <= x+i < rows and 0 <= y+j < rows:
                        self.grid.grid[x+i][y+j].make_obstacle()
    
    def _generate_urban(self):
        """
        Operational urban environment.
        More realistic than Manhattan maze:
        - Wide main roads (escape routes)
        - Narrow alleys (ambush risk)
        - Open plazas (danger zones)
        - Varying building sizes
        """
        rows = self.ROWS
        
        # Main roads (cross pattern) - guaranteed escape routes
        road_width = 8  # Wide enough for maneuvering
        mid = rows // 2
        
        # Don't block the main roads
        blocked = set()
        for i in range(rows):
            for w in range(road_width):
                blocked.add((mid - road_width//2 + w, i))  # Vertical road
                blocked.add((i, mid - road_width//2 + w))  # Horizontal road
        
        # City blocks with varying sizes
        block_configs = [
            (8, 12),   # Small blocks
            (12, 16),  # Medium blocks
            (16, 22),  # Large blocks
        ]
        
        # Quadrant-based generation (avoid roads)
        quadrants = [
            (2, 2, mid - road_width//2 - 2, mid - road_width//2 - 2),
            (mid + road_width//2 + 2, 2, rows - 2, mid - road_width//2 - 2),
            (2, mid + road_width//2 + 2, mid - road_width//2 - 2, rows - 2),
            (mid + road_width//2 + 2, mid + road_width//2 + 2, rows - 2, rows - 2),
        ]
        
        for qx1, qy1, qx2, qy2 in quadrants:
            # Fill quadrant with buildings
            x = qx1
            while x < qx2 - 5:
                y = qy1
                while y < qy2 - 5:
                    # Random building size
                    min_size, max_size = block_configs[np.random.randint(0, 3)]
                    bw = np.random.randint(min_size - 4, min_size)
                    bh = np.random.randint(min_size - 4, min_size)
                    
                    # Ensure we don't exceed quadrant
                    bw = min(bw, qx2 - x - 2)
                    bh = min(bh, qy2 - y - 2)
                    
                    if bw > 3 and bh > 3:
                        for i in range(bw):
                            for j in range(bh):
                                px, py = x + i, y + j
                                if (px, py) not in blocked and 0 <= px < rows and 0 <= py < rows:
                                    self.grid.grid[px][py].make_obstacle()
                    
                    y += bh + np.random.randint(4, 8)  # Street width varies
                x += np.random.randint(8, 14)
    
    # ==========================================
    # ENTITY SPAWNING
    # ==========================================
    
    def _spawn_agent_and_target(self):
        """Spawn agent and target in valid positions with safety margins."""
        agent_size = self.agent.size
        safety_margin = 30  # Extra space around agent (pixels)
        
        # Agent - spawn with room to maneuver
        for _ in range(100):
            x = np.random.randint(100, self.SCREEN_WIDTH - 100)
            y = np.random.randint(100, self.SCREEN_HEIGHT - 100)
            self.agent.position = np.array([float(x), float(y)])
            self.agent.rect.center = self.agent.position
            
            # Check not just collision, but that there's ROOM around the agent
            # Create an inflated rect to check for nearby walls
            safe_zone = self.agent.rect.inflate(safety_margin * 2, safety_margin * 2)
            
            if safe_zone.collidelist(self.walls) == -1:
                break
        
        # Target (must be far from agent, also away from edges, also with room)
        target_margin = 20
        for _ in range(100):
            x = np.random.randint(100, self.SCREEN_WIDTH - 100)
            y = np.random.randint(100, self.SCREEN_HEIGHT - 100)
            self.target.topleft = (x, y)
            
            dist_to_agent = np.linalg.norm(
                np.array(self.target.center) - self.agent.position
            )
            
            # Inflated check for target too
            target_safe = self.target.inflate(target_margin * 2, target_margin * 2)
            
            if (target_safe.collidelist(self.walls) == -1 and 
                not self.target.colliderect(self.agent.rect) and
                dist_to_agent > 300):
                break
    
    def _spawn_interceptor(self):
        """Spawns a missile near the Hazard Source."""
        # [CHANGE THIS LINE] - It used to be self.target.center
        tx, ty = self.hazard_source
        
        for _ in range(20): # Try 20 times to find a clear spot
            # Spawn in a small radius around target (Launch Silo)
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(20, 50) 
            
            x = tx + np.cos(angle) * dist
            y = ty + np.sin(angle) * dist
            
            # Check map bounds
            if not (0 < x < self.SCREEN_WIDTH and 0 < y < self.SCREEN_HEIGHT):
                continue
            
            # Check walls (Don't spawn inside a building)
            test_rect = pygame.Rect(x - 8, y - 8, 16, 16)
            if test_rect.collidelist(self.walls) == -1:
                # Add the missile
                # Assuming Interceptor class is imported!
                self.interceptors.append(Interceptor(x, y, size=16))
                break
    
    def _recalc_path(self):
        """A* pathfinding (non-arena maps)."""
        start = self.grid.get_node_from_pos(self.agent.position)
        end = self.grid.get_node_from_pos(self.target.center)
        
        if start and end and not start.is_obstacle:
            # Reset node visuals for new pathfinding
            for row in self.grid.grid:
                for node in row:
                    node.reset_visuals()
                    node.update_neighbors(self.grid.grid)
            
            path = a_star_algorithm(None, self.grid, start, end)
            if path:
                self.agent.path = path
    
    # ==========================================
    # RENDERING
    # ==========================================
    
    def render(self):
        if self.render_mode != "human":
            return
        
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Drone Tactical Trainer")
            
            # Initialize visuals
            try:
                from visuals import Visuals, ExplosionManager
                self.visuals = Visuals()
                self.explosion_manager = ExplosionManager(self.visuals)
                self.use_fancy_visuals = True
            except ImportError:
                self.use_fancy_visuals = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                quit()
        
        # Background
        if self.use_fancy_visuals:
            self.visuals.draw_background(self.screen, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        else:
            self.screen.fill((240, 240, 240))
        
        # Walls
        if self.use_fancy_visuals:
            for wall in self.walls:
                self.visuals.draw_wall(self.screen, wall)
        else:
            self.grid.draw(self.screen)
        
        # [ADD THIS] Draw Hazard Source (Red)
        if self.use_fancy_visuals:
            # Draw simple red marker for SAM site
            pygame.draw.circle(self.screen, (200, 50, 50), (int(self.hazard_source[0]), int(self.hazard_source[1])), 15)
        else:
            pygame.draw.circle(self.screen, (200, 0, 0), (int(self.hazard_source[0]), int(self.hazard_source[1])), 15)
        
        # Target / Base
        if self.use_fancy_visuals:
            self.visuals.draw_base(self.screen, self.target)
        else:
            pygame.draw.rect(self.screen, (0, 200, 0), self.target)
            pygame.draw.rect(self.screen, (0, 150, 0), self.target, 3)
        
        # Interceptors / Missiles
        for interceptor in self.interceptors:
            if self.use_fancy_visuals:
                if interceptor.alive:
                    self.visuals.draw_missile(self.screen, interceptor.position, 
                                              interceptor.velocity, interceptor.size)
                else:
                    # Dead interceptor = add explosion (only once)
                    if not hasattr(interceptor, '_exploded'):
                        self.explosion_manager.add(interceptor.position[0], interceptor.position[1])
                        interceptor._exploded = True
            else:
                interceptor.draw(self.screen)
        
        # Update and draw explosions
        if self.use_fancy_visuals:
            self.explosion_manager.update(0.016)
            self.explosion_manager.draw(self.screen)
        
        # Agent / Drone
        if self.use_fancy_visuals:
            self.visuals.draw_drone(self.screen, self.agent.position, 
                                    self.agent.velocity, self.agent.size,
                                    color=self.agent.color if hasattr(self.agent, 'color') else None)
        else:
            self.agent.draw_lidar(self.screen, self.walls)
            self.agent.draw(self.screen)

        # --- DUMMY DRONE RENDERING (Injected) ---
        if self.dummy.active:
            if self.use_fancy_visuals:
                # Use same drone sprite but with Blue color
                self.visuals.draw_drone(self.screen, self.dummy.position,
                                        self.dummy.velocity, self.dummy.size,
                                        color=self.dummy.color)
            else:
                pygame.draw.circle(self.screen, self.dummy.color, 
                                   self.dummy.rect.center, 7)
        # ----------------------------------------
        
        # HUD
        font = pygame.font.SysFont("Arial", 14)
        alive_interceptors = sum(1 for i in self.interceptors if i.alive)
        hud = f"Map: {self.current_map_type} | Threats: {alive_interceptors} | Step: {self.current_step}"
        
        # HUD background for readability
        text = font.render(hud, True, (255, 255, 255))
        text_bg = pygame.Surface((text.get_width() + 10, text.get_height() + 6))
        text_bg.fill((0, 0, 0))
        text_bg.set_alpha(150)
        self.screen.blit(text_bg, (5, 5))
        self.screen.blit(text, (10, 8))
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None
