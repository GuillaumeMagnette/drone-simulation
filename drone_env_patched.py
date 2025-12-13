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

class DummyDrone:
    def __init__(self, x, y):
        self.position = np.array([float(x), float(y)])
        self.velocity = np.array([0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0])
        
        self.size = 14
        self.rect = pygame.Rect(x, y, 14, 14)
        
        self.active = False
        self.respawn_timer = 0.0 # [NEW] Dead timer
        self.color = (100, 100, 255) 
        
        # PHYSICS TUNING
        self.max_speed = 500.0   # Significantly faster than Agent (360) to catch up
        self.max_force = 4000.0  # Snappy acceleration
        self.desired_separation = 45.0

    def update(self, dt, walls, leader_pos, leader_vel, other_drones):
        # [NEW] Respawn Logic
        if not self.active:
            if self.respawn_timer > 0:
                self.respawn_timer -= dt
                if self.respawn_timer <= 0:
                    # Respawn behind leader
                    self.active = True
                    offset = np.array([-30.0, 30.0]) if np.random.random() > 0.5 else np.array([-30.0, -30.0])
                    self.position = leader_pos + offset
            return

        self.acceleration[:] = 0
        
        # [NEW] Teleport Leash (Anti-Stuck)
        to_leader = leader_pos - self.position
        dist_leader = np.linalg.norm(to_leader)
        
        if dist_leader > 250.0:
            # We are stuck or lost. Teleport closer.
            self.position = leader_pos - (leader_vel * 0.1) # Teleport behind
            self.velocity[:] = 0
            return # Skip physics this frame

        # --- 1. SEPARATION ---
        sep_force = np.array([0.0, 0.0])
        count = 0
        
        for other in other_drones:
            if other is self or not other.active: continue
            d = np.linalg.norm(self.position - other.position)
            if 0 < d < self.desired_separation:
                diff = self.position - other.position
                diff /= d
                sep_force += diff
                count += 1
        
        if count > 0:
            sep_force = (sep_force / count) * self.max_speed * 2.0

        # --- 2. COHESION (Slot Following) ---
        target_pos = leader_pos
        leader_speed = np.linalg.norm(leader_vel)
        
        # Dynamic Slot: If leader moving, fly to wingman position
        if leader_speed > 10:
            leader_dir = leader_vel / leader_speed
            # Aim for a spot 40px behind and slightly to side?
            # Actually, just aiming 30px behind is stable enough
            target_pos = leader_pos - (leader_dir * 30.0)
            
        arrive = target_pos - self.position
        dist_arrive = np.linalg.norm(arrive)
        seek_force = np.array([0.0, 0.0])
        
        if dist_arrive > 0:
            desired_vel = (arrive / dist_arrive) * self.max_speed
            seek_force = desired_vel - self.velocity

        # --- 3. APPLY FORCES ---
        total_force = (sep_force * 3.0) + (seek_force * 1.5)
        
        # Clamp Force
        if np.linalg.norm(total_force) > self.max_force:
            total_force = (total_force / np.linalg.norm(total_force)) * self.max_force

        self.acceleration += total_force
        self.velocity += self.acceleration * dt
        self.velocity *= 0.90 # High Drag for stability

        # --- 4. PHYSICS (Improved Wall Slide) ---
        # Move X
        self.position[0] += self.velocity[0] * dt
        self.rect.centerx = int(self.position[0])
        
        # Wall Check X
        block_hit = self.rect.collidelist(walls)
        if block_hit != -1:
            # If hit, push out
            self.position[0] -= self.velocity[0] * dt
            self.velocity[0] *= -0.5
            self.rect.centerx = int(self.position[0])
            
        # Move Y
        self.position[1] += self.velocity[1] * dt
        self.rect.centery = int(self.position[1])
        
        # Wall Check Y
        block_hit = self.rect.collidelist(walls)
        if block_hit != -1:
            self.position[1] -= self.velocity[1] * dt
            self.velocity[1] *= -0.5
            self.rect.centery = int(self.position[1])
            
        # Bounds
        self.position = np.clip(self.position, 10, 790)
        self.rect.center = self.position

    def die(self):
        self.active = False
        self.respawn_timer = 5.0 # Gone for 5 seconds
        self.position = np.array([-1000.0, -1000.0])
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
        # --- SWARM (Replaces self.dummy) ---
        self.swarm_size = 2
        self.swarm = [DummyDrone(0,0) for _ in range(self.swarm_size)]
        
        # --- SPACES ---
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        # 36 inputs: Lidar(8), Vel(2), GPS(2), Threats(8), Proj(8), Neighbors(8)
        # CHANGE: Increase shape from 36 to 40
        # [0-35] Existing sensors
        # [36-39] Precision Tracking (Nearest Missile)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(40,), dtype=np.float32)
        
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
        """
        Tactical Spawning: Places the SAM site between Agent and Target.
        Simulates an 'Area Denial' weapon covering the approach.
        """
        if self.current_map_type == 'arena':
            self.hazard_source = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
            return

        # Vector from Agent -> Target
        start = self.agent.position
        end = np.array(self.target.center)
        vec = end - start
        dist = np.linalg.norm(vec)
        
        # If target is too close, just spawn random
        if dist < 300:
            self.hazard_source = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
            return

        # Target Point: Somewhere in the middle 50% of the path
        # This puts the SAM "In the way"
        t = np.random.uniform(0.3, 0.7)
        midpoint = start + vec * t
        
        # Add noise so it's not perfectly on the line (Ambush position)
        noise_x = np.random.randint(-150, 150)
        noise_y = np.random.randint(-150, 150)
        
        spawn_x = int(midpoint[0] + noise_x)
        spawn_y = int(midpoint[1] + noise_y)
        
        # Clamp to screen
        spawn_x = max(50, min(self.SCREEN_WIDTH - 50, spawn_x))
        spawn_y = max(50, min(self.SCREEN_HEIGHT - 50, spawn_y))
        
        # Ensure it's not inside a wall
        test_rect = pygame.Rect(spawn_x - 20, spawn_y - 20, 40, 40)
        
        # If the calculated "Tactical Spot" is inside a wall, 
        # Spiral out until we find an open spot
        found = False
        for radius in range(10, 200, 20): # Search expanding circle
            if found: break
            for angle in np.linspace(0, 6.28, 8):
                cx = spawn_x + np.cos(angle) * radius
                cy = spawn_y + np.sin(angle) * radius
                
                # Check bounds
                if not (50 < cx < self.SCREEN_WIDTH-50 and 50 < cy < self.SCREEN_HEIGHT-50):
                    continue
                    
                check_rect = pygame.Rect(cx-20, cy-20, 40, 40)
                if check_rect.collidelist(self.walls) == -1:
                    self.hazard_source = (cx, cy)
                    found = True
                    break
        
        if not found:
            # Fallback to random if the tactical calculation failed
            self.hazard_source = (np.random.randint(100, 700), np.random.randint(100, 700))

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
        else: 
            # URBAN OPTIMIZATION
            self._generate_urban()
            # Use the optimized big rectangles for physics!
            self.walls = self.urban_walls 

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
        
        if self.use_bait:
            for i, drone in enumerate(self.swarm):
                drone.active = True
                # Spawn in a small circle around agent
                angle = (2 * np.pi / self.swarm_size) * i
                offset = np.array([np.cos(angle), np.sin(angle)]) * 30
                drone.position = self.agent.position + offset
                drone.rect.center = drone.position
        else:
            for drone in self.swarm:
                drone.active = False
                drone.position = np.array([-100.0, -100.0])
        
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
        """
        Moves the target to a new random location.
        Robust version for Dense Urban Maps.
        """
        found = False
        attempts = 0
        
        # Try up to 500 times to find a valid spot
        # (Urban maps have ~40% walkable area, so this will find one quickly)
        while attempts < 500:
            attempts += 1
            
            # Margin of 20px from screen edges
            x = np.random.randint(20, self.SCREEN_WIDTH - 20)
            y = np.random.randint(20, self.SCREEN_HEIGHT - 20)
            
            # Create a test rect
            # We inflate it slightly to ensure it's not touching a wall pixel-perfectly
            candidate_rect = pygame.Rect(x, y, 40, 40).inflate(10, 10)
            
            # 1. Check Wall Collision
            if candidate_rect.collidelist(self.walls) != -1: 
                continue
            
            # 2. Check Distance (Force agent to travel)
            # In Scavenger mode, we want the agent to move at least 200px
            dist = np.linalg.norm(np.array([x,y]) - self.agent.position)
            if dist < 200:
                continue
            
            # If we get here, it's valid
            self.target.topleft = (x, y)
            found = True
            break
                
        # Emergency Fallback (Center Plaza) if 500 tries fail
        if not found:
            self.target.center = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)

    def _get_obs(self):
        """
        Generates the 40-dimensional observation vector.
        [0-7]   Lidar (Walls)
        [8-9]   Velocity (Agent)
        [10-11] GPS (Vector to Target)
        [12-19] Threat Sectors (General Awareness)
        [20-27] Projectile Sectors (Reserved)
        [28-35] Neighbor Sectors (Dummy/Ally)
        [36-39] Precision Tracking (Most Dangerous Missile)
        """
        
        # --- 1. LIDAR (8) ---
        lidar = self.agent.cast_rays(self.walls, num_rays=8, max_dist=200)
        
        # --- 2. VELOCITY (2) ---
        vel = self.agent.velocity / self.agent.max_speed
        
        # --- 3. GPS (2) ---
        # Always active for Scavenger Hunt / Mission modes
        if self.current_map_type == 'arena':
            # Optimization: Direct line in empty space
            waypoint = np.array(self.target.center, dtype=float)
        elif self.agent.path:
            # Navigation: Follow A* nodes
            node = self.agent.path[0]
            waypoint = np.array([node.x + 7, node.y + 7])
        else:
            # Fallback: Direct line
            waypoint = np.array(self.target.center, dtype=float)
        
        vec_to_wp = waypoint - self.agent.position
        dist_to_wp = np.linalg.norm(vec_to_wp)
        if dist_to_wp > 0: vec_to_wp /= dist_to_wp
        
        # --- 4. PRECISION TRACKING (4) - THE RWR UPGRADE ---
        # Select the single most dangerous missile based on Closing Speed & Distance
        most_dangerous_missile = None
        highest_danger_score = -float('inf')
        
        for i in self.interceptors:
            if not i.alive: continue
            
            # Vector pointing FROM Agent TO Missile
            vec_to_missile = i.position - self.agent.position
            dist = np.linalg.norm(vec_to_missile) + 0.1 # Safety epsilon
            
            # Relative Velocity (Missile Vel - Agent Vel)
            rel_vel = i.velocity - self.agent.velocity
            
            # Calculate Closing Speed via Dot Product
            # Direction from Agent to Missile
            dir_to_missile = vec_to_missile / dist
            
            # If Missile is flying AT Agent, dot(rel_vel, dir_to_missile) is NEGATIVE.
            # We want positive closing speed, so we negate the dot product.
            closing_speed = -np.dot(rel_vel, dir_to_missile)
            
            # SCORING HEURISTIC (Time-To-Impact)
            if closing_speed > 0:
                # It is closing in. High speed + Low Dist = MAX DANGER.
                score = closing_speed / dist
            else:
                # It is flying away. Low score.
                # We subtract distance so far-away missiles score lower than close ones.
                score = -dist 
            
            if score > highest_danger_score:
                highest_danger_score = score
                most_dangerous_missile = i
        
        # Construct the 4-float vector
        missile_state = np.zeros(4, dtype=np.float32)
        
        if most_dangerous_missile:
            # 1. Relative Position (Normalized by Radar Range 400)
            rel_pos = (most_dangerous_missile.position - self.agent.position) / 400.0
            
            # 2. Relative Velocity (Normalized by Max Missile Speed 600)
            # Crucial for the agent to know if it needs to lead the dodge
            rel_vel_norm = (most_dangerous_missile.velocity - self.agent.velocity) / 600.0
            
            missile_state[0] = np.clip(rel_pos[0], -1, 1)
            missile_state[1] = np.clip(rel_pos[1], -1, 1)
            missile_state[2] = np.clip(rel_vel_norm[0], -1, 1)
            missile_state[3] = np.clip(rel_vel_norm[1], -1, 1)

        # --- 5. SECTOR SENSORS (24) ---
        # General awareness of threats, projectiles, and neighbors
        visible_threats = []
        for interceptor in self.interceptors:
            if not interceptor.alive: continue
            if self.current_map_type == 'arena':
                visible_threats.append(interceptor)
            elif self.agent._has_line_of_sight(interceptor.position, self.walls):
                visible_threats.append(interceptor)
        
        threat_sectors = self.agent.get_sector_readings(visible_threats, radius=400.0, num_sectors=8)
        
        # Projectiles (Placeholder/Reserved)
        projectile_sectors = np.ones(8, dtype=np.float32)
        
        # Neighbors (Dummy/Bait)
        visible_neighbors = []
        if self.use_bait:
            # Add all active swarm members
            visible_neighbors = [d for d in self.swarm if d.active]
            
        neighbor_sectors = self.agent.get_sector_readings(visible_neighbors, radius=300.0, num_sectors=8)
        
        # --- 6. ASSEMBLE (Total: 40) ---
        return np.concatenate([
            lidar,              # 0-7
            vel,                # 8-9
            vec_to_wp,          # 10-11
            threat_sectors,     # 12-19
            projectile_sectors, # 20-27
            neighbor_sectors,   # 28-35
            missile_state       # 36-39 (PRECISION TRACKING)
        ]).astype(np.float32)
    

    def step(self, action):
        self.current_step += 1
        self.repath_steps += 1
        
        # --- 0. SAFETY SHIELD (Envelope Protection) ---
        # "Industry Standard" Override
        # Calculate where the requested action sends us
        proposed_force = action * self.agent.max_force
        
        # Predict position in next frame (0.016s)
        # We look a bit further ahead (e.g. 3 frames) for safety buffer
        lookahead = 0.05 
        pred_vel = self.agent.velocity + (proposed_force / self.agent.mass) * lookahead
        pred_pos = self.agent.position + pred_vel * lookahead
        pred_rect = self.agent.rect.copy()
        pred_rect.center = pred_pos
        
        # Check collision of PREDICTED state
        shield_activated = False
        
        # Check Wall Collision
        hit_future_wall = False
        if (pred_rect.left < 0 or pred_rect.right > self.SCREEN_WIDTH or
            pred_rect.top < 0 or pred_rect.bottom > self.SCREEN_HEIGHT):
            hit_future_wall = True
        elif pred_rect.collidelist(self.walls) != -1:
            hit_future_wall = True
            
        if hit_future_wall:
            shield_activated = True
            
            # EMERGENCY BRAKE / REFLECTION
            # 1. Kill current velocity (Brake)
            # We apply a force opposite to velocity to stop movement
            # But simpler: we just override the force to push AWAY from the wall
            # Since we don't have normal vectors easily, we just apply 
            # MAX FORCE opposite to current velocity (Emergency Stop)
            if np.linalg.norm(self.agent.velocity) > 0.1:
                opp_dir = -self.agent.velocity / np.linalg.norm(self.agent.velocity)
                # Overwrite the action!
                proposed_force = opp_dir * self.agent.max_force * 1.5 # 150% Power braking
                
        # --- 1. PHYSICS ---
        force = action * self.agent.max_force
        self.agent.apply_force(proposed_force) # (Using the proposed_force from Safety Shield)
        self.agent.update_physics(0.016)
        self.agent.rect.center = self.agent.position
        
        # [NEW] Update Swarm Physics
        if self.use_bait:
            for drone in self.swarm:
                if drone.active:
                    # Pass the whole list so they can separate from each other
                    drone.update(
                        0.016, 
                        self.walls, 
                        self.agent.position, 
                        self.agent.velocity, 
                        self.swarm  # <--- THIS WAS MISSING
                    )
        
        # --- 2. WALL COLLISION ---
        hit_wall = False
        if (self.agent.rect.left < 0 or self.agent.rect.right > self.SCREEN_WIDTH or
            self.agent.rect.top < 0 or self.agent.rect.bottom > self.SCREEN_HEIGHT): hit_wall = True   
        if not hit_wall and self.agent.rect.collidelist(self.walls) != -1: hit_wall = True

        # --- 3. MISSILE SPAWNING (Using Hazard Source) ---
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
                # [FIX]: Check distance to HAZARD SOURCE, not target
                dist_to_hazard = np.linalg.norm(self.agent.position - np.array(self.hazard_source))
                too_close = dist_to_hazard < 75
                can_fire = not too_close
                if can_fire and self.current_map_type != 'arena':
                    can_fire = self.agent._has_line_of_sight(self.hazard_source, self.walls)
                
                if can_fire:
                    self._spawn_interceptor()
                    self.total_missiles_spawned += 1

        # --- 4. ENTITY UPDATES & BAIT REWARD ---
        hit_interceptor = False
        # Create target list: Agent + All Active Drones
        potential_targets = [self.agent]
        if self.use_bait:
            potential_targets.extend([d for d in self.swarm if d.active])
        
        # [NEW] Track bait success
        bait_successful = False 
        
        for interceptor in self.interceptors:
            if not interceptor.alive: continue
            
            dist_to_agent = np.linalg.norm(interceptor.position - self.agent.position)
            interceptor.update(0.016, potential_targets, self.walls)
            
            # 1. Check Agent Hit
            if interceptor.alive and self.agent.rect.colliderect(interceptor.rect):
                hit_interceptor = True
                interceptor._die()
            
            # 2. Check Swarm Hits
            if self.use_bait and interceptor.alive:
                # Check collision with ANY drone in the swarm
                # We use collidelist for speed
                swarm_rects = [d.rect for d in self.swarm if d.active]
                hit_index = interceptor.rect.collidelist(swarm_rects)
                
                if hit_index != -1:
                    # We hit a drone!
                    interceptor._die()

                    
                    # Visual explosion
                    hit_drone = [d for d in self.swarm if d.active][hit_index]

                    # [ADD THIS LINE HERE] -----------------------------
                    hit_drone.die() # 2. Kill Drone (Starts 5s timer)
                    # -------------------------------------------------
                    
                    if self.render_mode == "human" and hasattr(self, 'explosion_manager'):
                        self.explosion_manager.add(hit_drone.position[0], hit_drone.position[1])
                    
                    # Reward check
                    if dist_to_agent < 200:
                        bait_successful = True

        # --- 5. PATHFINDING ---
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
        # 6. REWARDS & DEFCON SYSTEM
        # ============================================================
        
        obs = self._get_obs()
        reward = 0
        terminated = False
        truncated = False
        
        # Base Rewards
        cur_dist = np.linalg.norm(np.array(self.target.center) - self.agent.position)
        progress = self.last_dist - cur_dist
        self.last_dist = cur_dist
        reward += progress * 0.15
        self.cum_reward_dist += progress * 0.15
        
        reward -= 0.1 
        self.cum_reward_time -= 0.1

        # Penalty for relying on the Safety Shield
        if shield_activated:
            # -1.0 is enough to say "Bad Pilot", but not -50 "Dead Pilot"
            reward -= 1.0

        # [NEW] THE BETRAYAL BONUS
        if bait_successful:
            reward += 30.0
            # Optional: Log it so we know it happened
            # self.termination_reason = "Bait Success"

        # Threat Analysis
        closest_threat_dist = 9999.0
        prediction_time = 0.3
        future_agent_pos = self.agent.position + (self.agent.velocity * prediction_time)
        threat_detected = False
        
        for interceptor in self.interceptors:
            if not interceptor.alive: continue
            dist = np.linalg.norm(self.agent.position - interceptor.position)
            closest_threat_dist = min(closest_threat_dist, dist)
            
            # Defcon Trigger (350px)
            if dist < 350.0:
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
                            # [NEW] SHADOW CHECK
                            # Check if a dummy is blocking the line of fire
                            is_shadowed = False
                            if self.use_bait:
                                # Define the line of fire (Missile -> Agent)
                                p1 = interceptor.position
                                p2 = future_agent_pos
                                
                                for drone in self.swarm:
                                    if drone.active:
                                        # Does the drone block this line?
                                        if drone.rect.clipline(p1, p2):
                                            is_shadowed = True
                                            break
                            
                            # Only apply pain if NOT shadowed
                            if not is_shadowed:
                                intensity = 1.0 - (miss_dist / safety_radius)
                                reward -= intensity * 3.0 # The Pain
                            else:
                                # Optional: Small "Comfort" reward for being in the shadow?
                                # Usually 0.0 (Relief) is enough, but +0.1 speeds up learning.
                                reward += 0.1

                # --- NEW: TOP GUN BONUS (Evasion Geometry) ---
                # Reward moving perpendicular to the missile
                to_missile = interceptor.position - self.agent.position
                norm_to_m = np.linalg.norm(to_missile)
                if norm_to_m > 0:
                    to_missile /= norm_to_m
                    
                    my_speed = np.linalg.norm(self.agent.velocity)
                    if my_speed > 10:
                        my_dir = self.agent.velocity / my_speed
                        # Dot Product: 1.0 = Head on, 0.0 = Sidestep
                        alignment = np.dot(to_missile, my_dir)
                        
                        # If we are moving sideways or away (< 0.3 alignment)
                        if alignment < 0.3:
                            reward += 0.05

        # Penalties (Context Aware)
        is_safe = not threat_detected
        speed = np.linalg.norm(self.agent.velocity)
        
        # Stall Penalty (Only if safe)
        if is_safe and speed < 5 and cur_dist > 50.0:
             reward -= 0.5 if self.current_map_type == 'arena' else 1.0

        # Scrape Penalty (Only if safe)
        if is_safe and np.min(obs[0:8]) < 0.15: 
            reward -= 0.2
        
        # Crash Prediction (Reduced if threatened)
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
            if threat_detected: reward -= base_crash_pen * 0.5
            else: reward -= base_crash_pen

        # --- 7. TERMINATION ---
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
            # [FIXED] Scavenger Hunt Logic
            if self.dynamic_target:
                reward += 20 
                self.cum_reward_win += 20
                self._relocate_target() # Move target
                self._recalc_path()     # Update A* to new target
                self.last_dist = np.linalg.norm(np.array(self.target.center) - self.agent.position)
                # Does NOT terminate
            else:
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
        Generates a Randomized Manhattan grid using OPTIMIZED BIG RECTANGLES.
        Includes Grid Shifting and Random Gaps to prevent overfitting.
        """
        rows = self.ROWS
        pixel_size = 10
        
        # Configuration
        block_size = 14 
        street_width = 6 
        
        self.urban_walls = [] 
        
        step = block_size + street_width
        
        # 1. RANDOM GRID SHIFT (Prevents coordinate memorization)
        # Shift the whole city by 0-5 blocks (0-50px)
        shift_x = np.random.randint(0, 6)
        shift_y = np.random.randint(0, 6)
        
        for grid_x in range(2, rows - step, step):
            for grid_y in range(2, rows - step, step):
                
                # 2. RANDOM GAPS (Topology Variation)
                # 15% chance a building is missing (becomes a park/open lot)
                # This ensures the maze layout changes every episode
                if np.random.random() < 0.15:
                    continue

                # Apply shift to grid coordinates
                gx = grid_x + shift_x
                gy = grid_y + shift_y
                
                # Bounds check (don't draw off-screen)
                if gx >= rows - 2 or gy >= rows - 2:
                    continue

                # Define Building Rect
                bx = gx * pixel_size
                by = gy * pixel_size
                bw = block_size * pixel_size
                bh = block_size * pixel_size
                
                # Clip to screen size if necessary
                if bx + bw > self.SCREEN_WIDTH: bw = self.SCREEN_WIDTH - bx
                if by + bh > self.SCREEN_HEIGHT: bh = self.SCREEN_HEIGHT - by
                
                # Create Physics Wall
                big_wall = pygame.Rect(bx, by, bw, bh)
                self.urban_walls.append(big_wall)
                
                # Mark Grid Nodes (A* logic)
                # We iterate relative to the shifted position
                for i in range(block_size):
                    for j in range(block_size):
                        nx, ny = gx + i, gy + j
                        if 0 <= nx < rows and 0 <= ny < rows:
                            self.grid.grid[nx][ny].make_obstacle()

        # 3. Clear Central Plaza
        center_rect = pygame.Rect(300, 300, 200, 200)
        self.urban_walls = [w for w in self.urban_walls if not w.colliderect(center_rect)]
        
        center_grid = rows // 2
        margin = 10
        for x in range(center_grid - margin, center_grid + margin):
            for y in range(center_grid - margin, center_grid + margin):
                if 0 <= x < rows and 0 <= y < rows:
                    self.grid.grid[x][y].reset()


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

        # Draw Swarm
        if self.use_bait:
            for drone in self.swarm:
                if not drone.active: continue
                if self.use_fancy_visuals:
                    self.visuals.draw_drone(self.screen, drone.position,
                                            drone.velocity, drone.size,
                                            color=drone.color)
                else:
                    pygame.draw.circle(self.screen, drone.color, drone.rect.center, 7)
        
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
