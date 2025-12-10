"""
DRONE ENVIRONMENT 2.5D - MARL SATURATION ATTACK
===============================================

Sim 2.0: Multi-Agent + Altitude Physics.

Mission: 3 Drones must coordinate to strike a target protected by a Battery.
Tactics: Nap-of-the-Earth (NOE) flying, Pop-up attacks, Saturation.

Physics Model (2.5D):
---------------------
- Ground (Z=0) to Ceiling (Z=100).
- Buildings: Height = 50.0.
- Low Altitude (Z < 50): Collides with walls. Radar blocked by walls.
- High Altitude (Z > 50): Flies over walls. Always visible to Radar.

Observation Space (Per Agent = 45 floats):
------------------------------------------
[0-2]    Position (Norm X, Y, Z)
[3-5]    Velocity (Norm VX, VY, VZ)
[6-8]    Target GPS (Rel X, Y, Z)
[9-16]   Lidar (8 Rays) - Returns 1.0 if flying high (over walls)
[17-24]  Threat Sensors (8 Sectors)
[25-30]  Teammate Data (Rel X, Y, Z for 2 mates)
[31-36]  Precision RWR (Nearest Missile: Rel X, Y, Z, VX, VY, VZ)
[37]     Altitude State (1.0 = High, 0.0 = Low)
[38-44]  (Reserved/Padding)

Total Obs: 45 * 3 Agents = 135 floats.
Total Act: 3 * 3 Agents = 9 floats.
"""

import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import os
from grid import Grid
# --- IMPORTS FROM NEW FILES ---
from agent_marl import Agent25D
from interceptor_marl import Interceptor3D

# ==========================================
# CONFIGURATION
# ==========================================
# (These constants are also in agent/interceptor files, 
# but kept here for Env logic consistency)
BUILDING_HEIGHT = 50.0
CEILING = 100.0
GRAVITY = 15.0 # Downward force
LIFT_FORCE = 30.0 # Vertical thruster power

class DroneEnvMARL25D(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None):
        super().__init__()
        
        # --- CONFIGURATION ---
        # INCREASED SIZE (Tactical Depth)
        self.SCREEN_WIDTH = 1200 
        self.SCREEN_HEIGHT = 1200
        self.BUILDING_HEIGHT = 50.0
        self.CEILING = 100.0
        self.render_mode = render_mode
        self.screen = None
        
        # --- AGENTS (MARL) ---
        self.n_agents = 3
        # Initialize agents (using your Agent25D class)
        self.agents = [Agent25D(0, 0, i) for i in range(self.n_agents)]
        
        # Action Space: [Fx, Fy, Fz] * N_Agents
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_agents * 3,), dtype=np.float32)
        
        # Observation Space: 45 inputs per agent * N_Agents
        # We need slightly more inputs for Z-axis relative data
        self.obs_dim_per_agent = 45
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.n_agents * self.obs_dim_per_agent,), dtype=np.float32)
        
        # --- WORLD & LOGIC ---
        self.urban_walls = [] # Big Rect optimization
        self.target = pygame.Rect(0,0,40,40)
        self.hazard_source = (400, 400)
        self.interceptors = []
        self.missile_timer = 0.0
        self.max_steps = 2000
        
        # --- PHASE 4 LOGIC FLAGS ---
        self.use_bait = True # Implicit in MARL (Teammates are bait)
        self.respawn_threats = True
        self.dynamic_target = True # Scavenger mode
        
        # Update Grid Rows to match pixel size (10px)
        self.ROWS = self.SCREEN_WIDTH // 10 # 120 rows
        self.grid = Grid(self.ROWS, self.SCREEN_WIDTH)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.interceptors = []

                # [NEW] Default Configuration
        self.active_threats = True 
        
        # [NEW] Parse Options
        if options:
            self.active_threats = options.get('active_threats', True)
        
        # 1. Generate Randomized Manhattan Map (Phase 3 Logic)
        self._generate_urban_manhattan()
        
        # 2. Spawn Hazard (SAM Site)
        self._spawn_hazard()
        
        # 3. Spawn Target (Scavenger)
        self._relocate_target()
        
        # 4. Spawn Agents
        # Scatter them safely
        for ag in self.agents:
            ag.active = True
            ag.velocity[:] = 0
            ag.dead_timer = 0.0
            
            # Find safe spawn
            for _ in range(100):
                ax = np.random.randint(50, 750)
                ay = np.random.randint(50, 750)
                
                # Check walls
                t_rect = pygame.Rect(ax-10, ay-10, 20, 20)
                if t_rect.collidelist(self.urban_walls) == -1:
                    # Check Hazard Distance
                    if np.linalg.norm([ax - self.hazard_source[0], ay - self.hazard_source[1]]) > 300:
                        ag.position = np.array([float(ax), float(ay), 10.0]) # Start Low
                        ag.rect.center = (ax, ay)
                        ag.prev_dist = np.linalg.norm(ag.position[:2] - np.array(self.target.center))
                        break
                        
        return self._get_combined_obs(), {}

    def step(self, action):
        self.current_step += 1
        agent_actions = action.reshape((self.n_agents, 3))
        
        # ============================================================
        # 1. SAFETY SHIELD 2.5D (Diagonal Robust)
        # ============================================================
        processed_actions = []
        shield_penalties = [0.0] * self.n_agents
        
        for i, ag in enumerate(self.agents):
            user_action = agent_actions[i]
            
            if not ag.active:
                processed_actions.append(user_action)
                continue
            
            dt_pred = 0.12
            buffer = 4
            
            # State
            curr_pos = ag.position
            curr_vel = ag.velocity
            
            # Forces
            fx = user_action[0] * ag.max_force
            fy = user_action[1] * ag.max_force
            fz = user_action[2] * 30.0 - 15.0
            
            safe_fx = fx
            safe_fy = fy
            safe_fz = user_action[2]
            shield_active = False
            
            # --- Z CHECK ---
            pred_vz = curr_vel[2] + (fz / ag.mass) * dt_pred
            pred_z = curr_pos[2] + pred_vz * dt_pred
            
            if pred_z < 2.0:
                shield_active = True
                safe_fz = 1.0
            
            # --- XY WALL LOGIC ---
            if pred_z < self.BUILDING_HEIGHT:
                
                # 1. Check X-Axis Move
                pred_vx = curr_vel[0] + (fx / ag.mass) * dt_pred
                pred_x = curr_pos[0] + pred_vx * dt_pred
                test_rect_x = pygame.Rect(pred_x - 7, curr_pos[1] - 7, 14, 14).inflate(buffer, buffer)
                
                hit_x = False
                if (test_rect_x.left < 0 or test_rect_x.right > self.SCREEN_WIDTH): hit_x = True
                elif test_rect_x.collidelist(self.urban_walls) != -1: hit_x = True
                
                # 2. Check Y-Axis Move
                pred_vy = curr_vel[1] + (fy / ag.mass) * dt_pred
                pred_y = curr_pos[1] + pred_vy * dt_pred
                test_rect_y = pygame.Rect(curr_pos[0] - 7, pred_y - 7, 14, 14).inflate(buffer, buffer)
                
                hit_y = False
                if (test_rect_y.top < 0 or test_rect_y.bottom > self.SCREEN_HEIGHT): hit_y = True
                elif test_rect_y.collidelist(self.urban_walls) != -1: hit_y = True

                # 3. Check Diagonal (The Corner Case)
                # If X is safe AND Y is safe, we might still hit the corner if we move BOTH.
                hit_diag = False
                if not hit_x and not hit_y:
                    test_rect_diag = pygame.Rect(pred_x - 7, pred_y - 7, 14, 14).inflate(buffer, buffer)
                    if test_rect_diag.collidelist(self.urban_walls) != -1:
                        hit_diag = True

                # --- RESOLUTION ---
                
                if hit_x or hit_diag:
                    shield_active = True
                    safe_fx = 0.0
                    # Brake X
                    if abs(curr_vel[0]) > 5.0:
                        safe_fx = -np.sign(curr_vel[0]) * ag.max_force * 0.8
                
                if hit_y or hit_diag:
                    shield_active = True
                    safe_fy = 0.0
                    # Brake Y
                    if abs(curr_vel[1]) > 5.0:
                        safe_fy = -np.sign(curr_vel[1]) * ag.max_force * 0.8
                        
                # If Trapped (Both blocked OR Corner Strike), Climb
                if (hit_x and hit_y) or hit_diag:
                    safe_fz = 1.0

            # Re-normalize
            final_action = np.array([
                np.clip(safe_fx / ag.max_force, -1, 1),
                np.clip(safe_fy / ag.max_force, -1, 1),
                safe_fz
            ])
            
            if shield_active:
                shield_penalties[i] = -1.0
                processed_actions.append(final_action)
            else:
                processed_actions.append(user_action)

        # ============================================================
        # 2. PHYSICS & WAVE UPDATE (Same as before)
        # ============================================================
        team_alive = 0
        for i, ag in enumerate(self.agents):
            ag.update(0.016, processed_actions[i], self.urban_walls, self.hazard_source)
            if ag.active: team_alive += 1

        if team_alive == 0:
            for ag in self.agents:
                ag.spawn_at_safe_pos(self.urban_walls, self.hazard_source)
                # Reset prev_dist correctly
                ag.prev_dist = np.linalg.norm(ag.position[:2] - np.array(self.target.center))
            self.missile_timer = 2.0 
            self.interceptors = []
            team_alive = self.n_agents

        # ============================================================
        # 3. THREAT LOGIC (Same as before)
        # ============================================================
        if self.active_threats:
            self.missile_timer -= 0.016
            if self.missile_timer <= 0 and team_alive > 0:
                if len(self.interceptors) < 2: 
                    self.missile_timer = 2.0 + np.random.random()
                    self.interceptors.append(Interceptor3D(self.hazard_source[0], self.hazard_source[1], 10.0))
        
        for m in self.interceptors:
            m.update(0.016, self.agents, self.urban_walls)
        self.interceptors = [m for m in self.interceptors if m.alive]

        # ============================================================
        # 4. REWARDS
        # ============================================================
        reward = 0.0
        terminated = False
        truncated = False
        
        # A. Mission
        targets_collected = 0
        for ag in self.agents:
            if ag.active:
                d = np.linalg.norm(ag.position[:2] - np.array(self.target.center))
                if d < 30 and ag.position[2] < 20:
                    targets_collected += 1
        
        if targets_collected > 0:
            reward += 20.0 * targets_collected 
            self._relocate_target()
            for ag in self.agents:
                ag.prev_dist = np.linalg.norm(ag.position[:2] - np.array(self.target.center))

        # B. Individual Progress (Boosted)
        for ag in self.agents:
            if ag.active:
                d = np.linalg.norm(ag.position[:2] - np.array(self.target.center))
                diff = ag.prev_dist - d
                ag.prev_dist = d
                # [FIX] Increased multiplier to 2.0 to incentivize speed
                reward += diff * 2.0 

        # C. Tactical Rewards

        # [NEW] Calculate Swarm Centroid
        active_positions = [a.position for a in self.agents if a.active]
        if len(active_positions) > 1:
            centroid = np.mean(active_positions, axis=0)
        else:
            centroid = None

        for i, ag in enumerate(self.agents):

            # [NEW] Check for Death Event first
            if ag.events["crashed"]:
                reward -= 50.0
                if self.render_mode == "human" and hasattr(self, 'explosion_manager'):
                    # Visual feedback for crash
                    self.explosion_manager.add(ag.rect.centerx, ag.rect.centery)

            if not ag.active: continue
            
            reward += shield_penalties[i]

            # 1. Straggler Penalty (The Pull) -> Keep within 400px of group
            if centroid is not None:
                dist_to_group = np.linalg.norm(ag.position - centroid)
                if dist_to_group > 400.0: # Loosened slightly to allow spreading
                    excess = dist_to_group - 400.0
                    reward -= (excess / 100.0) * 0.1

            # 2. [NEW] WOLF PACK LOGIC (The Push & Flank)
            for neighbor in self.agents:
                if neighbor is ag or not neighbor.active: continue
                
                # A. Personal Space (Collision Avoidance)
                # If too close to a friend, get away.
                dist_neighbor = np.linalg.norm(ag.position - neighbor.position)
                if dist_neighbor < 40.0: # 40px bubble
                    # Exponential penalty the closer they get
                    intensity = 1.0 - (dist_neighbor / 40.0)
                    reward -= intensity * 0.5
                
                # B. Attack Angle Diversity (Encirclement)
                # Only matters if we are attacking (close to target)
                dist_to_target = np.linalg.norm(ag.position[:2] - np.array(self.target.center))
                
                if dist_to_target < 400:
                    # Calculate vector from Target to Me and Target to Neighbor
                    vec_me = ag.position[:2] - np.array(self.target.center)
                    vec_neighbor = neighbor.position[:2] - np.array(self.target.center)
                    
                    # Normalize
                    norm_me = np.linalg.norm(vec_me)
                    norm_n = np.linalg.norm(vec_neighbor)
                    
                    if norm_me > 0 and norm_n > 0:
                        dir_me = vec_me / norm_me
                        dir_n = vec_neighbor / norm_n
                        
                        # Dot Product:
                        # 1.0 = Same angle (Bad, Conga line)
                        # -1.0 = Opposite sides (Good, Pincer attack)
                        # 0.0 = 90 degrees (Good, Flanking)
                        angle_alignment = np.dot(dir_me, dir_n)
                        
                        # If alignment > 0.5 (less than 60 degrees separation), punish
                        # This forces them to spread out around the circle
                        if angle_alignment > 0.5:
                            reward -= 0.05

            
            # [FIX] STAGNATION PENALTY (The "Anti-Stuck" Logic)
            # If agent is pushing hard (Action > 0.5) but moving slow (Vel < 20),
            # it means they are pushing against a wall. Punish it.
            input_force = np.linalg.norm(agent_actions[i][:2])
            actual_speed = np.linalg.norm(ag.velocity[:2])
            
            if input_force > 0.5 and actual_speed < 20.0:
                reward -= 0.5 # Stop grinding the wall!
            
            # Altitude Tactics
            is_high = ag.position[2] > self.BUILDING_HEIGHT
            if not is_high:
                reward += 0.02
            else:
                # [FIX] Increased penalty slightly to encourage maze solving
                # even in Flight School mode.
                reward -= 0.1 
            
            # ... (Rest of Threat/Combat Logic remains the same) ...
            closest_m = None
            min_d = 9999
            for m in self.interceptors:
                d = np.linalg.norm(m.position - ag.position)
                if d < min_d:
                    min_d = d
                    closest_m = m
            
            if closest_m and min_d < 400:
                if is_high: reward -= 1.0 # Radar Lock
                
                to_agent = ag.position - closest_m.position
                to_agent /= (np.linalg.norm(to_agent) + 0.1)
                m_dir = closest_m.velocity / (np.linalg.norm(closest_m.velocity) + 0.1)
                
                if np.dot(m_dir, to_agent) > 0.9:
                    is_shadowed = False
                    if not closest_m._has_los(ag, self.urban_walls): is_shadowed = True
                    if not is_shadowed:
                        for tm in self.agents:
                            if tm is ag or not tm.active: continue
                            if self._check_meat_shield(closest_m.position, ag.position, tm.position):
                                is_shadowed = True; break
                    
                    if is_shadowed: reward += 0.5
                    else: reward -= 3.0
                
                # Top Gun
                to_m = closest_m.position - ag.position
                dist_m = np.linalg.norm(to_m)
                if dist_m > 0:
                    to_m /= dist_m
                    mspd = np.linalg.norm(ag.velocity)
                    if mspd > 10:
                        align = np.abs(np.dot(to_m, ag.velocity/mspd))
                        if align < 0.3: reward += 0.05

        reward -= 0.1
        if team_alive == 0: reward -= 50.0 
        if self.current_step >= self.max_steps: truncated = True
        
        return self._get_combined_obs(), reward, terminated, truncated, {}

    # --- HELPERS ---

    def _generate_urban_manhattan(self):
        """Phase 3 Manhattan Logic adapted for 2.5D"""
        rows = self.ROWS
        pixel_size = 10
        block_size = 14
        street_width = 6
        self.urban_walls = []
        
        shift_x = np.random.randint(0, 6)
        shift_y = np.random.randint(0, 6)
        step = block_size + street_width
        
        for grid_x in range(2, rows - step, step):
            for grid_y in range(2, rows - step, step):
                if np.random.random() < 0.15: continue # Random gaps
                
                gx, gy = grid_x + shift_x, grid_y + shift_y
                if gx >= rows-2 or gy >= rows-2: continue
                
                bx = gx * pixel_size
                by = gy * pixel_size
                bw = block_size * pixel_size
                bh = block_size * pixel_size
                
                big_wall = pygame.Rect(bx, by, bw, bh)
                self.urban_walls.append(big_wall)
        
        # Clear Center
        center_rect = pygame.Rect(300, 300, 200, 200)
        self.urban_walls = [w for w in self.urban_walls if not w.colliderect(center_rect)]

    def _spawn_hazard(self):
        # Tactical Spawning from Phase 3
        # Try to spawn between agents (avg pos) and target? 
        # For now, random valid location is safer for MARL init
        for _ in range(100):
            x = np.random.randint(200, self.SCREEN_WIDTH - 200)
            y = np.random.randint(200, self.SCREEN_HEIGHT - 200)
            t_rect = pygame.Rect(x-20, y-20, 40, 40)
            if t_rect.collidelist(self.urban_walls) == -1:
                self.hazard_source = (x, y)
                return
        self.hazard_source = (400, 400)

    def _relocate_target(self):
        """Spawns the target in a valid location, scaling to map size."""
        margin = 50
        
        for _ in range(100):
            x = np.random.randint(margin, self.SCREEN_WIDTH - margin)
            y = np.random.randint(margin, self.SCREEN_HEIGHT - margin)
            
            t_rect = pygame.Rect(x-10, y-10, 20, 20)
            
            # 1. Check Wall Collision
            if t_rect.collidelist(self.urban_walls) != -1:
                continue
                
            # 2. Check Distance from Hazard (Don't spawn on the gun)
            if np.linalg.norm([x - self.hazard_source[0], y - self.hazard_source[1]]) > 300:
                self.target = pygame.Rect(x-20, y-20, 40, 40)
                return
        
        # Fallback
        self.target = pygame.Rect(self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2, 40, 40)

    def _check_meat_shield(self, m_pos, a_pos, t_pos):
        # Check if Teammate (t) is within distance to line segment M->A
        # Simple point-to-line distance in 3D
        line_vec = a_pos - m_pos
        line_len = np.linalg.norm(line_vec)
        line_dir = line_vec / line_len
        
        to_t = t_pos - m_pos
        proj = np.dot(to_t, line_dir)
        
        if proj < 0 or proj > line_len: return False # Behind missile or behind agent
        
        # Closest point on line
        closest_p = m_pos + line_dir * proj
        dist = np.linalg.norm(t_pos - closest_p)
        
        return dist < 20.0 # Teammate radius covers line

    def _get_combined_obs(self):
        obs_list = []
        for i in range(self.n_agents):
            obs_list.append(self._get_single_agent_obs(i))
        return np.concatenate(obs_list)

    def _get_single_agent_obs(self, agent_idx):
        me = self.agents[agent_idx]
        
        # If dead, return zeros
        if not me.active: 
            return np.zeros(self.obs_dim_per_agent, dtype=np.float32)
            
        # Normalization Constants
        # Map: 1200x1200, Ceiling: 100
        world_scale = np.array([self.SCREEN_WIDTH, self.SCREEN_HEIGHT, self.CEILING], dtype=np.float32)
        vel_scale = np.array([360.0, 360.0, 100.0], dtype=np.float32)
        
        # --- 1. SELF STATE [0-5] ---
        pos_norm = me.position / world_scale
        vel_norm = me.velocity / vel_scale
        
        # --- 2. GPS [6-8] ---
        # Vector pointing to Green Target
        t_vec_xy = np.array(self.target.center) - me.position[:2]
        t_vec_z = 0.0 - me.position[2] # Target is on ground (Z=0)
        
        t_dist = np.linalg.norm(t_vec_xy)
        t_dir_xy = (t_vec_xy / (t_dist + 0.1))
        t_dir_z = t_vec_z / 100.0
        
        gps = np.array([t_dir_xy[0], t_dir_xy[1], t_dir_z])
        
        # --- 3. LIDAR [9-16] ---
        # Only sees walls if we are below building height
        lidar = np.ones(8, dtype=np.float32)
        if me.position[2] < self.BUILDING_HEIGHT:
            lidar = self._cast_lidar(me.position[:2])
            
        # --- 4. RWR (CLOSEST THREAT - SMART SORTING) [17-22] ---
        # We calculate a "Danger Score" based on Closing Speed and Distance.
        
        most_dangerous_missile = None
        highest_danger_score = -float('inf')
        
        for m in self.interceptors:
            # 1. Vector to Missile (3D)
            vec_to_missile = m.position - me.position
            dist = np.linalg.norm(vec_to_missile) + 0.1 # Safety epsilon
            
            # 2. Relative Velocity (3D)
            # (Missile Velocity - My Velocity)
            rel_vel = m.velocity - me.velocity
            
            # 3. Calculate Closing Speed (Doppler)
            # Project relative velocity onto the direction vector
            dir_to_missile = vec_to_missile / dist
            
            # Dot Product:
            # If missile flying AT me: Dot is Negative.
            # If missile flying AWAY: Dot is Positive.
            # We want Positive "Closing Speed", so we negate.
            closing_speed = -np.dot(rel_vel, dir_to_missile)
            
            # 4. Danger Score Formula
            if closing_speed > 0:
                # IT IS COMING AT US.
                # High Speed / Low Distance = Max Danger
                score = closing_speed / dist
            else:
                # IT IS FLYING AWAY.
                # Low score, but keep closer ones slightly higher priority just in case
                score = -dist 
            
            if score > highest_danger_score:
                highest_danger_score = score
                most_dangerous_missile = m
        
        # Build the Observation Vector
        missile_data = np.zeros(6, dtype=np.float32)
        
        if most_dangerous_missile:
            # Relative Pos (Normalized ~400)
            rel_pos = (most_dangerous_missile.position - me.position) / 400.0
            
            # Relative Vel (Normalized ~600)
            rel_vel = (most_dangerous_missile.velocity - me.velocity) / 600.0
            
            missile_data = np.concatenate([rel_pos, rel_vel])
            # Clip for Neural Network stability
            missile_data = np.clip(missile_data, -1.0, 1.0)

        # --- 5. TEAMMATES [23-34] ---
        # We track up to 2 teammates. If they are dead or don't exist, use 0.
        mates = []
        for i, ag in enumerate(self.agents):
            if i == agent_idx: continue # Skip self
            
            if ag.active:
                # Relative Pos
                p_diff = (ag.position - me.position) / 800.0
                # [NEW] Relative Vel (Crucial for formation!)
                v_diff = (ag.velocity - me.velocity) / 360.0
                
                # Add 6 values per teammate (3 Pos + 3 Vel)
                mates.extend(np.concatenate([p_diff, v_diff]))
            else:
                mates.extend(np.zeros(6)) # Dead teammate filler
        
        # Ensure we handled exactly 2 teammates (12 floats)
        # If n_agents > 3, this list grows, so we slice or pad.
        # For n_agents=3, this loop runs 2 times -> 12 floats.
        mates_arr = np.array(mates, dtype=np.float32)

        # --- 6. THREAT SECTORS [35-42] ---
        # 8-Directional "Radar Warning"
        # 1.0 = Safe (Far/No Threat), 0.0 = Danger (Close)
        threat_sectors = np.ones(8, dtype=np.float32)
        
        detection_radius = 400.0
        sector_angle = (2 * np.pi) / 8
        
        for m in self.interceptors:
            # Check 3D Distance
            dist = np.linalg.norm(m.position - me.position)
            
            # If out of range, ignore
            if dist > detection_radius: continue
            
            # Check LOS (If behind a wall and low, radar doesn't see it)
            # We reuse the helper method we defined earlier
            if not self._check_los_between(me, m): continue
            
            # Calculate Angle in 2D plane
            vec = m.position[:2] - me.position[:2]
            angle = np.arctan2(vec[1], vec[0]) # Returns -pi to pi
            
            # Normalize angle to 0 to 2pi
            if angle < 0: angle += 2 * np.pi
            
            # Determine Sector Index (0 to 7)
            idx = int(angle / sector_angle)
            idx = idx % 8 
            
            # Normalize Distance (0.0 = Impact, 1.0 = Far)
            norm_dist = dist / detection_radius
            
            # Update sector: Keep the CLOSEST threat in that sector
            if norm_dist < threat_sectors[idx]:
                threat_sectors[idx] = norm_dist
        
        # --- 7. ALTITUDE STATE [43] ---
        is_high = 1.0 if me.position[2] > self.BUILDING_HEIGHT else 0.0

        
        # --- ASSEMBLE ---
        # 6 + 3 + 8 + 6 + 12 + 8 + 1 = 44 inputs
        obs = np.concatenate([
            pos_norm,       # 0-2
            vel_norm,       # 3-5
            gps,            # 6-8
            lidar,          # 9-16
            missile_data,   # 17-22
            mates_arr,   # 23-34
            threat_sectors, # 35-42
            [is_high]       # 43
        ])
        
        # Pad to 45
        padding = np.zeros(self.obs_dim_per_agent - len(obs))
        return np.concatenate([obs, padding]).astype(np.float32)

    def _cast_lidar(self, pos_xy):
        # ... (Standard Raycast logic against self.urban_walls) ...
        readings = []
        angles = np.linspace(0, 6.28, 8, endpoint=False)
        for ang in angles:
            dx, dy = np.cos(ang), np.sin(ang)
            p1 = pos_xy
            p2 = pos_xy + np.array([dx, dy]) * 200.0
            min_d = 1.0
            for w in self.urban_walls:
                hit = w.clipline(p1, p2)
                if hit:
                    d = np.linalg.norm(np.array(hit[0]) - p1) / 200.0
                    if d < min_d: min_d = d
            readings.append(min_d)
        return np.array(readings, dtype=np.float32)
    
    def _check_los_between(self, ent1, ent2):
        """
        2.5D Line of Sight.
        - If EITHER entity is above Building Height, LOS is OPEN (Radar sees over walls).
        - If BOTH are Low, walls block LOS.
        """
        # 1. Altitude Check (Radar Horizon)
        if ent1.position[2] > self.BUILDING_HEIGHT or ent2.position[2] > self.BUILDING_HEIGHT:
            return True
            
        # 2. Wall Check (Urban Canyon)
        p1 = ent1.position[:2]
        p2 = ent2.position[:2]
        for w in self.urban_walls:
            if w.clipline(p1, p2):
                return False
        return True

    def _generate_urban_manhattan(self):
        """
        Phase 3 Manhattan Logic adapted for 2.5D.
        FIXED: Scales correctly to 1200x1200 map.
        """
        self.urban_walls = []
        
        # [FIX] Use dynamic rows (1200 / 10 = 120)
        rows = self.ROWS 
        pixel_size = 10
        
        # Config
        block_size = 14  # 140px buildings
        street_width = 6 # 60px streets
        step = block_size + street_width
        
        # Random Shift (Prevent overfitting coordinates)
        shift_x = np.random.randint(0, 6)
        shift_y = np.random.randint(0, 6)
        
        # Iterate over the FULL map
        for grid_x in range(2, rows - step, step):
            for grid_y in range(2, rows - step, step):
                
                # [FIX] Higher Density: Only 5% chance of a gap (was 15%)
                # This forces the agent to deal with walls constantly.
                if np.random.random() < 0.05: 
                    continue 
                
                gx, gy = grid_x + shift_x, grid_y + shift_y
                
                # Bounds check
                if gx >= rows-2 or gy >= rows-2: continue
                
                bx = gx * pixel_size
                by = gy * pixel_size
                bw = block_size * pixel_size
                bh = block_size * pixel_size
                
                big_wall = pygame.Rect(bx, by, bw, bh)
                self.urban_walls.append(big_wall)
        
        # Clear Center Plaza (Safe spawn zone)
        # Scaled up slightly for the larger map
        center_x = self.SCREEN_WIDTH // 2
        center_y = self.SCREEN_HEIGHT // 2
        center_rect = pygame.Rect(center_x - 150, center_y - 150, 300, 300)
        
        self.urban_walls = [w for w in self.urban_walls if not w.colliderect(center_rect)]


    def render(self):
        if self.render_mode != "human": return
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Drone Swarm 2.5D - MARL")
            
        self.screen.fill((30, 30, 30)) # Dark Grey Floor (Ground Level)
        
        # 1. Draw Buildings
        for w in self.urban_walls:
            # Base
            pygame.draw.rect(self.screen, (50, 50, 60), w)
            # Roof (Lighter, slightly offset to fake 3D perspective?)
            # Let's keep it simple: Inner rect
            pygame.draw.rect(self.screen, (70, 70, 80), w.inflate(-10, -10))
            
        # 2. Draw Hazard Source (SAM Site)
        pygame.draw.circle(self.screen, (200, 0, 0), self.hazard_source, 15)
        
        # 3. Draw Target
        pygame.draw.circle(self.screen, (0, 255, 0), self.target.center, 20)
        pygame.draw.circle(self.screen, (0, 150, 0), self.target.center, 25, 2)
        
        # 4. Draw Agents
        for ag in self.agents:
            if not ag.active: continue
            
            # Altitude Visuals
            # Scale: Z=0 -> 10px, Z=100 -> 18px (Parallax effect)
            scale = 10 + (ag.position[2] / 12.0)
            
            # Color: 
            # Green = NOE (Safe from Radar)
            # Cyan = High Altitude (Radar Lock!)
            c = (0, 255, 0) if ag.position[2] < self.BUILDING_HEIGHT else (0, 255, 255)
            
            cx, cy = int(ag.position[0]), int(ag.position[1])
            
            # Draw Drone Body
            pygame.draw.circle(self.screen, c, (cx, cy), int(scale))
            
            # Draw Height Bar (Yellow bar next to drone)
            # Height of bar represents Z
            h_bar = int(ag.position[2] / 3)
            pygame.draw.rect(self.screen, (255, 255, 0), (cx + 12, cy - h_bar, 4, h_bar))
            
            # Draw Agent ID
            # font = pygame.font.SysFont(None, 20)
            # img = font.render(str(ag.uid), True, (0,0,0))
            # self.screen.blit(img, (cx-3, cy-5))

        # 5. Draw Missiles
        for m in self.interceptors:
            cx, cy = int(m.position[0]), int(m.position[1])
            
            # Missile Altitude Visualization
            # Red = Low, Pink = High
            c = (255, 0, 0) if m.position[2] < self.BUILDING_HEIGHT else (255, 100, 100)
            pygame.draw.circle(self.screen, c, (cx, cy), 6)
            
        pygame.display.flip()
        self.clock.tick(60)
        
    def close(self):
        pygame.quit()