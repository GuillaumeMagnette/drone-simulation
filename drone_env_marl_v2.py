"""
DRONE ENVIRONMENT 2.5D - EMERGENT SWARM TACTICS (v2)
=====================================================

Philosophy Change: Outcome-Based Rewards, Not Behavior-Based.

We DO NOT reward:
- Spreading out (angle diversity)
- Staying close to team (cohesion)  
- Approaching from different angles (flanking)

We DO reward:
- Hitting the target (mission success)
- Hitting the target SIMULTANEOUSLY (saturation bonus)
- Staying alive (implicit: dead agents can't score)

The wolf pack behavior should EMERGE from:
1. Saturation bonus incentivizes synchronized arrival
2. Missile threats incentivize spreading (don't all die at once)
3. The tension between these creates natural flanking

Observation Space (Per Agent = 50 floats):
------------------------------------------
[0-2]    Position (Norm X, Y, Z)
[3-5]    Velocity (Norm VX, VY, VZ)
[6-8]    Target Vector (Direction + Distance encoded)
[9-16]   Lidar (8 Rays)
[17-22]  Nearest Missile (Rel Pos + Rel Vel)
[23-40]  Teammate Data (2 mates × 9 floats each):
         - Relative Position (3)
         - Relative Velocity (3)  
         - Their distance to target (1) <- NEW: coordination signal
         - Their altitude state (1) <- NEW: tactical awareness
         - Is alive flag (1)
[41-48]  Threat Sectors (8 directions)
[49]     My Altitude State (high/low)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

# ==========================================
# CONFIGURATION
# ==========================================
BUILDING_HEIGHT = 50.0
CEILING = 100.0
GRAVITY = 15.0
LIFT_FORCE = 30.0


class Agent25D:
    """Simplified agent with event bus for crash detection."""
    
    def __init__(self, x, y, uid):
        self.uid = uid
        self.active = True
        self.stealth_timer = 0.0
        self.events = {"crashed": False, "scored": False}
        
        self.position = np.array([float(x), float(y), 10.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0, 0.0])
        
        self.size = 14
        self.rect = pygame.Rect(x, y, 14, 14)
        
        self.mass = 1.0
        self.max_speed_xy = 360.0
        self.max_speed_z = 100.0
        self.max_force = 2500.0
        self.friction = 0.95

    def update(self, dt, action, walls, screen_size):
        self.events["crashed"] = False
        self.events["scored"] = False
        
        if not self.active:
            if self.stealth_timer > 0:
                self.stealth_timer -= dt
            return
            
        if self.stealth_timer > 0:
            self.stealth_timer -= dt

        # Physics
        fx = action[0] * self.max_force
        fy = action[1] * self.max_force
        fz = action[2] * LIFT_FORCE - GRAVITY * self.mass
        
        self.acceleration = np.array([fx, fy, fz]) / self.mass
        self.velocity += self.acceleration * dt
        
        # Speed limits
        v_xy = self.velocity[:2]
        speed_xy = np.linalg.norm(v_xy)
        if speed_xy > self.max_speed_xy:
            self.velocity[:2] = (v_xy / speed_xy) * self.max_speed_xy
        self.velocity[2] = np.clip(self.velocity[2], -self.max_speed_z, self.max_speed_z)
        
        # Apply friction only to XY (horizontal drag)
        # Z-axis has no friction - drones have strong vertical authority
        self.velocity[0] *= self.friction
        self.velocity[1] *= self.friction
        # self.velocity[2] keeps its value (no Z friction)
        
        next_pos = self.position + self.velocity * dt
        
        # Z constraints
        if next_pos[2] < 5:
            next_pos[2] = 5
            self.velocity[2] = 0
        elif next_pos[2] > CEILING:
            next_pos[2] = CEILING
            self.velocity[2] = 0
            
        # Wall collision (only when flying low)
        hit_wall = False
        if next_pos[2] <= BUILDING_HEIGHT:
            test_rect = pygame.Rect(next_pos[0]-7, next_pos[1]-7, 14, 14)
            if (test_rect.left < 0 or test_rect.right > screen_size or
                test_rect.top < 0 or test_rect.bottom > screen_size):
                hit_wall = True
            elif test_rect.collidelist(walls) != -1:
                hit_wall = True
                
        if hit_wall:
            self.active = False
            self.events["crashed"] = True
            self.position = np.array([-1000.0, -1000.0, 0.0])
        else:
            self.position = next_pos
            
        self.rect.center = (int(self.position[0]), int(self.position[1]))

    def respawn(self, walls, hazard_pos, screen_size):
        """Respawn at safe location."""
        self.active = True
        self.stealth_timer = 2.0
        self.velocity[:] = 0
        self.acceleration[:] = 0
        
        for _ in range(200):
            rx = np.random.randint(50, screen_size - 50)
            ry = np.random.randint(50, screen_size - 50)
            
            test_rect = pygame.Rect(rx-15, ry-15, 30, 30)
            if test_rect.collidelist(walls) != -1:
                continue
                
            dist = np.linalg.norm([rx - hazard_pos[0], ry - hazard_pos[1]])
            if dist > 350:
                self.position = np.array([float(rx), float(ry), 10.0])
                self.rect.center = (rx, ry)
                return
                
        self.position = np.array([50.0, 50.0, 10.0])


class Interceptor3D:
    """Homing missile with LOS-based tracking."""
    
    def __init__(self, x, y, z):
        self.position = np.array([float(x), float(y), float(z)])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.alive = True
        self.lifetime = 4.0
        self.speed = 550.0
        self.turn_rate = 1200.0
        self.target_ref = None
        self.size = 10
        self.rect = pygame.Rect(x, y, 10, 10)

    def update(self, dt, agents, walls):
        if not self.alive:
            return
            
        self.lifetime -= dt
        if self.lifetime <= 0:
            self.alive = False
            return
            
        # Re-acquire target if lost
        if self.target_ref and self.target_ref.active:
            if not self._has_los(self.target_ref, walls):
                self.target_ref = None
                
        if not self.target_ref:
            closest_dist = 9999
            for ag in agents:
                if not ag.active:
                    continue
                dist = np.linalg.norm(ag.position - self.position)
                if dist < closest_dist and self._has_los(ag, walls):
                    closest_dist = dist
                    self.target_ref = ag
                    
        # Pursuit physics
        accel = np.array([0.0, 0.0, 0.0])
        if self.target_ref:
            des_dir = self.target_ref.position - self.position
            dist = np.linalg.norm(des_dir)
            if dist > 0:
                accel = (des_dir / dist) * self.turn_rate
                
        self.velocity += accel * dt
        speed = np.linalg.norm(self.velocity)
        if speed > self.speed:
            self.velocity = (self.velocity / speed) * self.speed
            
        self.position += self.velocity * dt
        
        if self.position[2] < 5:
            self.alive = False
            
        # Wall collision when low
        if self.position[2] < BUILDING_HEIGHT:
            self.rect.center = (int(self.position[0]), int(self.position[1]))
            if self.rect.collidelist(walls) != -1:
                self.alive = False
                
        # Hit detection
        for ag in agents:
            if ag.active and np.linalg.norm(ag.position - self.position) < 20:
                self.alive = False
                ag.active = False
                ag.events["crashed"] = True

    def _has_los(self, agent, walls):
        # 2.5D Line of Sight:
        # - If EITHER entity is above building height, LOS is open (can see over buildings)
        # - If BOTH are below building height, walls can block LOS
        
        # High altitude = always visible (flying above buildings)
        if agent.position[2] > BUILDING_HEIGHT:
            return True
        if self.position[2] > BUILDING_HEIGHT:
            return True
            
        # Both are low - check wall occlusion
        p1 = (self.position[0], self.position[1])
        p2 = (agent.position[0], agent.position[1])
        for w in walls:
            if w.clipline(p1, p2):
                return False
        return True


class DroneEnvMARL_V2(gym.Env):
    """
    Emergent Swarm Tactics Environment.
    
    Key Design Principles:
    1. Sparse rewards focused on outcomes, not behaviors
    2. Saturation bonus rewards simultaneous target hits
    3. Observations include teammate intent (velocity + target distance)
    4. Curriculum support via options dict
    """
    
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None, n_agents=3):
        super().__init__()
        
        # Configuration
        self.SCREEN_SIZE = 1200
        self.BUILDING_HEIGHT = BUILDING_HEIGHT
        self.CEILING = CEILING
        self.render_mode = render_mode
        self.screen = None
        
        # Agents
        self.n_agents = n_agents
        self.agents = [Agent25D(0, 0, i) for i in range(self.n_agents)]
        
        # Action: [Fx, Fy, Fz] per agent
        self.action_space = spaces.Box(
            low=-1, high=1, 
            shape=(self.n_agents * 3,), 
            dtype=np.float32
        )
        
        # Observation: 50 floats per agent
        self.obs_dim_per_agent = 50
        self.observation_space = spaces.Box(
            low=-1, high=1,
            shape=(self.n_agents * self.obs_dim_per_agent,),
            dtype=np.float32
        )
        
        # World
        self.urban_walls = []
        self.target = pygame.Rect(0, 0, 40, 40)
        self.hazard_source = (600, 600)
        self.interceptors = []
        self.missile_timer = 0.0
        self.max_steps = 2000
        
        # Curriculum flags (set via reset options)
        self.active_threats = True
        self.max_missiles = 2
        
        # Statistics for this episode
        self.episode_stats = {
            "targets_hit": 0,
            "saturation_hits": 0,  # 2+ agents hitting simultaneously
            "deaths": 0,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.interceptors = []
        self.missile_timer = 2.0
        
        # Reset stats
        self.episode_stats = {"targets_hit": 0, "saturation_hits": 0, "deaths": 0}
        
        # Parse curriculum options
        if options:
            self.active_threats = options.get('active_threats', True)
            self.max_missiles = options.get('max_missiles', 2)
            
            # Dynamic agent count for curriculum
            new_n = options.get('n_agents', self.n_agents)
            if new_n != self.n_agents:
                self.n_agents = new_n
                self.agents = [Agent25D(0, 0, i) for i in range(self.n_agents)]
                self.action_space = spaces.Box(
                    low=-1, high=1,
                    shape=(self.n_agents * 3,),
                    dtype=np.float32
                )
                self.observation_space = spaces.Box(
                    low=-1, high=1,
                    shape=(self.n_agents * self.obs_dim_per_agent,),
                    dtype=np.float32
                )
        
        # Generate world
        self._generate_urban_manhattan()
        self._spawn_hazard()
        self._relocate_target()
        
        # Spawn agents safely
        for ag in self.agents:
            ag.active = True
            ag.velocity[:] = 0
            ag.stealth_timer = 0.0
            
            for _ in range(100):
                ax = np.random.randint(50, self.SCREEN_SIZE - 50)
                ay = np.random.randint(50, self.SCREEN_SIZE - 50)
                
                t_rect = pygame.Rect(ax-10, ay-10, 20, 20)
                if t_rect.collidelist(self.urban_walls) == -1:
                    dist_to_hazard = np.linalg.norm([
                        ax - self.hazard_source[0],
                        ay - self.hazard_source[1]
                    ])
                    if dist_to_hazard > 300:
                        ag.position = np.array([float(ax), float(ay), 10.0])
                        ag.rect.center = (ax, ay)
                        break
                        
        return self._get_combined_obs(), {}

    def step(self, action):
        self.current_step += 1
        agent_actions = action.reshape((self.n_agents, 3))
        
        # ============================================================
        # 1. SAFETY SHIELD (Minimal - just prevent ground crash)
        # ============================================================
        processed_actions = []
        for i, ag in enumerate(self.agents):
            user_action = agent_actions[i].copy()
            
            if ag.active and ag.position[2] < 10:
                # Force climb if too low
                user_action[2] = max(user_action[2], 0.5)
                
            processed_actions.append(user_action)
        
        # ============================================================
        # 2. PHYSICS UPDATE
        # ============================================================
        for i, ag in enumerate(self.agents):
            ag.update(0.016, processed_actions[i], self.urban_walls, self.SCREEN_SIZE)
        
        team_alive = sum(1 for ag in self.agents if ag.active)
        
        # Team wipe -> respawn all
        if team_alive == 0:
            for ag in self.agents:
                ag.respawn(self.urban_walls, self.hazard_source, self.SCREEN_SIZE)
            self.missile_timer = 3.0
            self.interceptors = []
            team_alive = self.n_agents
        
        # ============================================================
        # 3. THREAT SPAWNING
        # ============================================================
        if self.active_threats:
            self.missile_timer -= 0.016
            if self.missile_timer <= 0 and team_alive > 0:
                if len(self.interceptors) < self.max_missiles:
                    self.missile_timer = 2.0 + np.random.random()
                    self.interceptors.append(Interceptor3D(
                        self.hazard_source[0],
                        self.hazard_source[1],
                        10.0
                    ))
        
        # Update missiles
        for m in self.interceptors:
            m.update(0.016, self.agents, self.urban_walls)
        self.interceptors = [m for m in self.interceptors if m.alive]
        
        # ============================================================
        # 4. REWARDS (OUTCOME-BASED, NOT BEHAVIOR-BASED)
        # ============================================================
        reward = 0.0
        
        # --- A. SATURATION BONUS (The Core Mechanic) ---
        # Count how many agents hit the target THIS FRAME
        agents_scoring = []
        for ag in self.agents:
            if ag.active:
                dist = np.linalg.norm(ag.position[:2] - np.array(self.target.center))
                # Must be close AND low (actually touching the target)
                if dist < 35 and ag.position[2] < 25:
                    agents_scoring.append(ag)
        
        n_scoring = len(agents_scoring)
        if n_scoring > 0:
            # SATURATION BONUS TABLE:
            # The reward scales NON-LINEARLY with simultaneous hits
            # This creates pressure to coordinate timing
            #
            # 1 agent:  20 points (baseline - hard to achieve in combat)
            # 2 agents: 60 points (3x baseline - coordination pays off)
            # 3 agents: 120 points (6x baseline - perfect saturation)
            #
            # Why this works:
            # - In flight school: 1 agent can score easily, gets 20
            # - In combat: SAM kills lone agents, so 1-agent scores are rare
            # - Spreading out lets 2-3 agents survive, triggering big bonus
            
            saturation_bonus = {
                1: 20.0,
                2: 100.0,
                3: 180.0,
            }
            reward += saturation_bonus.get(n_scoring, 20.0 * n_scoring)
            
            # Stats tracking
            self.episode_stats["targets_hit"] += n_scoring
            if n_scoring >= 2:
                self.episode_stats["saturation_hits"] += 1
            
            # Move target, mark agents as having scored
            self._relocate_target()
            for ag in agents_scoring:
                ag.events["scored"] = True
        
        # --- B. DEATH PENALTY (Simple, Clear) ---
        for ag in self.agents:
            if ag.events["crashed"]:
                reward -= 25.0
                self.episode_stats["deaths"] += 1
        
        # --- C. TIME PRESSURE (Forces Progress) ---
        # Small constant drain encourages efficiency
        reward -= 0.02
        
        # --- D. TEAM WIPE (Catastrophic Failure) ---
        if team_alive == 0:
            reward -= 30.0
        
        # ============================================================
        # 5. TERMINATION
        # ============================================================
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        info = {
            "episode_stats": self.episode_stats.copy(),
            "team_alive": team_alive,
        }
        
        return self._get_combined_obs(), reward, terminated, truncated, info

    # ============================================================
    # OBSERVATION BUILDING
    # ============================================================
    
    def _get_combined_obs(self):
        return np.concatenate([
            self._get_single_agent_obs(i) for i in range(self.n_agents)
        ])

    def _get_single_agent_obs(self, agent_idx):
        me = self.agents[agent_idx]
        
        if not me.active:
            return np.zeros(self.obs_dim_per_agent, dtype=np.float32)
        
        obs_parts = []
        
        # --- 1. SELF STATE [0-5] ---
        pos_norm = me.position / np.array([self.SCREEN_SIZE, self.SCREEN_SIZE, CEILING])
        vel_norm = me.velocity / np.array([360.0, 360.0, 100.0])
        obs_parts.extend(pos_norm)
        obs_parts.extend(vel_norm)
        
        # --- 2. TARGET VECTOR [6-8] ---
        target_vec = np.array(self.target.center) - me.position[:2]
        target_dist = np.linalg.norm(target_vec)
        if target_dist > 0:
            target_dir = target_vec / target_dist
        else:
            target_dir = np.array([0.0, 0.0])
        # Encode distance (normalized to ~800 max useful range)
        target_dist_norm = min(target_dist / 800.0, 1.0)
        obs_parts.extend([target_dir[0], target_dir[1], target_dist_norm])
        
        # --- 3. LIDAR [9-16] ---
        if me.position[2] < BUILDING_HEIGHT:
            lidar = self._cast_lidar(me.position[:2])
        else:
            lidar = np.ones(8, dtype=np.float32)  # Flying high, no walls
        obs_parts.extend(lidar)
        
        # --- 4. NEAREST MISSILE [17-22] ---
        missile_data = self._get_nearest_missile_obs(me)
        obs_parts.extend(missile_data)
        
        # --- 5. TEAMMATE DATA [23-40] ---
        # Each teammate: 9 floats
        # - Relative position (3)
        # - Relative velocity (3)
        # - Their distance to target, normalized (1)
        # - Their altitude state (1)
        # - Is alive (1)
        teammates = [ag for i, ag in enumerate(self.agents) if i != agent_idx]
        
        for tm in teammates[:2]:  # Max 2 teammates
            if tm.active:
                # Relative position
                rel_pos = (tm.position - me.position) / 600.0
                rel_pos = np.clip(rel_pos, -1, 1)
                
                # Relative velocity (crucial for predicting intent)
                rel_vel = (tm.velocity - me.velocity) / 400.0
                rel_vel = np.clip(rel_vel, -1, 1)
                
                # Their distance to target (coordination signal!)
                # "Is my teammate about to score?"
                tm_dist = np.linalg.norm(tm.position[:2] - np.array(self.target.center))
                tm_dist_norm = min(tm_dist / 800.0, 1.0)
                
                # Their altitude state
                tm_high = 1.0 if tm.position[2] > BUILDING_HEIGHT else 0.0
                
                obs_parts.extend(rel_pos)
                obs_parts.extend(rel_vel)
                obs_parts.append(tm_dist_norm)
                obs_parts.append(tm_high)
                obs_parts.append(1.0)  # Is alive
            else:
                obs_parts.extend([0.0] * 9)  # Dead teammate
        
        # Pad if fewer than 2 teammates
        while len(teammates) < 2:
            obs_parts.extend([0.0] * 9)
            teammates.append(None)  # Just for counting
        
        # --- 6. THREAT SECTORS [41-48] ---
        threat_sectors = self._get_threat_sectors(me)
        obs_parts.extend(threat_sectors)
        
        # --- 7. ALTITUDE STATE [49] ---
        is_high = 1.0 if me.position[2] > BUILDING_HEIGHT else 0.0
        obs_parts.append(is_high)
        
        # Convert to array and ensure correct size
        obs = np.array(obs_parts, dtype=np.float32)
        
        # Pad or truncate to exact size
        if len(obs) < self.obs_dim_per_agent:
            obs = np.concatenate([obs, np.zeros(self.obs_dim_per_agent - len(obs))])
        elif len(obs) > self.obs_dim_per_agent:
            obs = obs[:self.obs_dim_per_agent]
            
        return obs

    def _get_nearest_missile_obs(self, me):
        """Return observation for the most dangerous missile."""
        if not self.interceptors:
            return np.zeros(6, dtype=np.float32)
        
        most_dangerous = None
        highest_danger = -float('inf')
        
        for m in self.interceptors:
            vec_to_m = m.position - me.position
            dist = np.linalg.norm(vec_to_m) + 0.1
            
            rel_vel = m.velocity - me.velocity
            dir_to_m = vec_to_m / dist
            closing_speed = -np.dot(rel_vel, dir_to_m)
            
            if closing_speed > 0:
                danger = closing_speed / dist
            else:
                danger = -dist
                
            if danger > highest_danger:
                highest_danger = danger
                most_dangerous = m
        
        if most_dangerous is None:
            return np.zeros(6, dtype=np.float32)
            
        rel_pos = (most_dangerous.position - me.position) / 400.0
        rel_vel = (most_dangerous.velocity - me.velocity) / 600.0
        
        return np.clip(np.concatenate([rel_pos, rel_vel]), -1, 1)

    def _get_threat_sectors(self, me):
        """8-directional threat radar."""
        sectors = np.ones(8, dtype=np.float32)  # 1.0 = safe
        detection_radius = 400.0
        sector_angle = (2 * np.pi) / 8
        
        for m in self.interceptors:
            dist = np.linalg.norm(m.position - me.position)
            if dist > detection_radius:
                continue
                
            # Check LOS
            if me.position[2] <= BUILDING_HEIGHT and m.position[2] <= BUILDING_HEIGHT:
                p1 = me.position[:2]
                p2 = m.position[:2]
                blocked = any(w.clipline(p1, p2) for w in self.urban_walls)
                if blocked:
                    continue
            
            vec = m.position[:2] - me.position[:2]
            angle = np.arctan2(vec[1], vec[0])
            if angle < 0:
                angle += 2 * np.pi
                
            idx = int(angle / sector_angle) % 8
            norm_dist = dist / detection_radius
            
            if norm_dist < sectors[idx]:
                sectors[idx] = norm_dist
                
        return sectors

    def _cast_lidar(self, pos_xy):
        """8-ray lidar for wall detection."""
        readings = []
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        
        for ang in angles:
            dx, dy = np.cos(ang), np.sin(ang)
            p1 = pos_xy
            p2 = pos_xy + np.array([dx, dy]) * 200.0
            min_d = 1.0
            
            for w in self.urban_walls:
                hit = w.clipline(p1, p2)
                if hit:
                    d = np.linalg.norm(np.array(hit[0]) - p1) / 200.0
                    if d < min_d:
                        min_d = d
                        
            readings.append(min_d)
            
        return np.array(readings, dtype=np.float32)

    # ============================================================
    # WORLD GENERATION
    # ============================================================
    
    def _generate_urban_manhattan(self):
        """Generate city blocks with streets."""
        self.urban_walls = []
        
        pixel_size = 10
        block_size = 14  # 140px buildings
        street_width = 6  # 60px streets
        step = block_size + street_width
        rows = self.SCREEN_SIZE // pixel_size
        
        shift_x = np.random.randint(0, 6)
        shift_y = np.random.randint(0, 6)
        
        for grid_x in range(2, rows - step, step):
            for grid_y in range(2, rows - step, step):
                if np.random.random() < 0.08:  # 8% gaps
                    continue
                    
                gx, gy = grid_x + shift_x, grid_y + shift_y
                if gx >= rows - 2 or gy >= rows - 2:
                    continue
                    
                bx = gx * pixel_size
                by = gy * pixel_size
                bw = block_size * pixel_size
                bh = block_size * pixel_size
                
                self.urban_walls.append(pygame.Rect(bx, by, bw, bh))
        
        # Clear center plaza
        center = self.SCREEN_SIZE // 2
        plaza = pygame.Rect(center - 150, center - 150, 300, 300)
        self.urban_walls = [w for w in self.urban_walls if not w.colliderect(plaza)]

    def _spawn_hazard(self):
        """Spawn SAM site in valid location."""
        for _ in range(100):
            x = np.random.randint(200, self.SCREEN_SIZE - 200)
            y = np.random.randint(200, self.SCREEN_SIZE - 200)
            t_rect = pygame.Rect(x-20, y-20, 40, 40)
            if t_rect.collidelist(self.urban_walls) == -1:
                self.hazard_source = (x, y)
                return
        self.hazard_source = (600, 600)

    def _relocate_target(self):
        """Move target to new valid location."""
        for _ in range(100):
            x = np.random.randint(50, self.SCREEN_SIZE - 50)
            y = np.random.randint(50, self.SCREEN_SIZE - 50)
            
            t_rect = pygame.Rect(x-10, y-10, 20, 20)
            if t_rect.collidelist(self.urban_walls) != -1:
                continue
                
            # Not too close to hazard
            if np.linalg.norm([x - self.hazard_source[0], y - self.hazard_source[1]]) > 250:
                self.target = pygame.Rect(x-20, y-20, 40, 40)
                return
                
        self.target = pygame.Rect(
            self.SCREEN_SIZE//2 - 20,
            self.SCREEN_SIZE//2 - 20,
            40, 40
        )

    # ============================================================
    # RENDERING
    # ============================================================
    
    def render(self):
        if self.render_mode != "human":
            return
            
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_SIZE, self.SCREEN_SIZE))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Emergent Swarm Tactics v2")
            
        self.screen.fill((30, 30, 30))
        
        # Buildings
        for w in self.urban_walls:
            pygame.draw.rect(self.screen, (50, 50, 60), w)
            pygame.draw.rect(self.screen, (70, 70, 80), w.inflate(-10, -10))
            
        # SAM site
        pygame.draw.circle(self.screen, (200, 0, 0), self.hazard_source, 15)
        pygame.draw.circle(self.screen, (150, 0, 0), self.hazard_source, 25, 2)
        
        # Target
        pygame.draw.circle(self.screen, (0, 255, 0), self.target.center, 20)
        pygame.draw.circle(self.screen, (0, 150, 0), self.target.center, 30, 2)
        
        # Agents
        for ag in self.agents:
            if not ag.active:
                continue
                
            scale = 10 + (ag.position[2] / 12.0)
            color = (0, 255, 0) if ag.position[2] < BUILDING_HEIGHT else (0, 255, 255)
            
            cx, cy = int(ag.position[0]), int(ag.position[1])
            pygame.draw.circle(self.screen, color, (cx, cy), int(scale))
            
            # Height indicator
            h_bar = int(ag.position[2] / 3)
            pygame.draw.rect(self.screen, (255, 255, 0), (cx + 12, cy - h_bar, 4, h_bar))
            
            # Agent ID
            if pygame.font.get_init():
                font = pygame.font.SysFont(None, 18)
                label = font.render(str(ag.uid), True, (255, 255, 255))
                self.screen.blit(label, (cx - 4, cy - 5))
        
        # Missiles
        for m in self.interceptors:
            cx, cy = int(m.position[0]), int(m.position[1])
            color = (255, 0, 0) if m.position[2] < BUILDING_HEIGHT else (255, 100, 100)
            pygame.draw.circle(self.screen, color, (cx, cy), 6)
            
            # Missile trail
            if np.linalg.norm(m.velocity) > 0:
                trail_end = m.position[:2] - (m.velocity[:2] / np.linalg.norm(m.velocity[:2]) * 20)
                pygame.draw.line(self.screen, (255, 100, 0), 
                               (cx, cy), 
                               (int(trail_end[0]), int(trail_end[1])), 2)
        
        # Stats overlay
        if pygame.font.get_init():
            font = pygame.font.SysFont(None, 24)
            stats_text = f"Targets: {self.episode_stats['targets_hit']} | Saturation: {self.episode_stats['saturation_hits']} | Deaths: {self.episode_stats['deaths']}"
            label = font.render(stats_text, True, (255, 255, 255))
            self.screen.blit(label, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# ============================================================
# REGISTRATION (for gym.make())
# ============================================================
if __name__ == "__main__":
    # Quick test
    env = DroneEnvMARL_V2(render_mode="human", n_agents=3)
    obs, _ = env.reset(options={'active_threats': True})
    
    print(f"Observation shape: {obs.shape}")
    print(f"Action shape: {env.action_space.shape}")
    
    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        env.render()
        
        if term or trunc:
            print(f"Episode ended: {info}")
            obs, _ = env.reset()
            
    env.close()
