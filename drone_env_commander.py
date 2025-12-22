"""
DRONE ENVIRONMENT COMMANDER (v8 - MULTI-AGENT PHYSICS)
======================================================
Updates:
- Integrated SAMSite (Turret Physics).
- Multi-Agent Observations (Teammate awareness).
- "Tracked" Signal (Agent knows if SAM is aiming at it).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

from physics import Agent, Interceptor, SAMSite, BUILDING_HEIGHT
from navigator_v2 import Navigator

class DroneEnvCommander(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, render_mode=None, n_agents=1):
        super().__init__()
        
        self.SCREEN_SIZE = 1200
        self.n_agents = n_agents
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        self.PHYSICS_TICKS_PER_STEP = 30
        self.DT = 0.016
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_agents * 3,), dtype=np.float32)
        
        # Obs: [Self(4) + Target(3) + Threat(4) + Teammate(5) + SAM_Status(2)] = 18 floats
        # Padded to 20 for roundness
        self.obs_dim = 20
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.n_agents * self.obs_dim,), dtype=np.float32)
        
        self.agents = [Agent(i, 0, 0) for i in range(self.n_agents)]
        self.navigators = [Navigator() for _ in range(self.n_agents)]
        self.walls = []
        self.interceptors = []
        self.target_pos = np.array([600.0, 600.0, 10.0])
        self.sam_site = None # Created in reset
        
        self.current_step_count = 0
        self.max_steps = 400 
        
        # States
        self.previous_positions = [np.zeros(3) for _ in range(n_agents)]
        self.stuck_timers = [0 for _ in range(n_agents)]
        self.last_command_waypoints = [np.zeros(3) for _ in range(n_agents)]
        
        self._generate_urban_map()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step_count = 0
        self.interceptors = []
        self._generate_urban_map()
        
        # Create SAM at target
        self.sam_site = SAMSite(self.target_pos[0], self.target_pos[1])
        
        for i, nav in enumerate(self.navigators):
            nav.reset()
            self.stuck_timers[i] = 0
            self.previous_positions[i] = np.zeros(3)
        
        for ag in self.agents:
            ag.active = True
            ag.velocity[:] = 0
            ag.current_thrust_vector[:] = 0
            safe = False
            while not safe:
                rx = np.random.randint(50, self.SCREEN_SIZE - 50)
                ry = np.random.randint(50, self.SCREEN_SIZE - 50)
                if pygame.Rect(rx-15, ry-15, 30, 30).collidelist(self.walls) == -1:
                    if np.linalg.norm([rx - self.target_pos[0], ry - self.target_pos[1]]) > 500:
                        ag.reset(rx, ry)
                        safe = True
        return self._get_obs(), {}

    def step(self, action):
        self.current_step_count += 1
        rewards = 0.0
        actions = action.reshape((self.n_agents, 3))
        command_waypoints = []
        
        # 1. COMMANDS
        for i, ag in enumerate(self.agents):
            if not ag.active:
                command_waypoints.append(ag.position)
                continue
            
            # Map actions
            dist_cmd = (actions[i][0] + 1) * 200.0
            angle_cmd = actions[i][1] * np.pi
            alt_cmd = actions[i][2]
            
            vec_to_target = self.target_pos - ag.position
            base_angle = np.arctan2(vec_to_target[1], vec_to_target[0])
            final_angle = base_angle + angle_cmd
            
            wx = ag.position[0] + math.cos(final_angle) * dist_cmd
            wy = ag.position[1] + math.sin(final_angle) * dist_cmd
            wz = 80.0 if alt_cmd > 0 else 10.0
            
            wx = np.clip(wx, 0, self.SCREEN_SIZE)
            wy = np.clip(wy, 0, self.SCREEN_SIZE)
            command_waypoints.append(np.array([wx, wy, wz]))

        self.last_command_waypoints = command_waypoints

        # 2. PHYSICS LOOP
        for _ in range(self.PHYSICS_TICKS_PER_STEP):
            # SAM LOGIC
            new_missile = self.sam_site.update(self.DT, self.agents, self.walls)
            if new_missile:
                self.interceptors.append(new_missile)
            
            # AGENTS
            for i, ag in enumerate(self.agents):
                if ag.active:
                    force = self.navigators[i].get_control_force(
                        ag, command_waypoints[i], self.walls, self.interceptors
                    )
                    ag.update(self.DT, force, self.walls, self.SCREEN_SIZE)
            
            # MISSILES
            for m in self.interceptors:
                m.update(self.DT, self.walls)
                for ag in self.agents:
                    if ag.active and m.check_hit(ag):
                        ag.active = False
                        rewards -= 50.0
            self.interceptors = [m for m in self.interceptors if m.active]
            
            if self.render_mode == "human": self.render()

        # 3. REWARDS & LOGIC
        alive_count = 0
        dist_sum = 0
        
        for i, ag in enumerate(self.agents):
            if ag.active:
                alive_count += 1
                
                # Stuck check
                dist_moved = np.linalg.norm(ag.position - self.previous_positions[i])
                if dist_moved < 2.0: self.stuck_timers[i] += 1
                else: self.stuck_timers[i] = 0
                self.previous_positions[i] = ag.position.copy()
                
                if self.stuck_timers[i] > 10:
                    ag.active = False
                    rewards -= 20.0
                    continue

                d = np.linalg.norm(ag.position - self.target_pos)
                dist_sum += d
                if d < 40.0:
                    rewards += 100.0
                    ag.active = False 

        if alive_count > 0: 
            rewards -= (dist_sum / alive_count / self.SCREEN_SIZE)
        
        rewards -= 0.1
        terminated = (alive_count == 0)
        truncated = (self.current_step_count >= self.max_steps)
        
        return self._get_obs(), rewards, terminated, truncated, {"alive": alive_count}

    def _check_los(self, start, end):
        if start[2] > 50 or end[2] > 50: return True
        for w in self.walls:
            if w.clipline(start[:2], end[:2]): return False
        return True

    def _get_obs(self):
        obs_list = []
        for i, ag in enumerate(self.agents):
            if not ag.active:
                obs_list.extend(np.zeros(self.obs_dim))
                continue
            
            agent_obs = []
            
            # 1. Self (4)
            pos_norm = ag.position / self.SCREEN_SIZE
            agent_obs.extend(np.clip(pos_norm, 0, 1)) 
            agent_obs.append(1.0 if ag.position[2] > 50 else -1.0)
            
            # 2. Target (3)
            vec_t = self.target_pos - ag.position
            dist_t = np.linalg.norm(vec_t)
            angle_t = np.arctan2(vec_t[1], vec_t[0])
            norm_dist_t = min(dist_t / self.SCREEN_SIZE, 1.0)
            agent_obs.extend([norm_dist_t, np.cos(angle_t), np.sin(angle_t)])
            
            # 3. Threat (4)
            closest_m = None
            min_dist_m = 9999.0
            for m in self.interceptors:
                d = np.linalg.norm(m.position - ag.position)
                if d < min_dist_m and self._check_los(ag.position, m.position):
                    min_dist_m = d
                    closest_m = m
            
            if closest_m:
                vec_m = closest_m.position - ag.position
                ang_m = np.arctan2(vec_m[1], vec_m[0])
                norm_dist_m = min(min_dist_m / 500.0, 1.0)
                rel_vel = closest_m.velocity - ag.velocity
                dir_to_m = vec_m / (min_dist_m + 0.01)
                closing_speed = -np.dot(rel_vel, dir_to_m)
                norm_closing = np.clip(closing_speed / 1000.0, -1.0, 1.0)
                agent_obs.extend([norm_dist_m, np.cos(ang_m), np.sin(ang_m), norm_closing])
            else:
                agent_obs.extend([1.0, 0.0, 0.0, -1.0])
            
            # 4. Teammate Info (NEW - 3 floats)
            # Find nearest ally
            ally = None
            min_d_ally = 9999.0
            for j, other in enumerate(self.agents):
                if i == j or not other.active: continue
                d = np.linalg.norm(other.position - ag.position)
                if d < min_d_ally:
                    min_d_ally = d
                    ally = other
            
            if ally:
                # Relative vector
                vec_a = ally.position - ag.position
                dist_a = min(min_d_ally / self.SCREEN_SIZE, 1.0)
                ang_a = np.arctan2(vec_a[1], vec_a[0])
                agent_obs.extend([dist_a, np.cos(ang_a), np.sin(ang_a)])
            else:
                agent_obs.extend([0.0, 0.0, 0.0]) # No ally/dead

            # 5. SAM Status (NEW - 2 floats)
            # "Is the turret looking at me?"
            vec_to_sam = self.sam_site.pos - ag.position
            angle_to_sam = math.atan2(vec_to_sam[1], vec_to_sam[0])
            # The turret angle is absolute (0=East). 
            # We need to rotate it by 180 to compare with "Angle TO sam".
            # Actually, simpler: Vector from SAM to Agent vs SAM Angle.
            vec_sam_to_ag = ag.position - self.sam_site.pos
            angle_sam_to_ag = math.atan2(vec_sam_to_ag[1], vec_sam_to_ag[0])
            
            angle_diff = angle_sam_to_ag - self.sam_site.angle
            # Normalize -PI to PI
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
            
            # 1.0 = Focused on me, 0.0 = Looking away
            is_focused = max(0, 1.0 - abs(angle_diff))
            
            # SAM State: 1=Locking/Firing (Danger), 0=Scanning/Reloading (Safe)
            is_dangerous = 1.0 if self.sam_site.state in ["TRACKING", "LOCKING"] else 0.0
            
            agent_obs.extend([is_focused, is_dangerous])

            # Dynamic Padding (Goal 20)
            current_len = len(agent_obs)
            pad_len = self.obs_dim - current_len
            if pad_len > 0: agent_obs.extend([0.0]*pad_len)
            
            obs_list.extend(agent_obs)

        return np.clip(np.array(obs_list, dtype=np.float32), -1.0, 1.0)

    def _generate_urban_map(self):
        self.walls = []
        block_size = 100
        street_width = 80
        num_blocks = self.SCREEN_SIZE // (block_size + street_width)
        center = self.SCREEN_SIZE / 2
        
        for i in range(num_blocks):
            for j in range(num_blocks):
                x = i * (block_size + street_width) + street_width
                y = j * (block_size + street_width) + street_width
                if np.linalg.norm([x + block_size/2 - center, y + block_size/2 - center]) < 250: continue
                if np.random.random() < 0.15: continue
                w = block_size + np.random.randint(-10, 20)
                h = block_size + np.random.randint(-10, 20)
                self.walls.append(pygame.Rect(x, y, w, h))
        self.target_pos = np.array([center, center, 10.0])

    def render(self):
        if self.render_mode is None: return
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_SIZE, self.SCREEN_SIZE))
            self.clock = pygame.time.Clock()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.close(); return
            
        self.screen.fill((20, 20, 30)) 
        for w in self.walls:
            pygame.draw.rect(self.screen, (40, 40, 50), w.move(5, 5)) 
            pygame.draw.rect(self.screen, (70, 70, 80), w) 
            
        pygame.draw.circle(self.screen, (0, 100, 0), (int(self.target_pos[0]), int(self.target_pos[1])), 40, 2)
        pygame.draw.circle(self.screen, (0, 255, 0), (int(self.target_pos[0]), int(self.target_pos[1])), 10)
        
        # Draw SAM Turret
        sam_end = (int(self.sam_site.pos[0] + math.cos(self.sam_site.angle)*30),
                   int(self.sam_site.pos[1] + math.sin(self.sam_site.angle)*30))
        turret_color = (255, 0, 0) if self.sam_site.state in ["LOCKING", "FIRING"] else (100, 100, 100)
        pygame.draw.line(self.screen, turret_color, (int(self.sam_site.pos[0]), int(self.sam_site.pos[1])), sam_end, 3)
        
        for i, ag in enumerate(self.agents):
            if ag.active:
                cx, cy = int(ag.position[0]), int(ag.position[1])
                color = (0, 255, 255) if ag.position[2] > 50 else (255, 165, 0)
                pygame.draw.circle(self.screen, color, (cx, cy), 10)
                end_x = cx + ag.velocity[0] * 0.1
                end_y = cy + ag.velocity[1] * 0.1
                pygame.draw.line(self.screen, (255, 255, 255), (cx, cy), (end_x, end_y), 2)
                
                cmd_wp = self.last_command_waypoints[i]
                wx, wy = int(cmd_wp[0]), int(cmd_wp[1])
                pygame.draw.line(self.screen, (0, 100, 255), (wx-5, wy-5), (wx+5, wy+5), 2)
                pygame.draw.line(self.screen, (0, 100, 255), (wx-5, wy+5), (wx+5, wy-5), 2)
                
                nav = self.navigators[i]
                if nav.cached_path:
                    path_points = [(int(p[0]), int(p[1])) for p in nav.cached_path]
                    if len(path_points) > 1:
                        pygame.draw.lines(self.screen, (255, 255, 0), False, path_points, 2)

        for m in self.interceptors:
            mx, my = int(m.position[0]), int(m.position[1])
            pygame.draw.circle(self.screen, (255, 50, 50), (mx, my), 6)
            
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen: pygame.quit()