"""
DRONE ENVIRONMENT COMMANDER (v6 - DEBUG VISUALS)
================================================
Updates:
- Visualization of the Commander's "Ghost Waypoint" (Blue)
- Visualization of the Navigator's A* Path (Yellow)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

from physics import Agent, Interceptor, BUILDING_HEIGHT
from navigator import Navigator

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
        
        # [Dist, Angle, Alt_Mode]
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_agents * 3,), dtype=np.float32)
        self.obs_dim = 20
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.n_agents * self.obs_dim,), dtype=np.float32)
        
        self.agents = [Agent(i, 0, 0) for i in range(self.n_agents)]
        self.navigators = [Navigator() for _ in range(self.n_agents)]
        self.walls = []
        self.interceptors = []
        self.target_pos = np.array([600.0, 600.0, 10.0])
        
        self.current_step_count = 0
        self.max_steps = 400 
        
        # DEBUG STORAGE
        self.last_command_waypoints = [np.zeros(3) for _ in range(n_agents)]
        
        self._generate_urban_map()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step_count = 0
        self.interceptors = []
        self._generate_urban_map()
        
        for nav in self.navigators:
            nav.pathfinder = None 
        
        for ag in self.agents:
            ag.active = True
            ag.velocity[:] = 0
            ag.current_thrust_vector[:] = 0
            safe = False
            while not safe:
                rx = np.random.randint(50, self.SCREEN_SIZE - 50)
                ry = np.random.randint(50, self.SCREEN_SIZE - 50)
                if pygame.Rect(rx-15, ry-15, 30, 30).collidelist(self.walls) == -1:
                    if np.linalg.norm([rx - self.target_pos[0], ry - self.target_pos[1]]) > 400:
                        ag.reset(rx, ry)
                        safe = True
        return self._get_obs(), {}

    def step(self, action):
        self.current_step_count += 1
        rewards = 0.0
        actions = action.reshape((self.n_agents, 3))
        
        # 1. DECODE COMMANDS & STORE FOR RENDER
        command_waypoints = []
        for i, ag in enumerate(self.agents):
            if not ag.active:
                command_waypoints.append(ag.position)
                continue
                
            dist_cmd = (actions[i][0] + 1) * 200.0  # 0..400m
            angle_cmd = actions[i][1] * np.pi       # -PI..PI
            alt_cmd = actions[i][2]                 # >0 High
            
            vec_to_target = self.target_pos - ag.position
            base_angle = np.arctan2(vec_to_target[1], vec_to_target[0])
            final_angle = base_angle + angle_cmd
            
            wx = ag.position[0] + math.cos(final_angle) * dist_cmd
            wy = ag.position[1] + math.sin(final_angle) * dist_cmd
            wz = 80.0 if alt_cmd > 0 else 10.0
            
            wx = np.clip(wx, 0, self.SCREEN_SIZE)
            wy = np.clip(wy, 0, self.SCREEN_SIZE)
            
            wp = np.array([wx, wy, wz])
            command_waypoints.append(wp)
            
        # Store for debug rendering
        self.last_command_waypoints = command_waypoints

        # 2. PHYSICS LOOP
        for _ in range(self.PHYSICS_TICKS_PER_STEP):
            if np.random.random() < 0.005: self._spawn_threat()
            
            for i, ag in enumerate(self.agents):
                if ag.active:
                    force = self.navigators[i].get_control_force(
                        ag, command_waypoints[i], self.walls, self.interceptors
                    )
                    ag.update(self.DT, force, self.walls, self.SCREEN_SIZE)
            
            for m in self.interceptors:
                m.update(self.DT, self.walls)
                for ag in self.agents:
                    if ag.active and m.check_hit(ag):
                        ag.active = False
                        rewards -= 50.0
            self.interceptors = [m for m in self.interceptors if m.active]
            
            # Internal Render Loop
            if self.render_mode == "human":
                self.render()

        # 3. REWARDS
        alive_count = 0
        dist_sum = 0
        for ag in self.agents:
            if ag.active:
                alive_count += 1
                d = np.linalg.norm(ag.position - self.target_pos)
                dist_sum += d
                if d < 40.0:
                    rewards += 100.0
                    ag.active = False 

        if alive_count > 0: rewards -= (dist_sum / alive_count / self.SCREEN_SIZE)
        rewards -= 0.1
        terminated = (alive_count == 0)
        truncated = (self.current_step_count >= self.max_steps)
        
        return self._get_obs(), rewards, terminated, truncated, {"alive": alive_count}

    def _get_obs(self):
        obs_list = []
        for i, ag in enumerate(self.agents):
            if not ag.active:
                obs_list.extend(np.zeros(self.obs_dim))
                continue
            
            # --- FEATURE COLLECTION ---
            agent_obs = []
            
            # 1. Self State (3 pos + 1 alt = 4)
            pos_norm = ag.position / self.SCREEN_SIZE
            agent_obs.extend(np.clip(pos_norm, 0, 1)) 
            agent_obs.append(1.0 if ag.position[2] > 50 else -1.0)
            
            # 2. Target Vector (3)
            vec_t = self.target_pos - ag.position
            dist_t = np.linalg.norm(vec_t)
            angle_t = np.arctan2(vec_t[1], vec_t[0])
            norm_dist_t = min(dist_t / self.SCREEN_SIZE, 1.0)
            agent_obs.extend([norm_dist_t, np.cos(angle_t), np.sin(angle_t)])
            
            # 3. Nearest Threat (3)
            closest_m = None
            min_dist_m = 9999.0
            for m in self.interceptors:
                d = np.linalg.norm(m.position - ag.position)
                if d < min_dist_m:
                    min_dist_m = d
                    closest_m = m
            
            if closest_m:
                vec_m = closest_m.position - ag.position
                ang_m = np.arctan2(vec_m[1], vec_m[0])
                norm_dist_m = min(min_dist_m / 500.0, 1.0)
                agent_obs.extend([norm_dist_m, np.cos(ang_m), np.sin(ang_m)])
            else:
                agent_obs.extend([1.0, 0.0, 0.0])
            
            # --- DYNAMIC PADDING ---
            # Calculate how much space is left in the fixed-size vector
            current_len = len(agent_obs)
            pad_len = self.obs_dim - current_len
            
            if pad_len > 0:
                agent_obs.extend([0.0] * pad_len)
            
            obs_list.extend(agent_obs)

        # Final reshape and clip to ensure strict bounds
        return np.clip(np.array(obs_list, dtype=np.float32), -1.0, 1.0)


    def _spawn_threat(self):
        if not self.agents: return
        actives = [a for a in self.agents if a.active]
        if not actives: return
        target_ag = np.random.choice(actives)
        angle = np.random.uniform(0, 2*np.pi)
        dist = np.random.uniform(50, 200)
        mx = self.target_pos[0] + np.cos(angle) * dist
        my = self.target_pos[1] + np.sin(angle) * dist
        self.interceptors.append(Interceptor(mx, my, 10, target_ag))

    def _generate_urban_map(self):
        self.walls = []
        block_size = 100
        street_width = 50
        num_blocks = self.SCREEN_SIZE // (block_size + street_width)
        center = self.SCREEN_SIZE / 2
        
        for i in range(num_blocks):
            for j in range(num_blocks):
                x = i * (block_size + street_width) + street_width
                y = j * (block_size + street_width) + street_width
                if np.linalg.norm([x + block_size/2 - center, y + block_size/2 - center]) < 250: continue
                if np.random.random() < 0.15: continue
                w = block_size + np.random.randint(-20, 30)
                h = block_size + np.random.randint(-20, 30)
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
        
        for i, ag in enumerate(self.agents):
            if ag.active:
                cx, cy = int(ag.position[0]), int(ag.position[1])
                color = (0, 255, 255) if ag.position[2] > 50 else (255, 165, 0)
                pygame.draw.circle(self.screen, color, (cx, cy), 10)
                end_x = cx + ag.velocity[0] * 0.1
                end_y = cy + ag.velocity[1] * 0.1
                pygame.draw.line(self.screen, (255, 255, 255), (cx, cy), (end_x, end_y), 2)
                
                # --- VISUALIZATION UPGRADE ---
                # 1. Draw Commander's "Ghost Waypoint" (Blue X)
                cmd_wp = self.last_command_waypoints[i]
                wx, wy = int(cmd_wp[0]), int(cmd_wp[1])
                pygame.draw.line(self.screen, (0, 100, 255), (wx-5, wy-5), (wx+5, wy+5), 2)
                pygame.draw.line(self.screen, (0, 100, 255), (wx-5, wy+5), (wx+5, wy-5), 2)
                # Dotted line to command
                pygame.draw.line(self.screen, (0, 50, 150), (cx, cy), (wx, wy), 1)

                # 2. Draw A* Path (Yellow)
                # Access the navigator's cached path
                nav = self.navigators[i]
                if nav.cached_path:
                    path_points = [(int(p[0]), int(p[1])) for p in nav.cached_path]
                    # Draw lines connecting the path nodes
                    if len(path_points) > 1:
                        pygame.draw.lines(self.screen, (255, 255, 0), False, path_points, 2)

        for m in self.interceptors:
            mx, my = int(m.position[0]), int(m.position[1])
            pygame.draw.circle(self.screen, (255, 50, 50), (mx, my), 6)
            
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen: pygame.quit()