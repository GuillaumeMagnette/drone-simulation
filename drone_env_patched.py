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



class DroneEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None):
        super(DroneEnv, self).__init__()
        
        # --- SCREEN ---
        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 800
        self.PIXEL_SIZE = 10
        self.ROWS = self.SCREEN_WIDTH // self.PIXEL_SIZE
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # --- AGENT ---
        self.agent = Agent(200, 200, size=14)
        
        # --- SPACES ---
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        # 36 inputs for MARL-ready architecture
        self.observation_space = spaces.Box(low=-1, high=1, shape=(36,), dtype=np.float32)
        
        # --- WORLD ---
        self.grid = Grid(self.ROWS, self.SCREEN_WIDTH)
        self.target = pygame.Rect(600, 600, 40, 40)
        self.walls = []
        self.interceptors = []
        # --- MISSILE BATTERY CONFIG ---
        self.missile_timer = 0.0
        self.missile_interval = 1.0  # Seconds between launches
        self.max_interceptors = 5    # Default, overridden by curriculum in reset()

        self.projectiles = []  # Reserved for future use
        
        # --- CURRICULUM DEFAULTS ---
        self.default_map_type = 'arena'
        self.default_num_interceptors = 1
        self.current_map_type = 'arena'
        
        # --- CONFIG ---
        self.repath_interval = 20
        self.max_steps = 2000
        
        # --- LOGGING ---
        self.cum_reward_dist = 0
        self.cum_reward_time = 0
        self.cum_penalty_collision = 0
        self.cum_reward_win = 0
        self.episode_count = 0
        self.termination_reason = "Timeout"
        
        # Logging (can be disabled by setting log_file = None)
        
        self.log_file = "training_log.csv"
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write("Episode,Map,Interceptors,DistRew,TimeRew,CollPen,WinRew,Total,Reason\n")

    def reset(self, seed=None, options=None):
        """
        Reset environment with curriculum options.
        
        options = {
            'map_type': 'arena' | 'sparse' | 'urban',
            'num_interceptors': int
        }
        """
        super().reset(seed=seed)
        
        # Parse options
        map_type = self.default_map_type
        num_interceptors = self.default_num_interceptors
        
        if options:
            map_type = options.get('map_type', self.default_map_type)
            num_interceptors = options.get('num_interceptors', self.default_num_interceptors)
        
        self.current_map_type = map_type
        
        # Log previous episode (skip if logging disabled)
        if self.episode_count > 0 and self.log_file:
            total = self.cum_reward_dist + self.cum_reward_time + self.cum_penalty_collision + self.cum_reward_win
            with open(self.log_file, "a") as f:
                f.write(f"{self.episode_count},{self.current_map_type},{len(self.interceptors)},"
                       f"{self.cum_reward_dist:.2f},{self.cum_reward_time:.2f},"
                       f"{self.cum_penalty_collision:.2f},{self.cum_reward_win:.2f},"
                       f"{total:.2f},{self.termination_reason}\n")
        
        # Reset counters
        self.cum_reward_dist = 0
        self.cum_reward_time = 0
        self.cum_penalty_collision = 0
        self.cum_reward_win = 0
        self.episode_count += 1
        self.termination_reason = "Timeout"
        self.current_step = 0
        self.repath_steps = 0
        
        # Generate map
        # --- THE SPEED FIX ---
        # OLD: self.grid = Grid(self.ROWS, self.SCREEN_WIDTH)
        # NEW: Reuse the existing object
        self.grid.clear() 
        
        if map_type == 'arena':
            self._generate_arena()
        elif map_type == 'sparse':
            self._generate_sparse()
        else:  # 'urban'
            self._generate_urban()

        # --- NEW: APPLY COSTMAP ---
        # Do this AFTER generating walls, but BEFORE finding path
        self.grid.apply_costmap()
        # --------------------------
            
        self.walls = self.grid.get_obstacle_rects()
        
        
        # Reset entities
        self.interceptors = []
        self.missile_timer = 0.0  # Ready to fire immediately when LOS acquired
        self.projectiles = []
        
        # Set max interceptors based on curriculum (disables battery if 0)
        self.max_interceptors = num_interceptors
        
        # Spawn agent and target first
        self._spawn_agent_and_target()
        
        # NOTE: Don't spawn interceptors here anymore!
        # The missile battery in step() handles staggered spawning.
        # This creates sustained pressure instead of an initial burst.
        
        # Reset agent state
        self.agent.velocity[:] = 0
        self.agent.acceleration[:] = 0
        self.agent.path = []
        
        # Pathfinding (skip in arena)
        if map_type != 'arena':
            self._recalc_path()
        
        self.last_dist = np.linalg.norm(np.array(self.target.center) - self.agent.position)
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        self.repath_steps += 1
        
        # --- PHYSICS ---
        force = action * self.agent.max_force
        self.agent.apply_force(force)
        self.agent.update_physics(0.016)
        self.agent.rect.center = self.agent.position
        
        # --- WALL COLLISION (LETHAL) ---
        hit_wall = False
        
        if (self.agent.rect.left < 0 or self.agent.rect.right > self.SCREEN_WIDTH or
            self.agent.rect.top < 0 or self.agent.rect.bottom > self.SCREEN_HEIGHT):
            hit_wall = True
            
        if not hit_wall and self.agent.rect.collidelist(self.walls) != -1:
            hit_wall = True
        
        # --- MISSILE BATTERY LOGIC ---
        self.missile_timer -= 0.016
        if self.missile_timer <= 0:
            # Reset timer with slight randomness for variety
            self.missile_timer = self.missile_interval + np.random.uniform(-0.2, 0.3)
            
            # Count only ALIVE interceptors
            alive_interceptors = sum(1 for i in self.interceptors if i.alive)
            
            if alive_interceptors < self.max_interceptors:
                # Check distance - don't fire at point-blank range
                dist_to_agent = np.linalg.norm(
                    self.agent.position - np.array(self.target.center)
                )
                too_close = dist_to_agent < 75  # ~2 seconds of flight time
                
                # Only spawn if target has line of sight to agent
                # (Simulates "lock-on" - can't fire blind)
                can_fire = not too_close
                if can_fire and self.current_map_type != 'arena':
                    can_fire = self.agent._has_line_of_sight(
                        self.target.center, self.walls
                    )
                
                if can_fire:
                    self._spawn_interceptor()
        
        # --- UPDATE INTERCEPTORS ---
        hit_interceptor = False
        for interceptor in self.interceptors:
            if not interceptor.alive: continue
            
            interceptor.update(0.016, [self.agent], self.walls)
            if interceptor.alive and self.agent.rect.colliderect(interceptor.rect):
                hit_interceptor = True
                interceptor._die()
        
        # --- PATHFINDING ---
        if self.current_map_type != 'arena':
            if self.repath_steps >= self.repath_interval:
                self.repath_steps = 0
                self._recalc_path()
            
            if self.agent.path:
                node = self.agent.path[0]
                node_pos = np.array([node.x + 7, node.y + 7])
                if np.linalg.norm(node_pos - self.agent.position) < 20:
                    self.agent.path.pop(0)
            if not self.agent.path: self._recalc_path()
        
        # --- OBSERVATION ---
        obs = self._get_obs()
        
        # --- REWARDS ---
        reward = 0
        terminated = False
        truncated = False
        
        # 1. Progress Reward (Kept your tuning)
        cur_dist = np.linalg.norm(np.array(self.target.center) - self.agent.position)
        progress = self.last_dist - cur_dist
        self.last_dist = cur_dist
        reward += progress * 0.15
        self.cum_reward_dist += progress * 0.15
        
        # 2. Time Penalty (Kept your tuning)
        reward -= 0.1
        self.cum_reward_time -= 0.1
        
        # 3. Interceptor proximity
        for interceptor in self.interceptors:
            if not interceptor.alive: continue
            if self.current_map_type != 'arena':
                if not self.agent._has_line_of_sight(interceptor.position, self.walls):
                    continue
            dist = np.linalg.norm(self.agent.position - interceptor.position)
            if dist < 200.0:
                intensity = 1.0 - (dist / 200.0)
                reward -= intensity * 0.3
        
        # 4. Stall penalty
        speed = np.linalg.norm(self.agent.velocity)
        if speed < 5 and cur_dist > 50.0:
            reward -= 0.5 if self.current_map_type == 'arena' else 1.0
        
        # 5. Scrape Penalty (Corner Protection)
        # 0.15 is approx 30 pixels. Good for corners.
        if np.min(obs[0:8]) < 0.15:
            reward -= 0.5

        # --- 6. NEW: PREDICTIVE CRASH PENALTY (The Fix) ---
        # Look ahead 0.2 seconds based on current velocity
        lookahead_vec = self.agent.velocity * 0.2 
        future_pos = self.agent.position + lookahead_vec
        
        # Create a test rect at future position
        future_rect = self.agent.rect.copy()
        future_rect.center = future_pos
        
        # Check if that future position hits a wall
        # We also check screen bounds manually
        future_hit = False
        if (future_rect.left < 0 or future_rect.right > self.SCREEN_WIDTH or
            future_rect.top < 0 or future_rect.bottom > self.SCREEN_HEIGHT):
            future_hit = True
        elif future_rect.collidelist(self.walls) != -1:
            future_hit = True
            
        if future_hit:
            # We are on a collision course!
            # Punish based on speed (High speed crash = worse)
            reward -= 2.0 * (speed / self.agent.max_speed)
            # This tells the AI: "BRAKE NOW!"

        
        # --- TERMINAL STATES ---
        
        if hit_wall:
            reward -= 50 
            self.cum_penalty_collision -= 50
            terminated = True
            self.termination_reason = "Crashed"
            # NEW: Trigger explosion visual
            if self.render_mode == "human" and hasattr(self, 'explosion_manager'):
                self.explosion_manager.add(self.agent.position[0], self.agent.position[1])

        elif hit_interceptor:
            reward -= 100
            self.cum_penalty_collision -= 100
            terminated = True
            self.termination_reason = "Caught"
            # NEW: Trigger explosion visual
            if self.render_mode == "human" and hasattr(self, 'explosion_manager'):
                self.explosion_manager.add(self.agent.position[0], self.agent.position[1])
        
        elif self.agent.rect.colliderect(self.target):
            reward += 100
            self.cum_reward_win += 100
            terminated = True
            self.termination_reason = "Success"
        
        elif self.current_step >= self.max_steps:
            truncated = True
            self.termination_reason = "Timeout"
        
        if self.render_mode == "human":
            self.render()
        
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        """
        36-Input Observation Vector:
        [0-7]   Lidar - Wall distances (8 sectors)
        [8-9]   Velocity - Normalized current velocity
        [10-11] GPS - Direction to next waypoint
        [12-19] Threats - Interceptor distances (8 sectors)
        [20-27] Projectiles - Bullet distances (8 sectors, reserved)
        [28-35] Neighbors - Friendly drone distances (8 sectors, MARL)
        """
        # 1. Lidar (8)
        lidar = self.agent.cast_rays(self.walls, num_rays=8, max_dist=200)
        
        # 2. Velocity (2)
        vel = self.agent.velocity / self.agent.max_speed
        
        # 3. GPS (2) - Direction to waypoint or target
        if self.current_map_type == 'arena':
            waypoint = np.array(self.target.center, dtype=float)
        elif self.agent.path:
            node = self.agent.path[0]
            waypoint = np.array([node.x + 7, node.y + 7])
        else:
            waypoint = np.array(self.target.center, dtype=float)
        
        vec_to_wp = waypoint - self.agent.position
        dist_to_wp = np.linalg.norm(vec_to_wp)
        if dist_to_wp > 0:
            vec_to_wp /= dist_to_wp
        
        # 4. Threat Sectors (8) - Interceptors
        visible_threats = []
        for interceptor in self.interceptors:
            if not interceptor.alive:
                continue
            if self.current_map_type == 'arena':
                visible_threats.append(interceptor)
            elif self.agent._has_line_of_sight(interceptor.position, self.walls):
                visible_threats.append(interceptor)
        
        threat_sectors = self.agent.get_sector_readings(visible_threats, radius=400.0, num_sectors=8)
        
        # 5. Projectile Sectors (8) - Reserved, all safe for now
        projectile_sectors = np.ones(8, dtype=np.float32)
        
        # 6. Neighbor Sectors (8) - Reserved for MARL, all safe for now
        neighbor_sectors = np.ones(8, dtype=np.float32)
        
        return np.concatenate([
            lidar,              # 0-7
            vel,                # 8-9
            vec_to_wp,          # 10-11
            threat_sectors,     # 12-19
            projectile_sectors, # 20-27
            neighbor_sectors    # 28-35
        ]).astype(np.float32)

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
        """Spawns a missile near the Target."""
        tx, ty = self.target.center
        
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
