import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from grid import Grid
from agent import Agent
from projectile import Projectile
from algorithm import a_star_algorithm
from swarm import FollowerDrone
import os

class DroneEnv(gym.Env):
    """
    Combat Drone Environment.
    Task: Navigate maze, dodge bullets, reach target.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None):
        super(DroneEnv, self).__init__()

        # --- CONFIGURATION ---
        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 800
        self.PIXEL_SIZE = 10      # Size of one square (10px)
        self.ROWS = int(self.SCREEN_WIDTH / self.PIXEL_SIZE) # 80 Rows

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # NEW: Make it slightly smaller (34px) to fit through doors easily
        agent_size = 14
        self.agent = Agent(200, 200, agent_size)

        self.num_followers = 5

        # --- ACTION SPACE ---
        # [Force_X, Force_Y]
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # --- OBSERVATION SPACE ---
        # 1. Lidar (8)
        # 2. Velocity (2)
        # 3. GPS to Next Waypoint (2)
        # 4. Vector to Closest Bullet (2) <--- NEW
        # Total = 14 inputs
        self.observation_space = spaces.Box(low=-1, high=1, shape=(14,), dtype=np.float32)

        # --- ENV SETUP ---
        # PASS 'ROWS' (80), NOT PIXEL_SIZE (10)
        self.grid = Grid(self.ROWS, self.SCREEN_WIDTH) 

        self.target = pygame.Rect(600, 600, 40, 40)
        
        self.projectiles = []
        self.shoot_timer = 0.0
        self.walls = []

        # --- LOGGING VARS ---
        self.cum_reward_dist = 0
        self.cum_reward_time = 0
        self.cum_penalty_collision = 0
        self.cum_reward_win = 0 # <--- ADD THIS
        self.episode_count = 0
        self.termination_reason = "Timeout" # Default reason

        # --- TIMEOUT CONFIG ---
        self.max_steps = 2000  # 2000 frames @ 60fps = ~33 seconds
        self.current_step = 0

        # --- PATHFINDING TIMER ---
        self.repath_steps = 0
        self.repath_interval = 20 # Run A* every 20 frames (approx 0.3 seconds)
        
        # --- LOG FILE SETUP ---
        self.log_file = "training_log.csv"
        # Create file with header if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write("Episode,DistReward,TimeReward,CollReward,TotalReward,Reason\n")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Log Previous Episode Stats
        if self.episode_count > 0:
            total = self.cum_reward_dist + self.cum_reward_time + self.cum_penalty_collision + self.cum_reward_win
            
            # Console Print
            print(f"Ep {self.episode_count} | Dist: {self.cum_reward_dist:.1f} | Time: {self.cum_reward_time:.1f} | Coll: {self.cum_penalty_collision} | TOT: {total:.1f} | {self.termination_reason}")
            
            # File Write (Append Mode)
            with open(self.log_file, "a") as f:
                f.write(f"{self.episode_count},{self.cum_reward_dist:.2f},{self.cum_reward_time:.2f},{self.cum_penalty_collision},{total:.2f},{self.termination_reason}\n")

        # 2. Reset Counters & State
        self.cum_reward_dist = 0
        self.cum_reward_time = 0
        self.cum_penalty_collision = 0
        self.cum_reward_win = 0 # <--- RESET THIS
        self.episode_count += 1
        self.termination_reason = "Timeout"
        
        self.current_step = 0
        self.repath_steps = 0 # <--- Reset Repath Timer
        
        # 3. Environment Reset
        self.grid = Grid(self.ROWS, self.SCREEN_WIDTH)
        self._randomize_walls()
        self.walls = self.grid.get_obstacle_rects()
        
        self.projectiles = []
        self.shoot_timer = 2.0 
        
        self.agent.velocity[:] = 0
        self.agent.acceleration[:] = 0
        self.agent.path = []
        
        self._spawn_entities()
        self._recalc_path()
        
        # 4. Initialize Differential Tracker
        self.last_dist = np.linalg.norm(np.array(self.target.center) - self.agent.position)
        
        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        # 1. Increment Counters
        self.current_step += 1
        self.repath_steps += 1 # <--- Increment Repath Timer
        
        # 2. Apply Physics
        force = action * self.agent.max_force
        self.agent.apply_force(force)
        self.agent.update_physics(0.016)
        self.agent.rect.center = self.agent.position
        
        self.agent._handle_collisions(self.walls)

        # --- UPDATE SWARM ---
        # Fixed dt = 0.016
        for drone in self.followers:
            drone.update(0.016, self.agent.position, self.followers, self.walls)
        
        # 3. Combat Logic
        self.shoot_timer -= 0.016
        if self.shoot_timer <= 0:
            self.shoot_timer = 2.0
            p = Projectile(self.target.center, self.agent.position)
            self.projectiles.append(p)
            
        hit_bullet = False
        for p in self.projectiles[:]:
            p.update(0.016, self.walls)
            if not p.active:
                self.projectiles.remove(p)
                continue
            
            if self.agent.rect.colliderect(p.rect):
                hit_bullet = True
                self.projectiles.remove(p)

        # 4. Path Logic (Dynamic Updates)
        
        # A. Periodic Repathing (The Fix)
        if self.repath_steps >= self.repath_interval:
            self.repath_steps = 0
            self._recalc_path()

        # B. Waypoint Management
        if self.agent.path:
            target_node = self.agent.path[0]
            t_pos = np.array([target_node.x + self.agent.size/2, target_node.y + self.agent.size/2])
            
            # If close to node, pop it
            if np.linalg.norm(t_pos - self.agent.position) < 30:
                self.agent.path.pop(0)
        
        # C. Emergency Recalc (if path empty but not at target)
        if not self.agent.path: 
            self._recalc_path()

        # 5. Reward Calculation
        step_reward = 0
        terminated = False
        truncated = False
        
        # Progress Reward
        current_dist = np.linalg.norm(np.array(self.target.center) - self.agent.position)
        progress = self.last_dist - current_dist
        self.last_dist = current_dist
        
        r_dist = progress * 0.1 
        step_reward += r_dist
        self.cum_reward_dist += r_dist
        
        # Time Penalty
        r_time = -0.05 
        step_reward += r_time
        self.cum_reward_time += r_time
        
        # Combat Penalty
        if hit_bullet:
            r_coll = -50
            step_reward += r_coll
            self.cum_penalty_collision += r_coll
            terminated = True
            self.termination_reason = "Died"
            print("   -> DIED (Shot)")
            
        # Victory Reward
        if self.agent.rect.colliderect(self.target):
            r_win = 100
            step_reward += r_win
            self.cum_reward_win += r_win # <--- ADD THIS
            terminated = True
            self.termination_reason = "Success"
            print("   -> SUCCESS")

        # 6. Timeout Check
        if not terminated:
            if self.current_step >= self.max_steps:
                truncated = True
                self.termination_reason = "Timeout"

        # 7. Finalize
        observation = self._get_obs()
        info = {}
        
        if self.render_mode == "human":
            self.render()
            
        return observation, step_reward, terminated, truncated, info

    def _get_obs(self):
        # 1. Lidar (8)
        lidar = self.agent.cast_rays(self.walls, num_rays=8, max_dist=200)
        
        # 2. Velocity (2)
        vel = self.agent.velocity / self.agent.max_speed
        
        # 3. GPS (2)
        if self.agent.path:
            node = self.agent.path[0]
            wp = np.array([node.x + self.agent.size/2, node.y + self.agent.size/2])
        else:
            wp = np.array(self.target.center)
        vec_to_wp = wp - self.agent.position
        dist_wp = np.linalg.norm(vec_to_wp)
        if dist_wp > 0: vec_to_wp /= dist_wp
        
        # 4. Bullet Sensor (2) <--- UPDATED WITH LINE OF SIGHT
        # Find closest bullet THAT IS VISIBLE
        closest_bullet_vec = np.array([0.0, 0.0])
        min_dist = 400.0 # Sensing range
        
        for p in self.projectiles:
            vec = p.position - self.agent.position
            dist = np.linalg.norm(vec)
            
            # Check distance first (Cheap)
            if dist < min_dist:
                # Check Line of Sight (Expensive but necessary)
                # This prevents the agent from dodging bullets behind walls
                if self.agent._has_line_of_sight(p.position, self.walls):
                    min_dist = dist
                    closest_bullet_vec = vec / dist # Normalized direction to threat
                
        return np.concatenate([lidar, vel, vec_to_wp, closest_bullet_vec]).astype(np.float32)

    def _randomize_walls(self):
        rows = self.grid.rows # Should be 80 now
        
        block_size = 10   # 10 * 10px = 100px buildings
        street_width = 6  # 6 * 10px = 60px streets
        
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
        
        # 1. Spawn Leader (Existing Code)
        while True:
            rx = np.random.randint(50, self.SCREEN_WIDTH - 50)
            ry = np.random.randint(50, self.SCREEN_HEIGHT - 50)
            self.agent.position = np.array([float(rx), float(ry)])
            self.agent.rect.center = self.agent.position
            if self.agent.rect.collidelist(self.walls) == -1: break
            
        # 2. Spawn Followers
        self.followers = []
        for _ in range(self.num_followers):
            drone = FollowerDrone(self.agent.position[0], self.agent.position[1], self.agent.size)
            # Add tiny random offset so they separate quickly
            offset = np.random.uniform(-20, 20, size=2)
            drone.position += offset
            self.followers.append(drone)
            
        # 3. Target
        while True:
            tx = np.random.randint(50, self.SCREEN_WIDTH - 50)
            ty = np.random.randint(50, self.SCREEN_HEIGHT - 50)
            self.target.topleft = (tx, ty)
            if (self.target.collidelist(self.walls) == -1 and 
                not self.target.colliderect(self.agent.rect)): break

    def _recalc_path(self):
        # Clear visuals
        for row in self.grid.grid:
            for node in row: 
                node.reset_visuals()
                node.update_neighbors(self.grid.grid)
                
        start_node = self.grid.get_node_from_pos(self.agent.position)
        end_node = self.grid.get_node_from_pos(self.target.center)
        
        if start_node and end_node:
            self.agent.path = a_star_algorithm(None, self.grid, start_node, end_node)

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                exit()

        self.screen.fill((255, 255, 255))
        self.grid.draw(self.screen)
        
        if self.agent.path:
            for node in self.agent.path: node.make_path()
            
        pygame.draw.rect(self.screen, (255, 0, 0), self.target)

        # Draw Followers
        for drone in self.followers:
            drone.draw(self.screen)
            
        # Draw Leader
        self.agent.draw_lidar(self.screen, self.walls)
        self.agent.draw(self.screen)
        
        # Draw Projectiles
        for p in self.projectiles:
            p.draw(self.screen)
            
        self.agent.draw_lidar(self.screen, self.walls)
        self.agent.draw(self.screen)
        
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            pygame.quit()