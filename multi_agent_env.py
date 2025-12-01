import numpy as np
import pygame
from grid import Grid
from agent import Agent
from turret import Turret
from algorithm import a_star_algorithm

class MultiAgentSwarm:
    def __init__(self, num_drones=6):
        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 800
        self.GRID_SIZE = 10
        self.ROWS = 80
        
        self.grid = Grid(self.ROWS, self.SCREEN_WIDTH)
        
        # --- SWARM SETUP ---
        self.agents = []
        self.num_drones = num_drones
        agent_size = 14
        
        # Create N independent agents
        for i in range(self.num_drones):
            self.agents.append(Agent(0, 0, agent_size)) # Pos set in reset
            
        self.turrets = []
        self.projectiles = []
        self.walls = []
        self.target = pygame.Rect(0,0,40,40)
        
        # Init Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 12)

    def reset(self):
        # 1. Map Gen
        self.grid = Grid(self.ROWS, self.SCREEN_WIDTH)
        self._generate_city()
        self.walls = self.grid.get_obstacle_rects()
        
        self.projectiles = []
        self.turrets = []
        
        # 2. Spawn Turrets
        for _ in range(5): 
            self._spawn_turret()
            
        # 3. Spawn Target
        self._spawn_target()
        
        # 4. Spawn Swarm & REVIVE
        start_x, start_y = self._find_safe_spot()
        
        for i, agent in enumerate(self.agents):
            offset_x = np.random.uniform(-30, 30)
            offset_y = np.random.uniform(-30, 30)
            
            agent.position = np.array([float(start_x + offset_x), float(start_y + offset_y)])
            
            # Reset State
            agent.color = (0, 255, 0)
            agent.velocity[:] = 0
            agent.acceleration[:] = 0
            agent.path = []
            agent.repath_timer = 0
            
            self._recalc_path(agent)

    def step(self, model):
        dt = 0.016
        
        # Handle User Quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.reset()

        # --- UPDATE LOOP ---
        
        # 1. Update Turrets
        for t in self.turrets:
            t.update(dt, self.agents, self.projectiles, self.walls)
            
        # 2. Update Projectiles
        for p in self.projectiles[:]:
            p.update(dt, self.walls)
            if not p.active: self.projectiles.remove(p)
        
        # 3. UPDATE SWARM
        active_agents = [a for a in self.agents if a.color != (0,0,0)]
        
        if len(active_agents) == 0:
            print("SWARM ELIMINATED. Resetting...")
            self.reset()
            return True

        for agent in active_agents:
            # A. Get Observation (Updated for 20 inputs)
            obs = self._get_obs(agent)
            
            # B. Ask Brain
            action, _ = model.predict(obs, deterministic=True) 
            
            # C. Physics
            force = action * agent.max_force
            agent.apply_force(force)
            
            # D. ALLY REPULSION (Hardcoded Safety Layer)
            for buddy in active_agents:
                if buddy is agent: continue
                dist = np.linalg.norm(agent.position - buddy.position)
                min_dist = 20.0 
                if 0 < dist < min_dist:
                    push = (agent.position - buddy.position) / dist
                    agent.apply_force(push * 2000.0) 

            # E. Update & Collide Walls
            agent.update_physics(dt)
            agent._handle_collisions(self.walls)
            
            # F. Check Projectile Hits
            for p in self.projectiles:
                if agent.rect.colliderect(p.rect):
                    agent.color = (0, 0, 0) # Mark dead
                    print("Agent DOWN!")
            
            # G. CHECK WIN CONDITION
            if agent.rect.colliderect(self.target):
                print(">>> TARGET SECURED! Mission Success. Resetting... <<<")
                self.reset()
                return True

            # H. Pathfinding
            self._manage_path(agent)

        # 4. Draw
        self._render()
        self.clock.tick(60)
        return True

    def _get_obs(self, agent):
        # 1. Lidar (8)
        lidar = agent.cast_rays(self.walls, num_rays=8, max_dist=200)
        
        # 2. Velocity (2)
        vel = agent.velocity / agent.max_speed
        
        # 3. GPS (2)
        if agent.path:
            node = agent.path[0]
            wp = np.array([node.x + 7, node.y + 7])
        else:
            wp = np.array(self.target.center)
        vec_to_wp = wp - agent.position
        dist = np.linalg.norm(vec_to_wp)
        if dist > 0: vec_to_wp /= dist
        
        # --- UPDATE: 8 SECTORS ---
        # 4. Bullet Sectors (8)
        visible_bullets = []
        for p in self.projectiles:
            if agent._has_line_of_sight(p.position, self.walls):
                visible_bullets.append(p)
        
        bul_sec = agent.get_sector_readings(visible_bullets, radius=300.0, num_sectors=8)
        
        # 5. Neighbor Sectors (8)
        # Filter neighbors (exclude self and dead ones)
        neighbors = [a for a in self.agents if a is not agent and a.color != (0,0,0)]
        nei_sec = agent.get_sector_readings(neighbors, radius=150.0, num_sectors=8)
        
        # Combine to 28 inputs
        return np.concatenate([lidar, vel, vec_to_wp, bul_sec, nei_sec]).astype(np.float32)

    def _manage_path(self, agent):
        # Repath Timer
        agent.repath_timer += 1
        if agent.repath_timer > 30: 
            agent.repath_timer = np.random.randint(0, 10) 
            self._recalc_path(agent)
            
        # Pop waypoints
        if agent.path:
            t_node = agent.path[0]
            t_pos = np.array([t_node.x+7, t_node.y+7])
            if np.linalg.norm(t_pos - agent.position) < 20:
                agent.path.pop(0)
        elif np.linalg.norm(self.target.center - agent.position) > 50:
             self._recalc_path(agent)

    def _recalc_path(self, agent):
        start = self.grid.get_node_from_pos(agent.position)
        end = self.grid.get_node_from_pos(self.target.center)
        if start and end and not start.is_obstacle:
            path = a_star_algorithm(None, self.grid, start, end)
            if path: agent.path = path

    def _generate_city(self):
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

    def _find_safe_spot(self):
        while True:
            x = np.random.randint(50, 750)
            y = np.random.randint(50, 750)
            rect = pygame.Rect(x, y, 40, 40)
            if rect.collidelist(self.walls) == -1: return x, y

    def _spawn_target(self):
        while True:
            x, y = self._find_safe_spot()
            self.target.topleft = (x, y)
            if self.target.collidelist(self.walls) == -1: break

    def _spawn_turret(self):
        while True:
            x = np.random.randint(50, 750)
            y = np.random.randint(50, 750)
            gx = (x // 10) * 10
            gy = (y // 10) * 10
            t = Turret(gx, gy, 20)
            self.turrets.append(t)
            self.walls.append(t.rect)
            break

    def _render(self):
        self.screen.fill((255, 255, 255))
        self.grid.draw(self.screen)
        pygame.draw.rect(self.screen, (0, 255, 0), self.target)
        for t in self.turrets: t.draw(self.screen)
        for p in self.projectiles: p.draw(self.screen)
        
        for agent in self.agents:
            if agent.color != (0,0,0):
                # 1. Draw Body
                agent.draw(self.screen)
                
                # --- NEW: DEBUG SECTOR SENSORS ---
                # Recalculate sensors just for visualization
                # (Filter for visible bullets)
                vis_bullets = [p for p in self.projectiles if agent._has_line_of_sight(p.position, self.walls)]
                sectors = agent.get_sector_readings(vis_bullets, radius=300.0)
                
                # Draw lines for N, E, S, W
                # sectors[0]=N, 1=E, 2=S, 3=W
                # Value 1.0 = Safe, 0.0 = Danger
                # We want line length to be proportional to DANGER (1 - value)
                
                directions = [
                    (0, -1), # North
                    (1, 0),  # East
                    (0, 1),  # South
                    (-1, 0)  # West
                ]
                
                cx, cy = agent.position
                max_draw_len = 50.0
                
                for i, val in enumerate(sectors):
                    danger = 1.0 - val # Invert so 0.9 danger = long line
                    if danger > 0.1: # Only draw if there is some danger
                        dx, dy = directions[i]
                        end_x = cx + (dx * max_draw_len * danger)
                        end_y = cy + (dy * max_draw_len * danger)
                        
                        # Draw Red 'Whiskers' pointing at danger
                        pygame.draw.line(self.screen, (255, 0, 0), (cx, cy), (end_x, end_y), 2)
        
        pygame.display.flip()
        
    def close(self):
        if self.screen: pygame.quit()