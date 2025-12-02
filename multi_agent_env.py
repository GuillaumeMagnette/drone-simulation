"""
MULTI-AGENT SWARM ENVIRONMENT
=============================

Visual validation tool for swarm behavior.
Uses the same 36-input observation as single-agent training.

This environment:
- Runs multiple drones with the same trained brain
- Tests if evasion skills transfer to swarm scenarios
- Validates the neighbor sensor (obs[28-35]) works correctly
- Prepares for true MARL training

NO "magic force field" - drones must use their sensors to avoid each other.

Usage:
    python multi_agent_env.py
"""

import numpy as np
import pygame
from grid import Grid
from agent import Agent
from interceptor import Interceptor
from algorithm import a_star_algorithm


class MultiAgentSwarm:
    def __init__(self, num_drones=4, num_interceptors=2, map_type='sparse'):
        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 800
        self.ROWS = 80
        
        self.num_drones = num_drones
        self.num_interceptors = num_interceptors
        self.map_type = map_type
        
        # World
        self.grid = Grid(self.ROWS, self.SCREEN_WIDTH)
        self.walls = []
        self.target = pygame.Rect(0, 0, 40, 40)
        
        # Entities
        self.agents = []
        self.interceptors = []
        
        # Create drones
        for _ in range(num_drones):
            self.agents.append(Agent(0, 0, size=14))
        
        # Collision config
        self.collision_radius = 14.0  # Drones crash if this close
        
        # Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 14)
        pygame.display.set_caption("Swarm Validator")
        
        # Stats
        self.total_successes = 0
        self.total_crashes = 0
        self.total_caught = 0
        self.total_friendly_fire = 0

    def reset(self):
        """Reset the swarm scenario."""
        # Generate map
        self.grid = Grid(self.ROWS, self.SCREEN_WIDTH)
        
        if self.map_type == 'arena':
            pass  # Empty
        elif self.map_type == 'sparse':
            self._generate_sparse()
        else:
            self._generate_urban()
        
        self.walls = self.grid.get_obstacle_rects()
        
        # Reset interceptors
        self.interceptors = []
        
        # Spawn target
        self._spawn_target()
        
        # Spawn drones in formation
        start_x, start_y = self._find_safe_spot()
        
        for i, agent in enumerate(self.agents):
            angle = (2 * np.pi / self.num_drones) * i
            offset = 35  # Formation spread
            
            agent.position = np.array([
                start_x + np.cos(angle) * offset,
                start_y + np.sin(angle) * offset
            ], dtype=float)
            agent.rect.center = agent.position
            
            agent.color = (0, 255, 0)  # Alive
            agent.velocity[:] = 0
            agent.acceleration[:] = 0
            agent.path = []
            
            # Individual pathfinding
            if self.map_type != 'arena':
                self._recalc_path(agent)
        
        # Spawn interceptors after drones
        for _ in range(self.num_interceptors):
            self._spawn_interceptor()

    def step(self, model):
        """
        Run one simulation step.
        Returns False to quit.
        """
        dt = 0.016
        
        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.reset()
                if event.key == pygame.K_ESCAPE:
                    return False
        
        # --- UPDATE INTERCEPTORS ---
        living_agents = [a for a in self.agents if a.color != (0, 0, 0)]
        
        for interceptor in self.interceptors:
            if interceptor.alive:
                interceptor.update(dt, living_agents, self.walls)
        
        # --- CHECK WIN/LOSS ---
        if len(living_agents) == 0:
            print(">>> SWARM ELIMINATED <<<")
            self.total_crashes += 1
            self.reset()
            return True
        
        # --- UPDATE DRONES ---
        for agent in living_agents[:]:  # Copy to allow modification
            # 1. Get observation (36 inputs)
            obs = self._get_obs(agent)
            
            # 2. Get action from brain
            action, _ = model.predict(obs, deterministic=True)
            
            # 3. Apply force
            force = action * agent.max_force
            agent.apply_force(force)
            
            # 4. Physics
            agent.update_physics(dt)
            agent.rect.center = agent.position
            
            # 5. Wall collision (lethal)
            hit_wall = False
            if (agent.rect.left < 0 or agent.rect.right > self.SCREEN_WIDTH or
                agent.rect.top < 0 or agent.rect.bottom > self.SCREEN_HEIGHT):
                hit_wall = True
            if agent.rect.collidelist(self.walls) != -1:
                hit_wall = True
            
            if hit_wall:
                agent.color = (0, 0, 0)
                print("[DRONE] Crashed into wall!")
                self.total_crashes += 1
                continue
            
            # 6. Interceptor collision (lethal)
            for interceptor in self.interceptors:
                if interceptor.alive and agent.rect.colliderect(interceptor.rect):
                    agent.color = (0, 0, 0)
                    interceptor._die()
                    print("[DRONE] Caught by interceptor!")
                    self.total_caught += 1
                    break
            
            if agent.color == (0, 0, 0):
                continue
            
            # 7. Friendly collision (BOTH die - true MARL)
            for buddy in living_agents:
                if buddy is agent or buddy.color == (0, 0, 0):
                    continue
                
                dist = np.linalg.norm(agent.position - buddy.position)
                if dist < self.collision_radius:
                    agent.color = (0, 0, 0)
                    buddy.color = (0, 0, 0)
                    print("[SWARM] Mid-air collision!")
                    self.total_friendly_fire += 2
                    break
            
            if agent.color == (0, 0, 0):
                continue
            
            # 8. Victory check
            if agent.rect.colliderect(self.target):
                print(">>> TARGET REACHED - SUCCESS! <<<")
                self.total_successes += 1
                self.reset()
                return True
            
            # 9. Pathfinding
            if self.map_type != 'arena':
                self._manage_path(agent)
        
        # --- RENDER ---
        self._render()
        self.clock.tick(60)
        
        return True

    def _get_obs(self, agent):
        """
        36-Input observation (matches drone_env.py):
        [0-7]   Lidar
        [8-9]   Velocity
        [10-11] GPS
        [12-19] Threats (interceptors)
        [20-27] Projectiles (empty)
        [28-35] Neighbors (other drones)
        """
        # 1. Lidar (8)
        lidar = agent.cast_rays(self.walls, num_rays=8, max_dist=200)
        
        # 2. Velocity (2)
        vel = agent.velocity / agent.max_speed
        
        # 3. GPS (2)
        if self.map_type == 'arena':
            waypoint = np.array(self.target.center, dtype=float)
        elif agent.path:
            node = agent.path[0]
            waypoint = np.array([node.x + 7, node.y + 7])
        else:
            waypoint = np.array(self.target.center, dtype=float)
        
        vec_to_wp = waypoint - agent.position
        dist = np.linalg.norm(vec_to_wp)
        if dist > 0:
            vec_to_wp /= dist
        
        # 4. Threat sectors (8) - Interceptors
        visible_threats = []
        for interceptor in self.interceptors:
            if interceptor.alive:
                if self.map_type == 'arena':
                    visible_threats.append(interceptor)
                elif agent._has_line_of_sight(interceptor.position, self.walls):
                    visible_threats.append(interceptor)
        
        threat_sectors = agent.get_sector_readings(visible_threats, radius=400.0, num_sectors=8)
        
        # 5. Projectile sectors (8) - Empty
        projectile_sectors = np.ones(8, dtype=np.float32)
        
        # 6. Neighbor sectors (8) - Other living drones
        neighbors = [a for a in self.agents 
                    if a is not agent and a.color != (0, 0, 0)]
        neighbor_sectors = agent.get_sector_readings(neighbors, radius=150.0, num_sectors=8)
        
        return np.concatenate([
            lidar, vel, vec_to_wp,
            threat_sectors, projectile_sectors, neighbor_sectors
        ]).astype(np.float32)

    def _manage_path(self, agent):
        """Handle pathfinding for an agent."""
        if not hasattr(agent, 'repath_timer'):
            agent.repath_timer = 0
        
        agent.repath_timer += 1
        if agent.repath_timer > 30:
            agent.repath_timer = np.random.randint(0, 10)
            self._recalc_path(agent)
        
        if agent.path:
            node = agent.path[0]
            node_pos = np.array([node.x + 7, node.y + 7])
            if np.linalg.norm(node_pos - agent.position) < 20:
                agent.path.pop(0)
        elif np.linalg.norm(np.array(self.target.center) - agent.position) > 50:
            self._recalc_path(agent)

    def _recalc_path(self, agent):
        """A* pathfinding for an agent."""
        start = self.grid.get_node_from_pos(agent.position)
        end = self.grid.get_node_from_pos(self.target.center)
        
        if start and end and not start.is_obstacle:
            path = a_star_algorithm(None, self.grid, start, end)
            if path:
                agent.path = path

    # --- MAP GENERATORS ---
    
    def _generate_sparse(self):
        rows = self.ROWS
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
        # Simplified urban - main roads + blocks
        rows = self.ROWS
        road_width = 8
        mid = rows // 2
        
        blocked = set()
        for i in range(rows):
            for w in range(road_width):
                blocked.add((mid - road_width//2 + w, i))
                blocked.add((i, mid - road_width//2 + w))
        
        # Fill quadrants
        for qx in [5, mid + road_width//2 + 2]:
            for qy in [5, mid + road_width//2 + 2]:
                for _ in range(4):
                    bw = np.random.randint(6, 12)
                    bh = np.random.randint(6, 12)
                    bx = qx + np.random.randint(0, 20)
                    by = qy + np.random.randint(0, 20)
                    
                    for i in range(bw):
                        for j in range(bh):
                            px, py = bx + i, by + j
                            if (px, py) not in blocked and 0 <= px < rows and 0 <= py < rows:
                                self.grid.grid[px][py].make_obstacle()

    # --- SPAWNING ---
    
    def _find_safe_spot(self):
        for _ in range(100):
            x = np.random.randint(80, 720)
            y = np.random.randint(80, 720)
            rect = pygame.Rect(x - 50, y - 50, 100, 100)
            if rect.collidelist(self.walls) == -1:
                return x, y
        return 400, 400
    
    def _spawn_target(self):
        for _ in range(100):
            x = np.random.randint(50, 750)
            y = np.random.randint(50, 750)
            self.target.topleft = (x, y)
            if self.target.collidelist(self.walls) == -1:
                break
    
    def _spawn_interceptor(self):
        for _ in range(100):
            x = np.random.randint(50, 750)
            y = np.random.randint(50, 750)
            
            # Not too close to any drone
            too_close = False
            for agent in self.agents:
                if np.linalg.norm(np.array([x, y]) - agent.position) < 300:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            test_rect = pygame.Rect(x - 8, y - 8, 16, 16)
            if test_rect.collidelist(self.walls) == -1:
                self.interceptors.append(Interceptor(x, y, size=16))
                break

    # --- RENDERING ---
    
    def _render(self):
        self.screen.fill((240, 240, 240))
        self.grid.draw(self.screen)
        
        # Target
        pygame.draw.rect(self.screen, (0, 200, 0), self.target)
        
        # Interceptors
        for interceptor in self.interceptors:
            interceptor.draw(self.screen)
        
        # Drones
        for agent in self.agents:
            if agent.color != (0, 0, 0):
                agent.draw(self.screen)
        
        # HUD
        alive = sum(1 for a in self.agents if a.color != (0, 0, 0))
        threats = sum(1 for i in self.interceptors if i.alive)
        hud = f"Drones: {alive}/{self.num_drones} | Threats: {threats} | " \
              f"Success: {self.total_successes} | Crashes: {self.total_crashes} | " \
              f"Caught: {self.total_caught} | Friendly: {self.total_friendly_fire}"
        text = self.font.render(hud, True, (0, 0, 0))
        self.screen.blit(text, (10, 10))
        
        pygame.display.flip()

    def close(self):
        pygame.quit()


# --- MAIN ---
if __name__ == "__main__":
    from stable_baselines3 import PPO
    
    print()
    print("=" * 50)
    print("  SWARM VALIDATOR")
    print("=" * 50)
    print("  [R]   Reset")
    print("  [ESC] Quit")
    print("=" * 50)
    print()
    
    # Create swarm
    swarm = MultiAgentSwarm(
        num_drones=4,
        num_interceptors=2,
        map_type='sparse'
    )
    
    # Load model
    model_path = "models/PPO_Tactical/drone_tactical"
    try:
        model = PPO.load(model_path, device='cpu')
        print(f"Loaded: {model_path}")
    except:
        print(f"Could not load {model_path}")
        print("Using random actions...")
        
        class RandomModel:
            def predict(self, obs, deterministic=True):
                return np.random.uniform(-1, 1, size=(2,)).astype(np.float32), None
        model = RandomModel()
    
    # Run
    swarm.reset()
    running = True
    while running:
        running = swarm.step(model)
    
    swarm.close()
