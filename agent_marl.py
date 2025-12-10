import numpy as np
import pygame

# Configuration
BUILDING_HEIGHT = 50.0
CEILING = 100.0
GRAVITY = 15.0 
LIFT_FORCE = 30.0 

class Agent25D:
    def __init__(self, x, y, uid):
        self.uid = uid
        self.active = True
        self.dead_timer = 0.0
        self.stealth_timer = 0.0 
        
        # Event Bus (To tell the Env what happened this frame)
        self.events = {"crashed": False}
        
        self.position = np.array([float(x), float(y), 10.0]) 
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0, 0.0])
        
        self.size = 14
        self.rect = pygame.Rect(x, y, 14, 14)
        self.prev_dist = 0.0
        
        self.mass = 1.0
        self.max_speed_xy = 360.0
        self.max_speed_z = 100.0
        self.max_force = 2500.0
        self.friction = 0.95 # Higher friction (less drift) as discussed
        
    def update(self, dt, action, walls, hazard_pos):
        # Reset events
        self.events["crashed"] = False

        if not self.active:
            if self.stealth_timer > 0: self.stealth_timer -= dt
            return

        if self.stealth_timer > 0:
            self.stealth_timer -= dt

        # Physics
        fx = action[0] * self.max_force
        fy = action[1] * self.max_force
        fz = action[2] * LIFT_FORCE 
        fz -= GRAVITY * self.mass
        
        self.acceleration = np.array([fx, fy, fz]) / self.mass
        self.velocity += self.acceleration * dt
        
        v_xy = self.velocity[:2]
        speed_xy = np.linalg.norm(v_xy)
        if speed_xy > self.max_speed_xy:
            self.velocity[:2] = (v_xy / speed_xy) * self.max_speed_xy
        self.velocity[2] = np.clip(self.velocity[2], -self.max_speed_z, self.max_speed_z)
        self.velocity *= self.friction
        
        next_pos = self.position + self.velocity * dt
        
        # Constraints
        if next_pos[2] < 5: 
            next_pos[2] = 5
            self.velocity[2] = 0
        elif next_pos[2] > CEILING: 
            next_pos[2] = CEILING
            self.velocity[2] = 0
            
        is_flying_high = (next_pos[2] > BUILDING_HEIGHT)
        
        # Wall Collision (Lethal)
        hit_wall = False
        if not is_flying_high:
            test_rect = pygame.Rect(next_pos[0]-7, next_pos[1]-7, 14, 14)
            # Use 1200 map size
            if (test_rect.left < 0 or test_rect.right > 1200 or
                test_rect.top < 0 or test_rect.bottom > 1200):
                hit_wall = True
            elif test_rect.collidelist(walls) != -1:
                hit_wall = True
                
        if hit_wall:
            # [CRITICAL CHANGE] DEATH ON IMPACT
            self.active = False
            self.events["crashed"] = True
            self.position = np.array([-1000.0, -1000.0, 0.0]) # Move to void
        else:
            self.position = next_pos
            
        self.rect.center = (int(self.position[0]), int(self.position[1]))

    def spawn_at_safe_pos(self, walls, hazard_pos):
        """Called externally by the Env to respawn this agent safely."""
        self.active = True
        self.stealth_timer = 3.0 
        self.velocity[:] = 0
        self.acceleration[:] = 0
        self.prev_dist = 0.0 # Fixed Lazarus exploit
        
        attempts = 0
        while attempts < 200:
            attempts += 1
            rx = np.random.randint(50, 1150)
            ry = np.random.randint(50, 1150)
            
            test_rect = pygame.Rect(rx-15, ry-15, 30, 30)
            if test_rect.collidelist(walls) != -1:
                continue
                
            dist = np.linalg.norm([rx - hazard_pos[0], ry - hazard_pos[1]])
            if dist > 350: 
                self.position = np.array([float(rx), float(ry), 10.0])
                self.rect.center = (rx, ry)
                return
        
        self.position = np.array([50.0, 50.0, 10.0])