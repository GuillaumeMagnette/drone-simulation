import pygame
import numpy as np
from projectile import Projectile

class Turret:
    def __init__(self, x, y, grid_size):
        self.rect = pygame.Rect(x, y, grid_size, grid_size)
        # Center coordinates for shooting
        self.center = np.array([x + grid_size/2, y + grid_size/2])
        self.color = (150, 0, 0) # Dark Red
        self.shoot_timer = 0.0
        self.shoot_interval = 1.0 # Fire every 1 second
        self.range = 350.0 # Range in pixels

    def update(self, dt, targets, projectiles, walls):
        self.shoot_timer -= dt
        
        if self.shoot_timer <= 0:
            closest_target = None
            min_dist = self.range
            
            for t in targets:
                # Ignore dead agents
                if t.color == (0, 0, 0): continue
                
                dist = np.linalg.norm(self.center - t.position)
                if dist < min_dist:
                    min_dist = dist
                    closest_target = t
            
            if closest_target:
                self.shoot_timer = self.shoot_interval + np.random.uniform(-0.1, 0.1)
                
                # --- FIX: SPAWN BULLET OUTSIDE TURRET ---
                # 1. Calculate direction vector
                vec_to_target = closest_target.position - self.center
                dist = np.linalg.norm(vec_to_target)
                
                if dist > 0:
                    direction = vec_to_target / dist
                    
                    # 2. Push spawn point out by (Turret Radius + Bullet Radius + Buffer)
                    # Turret is 20px wide (radius 10). Bullet is radius 5. Buffer 2.
                    spawn_offset = 18.0 
                    spawn_pos = self.center + (direction * spawn_offset)
                    
                    # 3. Fire
                    p = Projectile(spawn_pos, closest_target.position, speed=250.0)
                    projectiles.append(p)
                    
                    # Optional: Debug Print to confirm firing
                    print("TURRET FIRED!")

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        # Draw a black circle to look like a barrel
        pygame.draw.circle(screen, (0, 0, 0), (int(self.center[0]), int(self.center[1])), 5)