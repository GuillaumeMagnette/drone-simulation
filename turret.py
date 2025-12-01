import pygame
import numpy as np
from projectile import Projectile

class Turret:
    def __init__(self, x, y, grid_size):
        self.rect = pygame.Rect(x, y, grid_size, grid_size)
        # Center coordinates for shooting
        self.center = np.array([x + grid_size/2, y + grid_size/2])
        self.color = (150, 0, 0) # Dark Red
        # --- FIX: INITIAL DELAY ---
        # Don't shoot immediately. Wait 2 to 4 seconds.
        # This gives the agent time to accelerate and scan the room.
        self.shoot_timer = np.random.uniform(0.125, 0.25) 
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
                
                # Vector to target
                vec_to_target = closest_target.position - self.center
                dist = np.linalg.norm(vec_to_target)
                
                # --- FIX: FIRE EVEN IF CLOSE ---
                if dist > 0:
                    direction = vec_to_target / dist
                    
                    # Reduce spawn offset to ensure we hit point-blank targets
                    # Turret radius 10. Bullet radius 5. 
                    # If offset is too big, we spawn PAST the agent.
                    # If offset is too small, we hit ourselves.
                    
                    # Logic: If target is super close (< 30px), spawn bullet INSIDE turret
                    # but ignore turret collision for 5 frames (requires complex projectile logic).
                    
                    # SIMPLER FIX: Just spawn it.
                    spawn_offset = 15.0 
                    spawn_pos = self.center + (direction * spawn_offset)
                    
                    p = Projectile(spawn_pos, closest_target.position, speed=300.0)
                    projectiles.append(p)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        # Draw a black circle to look like a barrel
        pygame.draw.circle(screen, (0, 0, 0), (int(self.center[0]), int(self.center[1])), 5)