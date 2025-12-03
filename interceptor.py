import pygame
import numpy as np

class Interceptor:
    def __init__(self, x, y, size=16):
        self.position = np.array([float(x), float(y)], dtype=float)
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.acceleration = np.array([0.0, 0.0], dtype=float)
        
        # --- PHYSICS: GUIDED MISSILE TUNING ---
        self.mass = 1.0
        
        # Extremely Fast
        self.max_speed = 600.0   
        
        # Moderate Force: Takes time to change direction at high speed
        # This creates the "Wide Turn" arc.
        self.max_force = 1500.0  
        
        # NO DRAG: It never slows down naturally.
        self.friction = 1.0      
        
        self.size = size
        self.rect = pygame.Rect(0, 0, self.size, self.size)
        self.rect.center = self.position
        
        self.alive = True
        self.color = (255, 50, 0) # Orange-Red
        self.detection_range = 1000.0 # Infinite range

    def update(self, dt, blue_drones, walls):
        if not self.alive: return
            
        target = self._find_target(blue_drones)
        
        if target:
            # 1. PURE PURSUIT (Heat Seeking)
            # Aim directly at the target. No fancy prediction.
            desired = target.position - self.position
            dist = np.linalg.norm(desired)
            
            if dist > 0:
                # Normalize direction
                direction = desired / dist
                
                # Apply CONSTANT ACCELERATION
                # A rocket engine is either ON or OFF.
                # We apply max force continuously towards the target.
                self.acceleration += direction * (self.max_force / self.mass)

        # 2. Physics Integration
        self.velocity += self.acceleration * dt
        
        # Cap speed (Terminal Velocity)
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed
            
        # No Friction applied (Conservation of Momentum)
        
        self.position += self.velocity * dt
        self.acceleration[:] = 0
        self.rect.center = self.position
        
        # 3. Death Check (Missiles explode on impact)
        self._check_death(walls)
        
    def _find_target(self, blue_drones):
        closest = None
        min_dist = self.detection_range
        for drone in blue_drones:
            if hasattr(drone, 'color') and drone.color == (0,0,0): continue
            
            dist = np.linalg.norm(drone.position - self.position)
            if dist < min_dist:
                min_dist = dist
                closest = drone
        return closest

    def _check_death(self, walls):
        # Die if hitting a wall (Crash)
        if (self.rect.left < 0 or self.rect.right > 800 or
            self.rect.top < 0 or self.rect.bottom > 800):
            self._die()
            return
            
        if self.rect.collidelist(walls) != -1:
            self._die()
            
    def _die(self):
        self.alive = False
        self.color = (100, 100, 100) # Grey smoke
        self.velocity[:] = 0
        
    def draw(self, screen):
        if not self.alive: return
        
        # Draw Rocket Shape (Triangle aligned with velocity)
        if np.linalg.norm(self.velocity) > 1:
            angle = np.arctan2(self.velocity[1], self.velocity[0])
        else:
            angle = 0
            
        # Simple rotation math for visuals
        cx, cy = self.position
        size = self.size
        
        # Tip
        p1 = (cx + size * np.cos(angle), cy + size * np.sin(angle))
        # Back Left
        p2 = (cx + size/2 * np.cos(angle + 2.5), cy + size/2 * np.sin(angle + 2.5))
        # Back Right
        p3 = (cx + size/2 * np.cos(angle - 2.5), cy + size/2 * np.sin(angle - 2.5))
        
        pygame.draw.polygon(screen, self.color, [p1, p2, p3])