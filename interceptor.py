"""
INTERCEPTOR - Enemy Pursuit Drone
=================================

A "Smart" Red Team threat that actively chases Blue Drones.

Design Philosophy:
- SCARY acceleration (fast initial burst)
- Faster top speed (can't outrun in straight line)  
- Commits hard (high friction = wide turns, can't change direction)
- Mortal (dies on wall collision)

Tactical Counter: "The Juke"
- Bait interceptor toward wall/obstacle
- Sharp turn at last moment
- Interceptor crashes due to momentum commitment

This teaches the Blue Drone terrain exploitation and agility advantage.
"""

import pygame
import numpy as np


class Interceptor:
    def __init__(self, x, y, size=16):
        # --- PHYSICS (SCARY MISSILE) ---
        self.position = np.array([float(x), float(y)], dtype=float)
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.acceleration = np.array([0.0, 0.0], dtype=float)
        
        # Fast and aggressive, but commits to direction
        self.mass = 1.0             # Same mass (for fast acceleration)
        self.max_speed = 450.0      # FASTER than Blue (360)
        self.max_force = 3000.0     # MORE thrust (scary burst)
        self.friction = 0.95        # HIGH friction = commits hard, wide turns
        
        # Effective acceleration: 3000/1.0 = 3000 (faster than Blue's 2500)
        # But high friction means once moving, it's locked in that direction
        
        # --- COLLISION ---
        self.size = size
        self.rect = pygame.Rect(0, 0, self.size, self.size)
        self.rect.center = self.position
        
        # --- STATE ---
        self.alive = True
        self.color = (220, 30, 30)  # Bright menacing red
        
        # --- PURSUIT CONFIG ---
        self.detection_range = 600.0    # Sees further
        self.prediction_time = 0.3      # Shorter prediction = more aggressive
        
    def update(self, dt, blue_drones, walls):
        """Main update: pursue target, apply physics, check death."""
        if not self.alive:
            return
            
        # 1. Find closest living blue drone
        target = self._find_target(blue_drones)
        
        if target:
            # 2. Predict where target will be
            predicted_pos = self._predict_position(target)
            
            # 3. Steer toward predicted position
            steering = self._seek(predicted_pos)
            self._apply_force(steering)
        else:
            # No target - brake gently
            self._apply_force(-self.velocity * 0.5)
        
        # 4. Physics integration
        self._update_physics(dt)
        
        # 5. Check for lethal collisions
        self._check_death(walls)
        
    def _find_target(self, blue_drones):
        """Find closest living blue drone."""
        closest = None
        min_dist = self.detection_range
        
        for drone in blue_drones:
            # Skip dead drones
            if hasattr(drone, 'color') and drone.color == (0, 0, 0):
                continue
            if not hasattr(drone, 'position'):
                continue
                
            dist = np.linalg.norm(drone.position - self.position)
            if dist < min_dist:
                min_dist = dist
                closest = drone
                
        return closest
    
    def _predict_position(self, target):
        """Kinematic prediction - lead the target."""
        if hasattr(target, 'velocity'):
            return target.position + (target.velocity * self.prediction_time)
        return target.position.copy()
    
    def _seek(self, target_pos):
        """Craig Reynolds' Seek steering behavior."""
        desired = target_pos - self.position
        dist = np.linalg.norm(desired)
        
        if dist < 1.0:
            return np.array([0.0, 0.0])
        
        desired = (desired / dist) * self.max_speed
        steering = desired - self.velocity
        
        # Clamp force
        mag = np.linalg.norm(steering)
        if mag > self.max_force:
            steering = (steering / mag) * self.max_force
            
        return steering
    
    def _apply_force(self, force):
        """F = ma → a = F/m"""
        self.acceleration += force / self.mass
        
    def _update_physics(self, dt):
        """Euler integration."""
        self.velocity += self.acceleration * dt
        
        # Speed limit
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed
            
        # Friction
        self.velocity *= self.friction
        
        # Position update
        self.position += self.velocity * dt
        
        # Reset acceleration
        self.acceleration[:] = 0
        
        # Sync hitbox
        self.rect.center = self.position
        
    def _check_death(self, walls):
        """Interceptors die on wall/boundary collision."""
        SCREEN_SIZE = 800
        
        # Screen boundaries
        if (self.rect.left < 0 or self.rect.right > SCREEN_SIZE or
            self.rect.top < 0 or self.rect.bottom > SCREEN_SIZE):
            self._die()
            return
            
        # Wall collision
        if self.rect.collidelist(walls) != -1:
            self._die()
            
    def _die(self):
        """Mark as dead."""
        self.alive = False
        self.color = (80, 80, 80)  # Grey
        self.velocity[:] = 0
        
    def draw(self, screen):
        """Render interceptor."""
        if not self.alive:
            # Dead - draw X
            pygame.draw.line(screen, self.color,
                           (self.rect.left, self.rect.top),
                           (self.rect.right, self.rect.bottom), 2)
            pygame.draw.line(screen, self.color,
                           (self.rect.right, self.rect.top),
                           (self.rect.left, self.rect.bottom), 2)
            return
            
        # Body
        pygame.draw.rect(screen, self.color, self.rect)
        
        # Direction indicator
        speed = np.linalg.norm(self.velocity)
        if speed > 1.0:
            direction = self.velocity / speed
            end = self.position + direction * self.size * 1.5
            pygame.draw.line(screen, (255, 200, 0), self.position, end, 2)
