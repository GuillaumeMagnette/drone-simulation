import numpy as np
import pygame

class FollowerDrone:
    def __init__(self, x, y, size, color=(50, 150, 255)):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.acceleration = np.array([0.0, 0.0], dtype=float)
        self.size = size
        self.color = color
        self.rect = pygame.Rect(0, 0, size, size)
        
        # --- TUNING FOR SWARM ---
        # Leader max_speed is ~300. Followers need much more to catch up.
        self.max_speed = 420.0 
        
        # Leader max_force is 2000. Followers need snappy acceleration.
        self.max_force = 3000.0 
        
        # Less friction = easier gliding
        self.friction = 0.97

        self.separation_radius = 25.0

    def update(self, dt, leader_pos, swarm_mates, walls):
        # --- 1. FLOCKING FORCES ---
        
        # A. SEEK LEADER
        desired = leader_pos - self.position
        dist = np.linalg.norm(desired)
        
        force_seek = np.array([0.0, 0.0])
        if dist > 0:
            # NORMALIZED SEEK
            desired = (desired / dist) * self.max_speed
            force_seek = desired - self.velocity
            
            # --- CATCH UP BOOST ---
            # If we are far behind (>80px), double the seeking urge
            if dist > 80:
                force_seek *= 2.0
        
        # B. SEPARATION
        force_sep = np.array([0.0, 0.0])
        count = 0
        
        for mate in swarm_mates:
            if mate is self: continue
            
            dist_mate = np.linalg.norm(self.position - mate.position)
            if 0 < dist_mate < self.separation_radius:
                diff = self.position - mate.position
                diff /= dist_mate 
                force_sep += diff
                count += 1
        
        # Leader Separation (Don't crowd the boss)
        dist_leader = np.linalg.norm(self.position - leader_pos)
        if 0 < dist_leader < self.separation_radius + 5:
             diff = self.position - leader_pos
             diff /= dist_leader
             force_sep += diff * 2.0 
             count += 1

        if count > 0:
            force_sep /= count
            force_sep = (force_sep / np.linalg.norm(force_sep)) * self.max_speed
            
        # C. COMBINE
        # Seek = 1.0, Separation = 2.0 (High priority to avoid clumping)
        total_force = force_seek + (force_sep * 2.0)

        # Clamp
        force_mag = np.linalg.norm(total_force)
        if force_mag > self.max_force:
            total_force = (total_force / force_mag) * self.max_force

        # --- 2. PHYSICS ---
        self.acceleration += total_force
        self.velocity += self.acceleration * dt
        self.velocity *= self.friction
        self.position += self.velocity * dt
        self.acceleration[:] = 0
        
        # --- 3. COLLISION ---
        self.rect.center = self.position
        for wall in walls:
            if self.rect.colliderect(wall):
                dx = (self.position[0] - wall.centerx)
                dy = (self.position[1] - wall.centery)
                if abs(dx) > abs(dy):
                    self.velocity[0] *= -0.5
                    if dx > 0: self.position[0] = wall.right + self.size/2 + 1
                    else: self.position[0] = wall.left - self.size/2 - 1
                else:
                    self.velocity[1] *= -0.5
                    if dy > 0: self.position[1] = wall.bottom + self.size/2 + 1
                    else: self.position[1] = wall.top - self.size/2 - 1
                    
        self.rect.center = self.position

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)