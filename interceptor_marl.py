import numpy as np
import pygame

# Configuration
BUILDING_HEIGHT = 50.0

class Interceptor3D:
    def __init__(self, x, y, z):
        self.position = np.array([float(x), float(y), float(z)])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.alive = True
        self.lifetime = 4.0
        self.speed = 550.0
        self.turn_rate = 1200.0
        self.target_ref = None
        self.size = 10
        self.rect = pygame.Rect(x, y, 10, 10)
        
    def update(self, dt, agents, walls):
        if not self.alive: return
        self.lifetime -= dt
        if self.lifetime <= 0:
            self.alive = False
            return
            
        # Radar Logic
        if self.target_ref and self.target_ref.active:
            if not self._has_los(self.target_ref, walls):
                self.target_ref = None 
                
        if not self.target_ref:
            closest_dist = 9999
            for ag in agents:
                if not ag.active: continue
                dist = np.linalg.norm(ag.position - self.position)
                if dist < closest_dist:
                    if self._has_los(ag, walls):
                        closest_dist = dist
                        self.target_ref = ag
                        
        # Physics
        accel = np.array([0.0, 0.0, 0.0])
        if self.target_ref:
            des_dir = (self.target_ref.position - self.position)
            dist = np.linalg.norm(des_dir)
            if dist > 0:
                accel = (des_dir / dist) * self.turn_rate
                
        self.velocity += accel * dt
        speed = np.linalg.norm(self.velocity)
        if speed > self.speed:
            self.velocity = (self.velocity / speed) * self.speed
            
        self.position += self.velocity * dt
        if self.position[2] < 5: self.alive = False 
        
        # Collisions
        if self.position[2] < BUILDING_HEIGHT:
            self.rect.center = (int(self.position[0]), int(self.position[1]))
            if self.rect.collidelist(walls) != -1: self.alive = False
                
        for ag in agents:
            if ag.active:
                if np.linalg.norm(ag.position - self.position) < 20:
                    self.alive = False
                    ag.active = False
                    ag.dead_timer = 5.0
                    
    def _has_los(self, agent, walls):
        if agent.position[2] > BUILDING_HEIGHT: return True
        p1 = (self.position[0], self.position[1])
        p2 = (agent.position[0], agent.position[1])
        for w in walls:
            if w.clipline(p1, p2): return False
        return True