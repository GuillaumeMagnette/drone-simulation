"""
physics.py - High-Fidelity 2.5D Simulation
Simulates Drone Tilt Dynamics (Inertial Lag).
"""

import numpy as np
import pygame

# ==========================================
# CONFIGURATION
# ==========================================
BUILDING_HEIGHT = 50.0
CEILING = 100.0
GRAVITY = 40.0        # Increased to match pixel scale (feels less floaty)
DRAG = 0.97           # Reduced drag slightly to allow momentum
MAX_SPEED_XY = 450.0  # Fast top speed
MAX_SPEED_Z = 100.0
MAX_ACCEL = 2500.0    # MASSIVE increase: Needed to overcome Drag at high speeds

# Navigator Interface Limit
MAX_FORCE = 5000.0    

AGENT_RADIUS = 7.0
TILT_SPEED = 30.0     # Increased: Drone rotates/tilts faster (snappier response)

class Agent:
    def __init__(self, uid, x, y):
        self.uid = uid
        self.active = True
        
        # State
        self.position = np.array([float(x), float(y), 10.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        
        # Dynamics Internal State (The actual thrust vector)
        self.current_thrust_vector = np.array([0.0, 0.0, 0.0])
        
        self.mass = 1.0
        self.radius = AGENT_RADIUS

    def update(self, dt, desired_force_vector, walls, screen_size):
        if not self.active: return

        # --- 1. ATTITUDE CONTROL (Simulating Tilt Lag) ---
        # Normalize desired force to an acceleration vector
        target_accel = desired_force_vector / self.mass
        
        # Clamp target acceleration (Physical motor limits)
        accel_mag = np.linalg.norm(target_accel)
        if accel_mag > MAX_ACCEL:
            target_accel = (target_accel / accel_mag) * MAX_ACCEL

        # Smoothly interpolate current thrust towards target thrust
        # Higher TILT_SPEED = Faster reaction
        lerp_factor = 1.0 - np.exp(-TILT_SPEED * dt)
        self.current_thrust_vector += (target_accel - self.current_thrust_vector) * lerp_factor

        # --- 2. PHYSICS INTEGRATION ---
        acceleration = self.current_thrust_vector.copy()
        acceleration[2] -= GRAVITY 

        self.velocity += acceleration * dt
        self.velocity *= DRAG 
        
        # --- 3. HARD SPEED LIMITS ---
        v_xy = np.linalg.norm(self.velocity[:2])
        if v_xy > MAX_SPEED_XY:
            self.velocity[:2] = (self.velocity[:2] / v_xy) * MAX_SPEED_XY
        self.velocity[2] = np.clip(self.velocity[2], -MAX_SPEED_Z, MAX_SPEED_Z)
        
        # --- 4. POSITION UPDATE ---
        next_pos = self.position + self.velocity * dt
        
        # --- 5. COLLISION ---
        # Ground / Ceiling
        if next_pos[2] < 5:
            next_pos[2] = 5
            self.velocity[2] = 0
            self.current_thrust_vector[2] = max(0, self.current_thrust_vector[2])
        elif next_pos[2] > CEILING:
            next_pos[2] = CEILING
            self.velocity[2] = 0

        # Walls
        # We handle Z separately to allow sliding UP a wall even if pressing against it.
        if next_pos[2] <= BUILDING_HEIGHT:
            test_rect = pygame.Rect(next_pos[0]-7, next_pos[1]-7, 14, 14)
            hit_wall = False
            
            if test_rect.collidelist(walls) != -1:
                hit_wall = True
            if not (0 <= next_pos[0] <= screen_size and 0 <= next_pos[1] <= screen_size):
                hit_wall = True
                
            if hit_wall:
                # 1. Bounce/Dampen XY Velocity (Energy loss)
                self.velocity[:2] *= -0.5
                
                # 2. Reject ONLY the XY movement (Slide logic)
                # We revert X and Y to the previous valid position
                next_pos[0] = self.position[0]
                next_pos[1] = self.position[1]
                
                # 3. Keep the Z movement! 
                # (next_pos[2] is left untouched, allowing it to climb)
                
                # 4. Destabilize XY thrust (Crash effect)
                self.current_thrust_vector[:2] *= 0.5 

        self.position = next_pos
    
    def reset(self, x, y):
        self.active = True
        self.position = np.array([float(x), float(y), 10.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.current_thrust_vector = np.array([0.0, 0.0, 0.0])

class Interceptor:
    def __init__(self, x, y, z, target_agent):
        self.position = np.array([float(x), float(y), float(z)])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.speed = 550.0  # Fast missile
        self.target = target_agent
        self.active = True
        self.lifetime = 4.0
        self.radius = 5.0

    def update(self, dt, walls):
        if not self.active: return
        self.lifetime -= dt
        if self.lifetime <= 0: self.active = False; return

        if self.target and self.target.active:
            direction = self.target.position - self.position
            dist = np.linalg.norm(direction)
            if dist > 0:
                self.velocity = (direction / dist) * self.speed
        
        self.position += self.velocity * dt
        
        if self.position[2] < 0: self.active = False; return

        if walls and self.position[2] < BUILDING_HEIGHT:
            m_rect = pygame.Rect(self.position[0]-3, self.position[1]-3, 6, 6)
            if m_rect.collidelist(walls) != -1:
                self.active = False
                return

    def check_hit(self, agent):
        if not self.active or not agent.active: return False
        dist = np.linalg.norm(self.position - agent.position)
        return dist < (self.radius + agent.radius)