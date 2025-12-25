"""
physics.py - High-Fidelity 2.5D Simulation
Includes: Drones (Tilt Lag), Missiles (Inertia), and SAM Site (Turret Physics).
"""

import numpy as np
import pygame
import math

# ==========================================
# CONFIGURATION
# ==========================================
BUILDING_HEIGHT = 50.0
CEILING = 100.0
GRAVITY = 40.0        
DRAG = 0.97           
MAX_SPEED_XY = 450.0  
MAX_SPEED_Z = 100.0
MAX_ACCEL = 2500.0    

MAX_FORCE = 5000.0    
AGENT_RADIUS = 7.0
TILT_SPEED = 30.0     

class Agent:
    def __init__(self, uid, x, y):
        self.uid = uid
        self.active = True
        self.position = np.array([float(x), float(y), 10.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.current_thrust_vector = np.array([0.0, 0.0, 0.0])
        self.mass = 1.0
        self.radius = AGENT_RADIUS

    def update(self, dt, desired_force_vector, walls, screen_size):
        if not self.active: return

        # 1. ATTITUDE CONTROL
        target_accel = desired_force_vector / self.mass
        accel_mag = np.linalg.norm(target_accel)
        if accel_mag > MAX_ACCEL:
            target_accel = (target_accel / accel_mag) * MAX_ACCEL

        lerp_factor = 1.0 - np.exp(-TILT_SPEED * dt)
        self.current_thrust_vector += (target_accel - self.current_thrust_vector) * lerp_factor

        # 2. PHYSICS
        acceleration = self.current_thrust_vector.copy()
        acceleration[2] -= GRAVITY 
        self.velocity += acceleration * dt
        self.velocity *= DRAG 
        
        v_xy = np.linalg.norm(self.velocity[:2])
        if v_xy > MAX_SPEED_XY:
            self.velocity[:2] = (self.velocity[:2] / v_xy) * MAX_SPEED_XY
        self.velocity[2] = np.clip(self.velocity[2], -MAX_SPEED_Z, MAX_SPEED_Z)
        
        next_pos = self.position + self.velocity * dt
        
        # 3. COLLISION
        if next_pos[2] < 5:
            next_pos[2] = 5
            self.velocity[2] = 0
            self.current_thrust_vector[2] = max(0, self.current_thrust_vector[2])
        elif next_pos[2] > CEILING:
            next_pos[2] = CEILING
            self.velocity[2] = 0

        # Walls (Split Axis)
        if next_pos[2] <= BUILDING_HEIGHT:
            test_rect = pygame.Rect(next_pos[0]-7, next_pos[1]-7, 14, 14)
            hit_wall = False
            
            if test_rect.collidelist(walls) != -1: hit_wall = True
            if not (0 <= next_pos[0] <= screen_size and 0 <= next_pos[1] <= screen_size): hit_wall = True
                
            if hit_wall:
                self.velocity[:2] *= -0.5
                next_pos[0] = self.position[0]
                next_pos[1] = self.position[1]
                self.current_thrust_vector[:2] *= 0.5 

        self.position = next_pos
    
    def reset(self, x, y):
        self.active = True
        self.position = np.array([float(x), float(y), 10.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.current_thrust_vector = np.array([0.0, 0.0, 0.0])

class Interceptor:
    def __init__(self, x, y, z, target_agent, heading_vector):
        self.position = np.array([float(x), float(y), float(z)])
        self.speed = 900.0 
        self.target = target_agent
        self.active = True
        self.lifetime = 6.0
        self.radius = 5.0
        
        # Start moving in the direction the turret was facing
        norm = np.linalg.norm(heading_vector)
        if norm > 0:
            self.velocity = (heading_vector / norm) * self.speed
        else:
            self.velocity = np.array([0.0, 0.0, self.speed])

    def update(self, dt, walls):
        if not self.active: return
        self.lifetime -= dt
        if self.lifetime <= 0: self.active = False; return

        if self.target and self.target.active:
            # INERTIAL GUIDANCE (Realism Update)
            vec_to_target = self.target.position - self.position
            dist = np.linalg.norm(vec_to_target)
            
            if dist > 0:
                desired_dir = vec_to_target / dist
                desired_vel = desired_dir * self.speed
                
                # Turn Rate Limit (The missile cannot snap turn)
                # 3.0 rad/s allows dodging if timed right
                TURN_RATE = 3.0 
                lerp = 1.0 - np.exp(-TURN_RATE * dt)
                self.velocity += (desired_vel - self.velocity) * lerp
                
                # Renormalize speed
                current_speed = np.linalg.norm(self.velocity)
                if current_speed > 0:
                    self.velocity = (self.velocity / current_speed) * self.speed
        
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

class SAMSite:
    """
    Realistic Defense System.
    Has a turret that must physically rotate to track targets.
    Has reload time and lock-on delay.
    """
    def __init__(self, x, y):
        self.pos = np.array([float(x), float(y), 10.0])
        self.angle = 0.0 # Radians
        self.state = "SCANNING" # SCANNING, TRACKING, FIRING, RELOADING
        
        # Realism Constants (HARD MODE)
        self.ROTATION_SPEED = 3.0  # ~170 deg/sec. Impossible to outrun.
        self.LOCK_TIME_REQUIRED = 0.1 # Fast lock
        self.RELOAD_TIME = 2.0     # Fast reload
        self.RANGE = 700.0         # Covers almost the whole map
        
        # Internal timers
        self.lock_timer = 0.0
        self.reload_timer = 0.0
        self.current_target = None

    def update(self, dt, agents, walls):
        """
        Returns: An Interceptor object if fired, else None
        """
        missile_to_spawn = None
        
        if self.state == "RELOADING":
            self.reload_timer -= dt
            if self.reload_timer <= 0:
                self.state = "SCANNING"
            return None

        # 1. Target Selection (Visual check)
        visible_targets = []
        for ag in agents:
            if not ag.active: continue
            dist = np.linalg.norm(ag.position - self.pos)
            if dist > self.RANGE: continue
            
            # Line of Sight Check
            has_los = True
            if ag.position[2] < BUILDING_HEIGHT: # Only check walls if low
                for w in walls:
                    if w.clipline(self.pos[:2], ag.position[:2]):
                        has_los = False; break
            
            if has_los:
                visible_targets.append((ag, dist))
        
        # Logic: Keep tracking current if visible, else pick closest
        if self.current_target and self.current_target.active:
            # Verify current is still visible
            still_visible = any(t[0] == self.current_target for t in visible_targets)
            if not still_visible:
                self.current_target = None
                self.lock_timer = 0.0
                self.state = "SCANNING"
        
        if not self.current_target and visible_targets:
            # Pick closest
            visible_targets.sort(key=lambda x: x[1])
            self.current_target = visible_targets[0][0]
            self.state = "TRACKING"

        # 2. Turret Rotation
        if self.current_target:
            vec_to_target = self.current_target.position - self.pos
            target_angle = math.atan2(vec_to_target[1], vec_to_target[0])
            
            # Smallest angle difference
            diff = target_angle - self.angle
            while diff > math.pi: diff -= 2*math.pi
            while diff < -math.pi: diff += 2*math.pi
            
            # Rotate
            rotation = np.clip(diff, -self.ROTATION_SPEED * dt, self.ROTATION_SPEED * dt)
            self.angle += rotation
            
            # Check if aimed (within 5 degrees)
            if abs(diff) < 0.1:
                self.state = "LOCKING"
                self.lock_timer += dt
                if self.lock_timer >= self.LOCK_TIME_REQUIRED:
                    # FIRE!
                    self.state = "RELOADING"
                    self.reload_timer = self.RELOAD_TIME
                    self.lock_timer = 0.0
                    
                    # Spawn missile moving in current turret direction
                    heading = np.array([math.cos(self.angle), math.sin(self.angle), 0.2])
                    missile_to_spawn = Interceptor(self.pos[0], self.pos[1], 10, self.current_target, heading)
            else:
                self.state = "TRACKING"
                self.lock_timer = max(0, self.lock_timer - dt) # Lose lock if target moves too fast
        else:
            # Scan pattern (spin slowly)
            self.angle += 0.5 * dt 
            self.state = "SCANNING"
            
        return missile_to_spawn