"""
VISUAL ASSETS MODULE
====================

Pretty graphics for the drone simulation.
Uses pygame drawing - no external image files needed.

Usage:
    from visuals import Visuals
    
    vis = Visuals()
    vis.draw_drone(screen, position, angle, color)
    vis.draw_missile(screen, position, angle)
    vis.draw_explosion(screen, position, frame)
    vis.draw_base(screen, rect)
    vis.draw_wall(screen, rect)
"""

import pygame
import numpy as np
import math


class Visuals:
    def __init__(self):
        """Initialize visual assets."""
        self.explosions = {}  # Track active explosions {id: frame}
        self.explosion_duration = 20  # frames
        
        # Colors
        self.DRONE_BLUE = (30, 144, 255)      # Dodger blue
        self.DRONE_OUTLINE = (20, 100, 180)
        self.MISSILE_RED = (220, 50, 30)
        self.MISSILE_ORANGE = (255, 140, 0)
        self.EXPLOSION_COLORS = [
            (255, 255, 200),  # White-yellow center
            (255, 200, 50),   # Yellow
            (255, 140, 0),    # Orange
            (255, 80, 0),     # Red-orange
            (200, 50, 0),     # Dark red
            (100, 30, 0),     # Brown
            (50, 50, 50),     # Smoke grey
        ]
        self.WALL_COLOR = (60, 60, 70)
        self.WALL_HIGHLIGHT = (80, 80, 90)
        self.WALL_SHADOW = (40, 40, 50)
        self.BASE_GREEN = (34, 139, 34)
        self.BASE_DARK = (20, 100, 20)
        
    def draw_drone(self, screen, position, velocity, size=14, color=None):
        """
        Draw a quadcopter-style drone.
        
        Args:
            screen: Pygame surface
            position: (x, y) center position
            velocity: (vx, vy) for rotation
            size: Drone size in pixels
            color: Override color (default: blue)
        """
        x, y = position
        
        # Calculate rotation from velocity
        if np.linalg.norm(velocity) > 1:
            angle = math.atan2(velocity[1], velocity[0])
        else:
            angle = 0
            
        color = color or self.DRONE_BLUE
        outline = self.DRONE_OUTLINE
        
        # Body (rounded rectangle approximation)
        body_size = size * 0.6
        body_rect = pygame.Rect(x - body_size/2, y - body_size/2, body_size, body_size)
        pygame.draw.rect(screen, color, body_rect, border_radius=3)
        pygame.draw.rect(screen, outline, body_rect, width=2, border_radius=3)
        
        # Arms (4 diagonal lines)
        arm_length = size * 0.8
        for i in range(4):
            arm_angle = angle + math.pi/4 + i * math.pi/2
            ax = x + math.cos(arm_angle) * arm_length
            ay = y + math.sin(arm_angle) * arm_length
            pygame.draw.line(screen, outline, (x, y), (ax, ay), 3)
            
            # Rotors (circles at end of arms)
            rotor_radius = size * 0.25
            pygame.draw.circle(screen, (200, 200, 200), (int(ax), int(ay)), int(rotor_radius))
            pygame.draw.circle(screen, (150, 150, 150), (int(ax), int(ay)), int(rotor_radius), 1)
            
            # Rotor blur (spinning effect)
            blur_radius = size * 0.35
            blur_surface = pygame.Surface((int(blur_radius*2), int(blur_radius*2)), pygame.SRCALPHA)
            pygame.draw.circle(blur_surface, (200, 200, 200, 80), 
                             (int(blur_radius), int(blur_radius)), int(blur_radius))
            screen.blit(blur_surface, (ax - blur_radius, ay - blur_radius))
        
        # Direction indicator (front)
        front_x = x + math.cos(angle) * size * 0.4
        front_y = y + math.sin(angle) * size * 0.4
        pygame.draw.circle(screen, (255, 100, 100), (int(front_x), int(front_y)), 3)
        
    def draw_missile(self, screen, position, velocity, size=16):
        """
        Draw a missile/rocket with flame trail.
        
        Args:
            screen: Pygame surface
            position: (x, y) center position  
            velocity: (vx, vy) for rotation
            size: Missile size in pixels
        """
        x, y = position
        
        # Calculate rotation from velocity
        speed = np.linalg.norm(velocity)
        if speed > 1:
            angle = math.atan2(velocity[1], velocity[0])
        else:
            angle = 0
            
        # Missile body points (triangle)
        nose_x = x + math.cos(angle) * size
        nose_y = y + math.sin(angle) * size
        
        back_left_x = x + math.cos(angle + 2.6) * size * 0.6
        back_left_y = y + math.sin(angle + 2.6) * size * 0.6
        
        back_right_x = x + math.cos(angle - 2.6) * size * 0.6
        back_right_y = y + math.sin(angle - 2.6) * size * 0.6
        
        # Flame trail (behind missile)
        if speed > 10:
            flame_length = size * 0.8 * min(speed / 200, 1.5)
            flame_x = x - math.cos(angle) * flame_length
            flame_y = y - math.sin(angle) * flame_length
            
            # Outer flame (orange)
            flame_points = [
                (back_left_x, back_left_y),
                (flame_x, flame_y),
                (back_right_x, back_right_y),
            ]
            pygame.draw.polygon(screen, self.MISSILE_ORANGE, flame_points)
            
            # Inner flame (yellow)
            inner_flame_x = x - math.cos(angle) * flame_length * 0.5
            inner_flame_y = y - math.sin(angle) * flame_length * 0.5
            inner_points = [
                (x - math.cos(angle - 0.3) * size * 0.3, y - math.sin(angle - 0.3) * size * 0.3),
                (inner_flame_x, inner_flame_y),
                (x - math.cos(angle + 0.3) * size * 0.3, y - math.sin(angle + 0.3) * size * 0.3),
            ]
            pygame.draw.polygon(screen, (255, 255, 100), inner_points)
        
        # Missile body
        body_points = [(nose_x, nose_y), (back_left_x, back_left_y), (back_right_x, back_right_y)]
        pygame.draw.polygon(screen, self.MISSILE_RED, body_points)
        pygame.draw.polygon(screen, (150, 30, 20), body_points, 2)
        
        # Fins
        fin_size = size * 0.3
        for side in [-1, 1]:
            fin_base_x = x + math.cos(angle + side * 2.3) * size * 0.4
            fin_base_y = y + math.sin(angle + side * 2.3) * size * 0.4
            fin_tip_x = fin_base_x + math.cos(angle + side * 1.8) * fin_size
            fin_tip_y = fin_base_y + math.sin(angle + side * 1.8) * fin_size
            pygame.draw.line(screen, (150, 30, 20), (fin_base_x, fin_base_y), (fin_tip_x, fin_tip_y), 2)

    def draw_explosion(self, screen, position, progress):
        """
        Draw an explosion effect.
        
        Args:
            screen: Pygame surface
            position: (x, y) center position
            progress: 0.0 to 1.0 (0=start, 1=end)
        """
        x, y = position
        
        if progress >= 1.0:
            return False  # Explosion finished
            
        # Explosion grows then fades
        max_radius = 40
        
        if progress < 0.3:
            # Growing phase
            radius = max_radius * (progress / 0.3)
            alpha = 255
        else:
            # Fading phase
            radius = max_radius
            alpha = int(255 * (1 - (progress - 0.3) / 0.7))
        
        # Draw multiple circles for layered effect
        num_layers = 5
        for i in range(num_layers):
            layer_progress = (i / num_layers)
            layer_radius = radius * (1 - layer_progress * 0.6)
            
            color_idx = min(int(progress * len(self.EXPLOSION_COLORS) + i), len(self.EXPLOSION_COLORS) - 1)
            color = self.EXPLOSION_COLORS[color_idx]
            
            # Create surface with alpha
            surf_size = int(layer_radius * 2) + 4
            if surf_size > 0:
                surf = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
                layer_alpha = max(0, min(255, int(alpha * (1 - layer_progress))))
                pygame.draw.circle(surf, (*color, layer_alpha), 
                                 (surf_size//2, surf_size//2), int(layer_radius))
                screen.blit(surf, (x - surf_size//2, y - surf_size//2))
        
        # Sparks
        if progress < 0.5:
            num_sparks = 8
            for i in range(num_sparks):
                spark_angle = (i / num_sparks) * 2 * math.pi + progress * 3
                spark_dist = radius * 1.2 * progress * 2
                spark_x = x + math.cos(spark_angle) * spark_dist
                spark_y = y + math.sin(spark_angle) * spark_dist
                spark_size = max(1, int(3 * (1 - progress * 2)))
                pygame.draw.circle(screen, (255, 255, 200), (int(spark_x), int(spark_y)), spark_size)
        
        return True  # Explosion still active
        
    def draw_wall(self, screen, rect):
        """
        Draw a wall block with 3D effect.
        
        Args:
            screen: Pygame surface
            rect: pygame.Rect for the wall
        """
        # Main wall
        pygame.draw.rect(screen, self.WALL_COLOR, rect)
        
        # Top highlight
        pygame.draw.line(screen, self.WALL_HIGHLIGHT, 
                        rect.topleft, rect.topright, 2)
        pygame.draw.line(screen, self.WALL_HIGHLIGHT,
                        rect.topleft, rect.bottomleft, 2)
        
        # Bottom shadow
        pygame.draw.line(screen, self.WALL_SHADOW,
                        rect.bottomleft, rect.bottomright, 2)
        pygame.draw.line(screen, self.WALL_SHADOW,
                        rect.topright, rect.bottomright, 2)
        
        # Brick pattern (optional, for larger walls)
        if rect.width > 15 and rect.height > 15:
            brick_h = 8
            brick_w = 16
            for row in range(0, rect.height, brick_h):
                offset = (row // brick_h % 2) * (brick_w // 2)
                for col in range(-offset, rect.width, brick_w):
                    bx = rect.left + col
                    by = rect.top + row
                    if rect.left <= bx < rect.right and rect.top <= by < rect.bottom:
                        pygame.draw.line(screen, self.WALL_SHADOW,
                                       (bx, by), (min(bx + brick_w, rect.right), by), 1)
                        if col >= 0:
                            pygame.draw.line(screen, self.WALL_SHADOW,
                                           (bx, by), (bx, min(by + brick_h, rect.bottom)), 1)

    def draw_base(self, screen, rect):
        """
        Draw a military base / target.
        
        Args:
            screen: Pygame surface
            rect: pygame.Rect for the base
        """
        x, y = rect.center
        size = min(rect.width, rect.height)
        
        # Landing pad circle
        pygame.draw.circle(screen, self.BASE_DARK, (x, y), size//2 + 4)
        pygame.draw.circle(screen, (80, 80, 80), (x, y), size//2 + 2)
        pygame.draw.circle(screen, self.BASE_GREEN, (x, y), size//2 - 2)
        
        # H marking for helipad
        h_size = size // 3
        h_thickness = 3
        # Vertical bars of H
        pygame.draw.rect(screen, (240, 240, 240), 
                        (x - h_size//2 - h_thickness//2, y - h_size//2, h_thickness, h_size))
        pygame.draw.rect(screen, (240, 240, 240),
                        (x + h_size//2 - h_thickness//2, y - h_size//2, h_thickness, h_size))
        # Horizontal bar of H
        pygame.draw.rect(screen, (240, 240, 240),
                        (x - h_size//2, y - h_thickness//2, h_size, h_thickness))
        
        # Corner markers
        marker_size = 6
        corners = [
            (rect.left + 4, rect.top + 4),
            (rect.right - 4, rect.top + 4),
            (rect.left + 4, rect.bottom - 4),
            (rect.right - 4, rect.bottom - 4),
        ]
        for cx, cy in corners:
            pygame.draw.rect(screen, (255, 200, 0), 
                           (cx - marker_size//2, cy - marker_size//2, marker_size, marker_size))
            
    def draw_background(self, screen, width, height):
        """
        Draw a terrain background.
        
        Args:
            screen: Pygame surface
            width: Screen width
            height: Screen height
        """
        # Gradient green background (grass)
        for y in range(0, height, 4):
            green_val = 140 + int(20 * math.sin(y * 0.02))
            color = (80, green_val, 80)
            pygame.draw.rect(screen, color, (0, y, width, 4))


class ExplosionManager:
    """Manages multiple active explosions."""
    
    def __init__(self, visuals):
        self.visuals = visuals
        self.explosions = []  # [(x, y, progress), ...]
        
    def add(self, x, y):
        """Start a new explosion at position."""
        self.explosions.append([x, y, 0.0])
        
    def update(self, dt=0.016):
        """Update all explosions, remove finished ones."""
        speed = 2.0  # Explosion speed multiplier
        for exp in self.explosions:
            exp[2] += dt * speed
        self.explosions = [e for e in self.explosions if e[2] < 1.0]
        
    def draw(self, screen):
        """Draw all active explosions."""
        for x, y, progress in self.explosions:
            self.visuals.draw_explosion(screen, (x, y), progress)
