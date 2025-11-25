import pygame
import numpy as np

class Projectile:
    def __init__(self, start_pos, target_pos, speed=300.0):
        """
        A simple bullet that travels in a straight line.
        """
        self.position = np.array(start_pos, dtype=float)
        self.speed = speed
        self.radius = 5
        self.color = (255, 0, 0) # Red
        
        # Calculate direction ONCE upon creation
        vector = np.array(target_pos) - self.position
        distance = np.linalg.norm(vector)
        
        if distance > 0:
            self.direction = vector / distance
        else:
            self.direction = np.array([1.0, 0.0]) # Default right
            
        # Create a rect for collision
        # We center the rect on the float position
        self.rect = pygame.Rect(0, 0, self.radius * 2, self.radius * 2)
        self.rect.center = self.position
        
        # Mark for deletion (if it goes off screen)
        self.active = True

    # Update signature to accept walls
    def update(self, dt, walls):
        # Move in the straight line
        velocity = self.direction * self.speed
        self.position += velocity * dt
        
        # Update hitbox
        self.rect.center = self.position
        
        # 1. Check Wall Collisions (NEW)
        # If we hit a wall, we are no longer active
        if self.rect.collidelist(walls) != -1:
            self.active = False
            return # Stop processing

        # 2. Check Screen Edge (Existing logic moved here)
        # We assume screen size 800x800 for simplicity or pass it in
        if not (0 <= self.position[0] <= 800 and 0 <= self.position[1] <= 800):
            self.active = False

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.position[0]), int(self.position[1])), self.radius)