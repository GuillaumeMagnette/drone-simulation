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

    def update(self, dt):
        # Move in the straight line
        velocity = self.direction * self.speed
        self.position += velocity * dt
        
        # Update hitbox
        self.rect.center = self.position

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.position[0]), int(self.position[1])), self.radius)