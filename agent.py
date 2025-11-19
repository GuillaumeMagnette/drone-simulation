import pygame
import numpy as np # <-- 1. IMPORT NUMPY
# Import the algorithm so the agent can use it
from algorithm import a_star_algorithm 

# The Agent class is a blueprint for our player, enemies, etc.
class Agent:
    # The __init__ method is the constructor. It runs when we create a new Agent.
    def __init__(self, x, y, size):
        """
        Initializes the Agent.
        :param x: The starting x-coordinate.
        :param y: The starting y-coordinate.
        :param size: The width and height of the agent (as a square).
        """
        # --- 2. UPGRADE TO NUMPY FOR POSITION ---
        # We use floats for position to allow for smooth, fractional movement.
        self.position = np.array([x, y], dtype=float)
        
        # --- 3. ADD MOVEMENT PROPERTIES ---
        self.velocity = np.array([0.0, 0.0])
        self.speed = 200.0 # pixels per second
        # --- 1. ADD THE FORWARD VECTOR ---
        # This vector represents the direction the agent is "facing".
        # We initialize it to point upwards.
        self.forward_vector = np.array([0.0, -1.0])

        self.size = size
        
        
        # --- 1. STATE MACHINE SETUP ---
        self.state = "PLAYER_CONTROLLED"
        self.target_enemy = None # To store the enemy we are chasing
        self.path = []
        
        # --- NEW: RE-PATHING TIMER ---
        # We don't want to run A* every frame. 
        # We will calculate a new path every 0.5 seconds.
        self.repath_timer = 0.0
        self.repath_interval = 0.5 
        
        self.color_map = {
            "PLAYER_CONTROLLED": (255, 255, 255),
            "AUTOMATIC_CHASE": (100, 100, 255),
            "PATH_FOLLOWING": (255, 0, 255)
        }
        self.color = self.color_map[self.state]

        # Pygame uses Rect objects to store and manage an object's position and size.
        # This is extremely useful for drawing and, later, for collision detection.
        # The rect's position will be synced with the float position.
        # The rect's center is used for positioning to make rotation easier later.
        self.rect = pygame.Rect(0, 0, self.size, self.size)
        self.rect.center = self.position
    
    def switch_mode(self):
        # (Same as before)
        if self.state == "PLAYER_CONTROLLED" or self.state == "PATH_FOLLOWING":
            self.state = "AUTOMATIC_CHASE"
            self.target_enemy = None 
        else:
            self.state = "PLAYER_CONTROLLED"
        self.color = self.color_map[self.state]


    # --- THE UPDATED UPDATE METHOD ---
    # We added 'grid' as an argument because A* needs it!
    def update(self, dt, player_direction_vector, collidables, targets, grid):
        
        if self.state == "PLAYER_CONTROLLED":
            # Use 'collidables' for physics (walls + enemies)
            self._execute_movement(dt, player_direction_vector, collidables)
        
        elif self.state == "AUTOMATIC_CHASE":
            self.repath_timer -= dt
            
            # CHANGE 2: Look for the closest enemy in 'targets' (NOT collidables)
            if self.target_enemy is None:
                self.target_enemy = self._find_closest_enemy(targets)

            if self.target_enemy and self.repath_timer <= 0:
                self.repath_timer = self.repath_interval 
                
                start_pos = (self.position[0] + self.size//2, self.position[1] + self.size//2)
                start_node = grid.get_node_from_pos(start_pos)
                end_pos = self.target_enemy.center
                end_node = grid.get_node_from_pos(end_pos)
                
                if start_node and end_node and start_node != end_node:
                    
                    for row in grid.grid:
                        for node in row:
                            node.update_neighbors(grid.grid)
                            
                    new_path = a_star_algorithm(None, grid, start_node, end_node)
                    if new_path:
                        self.path = new_path

            if self.path:
                self._execute_path_following(dt)
            else:
                self.velocity = np.array([0.0, 0.0])

        elif self.state == "PATH_FOLLOWING":
            self._execute_path_following(dt)

    def _execute_path_following(self, dt):
        # (Keep the code from the previous lesson exactly as is)
        if not self.path:
            # If purely path following, switch to manual. 
            # If Chasing, we just wait for the next timer tick.
            if self.state == "PATH_FOLLOWING":
                self.state = "PLAYER_CONTROLLED"
                self.color = self.color_map[self.state]
            return

        target_node = self.path[0]
        target_x = target_node.x + self.size / 2
        target_y = target_node.y + self.size / 2
        target_pos = np.array([target_x, target_y])

        vector_to_target = target_pos - self.position
        distance = np.linalg.norm(vector_to_target)

        if distance < 5.0:
            self.path.pop(0)
            if self.path:
                self._execute_path_following(dt)
            else:
                self.velocity = np.array([0.0, 0.0])
            return

        direction = vector_to_target / distance
        self.forward_vector = direction
        self.velocity = direction * self.speed
        self.position += self.velocity * dt
        self.rect.center = self.position    
    
    def _find_closest_enemy(self, enemies):
        """A helper method to find the closest enemy from a list."""
        closest_enemy = None
        min_dist = float('inf')
        
        for enemy in enemies:
            dist = np.linalg.norm(np.array(enemy.center) - self.position)
            if dist < min_dist:
                min_dist = dist
                closest_enemy = enemy
        
        return closest_enemy
    
    def _execute_movement(self, dt, direction_vector, collidables):
        """
        A centralized method for all physics and movement logic.
        This is called by the state handlers.
        """
        # --- 3. CENTRALIZED MOVEMENT LOGIC (from last lesson) ---
        norm = np.linalg.norm(direction_vector)
        if norm > 0:
            normalized_direction = direction_vector / norm
        else:
            normalized_direction = np.array([0.0, 0.0])
        self.velocity = normalized_direction * self.speed

        # X-axis collision
        self.position[0] += self.velocity[0] * dt
        self.rect.centerx = round(self.position[0])
        for obstacle in collidables:
            if self.rect.colliderect(obstacle):
                if self.velocity[0] > 0: self.rect.right = obstacle.left
                elif self.velocity[0] < 0: self.rect.left = obstacle.right
                self.position[0] = self.rect.centerx

        # Y-axis collision
        self.position[1] += self.velocity[1] * dt
        self.rect.centery = round(self.position[1])
        for obstacle in collidables:
            if self.rect.colliderect(obstacle):
                if self.velocity[1] > 0: self.rect.bottom = obstacle.top
                elif self.velocity[1] < 0: self.rect.top = obstacle.bottom
                self.position[1] = self.rect.centery

    # def aim(self, target_pos):
    #     """
    #     Calculates the forward vector to face a target position.
    #     :param target_pos: A NumPy array representing the target's coordinates.
    #     """
    #     # --- 2. IMPLEMENT THE AIMING LOGIC ---
    #     vector_to_target = target_pos - self.position
        
    #     norm = np.linalg.norm(vector_to_target)
    #     if norm > 0:
    #         # Normalize the vector to get the pure direction
    #         self.forward_vector = vector_to_target / norm
            
    def draw(self, screen):
        """
        Draws the agent on a given screen.
        :param screen: The pygame screen (our canvas) to draw on.
        """
        # We use the built-in draw.rect function.
        # It needs:
        # 1. The screen to draw on.
        # 2. The color of the shape.
        # 3. The Rect object that defines the position and size.
        pygame.draw.rect(screen, self.color, self.rect)

        # --- 3. DRAW A LINE TO SHOW THE FORWARD VECTOR ---
        # Calculate the end point of the line
        line_end = self.position + self.forward_vector * self.size * 1.5
        pygame.draw.line(screen, (255, 0, 0), self.position, line_end, 2) # Red line
        
    
    def draw_awareness(self, screen, enemies, font):
        """
        Draws enemies and visualizes the agent's awareness of them.
        :param screen: The screen to draw on.
        :param enemies: A list of Rects representing enemies.
        :param font: A Pygame font object for rendering text.
        """
        # --- 4. IMPLEMENT THE DOT PRODUCT VISUALIZATION ---
        
        # A 90-degree Field of View cone threshold (cos(45 degrees))
        fov_threshold = 0.707 

        for enemy_rect in enemies:
            vector_to_enemy = np.array(enemy_rect.center) - self.position
            distance = np.linalg.norm(vector_to_enemy)
            
            dot_product = -1 # Default to -1 (behind)
            
            if distance > 0:
                direction_to_enemy = vector_to_enemy / distance
                dot_product = np.dot(self.forward_vector, direction_to_enemy)
            
            # Determine the enemy's color based on the dot product
            enemy_color = (255, 255, 0) # Yellow (default)
            if dot_product > fov_threshold:
                enemy_color = (0, 255, 0) # Green (in FoV)
            elif dot_product < -0.5: # A generous "behind" cone
                enemy_color = (255, 0, 0) # Red (behind)
            
            # Draw the enemy
            pygame.draw.rect(screen, enemy_color, enemy_rect)
            
            # Draw the debug text
            text = font.render(f"Dot: {dot_product:.2f}", True, (255, 255, 255))
            screen.blit(text, (enemy_rect.x, enemy_rect.y - 20))