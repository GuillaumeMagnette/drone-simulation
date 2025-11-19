import pygame
import numpy as np
from algorithm import a_star_algorithm 

class Agent:
    def __init__(self, x, y, size):
        """
        Set up the agent's physical properties, state machine, and visualization.
        """
        # TODO: Initialize Position using NumPy (float type)
        self.position = np.array([x,y], dtype=float)
        
        # TODO: Initialize Velocity (0,0) and Forward Vector (0, -1) using NumPy
        self.velocity = np.array([0,0])
        self.forward_vector = np.array([0,-1])
        
        # TODO: Set standard properties (speed, size)
        self.speed = 100
        self.size = size
        
        # --- FSM & Logic Setup ---
        # TODO: Set initial state to "PLAYER_CONTROLLED"
        self.state = "PLAYER_CONTROLLED"
        # TODO: Initialize target_enemy (None) and path (empty list)
        self.target_enemy = None
        self.path = []
        # TODO: Initialize repath_timer and repath_interval (e.g., 0.5)
        self.repath_timer = 0
        self.repath_interval = 0.5


        # --- Visualization ---
        # TODO: Create the color_map dictionary for the 3 states (White, Blue, Magenta)
        self.color_map = {"PLAYER_CONTROLLED" : (255, 255, 255), "AUTOMATIC_CHASE" : (0, 0, 255), "PATH_FOLLOWING" : (0, 255, 255)}
        # TODO: Set self.color based on current state
        self.color = self.color_map[self.state]
        
        # --- Pygame Rect ---
        # TODO: Create self.rect using pygame.Rect
        self.rect = pygame.Rect(0,0,self.size,self.size)
        # TODO: Center the rect on the position
        self.rect.center = self.position
        

    # --- STATE SWITCHING ---

    def switch_mode(self):
        """
        Toggles between PLAYER_CONTROLLED and AUTOMATIC_CHASE.
        """
        # TODO: If state is PLAYER or PATH_FOLLOWING, switch to CHASE.
        #       (Don't forget to reset target_enemy to None!)
        if self.state == "PLAYER_CONTROLLED" or self.state == "PATH_FOLLOWING":
            self.state = "AUTOMATIC_CHASE"
            self.target_enemy = None

        # TODO: Else, switch back to PLAYER_CONTROLLED.
        else:
            self.state = "PLAYER_CONTROLLED"
        # TODO: Update self.color
        self.color = self.color_map[self.state]
        

    def set_path(self, path_nodes):
        """
        Sets the path and switches state to PATH_FOLLOWING.
        """
        # TODO: Store the path_nodes in self.path
        self.path = path_nodes
        # TODO: If self.path is not empty: (Hint: use 'if self.path:')
        #       Change state to "PATH_FOLLOWING"
        #       Update self.color using self.color_map
        if self.path:
            self.state = "PATH_FOLLOWING"
            self.color = self.color_map[self.state]
        

    # --- THE BRAIN (MAIN LOOP) ---

    def update(self, dt, player_direction_vector, collidables, targets, grid):
        """
        The Main Router. Decides behavior based on current state.
        """
        # --- STATE 1: MANUAL CONTROL ---
        if self.state == "PLAYER_CONTROLLED":
            # Just move based on WASD inputs
            self._execute_movement(dt, player_direction_vector, collidables)
        
        # --- STATE 2: AI CHASE MODE ---
        elif self.state == "AUTOMATIC_CHASE":
            # 1. Manage the Timer (Don't think too hard every frame)
            self.repath_timer -= dt
            
            # 2. Find Target (Eyes)
            # If we don't have a target, look for the closest one
            if self.target_enemy is None:
                self.target_enemy = self._find_closest_enemy(targets)

            # 3. Calculate Plan (Brain)
            # Only run A* if we have a target AND the timer is ready
            if self.target_enemy and self.repath_timer <= 0:
                self.repath_timer = self.repath_interval # Reset timer
                
                # Convert Agent Pos to Start Node
                # We use +size//2 to get the center of the agent
                start_pos = (self.position[0] + self.size//2, self.position[1] + self.size//2)
                start_node = grid.get_node_from_pos(start_pos)
                
                # Convert Enemy Rect to End Node
                end_pos = self.target_enemy.center
                end_node = grid.get_node_from_pos(end_pos)
                
                # Safety check: Ensure nodes exist and aren't the same one
                if start_node and end_node and start_node != end_node:
                    
                    # REFRESH THE MAP: Tell nodes about current obstacles
                    for row in grid.grid:
                        for node in row:
                            node.update_neighbors(grid.grid)
                            
                    # RUN A*: Get the new path
                    new_path = a_star_algorithm(None, grid, start_node, end_node)
                    
                    if new_path:
                        self.path = new_path

            # 4. Execute Movement (Legs)
            if self.path:
                # Use the path following logic to move along the purple line
                self._execute_path_following(dt)
            else:
                # If no path found (or arrived), just stop
                self.velocity = np.array([0.0, 0.0])

        # --- STATE 3: FIXED PATH MODE ---
        elif self.state == "PATH_FOLLOWING":
            # Just follow the existing list until it ends
            self._execute_path_following(dt)

    # --- BEHAVIOR LOGIC (THE LEGS) ---

    def _execute_path_following(self, dt):
        """
        Moves along the A* path nodes.
        """
        # TODO: Check if self.path is empty (using 'not self.path')
        if not self.path:
            # TODO: If empty, set self.state to "PLAYER_CONTROLLED"
            self.state = "PLAYER_CONTROLLED"
            # TODO: Update self.color using self.color_map
            self.color = self.color_map[self.state]
            # TODO: Return (Stop running this function)
            return
        
        # --- 1. IDENTIFY TARGET ---
        # TODO: Get the first node: target_node = self.path[0]
        target_node = self.path[0]
        
        # TODO: Calculate target_x = target_node.x + (self.size / 2)
        target_x = target_node.x + (self.size / 2)
        # TODO: Calculate target_y = target_node.y + (self.size / 2)
        target_y = target_node.y + (self.size / 2)
        # TODO: Create a numpy array target_pos from these
        target_pos = np.array([target_x, target_y])
        
        # --- 2. CALCULATE MOVEMENT ---
        # TODO: Calculate vector_to_target (target_pos - self.position)
        vector_to_target = target_pos - self.position
        # TODO: Calculate distance (norm of vector_to_target)
        dist = np.linalg.norm(vector_to_target)
        
        # --- 3. CHECK ARRIVAL ---
        # TODO: If distance < 5.0: (We are close enough)
        if dist < 5.0:
            # TODO: Pop the first node: self.path.pop(0)
            self.path.pop(0)
            
            # TODO: If self.path is still not empty:
            if self.path:
                # TODO: Recursive call: self._execute_path_following(dt)
                self._execute_path_following(dt)
            # TODO: Else (Path finished):
            else:
                # TODO: Set self.velocity to [0,0]
                self.velocity = np.array([0,0])
            
            # TODO: Return (Important to stop processing this specific frame)
            return
        
        # --- 4. MOVE ---
        # TODO: Calculate direction = vector_to_target / distance
        direction = vector_to_target / dist
        # TODO: Update self.forward_vector = direction (So it looks where it's going)
        self.forward_vector = direction
        
        # TODO: Calculate self.velocity = direction * self.speed
        self.velocity = direction * self.speed
        
        # TODO: Update self.position += self.velocity * dt
        self.position += self.velocity * dt
        # TODO: Sync self.rect.center = self.position
        self.rect.center = self.position

    def _find_closest_enemy(self, enemies):
        """
        Returns the enemy Rect from the list that is closest to the agent.
        """
        closest_enemy = None
        min_dist = float('inf') # Start with an infinitely large distance
        
        # TODO: Loop through 'enemy' in 'enemies'
        for enemy in enemies:
            # TODO: Get enemy center as a numpy array -> np.array(enemy.center)
            enemy_center = np.array(enemy.center)
            # TODO: Calculate vector from self.position to enemy center
            vector_to_enemy_center = enemy_center - self.position
            # TODO: Calculate distance (norm)
            dist_to_enemy_center = np.linalg.norm(vector_to_enemy_center) 
            # TODO: If distance < min_dist:
                # Update min_dist
                # Update closest_enemy
            if dist_to_enemy_center < min_dist:
                min_dist = dist_to_enemy_center
                closest_enemy = enemy
        
        return closest_enemy

    def _execute_movement(self, dt, direction_vector, collidables):
        """
        Standard physics movement with collision detection.
        """
        # TODO: Calculate the norm of direction_vector
        norm = np.linalg.norm(direction_vector)
        
        # TODO: If norm > 0:
        #       normalized_direction = direction_vector / norm
        # TODO: Else:
        #       normalized_direction = [0, 0]
        if norm > 0:
            normalized_direction = direction_vector / norm
        else:
            normalized_direction = np.array([0.0, 0.0]) # This is a NumPy Array
        
        # TODO: Set self.velocity = normalized_direction * self.speed
        self.velocity = normalized_direction * self.speed
        
        # --- X AXIS MOVEMENT ---
        # TODO: Update self.position[0] by adding (velocity[0] * dt)
        self.position[0] += self.velocity[0] * dt 
        # TODO: Sync self.rect.centerx to round(self.position[0])
        self.rect.centerx = round(self.position[0])
        
        # TODO: Loop through 'obstacle' in 'collidables':
        for obstacle in collidables:
            # TODO: If self.rect.colliderect(obstacle):
            if self.rect.colliderect(obstacle):
                # TODO: If velocity[0] > 0: (Moving Right)
                if self.velocity[0] > 0:
                    self.rect.right = obstacle.left
                #       Set self.rect.right = obstacle.left
                # TODO: Elif velocity[0] < 0: (Moving Left)
                #       Set self.rect.left = obstacle.right
                elif self.velocity[0] < 0: 
                    self.rect.left = obstacle.right
                
                # CRITICAL: Sync float position back to the fixed rect
                self.position[0] = self.rect.centerx
        
        # --- Y AXIS MOVEMENT ---
        # TODO: Update self.position[1] by adding (velocity[1] * dt)
        self.position[1] += self.velocity[1] * dt
        # TODO: Sync self.rect.centery to round(self.position[1])
        self.rect.centery = round(self.position[1])
        
        # TODO: Loop through 'obstacle' in 'collidables':
        for obstacle in collidables:
            # TODO: If self.rect.colliderect(obstacle):
            if self.rect.colliderect(obstacle):
                # TODO: If velocity[1] > 0: (Moving Down)
                #       Set self.rect.bottom = obstacle.top
                if self.velocity[1] > 0:
                    self.rect.bottom = obstacle.top
                # TODO: Elif velocity[1] < 0: (Moving Up)
                elif self.velocity[1] < 0:
                #       Set self.rect.top = obstacle.bottom
                    self.rect.top = obstacle.bottom
                
                # CRITICAL: Sync float position back to the fixed rect
                self.position[1] = self.rect.centery


    # --- VISUALIZATION ---

    def draw(self, screen):
        """
        Draws the agent and its forward vector.
        """
        # 1. Draw the Agent's Body
        # We use the self.rect because it handles the integer pixel positions perfectly
        pygame.draw.rect(screen, self.color, self.rect)
        
        # 2. Draw the Forward Vector (The "Head")
        # We create a red line starting from the center and pointing outwards
        # We multiply by size * 1.5 to make the line stick out a bit
        line_start = self.position
        line_end = self.position + self.forward_vector * self.size * 1.5
        
        pygame.draw.line(screen, (255, 0, 0), line_start, line_end, 2)
    
    def draw_awareness(self, screen, enemies, font):
        """
        Debug method: Draws lines and text to show what the agent 'sees'.
        """
        # Threshold for 90-degree Field of View (cos(45))
        fov_threshold = 0.707 

        for enemy_rect in enemies:
            # 1. Get Vector to Enemy
            enemy_center = np.array(enemy_rect.center)
            vector_to_enemy = enemy_center - self.position
            distance = np.linalg.norm(vector_to_enemy)
            
            # Default dot product (if distance is 0, we can't calculate direction)
            dot_product = -1.0
            
            if distance > 0:
                # 2. Normalize to get Direction
                direction_to_enemy = vector_to_enemy / distance
                
                # 3. Calculate Dot Product
                # Compare Agent's Forward Vector vs Direction to Enemy
                dot_product = np.dot(self.forward_vector, direction_to_enemy)
            
            # 4. Visualization Logic
            # Green = In front (Seen)
            # Red = Behind
            # Yellow = Side
            enemy_color = (255, 255, 0) 
            if dot_product > fov_threshold:
                enemy_color = (0, 255, 0) 
            elif dot_product < 0:
                enemy_color = (255, 0, 0)

            # Draw the enemy with the status color
            pygame.draw.rect(screen, enemy_color, enemy_rect)
            
            # Draw the math value above the enemy
            text = font.render(f"{dot_product:.2f}", True, (0, 0, 0))
            screen.blit(text, (enemy_rect.x, enemy_rect.y - 20))