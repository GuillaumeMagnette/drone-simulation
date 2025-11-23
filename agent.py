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

        # NEW: Initialize danger distance
        self.danger_dist = 1000.0 
        
        
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
            "PLAYER_CONTROLLED": (0, 255, 0),
            "AUTOMATIC_CHASE": (100, 100, 255),
            "PATH_FOLLOWING": (255, 0, 255)
        }
        self.color = self.color_map[self.state]

        # --- RL BRAIN SETUP ---
        self.q_table = {} # The empty cheat sheet
        
        # Hyperparameters
        self.epsilon = 1.0      # 100% Random at start
        self.epsilon_min = 0.1  # Minimum exploration
        self.epsilon_decay = 0.995 # Fade out exploration over time
        
        self.alpha = 0.1        # Learning Rate
        self.gamma = 0.9        # Discount Factor

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

        # 1. Check Walls
        for obstacle in collidables:
            if self.rect.colliderect(obstacle):
                if self.velocity[0] > 0: self.rect.right = obstacle.left
                elif self.velocity[0] < 0: self.rect.left = obstacle.right
                self.position[0] = self.rect.centerx
        
        # 2. Check Screen Boundaries (NEW)
        # Assuming we pass screen_width/height or know them (800x800)
        # You can hardcode 800 for now or pass it in __init__
        SCREEN_LIMIT = 800 
        
        if self.rect.left < 0:
            self.rect.left = 0
            self.position[0] = self.rect.centerx
        elif self.rect.right > SCREEN_LIMIT:
            self.rect.right = SCREEN_LIMIT
            self.position[0] = self.rect.centerx

        # Y-axis collision
        self.position[1] += self.velocity[1] * dt
        self.rect.centery = round(self.position[1])

        # 1. Check Walls
        for obstacle in collidables:
            if self.rect.colliderect(obstacle):
                if self.velocity[1] > 0: self.rect.bottom = obstacle.top
                elif self.velocity[1] < 0: self.rect.top = obstacle.bottom
                self.position[1] = self.rect.centery

        # 2. Check Screen Boundaries (NEW)
        if self.rect.top < 0:
            self.rect.top = 0
            self.position[1] = self.rect.centery
        elif self.rect.bottom > SCREEN_LIMIT:
            self.rect.bottom = SCREEN_LIMIT
            self.position[1] = self.rect.centery

    def aim(self, target_pos):
        """
        Calculates the forward vector to face a target position.
        :param target_pos: A NumPy array representing the target's coordinates.
        """
        # --- 2. IMPLEMENT THE AIMING LOGIC ---
        vector_to_target = target_pos - self.position
        
        norm = np.linalg.norm(vector_to_target)
        if norm > 0:
            # Normalize the vector to get the pure direction
            self.forward_vector = vector_to_target / norm
            
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

    # --- RL INTERFACE ---

    def get_grid_state(self):
        """
        Converts the agent's continuous pixel position into discrete grid coordinates.
        Returns: (col, row) tuple which acts as the 'State' for the Q-Table.
        """
        # We use the size of the agent (which matches grid size) to divide
        grid_x = int(self.position[0] // self.size)
        grid_y = int(self.position[1] // self.size)
        return (grid_x, grid_y)
    
    # --- SENSORY INPUT (THE NEW EYES) ---

    def get_relative_state(self, targets, projectiles, walls):
        """
        Returns: 
        (target_dx, target_dy, danger_level, safety_direction, wall_x, wall_y)
        """
        # --- 1. TARGET SENSING (Existing Logic) ---
        target_dx = 0
        target_dy = 0
        target_pos = None
        
        if self.path:
            lookahead_index = min(len(self.path) - 1, 2) 
            target_node = self.path[lookahead_index]
            next_immediate_node = self.path[0]
            node_center = np.array([next_immediate_node.x + self.size//2, 
                                    next_immediate_node.y + self.size//2])
            if np.linalg.norm(node_center - self.position) < 30:
                self.path.pop(0)
            target_pos = (target_node.x + self.size // 2, target_node.y + self.size // 2)
        
        if target_pos is None:
            closest_enemy = self._find_closest_enemy(targets)
            if closest_enemy: target_pos = closest_enemy.center

        if target_pos:
            tx, ty = target_pos
            raw_dx = tx - self.rect.centerx
            raw_dy = ty - self.rect.centery
            if raw_dx < -10: target_dx = -1
            elif raw_dx > 10: target_dx = 1
            if raw_dy < -10: target_dy = -1
            elif raw_dy > 10: target_dy = 1

        # --- 2. THREAT SENSING (ESCAPE COMPASS) ---
        danger_level = 0
        safety_direction = 0 # 0=Safe, 1=Up, 2=Down, 3=Left, 4=Right
        self.is_in_danger_zone = False 
        self.danger_dist = 1000.0 # Reset distance

        closest_bullet = self._find_closest_projectile(projectiles)
        
        if closest_bullet:
            dist_to_bullet = np.linalg.norm(closest_bullet.position - self.position)
            
            if dist_to_bullet < 300:
                b_vel = closest_bullet.direction * closest_bullet.speed
                p1 = closest_bullet.position
                p2 = p1 + (b_vel * 1.5) 
                
                # GET DISTANCE AND NEAREST POINT (nx, ny)
                dist_from_trajectory, nx, ny = self._point_to_segment_dist(
                    self.position[0], self.position[1],
                    p1[0], p1[1],
                    p2[0], p2[1]
                )
                self.danger_dist = dist_from_trajectory
                
                # --- CALCULATE ESCAPE COMPASS ---
                # Vector from Line -> Agent
                esc_x = self.position[0] - nx
                esc_y = self.position[1] - ny
                
                # "Perfect Center" Fix: If vector is essentially zero, pick a perpendicular
                if abs(esc_x) < 0.1 and abs(esc_y) < 0.1:
                    # Perpendicular to bullet velocity (-y, x)
                    esc_x = -b_vel[1]
                    esc_y = b_vel[0]

                # Determine primary direction of escape
                if abs(esc_x) > abs(esc_y):
                    if esc_x < 0: safety_direction = 3 # LEFT
                    else:         safety_direction = 4 # RIGHT
                else:
                    if esc_y < 0: safety_direction = 1 # UP
                    else:         safety_direction = 2 # DOWN

                # --- ZONE LOGIC ---
                critical_radius = self.size + 10 
                warning_radius = critical_radius * 2.0 
                
                if dist_from_trajectory < critical_radius:
                    danger_level = 2 
                    self.is_in_danger_zone = True
                elif dist_from_trajectory < warning_radius:
                    danger_level = 1
                else:
                    safety_direction = 0 # Reset if safe

        # --- 3. WALL SENSING (Short Whiskers) ---
        wall_x = 0
        wall_y = 0
        step = 5 # Reduced step
        
        rect_right = self.rect.move(step, 0)
        if rect_right.right > 800 or rect_right.collidelist(walls) != -1: wall_x = 1
        rect_left = self.rect.move(-step, 0)
        if rect_left.left < 0 or rect_left.collidelist(walls) != -1: wall_x = -1
        rect_down = self.rect.move(0, step)
        if rect_down.bottom > 800 or rect_down.collidelist(walls) != -1: wall_y = 1
        rect_up = self.rect.move(0, -step)
        if rect_up.top < 0 or rect_up.collidelist(walls) != -1: wall_y = -1

        # RETURN UPDATED TUPLE (Added safety_direction)
        return (int(target_dx), int(target_dy), int(danger_level), int(safety_direction), int(wall_x), int(wall_y))

    def _point_to_segment_dist(self, px, py, x1, y1, x2, y2):
        """
        Calculates distance AND the nearest point on the segment.
        Returns: (distance, nearest_x, nearest_y)
        """
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0: 
            return np.sqrt((px - x1)**2 + (py - y1)**2), x1, y1

        t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)
        t = max(0, min(1, t))
        
        nearest_x = x1 + t * dx
        nearest_y = y1 + t * dy
        
        dist = np.sqrt((px - nearest_x)**2 + (py - nearest_y)**2)
        return dist, nearest_x, nearest_y

    def _find_closest_projectile(self, projectiles):
        """Helper to find the nearest bullet."""
        closest = None
        min_dist = float('inf')
        
        for p in projectiles:
            # Check distance to center
            dist = np.linalg.norm(p.position - self.position)
            if dist < min_dist:
                min_dist = dist
                closest = p
        return closest

    def move_discrete(self, action_index):
        """
        Translates a discrete index (0-3) into a movement vector.
        0: Up, 1: Down, 2: Left, 3: Right
        """
        # Create a vector based on the integer action
        direction = np.array([0.0, 0.0])
        
        if action_index == 0:   # UP
            direction[1] = -1
        elif action_index == 1: # DOWN
            direction[1] = 1
        elif action_index == 2: # LEFT
            direction[0] = -1
        elif action_index == 3: # RIGHT
            direction[0] = 1
            
        # Now we just pass this vector to our existing physics engine!
        # Note: We need to pass 'collidables' to this function or store it in self
        # For now, let's just calculate the vector.
        return direction
    
    def choose_action(self, state):
        """
        Epsilon-Greedy Logic:
        Sometimes do something random to learn.
        Mostly do what we know is best.
        """
        # 1. EXPLORE: Random Action
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 4) # Random 0, 1, 2, or 3
        
        # 2. EXPLOIT: Best Action
        # Check if this state exists in the table yet
        if state not in self.q_table:
            # If new state, initialize with zeros [Up, Down, Left, Right]
            self.q_table[state] = [0.0, 0.0, 0.0, 0.0]
            
        # Pick the index of the highest value
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state):
        """
        The Heart of Q-Learning.
        """
        # FIX: Ensure CURRENT state exists before reading it
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0, 0.0]

        # 1. Ensure the 'next_state' exists (You already have this)
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0, 0.0, 0.0, 0.0]

        # 2. Get the Old Q-Value
        old_value = self.q_table[state][action]
        
        # 3. Get the Max Future Q-Value
        next_max = np.max(self.q_table[next_state])
        
        # 4. The Bellman Equation
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        
        # 5. Update the Table
        self.q_table[state][action] = new_value