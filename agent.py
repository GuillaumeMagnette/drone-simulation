import pygame
import numpy as np
from algorithm import a_star_algorithm 

# --- PHASE 2: PHYSICS ENGINE ---
class PhysicsEntity:
    def __init__(self, x, y, mass=1.0, max_speed=300.0, max_force=800.0):
        # Position is float for precision
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.acceleration = np.array([0.0, 0.0], dtype=float)
        
        self.mass = mass
        self.max_speed = max_speed
        self.max_force = max_force
        # Friction: 1.0 = No Friction, 0.9 = High Friction. 
        # 0.96 feels like a slippery drone.
        self.friction = 0.96  

    def apply_force(self, force):
        # Newton's 2nd Law: F = ma -> a = F/m
        self.acceleration += force / self.mass

    def update_physics(self, dt):
        # 1. Update Velocity
        self.velocity += self.acceleration * dt
        
        # 2. Limit Speed (Terminal Velocity)
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed
            
        # 3. Apply Air Resistance / Friction
        self.velocity *= self.friction
        
        # 4. Update Position
        self.position += self.velocity * dt
        
        # 5. Reset Acceleration (Forces are instantaneous per frame)
        self.acceleration[:] = 0

# --- THE AGENT ---
class Agent(PhysicsEntity):
    def __init__(self, x, y, size):
        # Initialize Physics (Mass=1.5 makes it feel weighty)
        super().__init__(x, y, mass=1.5, max_speed=350.0, max_force=2000.0)
        
        self.size = size
        # Visuals: Agent is Green by default
        self.color = (0, 255, 0)
        self.forward_vector = np.array([0.0, -1.0])

        # Hitbox (Synced to position in update)
        self.rect = pygame.Rect(0, 0, self.size, self.size)
        self.rect.center = self.position

        # Navigation State
        self.path = []
        self.repath_timer = 0.0
        self.repath_interval = 0.2  # Re-run A* every 0.2s

        # Reflex/Safety State
        self.danger_dist = 1000.0 
        self.dodge_timer = 0.0
        self.locked_safety_dir = None
        
        # RL Hyperparameters (Kept for compatibility, though we are in Physics Mode now)
        self.q_table = {}
        self.epsilon = 0.1
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.alpha = 0.1
        self.gamma = 0.9

    def update(self, dt, walls, targets, grid, projectiles):
        # --- 1. SENSE ---
        _, reflex_action = self.get_relative_state(targets, projectiles, walls)
        steering_force = np.array([0.0, 0.0], dtype=float)

        # Find target
        kill_target = self._find_closest_enemy(targets)
        dist_to_target = float('inf')
        has_los = False
        
        if kill_target:
            dist_to_target = np.linalg.norm(np.array(kill_target.center) - self.position)
            has_los = self._has_line_of_sight(kill_target.center, walls)

        # --- 2. STATE DETERMINATION ---
        
        # Priority 1: REFLEX (Dodge)
        if reflex_action is not None:
            if reflex_action == 0: steering_force = np.array([0.0, -self.max_force])
            elif reflex_action == 1: steering_force = np.array([0.0, self.max_force])
            elif reflex_action == 2: steering_force = np.array([-self.max_force, 0.0])
            elif reflex_action == 3: steering_force = np.array([self.max_force, 0.0])
            self.color = (255, 0, 0) # RED

        # Priority 2: TERMINAL GUIDANCE (Attack)
        # Condition: We see the target AND (we are close OR we strictly finished the path)
        # FIX: Removed 'not self.path' as a primary trigger to prevent wall-ramming at start
        elif kill_target and has_los and dist_to_target < 250:
            steering_force = self.seek(kill_target.center)
            self.color = (255, 140, 0) # ORANGE
            self.path = [] # Clear path to prevent conflict

        # Priority 3: NAVIGATION (A* Pathfinding)
        else:
            self.color = (0, 255, 0) # GREEN
            
            # --- REPATHING LOGIC (Fixed: Runs independently of current path) ---
            self.repath_timer -= dt
            if self.repath_timer <= 0:
                self.repath_timer = self.repath_interval
                self._calculate_path(targets, grid)

            # --- PATH FOLLOWING ---
            if self.path:
                target_node = self.path[0]
                # Center of the grid node
                target_pos = np.array([target_node.x + self.size/2, target_node.y + self.size/2])
                
                # Distance to waypoint
                dist_to_node = np.linalg.norm(target_pos - self.position)
                
                # Radius of Satisfaction (20px)
                if dist_to_node < 20:
                    self.path.pop(0)
                    # If we popped the last node, next frame might trigger Terminal or Idle
                else:
                    steering_force = self.seek(target_pos)
            else:
                # No path found (surrounded by walls?) or waiting for A*
                # Gentle braking to avoid drifting forever
                steering_force = -self.velocity * 2.0 
                self.color = (100, 100, 100) # GREY

        # --- 3. APPLY FORCES ---
        self.apply_force(steering_force)

        # --- 4. PHYSICS INTEGRATION ---
        self.update_physics(dt)
        self.rect.center = self.position

        # --- 5. COLLISION RESOLUTION ---
        self._handle_collisions(walls)
        
        # Update Forward Vector
        if np.linalg.norm(self.velocity) > 1:
            self.forward_vector = self.velocity / np.linalg.norm(self.velocity)
                                                                 

    def seek(self, target_pos):
        """
        PD Controller (Proportional-Derivative).
        Calculates a force to steer towards the target while dampening oscillation.
        """
        # 1. Calculate Error (Vector to Target)
        desired = target_pos - self.position
        dist = np.linalg.norm(desired)
        
        if dist == 0: return np.array([0.0, 0.0])

        # Normalize and Scale to Max Speed
        desired = (desired / dist) * self.max_speed

        # 2. Calculate Steering Force
        # Steering = Desired_Velocity - Current_Velocity
        # (Current Velocity acts as the D-term/Damping)
        steer = desired - self.velocity
        
        # 3. Clamp Force (Physical Limit of Motors)
        steer_mag = np.linalg.norm(steer)
        if steer_mag > self.max_force:
            steer = (steer / steer_mag) * self.max_force
            
        return steer

    def _handle_collisions(self, walls):
        # 1. Screen Boundaries (Keep 'Bounce' or make lethal?)
        SCREEN_LIMIT = 800
        hit_boundary = False
        
        if self.rect.left < 0 or self.rect.right > SCREEN_LIMIT:
            hit_boundary = True
        if self.rect.top < 0 or self.rect.bottom > SCREEN_LIMIT:
            hit_boundary = True
            
        # 2. Wall Objects
        hit_wall = False
        if self.rect.collidelist(walls) != -1:
            hit_wall = True

        # --- REALISM SWITCH ---
        REALISTIC_CRASH = False # <--- Toggle this to TRUE to test realism

        if REALISTIC_CRASH:
            if hit_wall or hit_boundary:
                # In a real sim, we flag the agent as 'destroyed'
                # For this specific loop, we can just stop it dead to visualize the crash
                self.velocity[:] = 0
                self.color = (0, 0, 0) # Black = Dead
                # Ideally, you would signal 'main.py' to reset the episode here
        else:
            # --- OLD SLIDING LOGIC (Keep this for now for tuning) ---
            if self.rect.left < 0:
                self.position[0] = self.size / 2
                self.velocity[0] *= -0.5 
            elif self.rect.right > SCREEN_LIMIT:
                self.position[0] = SCREEN_LIMIT - self.size / 2
                self.velocity[0] *= -0.5

            if self.rect.top < 0:
                self.position[1] = self.size / 2
                self.velocity[1] *= -0.5
            elif self.rect.bottom > SCREEN_LIMIT:
                self.position[1] = SCREEN_LIMIT - self.size / 2
                self.velocity[1] *= -0.5
                
            self.rect.center = self.position

            for wall in walls:
                if self.rect.colliderect(wall):
                    dx = (self.position[0] - wall.centerx) / (wall.width / 2 + self.size / 2)
                    dy = (self.position[1] - wall.centery) / (wall.height / 2 + self.size / 2)
                    if abs(dx) > abs(dy):
                        if dx > 0: self.position[0] = wall.right + self.size/2
                        else: self.position[0] = wall.left - self.size/2
                        self.velocity[0] = 0
                    else:
                        if dy > 0: self.position[1] = wall.bottom + self.size/2
                        else: self.position[1] = wall.top - self.size/2
                        self.velocity[1] = 0
                    self.rect.center = self.position

                    
    
    def _calculate_path(self, targets, grid):
        """Wraps the A* logic."""
        if not targets: return
        
        target_enemy = self._find_closest_enemy(targets)
        if not target_enemy: return

        start_node = grid.get_node_from_pos(self.position)
        end_node = grid.get_node_from_pos(target_enemy.center)

        if start_node and end_node and start_node != end_node:
            # Clean grid visuals for new path
            for row in grid.grid:
                for node in row: 
                    node.reset_visuals()
                    node.update_neighbors(grid.grid)
            
            # Run A*
            new_path = a_star_algorithm(None, grid, start_node, end_node)
            if new_path:
                self.path = new_path

    # --- SENSORY INPUT (Phase 1 Logic - Preserved) ---

    def get_relative_state(self, targets, projectiles, walls):
        # --- 1. TARGET SENSING ---
        target_dx, target_dy = 0, 0
        target_pos = None
        if self.path:
            target_node = self.path[0] 
            target_pos = (target_node.x + self.size//2, target_node.y + self.size//2)
        elif targets:
            closest = self._find_closest_enemy(targets)
            if closest: target_pos = closest.center
            
        if target_pos:
            if target_pos[0] - self.rect.centerx < -10: target_dx = -1
            elif target_pos[0] - self.rect.centerx > 10: target_dx = 1
            if target_pos[1] - self.rect.centery < -10: target_dy = -1
            elif target_pos[1] - self.rect.centery > 10: target_dy = 1

        # --- 2. REFLEX LAYER ---
        reflex_action = None 
        wall_x = 0
        wall_y = 0
        
        sensor = self.rect.inflate(-6, -6)
        step = 5
        if sensor.move(0, -step).collidelist(walls) != -1 or self.rect.top < 0: wall_y = -1
        if sensor.move(0, step).collidelist(walls) != -1 or self.rect.bottom > 800: wall_y = 1
        if sensor.move(-step, 0).collidelist(walls) != -1 or self.rect.left < 0: wall_x = -1
        if sensor.move(step, 0).collidelist(walls) != -1 or self.rect.right > 800: wall_x = 1

        self.dodge_timer -= 0.016

        closest_bullet = self._find_closest_projectile(projectiles)
        if closest_bullet:
            if not self._has_line_of_sight(closest_bullet.position, walls):
                closest_bullet = None

        if closest_bullet:
            dist = np.linalg.norm(closest_bullet.position - self.position)
            
            if dist < 350:
                b_vel = closest_bullet.direction * closest_bullet.speed
                p1 = closest_bullet.position
                p2 = p1 + (b_vel * 1.5)
                dist_traj, _, _ = self._point_to_segment_dist(self.position[0], self.position[1], p1[0], p1[1], p2[0], p2[1])
                self.danger_dist = dist_traj

                enter_panic = self.size + 15
                exit_panic  = self.size + 60 
                
                panic_mode = False
                if dist_traj < enter_panic: panic_mode = True
                elif dist_traj > exit_panic: panic_mode = False
                
                if panic_mode:
                    current_lock_valid = False
                    if self.dodge_timer > 0 and self.locked_safety_dir is not None:
                        idx = self.locked_safety_dir
                        check_moves = [(0,-1), (0,1), (-1,0), (1,0)] 
                        dx, dy = check_moves[idx]
                        if sensor.move(dx*step, dy*step).collidelist(walls) == -1:
                            current_lock_valid = True
                            reflex_action = self.locked_safety_dir

                    if not current_lock_valid:
                        best_score = -1
                        best_action = None
                        moves = [(0, -10, 0), (0, 10, 1), (-10, 0, 2), (10, 0, 3)]
                        
                        for dx, dy, action_idx in moves:
                            if sensor.move(dx, dy).collidelist(walls) != -1: continue 
                            if sensor.move(dx, dy).left < 0 or sensor.move(dx, dy).right > 800: continue
                            if sensor.move(dx, dy).top < 0 or sensor.move(dx, dy).bottom > 800: continue 

                            test_x = self.position[0] + dx
                            test_y = self.position[1] + dy
                            score, _, _ = self._point_to_segment_dist(test_x, test_y, p1[0], p1[1], p2[0], p2[1])
                            
                            if score > best_score:
                                best_score = score
                                best_action = action_idx
                        
                        if best_action is not None:
                            reflex_action = best_action
                            self.locked_safety_dir = best_action
                            self.dodge_timer = 0.2 

        return (target_dx, target_dy, wall_x, wall_y), reflex_action
    
    # --- HELPER METHODS ---
    # --- NEW: LIDAR SENSORS (For Phase 3 Brain) ---
    def cast_rays(self, walls, num_rays=8, max_dist=200):
        """
        Casts 'num_rays' evenly spaced around the agent.
        Returns: A numpy array of distances [0.0 to 1.0], where 1.0 is far, 0.0 is touching.
        """
        # 1. Define angles (0 to 2pi)
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        
        # We start rays from the center of the agent
        cx, cy = self.position
        
        distances = []
        
        for angle in angles:
            # Calculate end point of the ray
            dx = np.cos(angle)
            dy = np.sin(angle)
            
            # End point is max_dist away
            end_x = cx + dx * max_dist
            end_y = cy + dy * max_dist
            
            closest_dist = max_dist
            
            # Ray line segment
            ray_start = (cx, cy)
            ray_end = (end_x, end_y)
            
            # Check intersection with every wall
            for wall in walls:
                # clipline returns the segment of the line INSIDE the rect
                # If it returns a value, we hit the wall
                clipped = wall.clipline(ray_start, ray_end)
                
                if clipped:
                    # clipped[0] is the entry point (closest to start)
                    # clipped[1] is the exit point
                    
                    # Calculate distance to entry point
                    hit_x, hit_y = clipped[0]
                    dist = np.sqrt((hit_x - cx)**2 + (hit_y - cy)**2)
                    
                    if dist < closest_dist:
                        closest_dist = dist
            
            # Normalize (1.0 = Max Range, 0.0 = Touching Wall)
            # In RL, inputs between 0 and 1 are best.
            norm_dist = closest_dist / max_dist
            distances.append(norm_dist)
            
        return np.array(distances, dtype=np.float32)

    def draw_lidar(self, screen, walls):
        """Visualizes the rays."""
        cx, cy = self.position
        lidar_data = self.cast_rays(walls) # Get the normalized distances
        angles = np.linspace(0, 2 * np.pi, len(lidar_data), endpoint=False)
        max_dist = 200
        
        for i, dist_norm in enumerate(lidar_data):
            angle = angles[i]
            actual_dist = dist_norm * max_dist
            
            end_x = cx + np.cos(angle) * actual_dist
            end_y = cy + np.sin(angle) * actual_dist
            
            # Color logic: Red if close, Green if far
            color = (255 * (1-dist_norm), 255 * dist_norm, 0)
            
            pygame.draw.line(screen, color, (cx, cy), (end_x, end_y), 1)
            pygame.draw.circle(screen, color, (int(end_x), int(end_y)), 3)

    def _find_closest_enemy(self, enemies):
        closest = None
        min_dist = float('inf')
        for enemy in enemies:
            dist = np.linalg.norm(np.array(enemy.center) - self.position)
            if dist < min_dist:
                min_dist = dist
                closest = enemy
        return closest
        
    def _point_to_segment_dist(self, px, py, x1, y1, x2, y2):
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
    
    def _has_line_of_sight(self, target_pos, walls):
        for wall in walls:
            if wall.clipline(self.rect.center, target_pos):
                return False 
        return True 

    def _find_closest_projectile(self, projectiles):
        closest = None
        min_dist = float('inf')
        for p in projectiles:
            dist = np.linalg.norm(p.position - self.position)
            if dist < min_dist:
                min_dist = dist
                closest = p
        return closest

    def aim(self, target_pos):
        """Visual aim helper."""
        vector_to_target = target_pos - self.position
        norm = np.linalg.norm(vector_to_target)
        if norm > 0:
            self.forward_vector = vector_to_target / norm

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        # Draw forward vector/Velocity
        line_end = self.position + self.forward_vector * self.size * 1.5
        pygame.draw.line(screen, (255, 0, 0), self.position, line_end, 2)
    
    def draw_awareness(self, screen, enemies, font):
        fov_threshold = 0.707 
        for enemy_rect in enemies:
            vector_to_enemy = np.array(enemy_rect.center) - self.position
            distance = np.linalg.norm(vector_to_enemy)
            dot_product = -1 
            if distance > 0:
                direction_to_enemy = vector_to_enemy / distance
                dot_product = np.dot(self.forward_vector, direction_to_enemy)
            
            enemy_color = (255, 255, 0)
            if dot_product > fov_threshold: enemy_color = (0, 255, 0)
            elif dot_product < -0.5: enemy_color = (255, 0, 0)
            
            pygame.draw.rect(screen, enemy_color, enemy_rect)

    # --- COMPATIBILITY STUBS (To prevent main.py crashing before refactor) ---
    def get_grid_state(self): return (0,0)
    def choose_action(self, state): return 0
    def move_discrete(self, action): return np.array([0,0])
    def learn(self, s, a, r, ns): pass