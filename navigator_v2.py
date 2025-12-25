"""
navigator_v2.py - Layer 2: Hybrid Pathfinding (A* + APF)

KEY FIX: Cost-based A* instead of binary blocked cells.
- Cells near walls have HIGHER cost, not blocked entirely
- A* will route through corridors but PREFER open areas
- APF is scaled down when following a valid path
"""

import numpy as np
import pygame
import heapq
from physics import BUILDING_HEIGHT, MAX_FORCE


class Pathfinder:
    def __init__(self, walls, screen_size, grid_size=30):  # Smaller grid = more resolution
        self.grid_size = grid_size
        self.width = int(screen_size // grid_size) + 1
        self.height = int(screen_size // grid_size) + 1
        self.walls = walls
        self.screen_size = screen_size
        
        # Cost grid: 0 = free, 1-10 = near wall, inf = blocked
        self.cost_grid = np.ones((self.width, self.height), dtype=float)
        self._bake_grid()
    

    def _bake_grid(self):
        """
        Build a COST grid instead of binary blocked grid.
        Cells inside walls = infinite cost (blocked)
        Cells in narrow corridors = infinite cost (blocked) 
        Cells near walls = elevated cost (discouraged)
        Cells far from walls = base cost (preferred)
        """
        MIN_CORRIDOR_WIDTH = 45  # Agent needs at least this much space to pass
        
        for x in range(self.width):
            for y in range(self.height):
                cx = x * self.grid_size + self.grid_size // 2
                cy = y * self.grid_size + self.grid_size // 2
                
                # Check if cell CENTER is inside a wall -> blocked
                center_rect = pygame.Rect(cx - 5, cy - 5, 10, 10)
                if center_rect.collidelist(self.walls) != -1:
                    self.cost_grid[x, y] = float('inf')
                    continue
                
                # Find clearance in each direction
                clearance = self._get_directional_clearance(cx, cy)
                
                # Check for narrow corridors (walls on opposite sides)
                horizontal_gap = clearance['left'] + clearance['right']
                vertical_gap = clearance['up'] + clearance['down']
                
                # If EITHER corridor dimension is too narrow, block it
                if horizontal_gap < MIN_CORRIDOR_WIDTH or vertical_gap < MIN_CORRIDOR_WIDTH:
                    self.cost_grid[x, y] = float('inf')
                    continue
                
                # Otherwise, use distance to nearest wall for cost
                min_dist = min(clearance.values())
                
                if min_dist < 15:
                    self.cost_grid[x, y] = 8.0  # Very close - high cost
                elif min_dist < 30:
                    self.cost_grid[x, y] = 4.0  # Close - medium cost
                elif min_dist < 50:
                    self.cost_grid[x, y] = 2.0  # Nearby - slight cost
                else:
                    self.cost_grid[x, y] = 1.0  # Open - base cost

    def _get_directional_clearance(self, px, py):
        """
        Get clearance distance in each cardinal direction.
        Returns dict with 'left', 'right', 'up', 'down' distances.
        """
        MAX_CHECK = 100  # Don't check beyond this
        
        clearance = {
            'left': MAX_CHECK,
            'right': MAX_CHECK,
            'up': MAX_CHECK,
            'down': MAX_CHECK
        }
        
        for w in self.walls:
            # Left clearance: wall is to the left of point
            if w.right <= px and w.top <= py <= w.bottom:
                dist = px - w.right
                clearance['left'] = min(clearance['left'], dist)
            
            # Right clearance: wall is to the right of point
            if w.left >= px and w.top <= py <= w.bottom:
                dist = w.left - px
                clearance['right'] = min(clearance['right'], dist)
            
            # Up clearance: wall is above point
            if w.bottom <= py and w.left <= px <= w.right:
                dist = py - w.bottom
                clearance['up'] = min(clearance['up'], dist)
            
            # Down clearance: wall is below point
            if w.top >= py and w.left <= px <= w.right:
                dist = w.top - py
                clearance['down'] = min(clearance['down'], dist)
        
        # Also check screen edges
        clearance['left'] = min(clearance['left'], px)
        clearance['right'] = min(clearance['right'], self.screen_size - px)
        clearance['up'] = min(clearance['up'], py)
        clearance['down'] = min(clearance['down'], self.screen_size - py)
        
        return clearance

    def _find_nearest_free_cell(self, start_idx):
        """BFS to find nearest non-blocked cell."""
        if self.cost_grid[start_idx[0], start_idx[1]] < float('inf'):
            return start_idx
        queue = [start_idx]
        visited = {start_idx}
        attempts = 0
        while queue and attempts < 500:
            cx, cy = queue.pop(0)
            attempts += 1
            for nx, ny in [(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1),
                           (cx+1, cy+1), (cx-1, cy-1), (cx+1, cy-1), (cx-1, cy+1)]:
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if (nx, ny) in visited:
                        continue
                    if self.cost_grid[nx, ny] < float('inf'):
                        return (nx, ny)
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return start_idx

    def get_path(self, start_pos, target_pos):
        start_idx = (int(start_pos[0] // self.grid_size), int(start_pos[1] // self.grid_size))
        end_idx = (int(target_pos[0] // self.grid_size), int(target_pos[1] // self.grid_size))
        
        # Clamp to grid bounds
        start_idx = (max(0, min(self.width-1, start_idx[0])), max(0, min(self.height-1, start_idx[1])))
        end_idx = (max(0, min(self.width-1, end_idx[0])), max(0, min(self.height-1, end_idx[1])))
        
        # Find valid start/end if inside walls
        start_idx = self._find_nearest_free_cell(start_idx)
        end_idx = self._find_nearest_free_cell(end_idx)
        
        # Short distance + clear line of sight = skip A*
        dist = np.linalg.norm(np.array(start_pos[:2]) - np.array(target_pos[:2]))
        if dist < 60 and self._has_line_of_sight(start_pos, target_pos):
            return [start_idx, end_idx]

        return self._astar(start_idx, end_idx)

    def _has_line_of_sight(self, start_pos, target_pos):
        """Check if agent body can pass directly - thick LOS check."""
        return self._thick_los(start_pos[:2], target_pos[:2], radius=15.0)
    
    def _thick_los(self, start, end, radius):
        """Check if a circle can travel from start to end without hitting walls."""
        start = np.array(start)
        end = np.array(end)
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 0.1:
            return True
        
        direction = direction / length
        perp = np.array([-direction[1], direction[0]])
        
        # Check center + both edges at full radius
        for offset in [0, radius, -radius]:
            line_start = start + perp * offset
            line_end = end + perp * offset
            for w in self.walls:
                if w.clipline(line_start, line_end):
                    return False
        return True

    def _astar(self, start, end):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == end:
                return self._reconstruct_path(came_from, current)
            
            x, y = current
            
            # 4-Directional (cardinal) moves - always allowed if cell is passable
            for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    cell_cost = self.cost_grid[nx, ny]
                    if cell_cost == float('inf'):
                        continue
                    
                    tentative_g = g_score[current] + cell_cost
                    if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                        came_from[(nx, ny)] = current
                        g_score[(nx, ny)] = tentative_g
                        f = tentative_g + self._heuristic((nx, ny), end)
                        heapq.heappush(open_set, (f, (nx, ny)))
            
            # 4-Directional (diagonal) moves - CORNER CUT CHECK
            for nx, ny in [(x+1, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1)]:
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    cell_cost = self.cost_grid[nx, ny]
                    if cell_cost == float('inf'):
                        continue
                    
                    # CORNER CUT CHECK: Both adjacent cardinal cells must be SAFE
                    # Not just passable - they need low cost (far from walls)
                    dx, dy = nx - x, ny - y
                    adj1_cost = self.cost_grid[x + dx, y]  # Horizontal neighbor
                    adj2_cost = self.cost_grid[x, y + dy]  # Vertical neighbor
                    
                    # Block diagonal if EITHER adjacent cell is:
                    # - Blocked (inf), OR
                    # - High cost (near wall) - agent won't fit diagonally
                    HIGH_COST_THRESHOLD = 4.0
                    if (adj1_cost == float('inf') or adj1_cost >= HIGH_COST_THRESHOLD or
                        adj2_cost == float('inf') or adj2_cost >= HIGH_COST_THRESHOLD):
                        continue  # Too risky - skip this diagonal
                    
                    # Diagonal cost includes the "squeeze" penalty from adjacent cells
                    move_cost = 1.414 * cell_cost
                    tentative_g = g_score[current] + move_cost
                    
                    if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                        came_from[(nx, ny)] = current
                        g_score[(nx, ny)] = tentative_g
                        f = tentative_g + self._heuristic((nx, ny), end)
                        heapq.heappush(open_set, (f, (nx, ny)))
        
        return None  # No path found

    def _heuristic(self, a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]


class Navigator:
    def __init__(self):
        # Tuned for cooperation with cost-based A*
        self.K_ATTRACT = 12.0           # Strong goal pull
        self.K_REPULSE_WALL = 1500.0    # Moderate wall push (A* handles routing)
        self.K_AVOID_MISSILE = 4500.0
        self.SENSE_RADIUS = 50.0        # Tighter sensing - trust the path more
        
        self.pathfinder = None
        self.last_forces = {k: np.zeros(3) for k in ['attract', 'repulse', 'avoid', 'lift']}
        
        self.cached_path = []
        self.cached_target_pos = np.array([-1.0, -1.0, -1.0])
        self.path_recalc_timer = 0
    
    def reset(self):
        """Clear all internal state for a new episode."""
        self.pathfinder = None # Will be rebuilt on first step
        self.cached_path = []
        self.cached_target_pos = np.array([-1.0, -1.0, -1.0])
        self.path_recalc_timer = 0
        self.current_waypoint_idx = 0
        
        # Clear debug forces
        for k in self.last_forces:
            self.last_forces[k] = np.zeros(3)

    def _snap_to_valid(self, pos):
        """If inside wall, push out to nearest edge."""
        snapped = pos.copy()
        snapped[0] = np.clip(snapped[0], 5, 1195)
        snapped[1] = np.clip(snapped[1], 5, 1195)
        
        rect = pygame.Rect(snapped[0]-2, snapped[1]-2, 4, 4)
        wall_idx = rect.collidelist(self.pathfinder.walls)
        if wall_idx == -1:
            return snapped
        
        w = self.pathfinder.walls[wall_idx]
        d = [abs(snapped[0]-w.left), abs(snapped[0]-w.right), 
             abs(snapped[1]-w.top), abs(snapped[1]-w.bottom)]
        min_d = min(d)
        
        margin = 10.0
        if min_d == d[0]:
            snapped[0] = w.left - margin
        elif min_d == d[1]:
            snapped[0] = w.right + margin
        elif min_d == d[2]:
            snapped[1] = w.top - margin
        else:
            snapped[1] = w.bottom + margin
        return snapped

    def _get_carrot_point(self, pos, path, target_z, lookahead=100.0):
        """
        Find furthest waypoint we can reach with our full body (not just center point).
        """
        if not path or len(path) < 1:
            return None
        
        # Convert grid indices to world coordinates
        world_path = []
        gs = self.pathfinder.grid_size
        for idx in path:
            wx = idx[0] * gs + gs // 2
            wy = idx[1] * gs + gs // 2
            world_path.append(np.array([wx, wy, target_z]))
        
        # Find closest waypoint on path
        min_dist = float('inf')
        closest_idx = 0
        for i, wp in enumerate(world_path):
            d = np.linalg.norm(pos[:2] - wp[:2])
            if d < min_dist:
                min_dist = d
                closest_idx = i
        
        # Advance if close to current waypoint
        WAYPOINT_REACH_DIST = 30.0
        target_idx = closest_idx
        if min_dist < WAYPOINT_REACH_DIST and closest_idx < len(world_path) - 1:
            target_idx = closest_idx + 1
        
        # Look ahead - find furthest waypoint we can ACTUALLY reach with our body
        AGENT_RADIUS = 12.0
        walls = self.pathfinder.walls
        for i in range(target_idx, min(target_idx + 5, len(world_path))):
            if self._thick_line_of_sight(pos[:2], world_path[i][:2], AGENT_RADIUS, walls):
                target_idx = i
            else:
                break
        
        return world_path[target_idx].copy()
    
    def _thick_line_of_sight(self, start, end, radius, walls):
        """
        Check if a circle of given radius can travel from start to end without hitting walls.
        Uses multiple parallel rays to simulate the agent's body width.
        """
        start = np.array(start)
        end = np.array(end)
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 0.1:
            return True
        
        # Normalize direction
        direction = direction / length
        
        # Perpendicular vector
        perp = np.array([-direction[1], direction[0]])
        
        # Check center line plus offset lines at +/- radius
        offsets = [0, radius, -radius]  # Center and both edges
        
        for offset in offsets:
            line_start = start + perp * offset
            line_end = end + perp * offset
            
            for w in walls:
                if w.clipline(line_start, line_end):
                    return False
        
        return True
    
    def _has_line_of_sight(self, start_pos, end_pos):
        """Thick LOS using pathfinder walls."""
        if self.pathfinder is None:
            return True
        return self._thick_line_of_sight(
            np.array(start_pos[:2]), 
            np.array(end_pos[:2]), 
            radius=15.0,
            walls=self.pathfinder.walls
        )

    # Update signature to accept 'true_objective'
    def get_control_force(self, agent, command_pos, walls, missiles, true_objective=None):
        if self.pathfinder is None:
            self.pathfinder = Pathfinder(walls, 1200, grid_size=30)
        
        for k in self.last_forces: self.last_forces[k] = np.zeros(3)
        
        pos = agent.position
        vel = agent.velocity

        # === 0. TERMINAL PHASE OVERRIDE (The "Kamikaze" Reflex) ===
        # Check distance to the REAL GOAL (Green Dot), not the Command (Blue X)
        active_target = command_pos # Default to Blue X
        
        if true_objective is not None:
            dist_to_obj = np.linalg.norm(true_objective[:2] - pos[:2])
            
            # If we are close to the WIN condition, ignore Commander, ignore Walls.
            # Dive on the Objective.
            if dist_to_obj < 60.0: # Increased range to catch them earlier
                active_target = true_objective # Override target
                
                # FULL AGGRESSION LOGIC
                vec = active_target - pos
                dist = np.linalg.norm(vec)
                
                if dist > 0:
                    desired_vel = (vec / dist) * 800.0 # Max Overdrive
                    total_force = (desired_vel - vel) * 20.0 
                    
                    z_err = active_target[2] - pos[2]
                    total_force[2] = (z_err * 40.0) - (vel[2] * 5.0) 
                    
                    f_mag = np.linalg.norm(total_force)
                    if f_mag > 5000.0:
                        total_force = (total_force / f_mag) * 5000.0
                        
                    self.last_forces['attract'] = total_force
                    return total_force

        # === STANDARD NAVIGATION LOGIC BELOW ===
        # Use active_target (which is usually command_pos) for the rest
        target_pos = command_pos

        # Effective target altitude (force high if inside building footprint)
        eff_z = target_pos[2]
        if pygame.Rect(pos[0]-5, pos[1]-5, 10, 10).collidelist(walls) != -1:
            eff_z = max(eff_z, BUILDING_HEIGHT + 15.0)

        # === PATHFINDING ===
        current_goal = target_pos.copy()
        
        if pos[2] > BUILDING_HEIGHT:
            # Flying high - direct to target, no pathfinding needed
            self.cached_path = []
        else:
            # Low altitude - need to navigate around buildings
            valid_target = self._snap_to_valid(target_pos)
            dist_change = np.linalg.norm(valid_target[:2] - self.cached_target_pos[:2])
            self.path_recalc_timer -= 1
            
            # Recalculate path if target moved or timer expired
            needs_recalc = (dist_change > 20.0 or 
                          self.path_recalc_timer <= 0 or 
                          len(self.cached_path) == 0)
            
            if needs_recalc:
                raw_path = self.pathfinder.get_path(pos, valid_target)
                self.cached_path = raw_path if raw_path else []
                self.cached_target_pos = valid_target.copy()
                self.path_recalc_timer = 15
            
            # Get carrot point from path
            if self.cached_path:
                carrot = self._get_carrot_point(pos, self.cached_path, target_pos[2], lookahead=100.0)
                if carrot is not None:
                    current_goal = carrot
                else:
                    current_goal = valid_target
            else:
                # NO PATH FOUND - target is unreachable by ground
                eff_z = BUILDING_HEIGHT + 20.0  # Override to fly high
                current_goal = target_pos.copy()

        # === FORCES ===
        
        # 1. ATTRACTION to goal
        vec = current_goal - pos
        dist = np.linalg.norm(vec[:2])
        if dist > 0:
            desired_speed = min(dist * 5.0, 450.0)
            desired_vel = (vec[:2] / dist) * desired_speed
            self.last_forces['attract'][:2] = (desired_vel - vel[:2]) * self.K_ATTRACT

        # 2. WALL REPULSION (reduced when on valid path)
        if pos[2] < BUILDING_HEIGHT - 2.0:
            # Trust the path more when we have one
            repulsion_scale = 0.4 if self.cached_path else 1.0
            
            sense_rect = pygame.Rect(
                pos[0] - self.SENSE_RADIUS, 
                pos[1] - self.SENSE_RADIUS,
                self.SENSE_RADIUS * 2, 
                self.SENSE_RADIUS * 2
            )
            
            for w in walls:
                if not w.colliderect(sense_rect):
                    continue
                    
                closest_x = max(w.left, min(pos[0], w.right))
                closest_y = max(w.top, min(pos[1], w.bottom))
                vec_away = pos[:2] - np.array([closest_x, closest_y])
                dist_to_wall = max(0.1, np.linalg.norm(vec_away))
                
                if dist_to_wall < self.SENSE_RADIUS:
                    force_mag = self.K_REPULSE_WALL * (1.0/dist_to_wall - 1.0/self.SENSE_RADIUS)
                    self.last_forces['repulse'][:2] += (vec_away / dist_to_wall) * force_mag * repulsion_scale

        # 3. MAP EDGE REPULSION
        edge_margin = 50.0
        screen = 1200
        if pos[0] < edge_margin:
            self.last_forces['repulse'][0] += self.K_REPULSE_WALL * (1/max(1, pos[0]) - 1/edge_margin)
        elif pos[0] > screen - edge_margin:
            self.last_forces['repulse'][0] -= self.K_REPULSE_WALL * (1/max(1, screen-pos[0]) - 1/edge_margin)
        if pos[1] < edge_margin:
            self.last_forces['repulse'][1] += self.K_REPULSE_WALL * (1/max(1, pos[1]) - 1/edge_margin)
        elif pos[1] > screen - edge_margin:
            self.last_forces['repulse'][1] -= self.K_REPULSE_WALL * (1/max(1, screen-pos[1]) - 1/edge_margin)

        # 4. MISSILE AVOIDANCE
        for m in missiles:
            if not m.active:
                continue
            vec_to_agent = pos - m.position
            dist_to_missile = np.linalg.norm(vec_to_agent)
            
            if dist_to_missile < 250:
                closing_vel = m.velocity - vel
                if np.dot(closing_vel, vec_to_agent / (dist_to_missile + 0.01)) > 0:
                    perp = np.cross(vec_to_agent, [0, 0, 1])
                    perp_norm = np.linalg.norm(perp)
                    if perp_norm > 0.1:
                        perp /= perp_norm
                    if np.dot(perp, vel) < 0:
                        perp = -perp
                    
                    intensity = (250 - dist_to_missile) / 250
                    self.last_forces['avoid'] += (perp * 5000 * intensity) + (vec_to_agent / dist_to_missile * 1000 * intensity)

        # === COMBINE FORCES ===
        f_xy = (self.last_forces['attract'][:2] + 
                self.last_forces['repulse'][:2] + 
                self.last_forces['avoid'][:2])
        
        # Clamp total XY force
        f_xy_mag = np.linalg.norm(f_xy)
        if f_xy_mag > 2000:
            f_xy = (f_xy / f_xy_mag) * 2000
        
        total_force = np.zeros(3)
        total_force[:2] = f_xy

        # Z control (altitude)
        z_error = eff_z - pos[2]
        # Stronger P-gain for descent (30.0)
        z_force = (z_error * 30.0) - (vel[2] * 10.0)
        
        # FIX: Allow aggressive diving. 
        # -100 means we can apply negative thrust (push down) + gravity
        z_force = max(-100.0, z_force) 
        
        total_lift = np.clip(z_force + (15.0 * agent.mass) + self.last_forces['avoid'][2], -1000, 1500)
        total_force[2] = total_lift
        self.last_forces['lift'][2] = total_lift

        return total_force


class NavigatorDebug(Navigator):
    """Same as Navigator but exposes debug info."""
    pass
