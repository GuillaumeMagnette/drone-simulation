"""
navigator.py - Layer 2: Hybrid Pathfinding (A* + APF)
STRATEGY CHANGE: Wall Inflation.
The A* Grid treats walls as 'Fatter' than they are.
This forces the path to stay in the center of the street, avoiding physics conflicts.
"""

import numpy as np
import pygame
import heapq
from physics import BUILDING_HEIGHT, MAX_FORCE

class Pathfinder:
    def __init__(self, walls, screen_size, grid_size=40):
        self.grid_size = grid_size
        self.width = int(screen_size // grid_size) + 1
        self.height = int(screen_size // grid_size) + 1
        self.walls = walls
        self.screen_size = screen_size
        self.grid = np.zeros((self.width, self.height), dtype=bool)
        self._bake_grid()

    def _bake_grid(self):
        """
        Mark cells as blocked.
        CRITICAL: We inflate the check to create a 'Safety Buffer' around walls.
        """
        for x in range(self.width):
            for y in range(self.height):
                cell_rect = pygame.Rect(x * self.grid_size, y * self.grid_size, 
                                        self.grid_size, self.grid_size)
                
                # INFLATION: Check a larger area than the cell itself.
                # If a wall is anywhere near this cell, mark it blocked.
                # This keeps the A* path away from the physical wall edges.
                safety_buffer = 20
                test_rect = cell_rect.inflate(safety_buffer, safety_buffer)
                
                if test_rect.collidelist(self.walls) != -1:
                    self.grid[x, y] = True

    def _find_nearest_free_cell(self, start_idx):
        if not self.grid[start_idx[0], start_idx[1]]: return start_idx
        queue = [start_idx]; visited = {start_idx}
        attempts = 0
        while queue and attempts < 500: # Increased search depth
            cx, cy = queue.pop(0); attempts += 1
            # Check 8 neighbors
            for nx, ny in [(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1),
                           (cx+1, cy+1), (cx-1, cy-1), (cx+1, cy-1), (cx-1, cy+1)]:
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if (nx, ny) in visited: continue
                    if not self.grid[nx, ny]: return (nx, ny)
                    visited.add((nx, ny)); queue.append((nx, ny))
        return start_idx

    def get_path(self, start_pos, target_pos):
        start_idx = (int(start_pos[0] // self.grid_size), int(start_pos[1] // self.grid_size))
        end_idx = (int(target_pos[0] // self.grid_size), int(target_pos[1] // self.grid_size))
        
        # Clamp
        start_idx = (max(0, min(self.width-1, start_idx[0])), max(0, min(self.height-1, start_idx[1])))
        end_idx = (max(0, min(self.width-1, end_idx[0])), max(0, min(self.height-1, end_idx[1])))
        
        # Validate Start/End
        start_idx = self._find_nearest_free_cell(start_idx)
        end_idx = self._find_nearest_free_cell(end_idx)
        
        # Line of Sight Check (Physics-based)
        if self._has_line_of_sight(start_pos, target_pos):
            return [start_idx, end_idx]

        return self._astar(start_idx, end_idx)

    def _has_line_of_sight(self, start_pos, target_pos):
        # Raycast against ACTUAL walls (not inflated grid)
        for w in self.walls:
            if w.clipline(start_pos[:2], target_pos[:2]): return False
        return True

    def _astar(self, start, end):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}; g_score = {start: 0}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == end: return self._reconstruct_path(came_from, current)
            x, y = current
            # 8-Directional movement
            for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1),
                           (x+1, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1)]:
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid[nx, ny]: continue 
                    dist = 1.414 if (nx!=x and ny!=y) else 1.0
                    tentative_g = g_score[current] + dist
                    if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                        came_from[(nx, ny)] = current
                        g_score[(nx, ny)] = tentative_g
                        f = tentative_g + self._heuristic((nx, ny), end)
                        heapq.heappush(open_set, (f, (nx, ny)))
        return None

    def _heuristic(self, a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def _reconstruct_path(self, came_from, current):
        path = [current]; 
        while current in came_from: current = came_from[current]; path.append(current)
        return path[::-1]


class Navigator:
    def __init__(self):
        self.K_ATTRACT = 8.0         
        self.K_REPULSE_WALL = 3000.0 
        self.K_AVOID_MISSILE = 4500.0
        self.SENSE_RADIUS = 80.0
        
        self.pathfinder = None
        self.last_forces = {k: np.zeros(3) for k in ['attract', 'repulse', 'avoid', 'lift']}
        
        self.cached_path = []
        self.cached_target_pos = np.array([-1.0, -1.0, -1.0])
        self.path_recalc_timer = 0
        self.current_waypoint_idx = 0

    def _snap_to_valid(self, pos):
        """If inside wall, push out to nearest edge."""
        snapped = pos.copy()
        snapped[0] = np.clip(snapped[0], 5, 1195)
        snapped[1] = np.clip(snapped[1], 5, 1195)
        
        rect = pygame.Rect(snapped[0]-2, snapped[1]-2, 4, 4)
        wall_idx = rect.collidelist(self.pathfinder.walls)
        if wall_idx == -1: return snapped
        
        w = self.pathfinder.walls[wall_idx]
        d = [abs(snapped[0]-w.left), abs(snapped[0]-w.right), 
             abs(snapped[1]-w.top), abs(snapped[1]-w.bottom)]
        min_d = min(d)
        
        margin = 10.0
        if min_d == d[0]: snapped[0] = w.left - margin
        elif min_d == d[1]: snapped[0] = w.right + margin
        elif min_d == d[2]: snapped[1] = w.top - margin
        else: snapped[1] = w.bottom + margin
        return snapped

    def get_control_force(self, agent, target_pos, walls, missiles):
        if self.pathfinder is None: self.pathfinder = Pathfinder(walls, 1200, grid_size=40)
        for k in self.last_forces: self.last_forces[k] = np.zeros(3)
        pos = agent.position; vel = agent.velocity

        # 0. FLOOR
        eff_z = target_pos[2]
        if pygame.Rect(pos[0]-5, pos[1]-5, 10, 10).collidelist(walls) != -1:
            eff_z = max(eff_z, BUILDING_HEIGHT + 15.0)

        # 1. PATHING
        current_goal = target_pos
        if pos[2] > BUILDING_HEIGHT:
            self.cached_path = []; current_goal = target_pos
        else:
            valid_target = self._snap_to_valid(target_pos)
            dist_change = np.linalg.norm(valid_target - self.cached_target_pos)
            self.path_recalc_timer -= 1
            
            if dist_change > 15.0 or self.path_recalc_timer <= 0 or not self.cached_path:
                raw_path = self.pathfinder.get_path(pos, valid_target)
                if raw_path:
                    self.cached_path = []
                    for idx in raw_path:
                        wx = idx[0]*40 + 20; wy = idx[1]*40 + 20
                        self.cached_path.append(np.array([wx, wy, target_pos[2]]))
                    self.current_waypoint_idx = 0
                self.cached_target_pos = valid_target.copy()
                self.path_recalc_timer = 20

            if self.cached_path:
                # Simple "Rabbit" following - chase the node
                if self.current_waypoint_idx >= len(self.cached_path):
                    self.current_waypoint_idx = len(self.cached_path)-1
                
                # Advance logic
                wp = self.cached_path[self.current_waypoint_idx]
                wp[2] = target_pos[2]
                
                dist = np.linalg.norm(pos[:2] - wp[:2])
                if dist < 40.0:
                    self.current_waypoint_idx += 1
                    if self.current_waypoint_idx < len(self.cached_path):
                        current_goal = self.cached_path[self.current_waypoint_idx]
                        current_goal[2] = target_pos[2]
                    else:
                        current_goal = valid_target
                else:
                    current_goal = wp
            else:
                current_goal = valid_target

        # 2. XY FORCES
        total_force = np.zeros(3)
        vec = current_goal - pos
        dist = np.linalg.norm(vec[:2])
        if dist > 0:
            speed = min(dist * 4.0, 450.0)
            self.last_forces['attract'][:2] = ((vec[:2]/dist)*speed - vel[:2]) * self.K_ATTRACT

        # 3. REPULSION (Walls)
        if pos[2] < BUILDING_HEIGHT - 2.0:
            agent_rect = pygame.Rect(pos[0]-self.SENSE_RADIUS, pos[1]-self.SENSE_RADIUS, self.SENSE_RADIUS*2, self.SENSE_RADIUS*2)
            for w in [w for w in walls if w.colliderect(agent_rect)]:
                cx = max(w.left, min(pos[0], w.right)); cy = max(w.top, min(pos[1], w.bottom))
                vec_away = pos[:2] - np.array([cx, cy]); d_w = max(0.1, np.linalg.norm(vec_away))
                if d_w < self.SENSE_RADIUS:
                    self.last_forces['repulse'][:2] += (vec_away/d_w) * self.K_REPULSE_WALL * (1.0/d_w - 1.0/self.SENSE_RADIUS)

        # 4. REPULSION (Map Edge)
        w = 1200; m = 60.0
        if pos[0] < m: self.last_forces['repulse'][0] += self.K_REPULSE_WALL * (1/max(1, pos[0]) - 1/m)
        elif pos[0] > w-m: self.last_forces['repulse'][0] -= self.K_REPULSE_WALL * (1/max(1, w-pos[0]) - 1/m)
        if pos[1] < m: self.last_forces['repulse'][1] += self.K_REPULSE_WALL * (1/max(1, pos[1]) - 1/m)
        elif pos[1] > w-m: self.last_forces['repulse'][1] -= self.K_REPULSE_WALL * (1/max(1, w-pos[1]) - 1/m)

        # 5. MISSILE
        for m in missiles:
            if not m.active: continue
            v_m = pos - m.position; d_m = np.linalg.norm(v_m)
            if d_m < 250:
                if np.dot(m.velocity-vel, v_m/(d_m+0.01)) > 0:
                    perp = np.cross(v_m, [0,0,1]); p_n = np.linalg.norm(perp)
                    if p_n > 0.1: perp /= p_n
                    if np.dot(perp, vel) < 0: perp = -perp
                    intense = (250-d_m)/250
                    self.last_forces['avoid'] += (perp*5000*intense) + ((v_m/d_m)*1000*intense)

        # 6. COMBINE
        f_xy = self.last_forces['attract'][:2] + self.last_forces['repulse'][:2] + self.last_forces['avoid'][:2]
        if np.linalg.norm(f_xy) > 1800: f_xy = (f_xy/np.linalg.norm(f_xy))*1800
        total_force[:2] = f_xy

        z_err = eff_z - pos[2]
        req_lift = max(-20.0, (z_err * 25.0) - (vel[2] * 10.0))
        total_lift = np.clip(req_lift + (15.0 * agent.mass) + self.last_forces['avoid'][2], -1000, 1500)
        total_force[2] = total_lift; self.last_forces['lift'][2] = total_lift

        return total_force

class NavigatorDebug(Navigator): pass