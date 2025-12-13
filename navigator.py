"""
navigator.py - Layer 2: Hybrid Pathfinding (A* + APF)
TUNED: Larger Grid (60px) + Aggressive Smoothing
"""

import numpy as np
import pygame
import heapq
from physics import BUILDING_HEIGHT, MAX_FORCE

class Pathfinder:
    def __init__(self, walls, screen_size, grid_size=60): # Grid 60 = Safer paths
        self.grid_size = grid_size
        self.width = int(screen_size // grid_size) + 1
        self.height = int(screen_size // grid_size) + 1
        self.walls = walls
        self.screen_size = screen_size
        self.grid = np.zeros((self.width, self.height), dtype=bool)
        self._bake_grid()

    def _bake_grid(self):
        for x in range(self.width):
            for y in range(self.height):
                cell_rect = pygame.Rect(x * self.grid_size, y * self.grid_size, 
                                        self.grid_size, self.grid_size)
                # Keep tolerance tight so we don't block valid paths
                test_rect = cell_rect.inflate(-10, -10)
                if test_rect.collidelist(self.walls) != -1:
                    self.grid[x, y] = True

    def _find_nearest_free_cell(self, start_idx):
        if not self.grid[start_idx[0], start_idx[1]]: return start_idx
        queue = [start_idx]; visited = {start_idx}
        attempts = 0
        while queue and attempts < 100:
            cx, cy = queue.pop(0); attempts += 1
            for nx, ny in [(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)]:
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if (nx, ny) in visited: continue
                    if not self.grid[nx, ny]: return (nx, ny)
                    visited.add((nx, ny)); queue.append((nx, ny))
        return start_idx

    def get_path(self, start_pos, target_pos):
        start_idx = (int(start_pos[0] // self.grid_size), int(start_pos[1] // self.grid_size))
        end_idx = (int(target_pos[0] // self.grid_size), int(target_pos[1] // self.grid_size))
        
        start_idx = (max(0, min(self.width-1, start_idx[0])), max(0, min(self.height-1, start_idx[1])))
        end_idx = (max(0, min(self.width-1, end_idx[0])), max(0, min(self.height-1, end_idx[1])))
        
        start_idx = self._find_nearest_free_cell(start_idx)
        end_idx = self._find_nearest_free_cell(end_idx)
        
        if self._has_line_of_sight(start_pos, target_pos):
            return [start_idx, end_idx]

        return self._astar(start_idx, end_idx)

    def _has_line_of_sight(self, start_pos, target_pos):
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
            for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid[nx, ny]: continue 
                    tentative_g = g_score[current] + 1.0 # Simple Manhattan weight
                    if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                        came_from[(nx, ny)] = current
                        g_score[(nx, ny)] = tentative_g
                        f = tentative_g + self._heuristic((nx, ny), end)
                        heapq.heappush(open_set, (f, (nx, ny)))
        return None

    def _heuristic(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1]) # Manhattan for grid

    def _reconstruct_path(self, came_from, current):
        path = [current]; 
        while current in came_from: current = came_from[current]; path.append(current)
        return path[::-1]


class Navigator:
    def __init__(self):
        self.K_ATTRACT = 8.0         # Increased pull
        self.K_REPULSE_WALL = 3000.0 # Reduced push (was 4000)
        self.K_AVOID_MISSILE = 4500.0
        self.SENSE_RADIUS = 80.0
        
        self.pathfinder = None
        self.last_forces = {k: np.zeros(3) for k in ['attract', 'repulse', 'avoid', 'lift']}
        
        self.cached_path = []
        self.cached_target_pos = np.array([-1.0, -1.0, -1.0])
        self.path_recalc_timer = 0
        self.current_waypoint_idx = 0

    def _snap_to_valid(self, pos):
        snapped_pos = pos.copy()
        snapped_pos[0] = np.clip(snapped_pos[0], 5, 1195) 
        snapped_pos[1] = np.clip(snapped_pos[1], 5, 1195)
        
        point_rect = pygame.Rect(snapped_pos[0]-2, snapped_pos[1]-2, 4, 4)
        wall_idx = point_rect.collidelist(self.pathfinder.walls)
        if wall_idx == -1: return snapped_pos
            
        w = self.pathfinder.walls[wall_idx]
        d_l = abs(snapped_pos[0] - w.left); d_r = abs(snapped_pos[0] - w.right)
        d_t = abs(snapped_pos[1] - w.top); d_b = abs(snapped_pos[1] - w.bottom)
        min_d = min(d_l, d_r, d_t, d_b)
        margin = 25.0 # Increased margin
        
        if min_d == d_l: snapped_pos[0] = w.left - margin
        elif min_d == d_r: snapped_pos[0] = w.right + margin
        elif min_d == d_t: snapped_pos[1] = w.top - margin
        else: snapped_pos[1] = w.bottom + margin
        return snapped_pos

    def get_control_force(self, agent, target_pos, walls, missiles):
        if self.pathfinder is None: self.pathfinder = Pathfinder(walls, 1200, grid_size=60)
        for k in self.last_forces: self.last_forces[k] = np.zeros(3)
        pos = agent.position; vel = agent.velocity

        # 0. FLOOR
        eff_z = target_pos[2]
        if pygame.Rect(pos[0]-5, pos[1]-5, 10, 10).collidelist(walls) != -1:
            eff_z = max(eff_z, BUILDING_HEIGHT + 15.0)

        # 1. WAYPOINT
        current_goal = target_pos 
        if pos[2] > BUILDING_HEIGHT:
            self.cached_path = []; current_goal = target_pos
        else:
            valid_target = self._snap_to_valid(target_pos)
            dist_change = np.linalg.norm(valid_target - self.cached_target_pos)
            self.path_recalc_timer -= 1
            
            if dist_change > 20.0 or self.path_recalc_timer <= 0 or not self.cached_path:
                raw_path_indices = self.pathfinder.get_path(pos, valid_target)
                if raw_path_indices:
                    self.cached_path = []
                    for idx in raw_path_indices:
                        wx = idx[0] * 60 + 30 # Center of 60px grid
                        wy = idx[1] * 60 + 30
                        self.cached_path.append(np.array([wx, wy, target_pos[2]]))
                    self.current_waypoint_idx = 0 
                self.cached_target_pos = valid_target.copy()
                self.path_recalc_timer = 20 
            
            if self.cached_path:
                if self.current_waypoint_idx >= len(self.cached_path): self.current_waypoint_idx = len(self.cached_path) - 1
                best_idx = self.current_waypoint_idx
                lookahead_limit = min(len(self.cached_path), self.current_waypoint_idx + 4)
                for i in range(len(self.cached_path) - 1, self.current_waypoint_idx, -1):
                    if i >= lookahead_limit: continue
                    wp = self.cached_path[i]
                    if self.pathfinder._has_line_of_sight(pos, wp): best_idx = i; break
                self.current_waypoint_idx = best_idx
                
                wp = self.cached_path[self.current_waypoint_idx]
                wp[2] = target_pos[2]
                
                # Loose waypoint acceptance (60px)
                if np.linalg.norm(pos[:2] - wp[:2]) < 60.0: 
                    self.current_waypoint_idx += 1
                    if self.current_waypoint_idx < len(self.cached_path):
                        current_goal = self.cached_path[self.current_waypoint_idx]
                        current_goal[2] = target_pos[2]
                    else: current_goal = valid_target
                else: current_goal = wp
            else: current_goal = valid_target

        # 2. XY FORCES
        total_force = np.zeros(3)
        vec_to_goal = current_goal - pos
        dist_xy = np.linalg.norm(vec_to_goal[:2])
        if dist_xy > 0:
            speed_req = min(dist_xy * 4.0, 450.0)
            f_xy = ((vec_to_goal[:2]/dist_xy) * speed_req - vel[:2]) * self.K_ATTRACT
            self.last_forces['attract'][:2] = f_xy

        if pos[2] < BUILDING_HEIGHT - 2.0:
            agent_rect = pygame.Rect(pos[0]-self.SENSE_RADIUS, pos[1]-self.SENSE_RADIUS, self.SENSE_RADIUS*2, self.SENSE_RADIUS*2)
            for w in [w for w in walls if w.colliderect(agent_rect)]:
                cx = max(w.left, min(pos[0], w.right)); cy = max(w.top, min(pos[1], w.bottom))
                vec_away = pos[:2] - np.array([cx, cy]); dist_wall = max(0.1, np.linalg.norm(vec_away))
                if dist_wall < self.SENSE_RADIUS:
                    self.last_forces['repulse'][:2] += (vec_away / dist_wall) * self.K_REPULSE_WALL * (1.0/dist_wall - 1.0/self.SENSE_RADIUS)

        screen_w = 1200; margin = 60.0
        if pos[0] < margin: self.last_forces['repulse'][0] += self.K_REPULSE_WALL * (1.0/max(1, pos[0]) - 1.0/margin)
        elif pos[0] > screen_w - margin: self.last_forces['repulse'][0] -= self.K_REPULSE_WALL * (1.0/max(1, screen_w - pos[0]) - 1.0/margin)
        if pos[1] < margin: self.last_forces['repulse'][1] += self.K_REPULSE_WALL * (1.0/max(1, pos[1]) - 1.0/margin)
        elif pos[1] > screen_w - margin: self.last_forces['repulse'][1] -= self.K_REPULSE_WALL * (1.0/max(1, screen_w - pos[1]) - 1.0/margin)

        for m in missiles:
            if not m.active: continue
            vec_m = pos - m.position; dist = np.linalg.norm(vec_m)
            if dist < 250:
                rel_v = m.velocity - vel
                if np.dot(rel_v, vec_m / (dist + 0.01)) > 0:
                    perp = np.cross(vec_m, np.array([0, 0, 1.0])); p_norm = np.linalg.norm(perp)
                    if p_norm > 0.1: perp /= p_norm
                    if np.dot(perp, vel) < 0: perp = -perp
                    intensity = (250 - dist) / 250
                    self.last_forces['avoid'] += (perp * 5000.0 * intensity) + ((vec_m/dist) * 1000.0 * intensity)

        force_xy = self.last_forces['attract'][:2] + self.last_forces['repulse'][:2] + self.last_forces['avoid'][:2]
        mag_xy = np.linalg.norm(force_xy)
        if mag_xy > 1800.0: force_xy = (force_xy / mag_xy) * 1800.0
        total_force[:2] = force_xy

        z_err = eff_z - pos[2]
        req_lift = max(-20.0, (z_err * 25.0) - (vel[2] * 10.0))
        total_lift = np.clip(req_lift + (15.0 * agent.mass) + self.last_forces['avoid'][2], -1000, 1500)
        total_force[2] = total_lift; self.last_forces['lift'][2] = total_lift

        return total_force

class NavigatorDebug(Navigator):
    pass