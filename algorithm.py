import pygame
import math
from queue import PriorityQueue

def heuristic(p1, p2):
    """
    Calculates the Euclidean distance between two points.
    """
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def reconstruct_path(came_from, current, draw_func):
    path = []
    while current in came_from:
        current = came_from[current]
        path.append(current)
        if draw_func: draw_func() 
    return path[::-1] # Return reversed path (Start -> End)

def a_star_algorithm(draw_func, grid, start, end):
    """
    The Core A* Algorithm.
    Cleaned up for Phase 3: No event polling, just calculation.
    """
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {} 

    g_score = {node: float("inf") for row in grid.grid for node in row}
    g_score[start] = 0

    f_score = {node: float("inf") for row in grid.grid for node in row}
    f_score[start] = heuristic((start.row, start.col), (end.row, end.col))

    open_set_hash = {start}

    while not open_set.empty():
        # --- CHANGE: REMOVED PYGAME EVENT LOOP HERE ---
        # We assume the external environment handles quitting.
        
        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            path = reconstruct_path(came_from, end, draw_func)
            end.make_end() 
            start.make_start() 
            return path

        for neighbor in current.neighbors:
            # New: Add the neighbor's weight (1.0 for open, 10.0 for near wall)
            temp_g_score = g_score[current] + neighbor.weight 
            # ----------------------

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + heuristic((neighbor.row, neighbor.col), (end.row, end.col))
                
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + heuristic((neighbor.row, neighbor.col), (end.row, end.col))
                
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open() 

        if current != start:
            current.make_closed()

        if draw_func: draw_func()

    return False