import pygame
import math
from queue import PriorityQueue

def heuristic(p1, p2):
    """
    Calculates the Euclidean distance between two points.
    We use Euclidean because we allow 8-way (diagonal) movement.
    """
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def reconstruct_path(came_from, current, draw_func):
    """
    Backtracks from the end node to the start node to rebuild the path.
    """
    path = []
    while current in came_from:
        current = came_from[current]
        path.append(current)
        # Optional: visual feedback as we reconstruct
        current.make_path() 
        # We define make_path() in Node later
        if draw_func: draw_func() 
    # CHANGE: Return the path reversed (Start -> End)
    return path[::-1]

def a_star_algorithm(draw_func, grid, start, end):
    """
    The Core A* Algorithm.
    :param draw_func: A lambda function to update the screen (for visualization).
    :param grid: The Grid object.
    :param start: The starting Node.
    :param end: The goal Node.
    """
    count = 0
    open_set = PriorityQueue()
    
    # We store (f_score, count, node). 
    # 'count' acts as a tie-breaker if two nodes have the same f_score.
    open_set.put((0, count, start))
    
    came_from = {} # Keeps track of the path: came_from[child] = parent

    # g_score: The cost of the cheapest path from start to current
    # Initialize all to infinity
    g_score = {node: float("inf") for row in grid.grid for node in row}
    g_score[start] = 0

    # f_score: g_score + heuristic (estimated cost to goal)
    # Initialize all to infinity
    f_score = {node: float("inf") for row in grid.grid for node in row}
    f_score[start] = heuristic((start.row, start.col), (end.row, end.col))

    # To keep track of items in the PriorityQueue (since we can't check it directly)
    open_set_hash = {start}

    while not open_set.empty():
        # Allow user to quit even while calculating
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Get the node with the lowest F-score
        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            # WE FOUND THE PATH!
            path = reconstruct_path(came_from, end, draw_func)
            end.make_end() # Visual polish
            start.make_start() # Visual polish
            return path

        # Check all valid neighbors (including diagonals)
        for neighbor in current.neighbors:
            # 1 is the cost to move to a neighbor. 
            # NOTE: For diagonals, technically cost should be 1.41, but 1 works for simple grids.
            temp_g_score = g_score[current] + 1 

            if temp_g_score < g_score[neighbor]:
                # We found a better path to this neighbor!
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + heuristic((neighbor.row, neighbor.col), (end.row, end.col))
                
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open() # Visual: Mark as "considered"

        # Visual: Mark current node as "closed" (already checked)
        if current != start:
            current.make_closed()

        # Update the screen to show the algorithm working
        if draw_func: draw_func()

    return False # No path found