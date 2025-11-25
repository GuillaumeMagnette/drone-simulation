import pygame

class Node:
    def __init__(self, row, col, grid_size, total_rows):
        self.row = row
        self.col = col
        self.x = row * grid_size
        self.y = col * grid_size
        self.color = (255, 255, 255) # White (walkable)
        self.neighbors = []
        self.grid_size = grid_size
        self.total_rows = total_rows
        self.is_obstacle = False
        
        # A* properties
        self.g_cost = float('inf')
        self.h_cost = float('inf')
        self.f_cost = float('inf')
        self.parent = None
        
    def draw(self, screen):
        """Draws the node on the screen."""
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.grid_size, self.grid_size))

    def make_obstacle(self):
        """Sets this node to be an obstacle."""
        self.color = (0, 0, 0) # Black
        self.is_obstacle = True

    def reset(self):
        """Resets the node to be walkable."""
        self.color = (255, 255, 255) # White
        self.is_obstacle = False
        # Also reset A* properties for a new pathfind
        self.g_cost = float('inf')
        self.h_cost = float('inf')
        self.f_cost = float('inf')
        self.parent = None
    def update_neighbors(self, grid):
        """
        Checks all 4 surrounding nodes (Up, Down, Left, Right).
        (Diagonals removed to prevent corner clipping)
        """
        self.neighbors = []

        # DOWN
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_obstacle:
            self.neighbors.append(grid[self.row + 1][self.col])

        # UP
        if self.row > 0 and not grid[self.row - 1][self.col].is_obstacle:
            self.neighbors.append(grid[self.row - 1][self.col])

        # RIGHT
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_obstacle:
            self.neighbors.append(grid[self.row][self.col + 1])

        # LEFT
        if self.col > 0 and not grid[self.row][self.col - 1].is_obstacle:
            self.neighbors.append(grid[self.row][self.col - 1])

        # --- DIAGONALS (DISABLED) ---
        # We disable these so the agent doesn't try to squeeze partially through corners.
        # This forces the path to take wide, safe 90-degree turns.
        
        # if (self.row < self.total_rows - 1 and self.col < self.total_rows - 1 and 
        #     not grid[self.row + 1][self.col + 1].is_obstacle):
        #     self.neighbors.append(grid[self.row + 1][self.col + 1])

        # if (self.row < self.total_rows - 1 and self.col > 0 and 
        #     not grid[self.row + 1][self.col - 1].is_obstacle):
        #     self.neighbors.append(grid[self.row + 1][self.col - 1])

        # if (self.row > 0 and self.col < self.total_rows - 1 and 
        #     not grid[self.row - 1][self.col + 1].is_obstacle):
        #     self.neighbors.append(grid[self.row - 1][self.col + 1])

        # if (self.row > 0 and self.col > 0 and 
        #     not grid[self.row - 1][self.col - 1].is_obstacle):
        #     self.neighbors.append(grid[self.row - 1][self.col - 1])
    
        # Add this NEW method
    def reset_visuals(self):
        """Resets color/path status but KEEPS obstacles."""
        if not self.is_obstacle:
            self.color = (255, 255, 255) # White
            # Reset A* internal parents so new path can be found
            self.parent = None
            self.g_cost = float('inf')
            self.h_cost = float('inf')
            self.f_cost = float('inf')
            
    def make_start(self):
        self.color = (255, 165, 0) # Orange

    def make_end(self):
        self.color = (64, 224, 208) # Turquoise

    # UPDATE these methods to use non-confusing colors
    def make_closed(self):
        self.color = (255, 200, 200) # Pale Red (Light Pink)

    def make_open(self):
        self.color = (200, 255, 200) # Pale Green (Mint)
        
    def make_path(self):
        self.color = (0, 0, 255) # Blue Path (Distinct from Green Player