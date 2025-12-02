import pygame
from node import Node

class Grid:
    def __init__(self, rows, width):
        self.rows = rows
        self.width = width
        self.grid_size = width // rows
        self.grid = self._create_grid()

    def _create_grid(self):
        """Creates the 2D list of Node objects."""
        grid = []
        for i in range(self.rows):
            grid.append([])
            for j in range(self.rows):
                node = Node(i, j, self.grid_size, self.rows)
                grid[i].append(node)
        return grid
    def apply_costmap(self):
        """
        Creates a 'Safety Cushion' around obstacles.
        Standard Nodes: Weight 1.0
        Danger Zones: Weight 10.0 (A* will avoid unless necessary)
        """
        # 1. Reset all weights to 1.0
        for row in self.grid:
            for node in row:
                node.weight = 1.0

        # 2. Find all walls
        walls = []
        for x in range(self.rows):
            for y in range(self.rows):
                if self.grid[x][y].is_obstacle:
                    walls.append((x, y))

        # 3. Inflate costs around walls
        # Padding = 2 cells (approx 20 pixels safety margin)
        padding = 2 
        
        for wx, wy in walls:
            for dx in range(-padding, padding + 1):
                for dy in range(-padding, padding + 1):
                    nx, ny = wx + dx, wy + dy
                    
                    # Bounds check
                    if 0 <= nx < self.rows and 0 <= ny < self.rows:
                        node = self.grid[nx][ny]
                        
                        # Only affect empty space, don't change walls
                        if not node.is_obstacle:
                            # Set high cost so A* treats it like "mud" or "danger"
                            node.weight = 10.0 
                            # Optional: Visual Debug
                            # if node.color == (255, 255, 255): node.color = (240, 240, 240)

    def draw_grid_lines(self, screen):
        """Draws the grey lines for the grid visualization."""
        for i in range(self.rows):
            # Draw horizontal line
            pygame.draw.line(screen, (128, 128, 128), (0, i * self.grid_size), (self.width, i * self.grid_size))
            for j in range(self.rows):
                # Draw vertical line
                pygame.draw.line(screen, (128, 128, 128), (j * self.grid_size, 0), (j * self.grid_size, self.width))

    def draw(self, screen):
        """Draws all the nodes and the grid lines."""
        # First, draw all the nodes
        for row in self.grid:
            for node in row:
                node.draw(screen)
        
        # Then, draw the grid lines on top
        self.draw_grid_lines(screen)
        
    def get_node_from_pos(self, pos):
            """Gets the grid row and col from a pixel position."""
            x, y = pos
            
            # FIX: Explicitly cast to int(). 
            # Even if x is 400.0, we need the integer 400.
            row = int(x // self.grid_size)
            col = int(y // self.grid_size)
            
            if 0 <= row < self.rows and 0 <= col < self.rows:
                return self.grid[row][col]
            return None
    def get_obstacle_rects(self):
        """Returns a list of Pygame Rects for every black node."""
        obstacles = []
        for row in self.grid:
            for node in row:
                if node.is_obstacle:
                    # Create a rect for this wall node
                    rect = pygame.Rect(node.x, node.y, self.grid_size, self.grid_size)
                    obstacles.append(rect)
        return obstacles