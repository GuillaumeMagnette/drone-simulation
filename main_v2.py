import pygame
import numpy as np
from agent import Agent
from grid import Grid
from algorithm import a_star_algorithm

pygame.init()
debug_font = pygame.font.SysFont("Arial", 16)

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("A* Smart Agent Sandbox")

# --- SETUP GRID & AGENT ---
GRID_SIZE = 20 
grid = Grid(GRID_SIZE, SCREEN_WIDTH)

agent_size = SCREEN_WIDTH // GRID_SIZE
player = Agent(0, 0, agent_size) 

clock = pygame.time.Clock()

# --- SETUP DUMMY ENEMY ---
# A red square for the agent to chase automatically
dummy_enemy = pygame.Rect(600, 600, 50, 50)
dummy_enemies = [dummy_enemy] 

# Variables for manual A* testing (Right Click + Spacebar)
start_node = None
end_node = None

running = True
while running:
    dt = clock.tick(60) / 1000.0
    
    # Helper for A* visualization
    def draw_wrapper():
        grid.draw(screen)
        player.draw(screen)
        pygame.display.flip()

    # --- EVENT HANDLING ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        # --- MOUSE INPUT (Obstacles & Goal) ---
        if pygame.mouse.get_pressed()[0]: # Left Click: Obstacles
            pos = pygame.mouse.get_pos()
            node = grid.get_node_from_pos(pos)
            if node: node.make_obstacle()

        elif pygame.mouse.get_pressed()[2]: # Right Click: Manual Goal
            pos = pygame.mouse.get_pos()
            node = grid.get_node_from_pos(pos)
            if node:
                end_node = node
                end_node.make_end()
                
        # --- KEYBOARD INPUT (Single Press) ---
        if event.type == pygame.KEYDOWN:
            
            # [E] Switch Modes (Manual <-> Chase)
            if event.key == pygame.K_e:
                player.switch_mode()
                print(f"Switched Mode to: {player.state}")

            # [C] Clear Grid
            if event.key == pygame.K_c: 
                start_node = None
                end_node = None
                grid = Grid(GRID_SIZE, SCREEN_WIDTH)

            # [SPACE] Manual A* Visualization (Optional Debugging)
            if event.key == pygame.K_SPACE:
                start_pos = (player.position[0] + agent_size//2, player.position[1] + agent_size//2)
                start_node = grid.get_node_from_pos(start_pos)
                start_node.make_start()

                if start_node and end_node:
                    for row in grid.grid:
                        for node in row:
                            node.update_neighbors(grid.grid)
                    
                    print("Starting Visual A*...")
                    path = a_star_algorithm(draw_wrapper, grid, start_node, end_node)
                    
                    # If we found a path manually, we can tell the agent to follow it
                    if path: 
                        player.set_path(path)

    # --- CONTINUOUS INPUT (WASD) ---
    # This was missing! We need this for Manual Mode.
    keys = pygame.key.get_pressed()
    player_direction_vector = np.array([0.0, 0.0])
    
    if keys[pygame.K_w]: player_direction_vector[1] -= 1
    if keys[pygame.K_s]: player_direction_vector[1] += 1
    if keys[pygame.K_a]: player_direction_vector[0] -= 1
    if keys[pygame.K_d]: player_direction_vector[0] += 1

    # 1. Get all wall rectangles from the grid
    walls = grid.get_obstacle_rects()
    
    # 2. Combine walls with the dummy enemy so we collide with BOTH
    # (The '+' operator joins two lists together)
    all_collidables = walls + dummy_enemies
    
    # 3. Pass the combined list to the agent
    player.update(dt, player_direction_vector, all_collidables, dummy_enemies, grid)

    # --- DRAWING ---
    screen.fill((255, 255, 255))
    
    grid.draw(screen)
    
    # Draw the dummy enemy (Red Square)
    pygame.draw.rect(screen, (255, 0, 0), dummy_enemy)
    
    player.draw(screen)
    pygame.display.flip()

pygame.quit()