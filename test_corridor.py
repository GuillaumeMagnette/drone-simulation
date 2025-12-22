"""
test_corridor.py - Stress test for corridor navigation
Creates a maze with narrow corridors and verifies the navigator can path through them.
"""

import pygame
import numpy as np
from physics import Agent, BUILDING_HEIGHT
from navigator_v2 import Navigator, Pathfinder

SCREEN_SIZE = 800
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
clock = pygame.time.Clock()
font = pygame.font.Font(None, 24)

# Create a CHALLENGING corridor maze
# Corridors are 50-60px wide (tight but passable)
walls = [
    # Outer frame
    pygame.Rect(50, 50, 700, 30),    # Top
    pygame.Rect(50, 720, 700, 30),   # Bottom
    pygame.Rect(50, 50, 30, 700),    # Left
    pygame.Rect(720, 50, 30, 700),   # Right
    
    # Internal maze walls creating corridors
    pygame.Rect(150, 50, 30, 300),   # Vertical wall 1
    pygame.Rect(150, 450, 30, 300),  # Vertical wall 1 cont.
    
    pygame.Rect(280, 150, 30, 400),  # Vertical wall 2
    pygame.Rect(280, 650, 30, 100),  # Vertical wall 2 cont.
    
    pygame.Rect(400, 50, 30, 250),   # Vertical wall 3
    pygame.Rect(400, 400, 30, 350),  # Vertical wall 3 cont.
    
    pygame.Rect(530, 200, 30, 400),  # Vertical wall 4
    
    # Horizontal connectors
    pygame.Rect(150, 350, 200, 30),  # Horizontal 1
    pygame.Rect(400, 500, 200, 30),  # Horizontal 2
    pygame.Rect(530, 150, 190, 30),  # Horizontal 3
]

# Corridor width check
print("=== CORRIDOR ANALYSIS ===")
print(f"Wall between 150-180 and 280-310: gap = {280-180} = 100px")
print(f"Wall between 280-310 and 400-430: gap = {400-310} = 90px")  
print(f"Wall between 400-430 and 530-560: gap = {530-430} = 100px")
print("These should be navigable.\n")

# Create entities
agent = Agent(0, 100, 400)  # Start on left side
nav = Navigator()
target_pos = np.array([650.0, 400.0, 10.0])  # Target on right side

# Bake pathfinder and show grid analysis
pathfinder = Pathfinder(walls, SCREEN_SIZE, grid_size=30)
blocked_count = np.sum(pathfinder.cost_grid == float('inf'))
high_cost_count = np.sum(pathfinder.cost_grid > 1)
total_cells = pathfinder.width * pathfinder.height
print(f"Grid: {pathfinder.width}x{pathfinder.height} = {total_cells} cells")
print(f"Blocked (inside walls): {blocked_count}")
print(f"Elevated cost (near walls): {high_cost_count}")
print(f"Free cells: {total_cells - blocked_count}")

# Test pathfinding
test_path = pathfinder.get_path(agent.position, target_pos)
if test_path:
    print(f"\n✓ A* found path with {len(test_path)} waypoints")
else:
    print("\n✗ A* FAILED to find path!")

running = True
show_grid = True
show_costs = False

while running:
    dt = 0.016
    
    # Input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_g:
                show_grid = not show_grid
            if event.key == pygame.K_c:
                show_costs = not show_costs
            if event.key == pygame.K_r:
                agent.reset(100, 400)
            if event.key == pygame.K_SPACE:
                # Toggle target altitude
                target_pos[2] = 80.0 if target_pos[2] < 50 else 10.0
    
    # Mouse sets target
    if pygame.mouse.get_pressed()[0]:
        mx, my = pygame.mouse.get_pos()
        target_pos[0] = mx
        target_pos[1] = my
    
    # Navigation
    force = nav.get_control_force(agent, target_pos, walls, [])
    agent.update(dt, force, walls, SCREEN_SIZE)
    
    # === RENDER ===
    screen.fill((20, 20, 30))
    
    # Draw cost grid
    if show_grid:
        for x in range(pathfinder.width):
            for y in range(pathfinder.height):
                cost = pathfinder.cost_grid[x, y]
                rx = x * pathfinder.grid_size
                ry = y * pathfinder.grid_size
                
                if cost == float('inf'):
                    color = (80, 20, 20)  # Blocked - dark red
                elif show_costs:
                    # Show cost gradient
                    if cost >= 8:
                        color = (60, 30, 30)  # High cost
                    elif cost >= 4:
                        color = (50, 40, 30)  # Medium cost
                    elif cost >= 2:
                        color = (40, 45, 35)  # Low cost
                    else:
                        color = (30, 50, 30)  # Free - greenish
                else:
                    color = (30, 35, 30)  # Free
                
                pygame.draw.rect(screen, color, 
                               (rx, ry, pathfinder.grid_size-1, pathfinder.grid_size-1))
    
    # Draw walls
    for w in walls:
        pygame.draw.rect(screen, (100, 100, 110), w)
        pygame.draw.rect(screen, (140, 140, 150), w, 2)
    
    # Draw A* path
    if nav.cached_path:
        gs = pathfinder.grid_size
        path_points = [(idx[0] * gs + gs//2, idx[1] * gs + gs//2) for idx in nav.cached_path]
        if len(path_points) > 1:
            pygame.draw.lines(screen, (255, 255, 0), False, path_points, 3)
        for px, py in path_points:
            pygame.draw.circle(screen, (255, 200, 0), (px, py), 5)
        
        # Draw carrot point (the actual goal the agent is chasing)
        carrot = nav._get_carrot_point(agent.position, nav.cached_path, target_pos[2])
        if carrot is not None:
            cx, cy = int(carrot[0]), int(carrot[1])
            pygame.draw.circle(screen, (0, 255, 255), (cx, cy), 8, 3)  # Cyan ring
            # Line from agent to carrot
            pygame.draw.line(screen, (0, 200, 200), 
                           (int(agent.position[0]), int(agent.position[1])), 
                           (cx, cy), 1)
    else:
        # NO PATH - draw direct line to target (will fly over)
        pygame.draw.line(screen, (255, 100, 100), 
                        (int(agent.position[0]), int(agent.position[1])),
                        (int(target_pos[0]), int(target_pos[1])), 2)
        # Show "FLY HIGH" indicator
        fly_text = font.render("FLY HIGH (no ground path)", True, (255, 150, 150))
        screen.blit(fly_text, (int(agent.position[0]) + 15, int(agent.position[1]) - 20))
    
    # Draw target
    color_t = (0, 255, 0) if target_pos[2] < 50 else (0, 255, 255)
    pygame.draw.circle(screen, color_t, (int(target_pos[0]), int(target_pos[1])), 12)
    pygame.draw.circle(screen, (255, 255, 255), (int(target_pos[0]), int(target_pos[1])), 12, 2)
    
    # Draw agent
    if agent.active:
        ax, ay = int(agent.position[0]), int(agent.position[1])
        color_a = (200, 255, 255) if agent.position[2] > BUILDING_HEIGHT else (255, 180, 50)
        pygame.draw.circle(screen, color_a, (ax, ay), 10)
        pygame.draw.circle(screen, (255, 255, 255), (ax, ay), 10, 2)
        
        # Velocity vector
        vx, vy = agent.velocity[0] * 0.1, agent.velocity[1] * 0.1
        pygame.draw.line(screen, (255, 100, 100), (ax, ay), (ax + vx, ay + vy), 2)
        
        # Force vectors (debug)
        scale = 0.02
        colors = {'attract': (0, 255, 0), 'repulse': (255, 0, 0), 'avoid': (255, 0, 255)}
        for name, color in colors.items():
            f = nav.last_forces[name]
            if np.linalg.norm(f) > 10:
                pygame.draw.line(screen, color, (ax, ay), 
                               (ax + f[0]*scale, ay + f[1]*scale), 2)
    
    # HUD
    path_status = f"Path: {len(nav.cached_path)} nodes" if nav.cached_path else "NO PATH - Flying High"
    texts = [
        f"Corridor Test | Alt: {agent.position[2]:.0f} | Vel: {np.linalg.norm(agent.velocity):.0f}",
        f"{path_status} | G=grid C=costs R=reset SPACE=altitude",
        f"Click to set target | Green=attract Red=repulse Cyan=carrot"
    ]
    for i, t in enumerate(texts):
        surf = font.render(t, True, (200, 200, 200))
        screen.blit(surf, (10, 10 + i * 20))
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
print("Test complete.")
