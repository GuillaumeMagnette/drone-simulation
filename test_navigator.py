"""
test_navigator.py - Visual Test for the APF Navigator

Controls:
- Left Click: Set target position (Green dot)
- Right Click (hold): Set high altitude target (fly over walls)
- Spacebar: Spawn a homing missile from corner
- R: Reset agent position
- D: Toggle debug force visualization

Verification Tests:
1. Wall Sliding: Move target behind a wall. Agent should curve around smoothly.
2. 2.5D Flight: Hold Right Click. Agent should fly over walls (turns cyan).
3. Evasion: Press Space. Agent should dodge missiles while pursuing target.
"""

import pygame
import numpy as np
from physics import Agent, Interceptor, BUILDING_HEIGHT
from navigator import Navigator, NavigatorDebug

# Setup
SCREEN_SIZE = 800
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
clock = pygame.time.Clock()
font = pygame.font.Font(None, 24)

# Create World - Urban environment with buildings
walls = [
    pygame.Rect(300, 300, 200, 200),  # Central block
    pygame.Rect(100, 100, 50, 400),   # Long wall
    pygame.Rect(600, 200, 100, 100),  # Small block
    pygame.Rect(500, 500, 150, 80),   # Wide block
    pygame.Rect(200, 600, 80, 120),   # Tall block
]

# Create Entities
agent = Agent(0, 100, 700)
nav = Navigator()
nav_debug = NavigatorDebug()
missiles = []

# Target (Mouse controls this)
target_pos = np.array([400.0, 400.0, 10.0])

# State
debug_mode = False
running = True

while running:
    dt = 0.016  # 60 FPS
    
    # Input
    mx, my = pygame.mouse.get_pos()
    keys = pygame.key.get_pressed()
    
    # Left Click: Set Target XY
    if pygame.mouse.get_pressed()[0]:
        target_pos[0] = mx
        target_pos[1] = my
        
    # Right Click: Toggle Altitude (High/Low)
    if pygame.mouse.get_pressed()[2]:
        target_pos[2] = 80.0
    else:
        target_pos[2] = 10.0

    # Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Spawn missile from random corner
                corner = np.random.choice([0, 1, 2, 3])
                spawn_pos = [
                    (0, 0),
                    (SCREEN_SIZE, 0),
                    (0, SCREEN_SIZE),
                    (SCREEN_SIZE, SCREEN_SIZE)
                ][corner]
                missiles.append(Interceptor(spawn_pos[0], spawn_pos[1], 10, agent))
            elif event.key == pygame.K_r:
                # Reset
                agent.reset(100, 700)
                missiles.clear()
            elif event.key == pygame.K_d:
                # Toggle debug
                debug_mode = not debug_mode

    # --- LOGIC LOOP ---
    
    # 1. Navigator calculates Desired Force
    if debug_mode:
        force = nav_debug.get_control_force(agent, target_pos, walls, missiles)
    else:
        force = nav.get_control_force(agent, target_pos, walls, missiles)
    
    # 2. Physics applies Force
    agent.update(dt, force, walls, SCREEN_SIZE)
    
    # 3. Update Missiles
    for m in missiles:
        m.update(dt, walls)  # <--- PASS WALLS HERE
        if m.check_hit(agent):
            print("HIT! Agent destroyed.")
            agent.active = False
    
    # Clean up dead missiles
    missiles = [m for m in missiles if m.active]
    
    # --- RENDER ---
    screen.fill((30, 30, 30))
    
    # Draw grid for reference
    for i in range(0, SCREEN_SIZE, 100):
        pygame.draw.line(screen, (50, 50, 50), (i, 0), (i, SCREEN_SIZE))
        pygame.draw.line(screen, (50, 50, 50), (0, i), (SCREEN_SIZE, i))
    
    # Draw Walls
    for w in walls:
        # Darker if agent is above building height
        if agent.position[2] > BUILDING_HEIGHT:
            color = (60, 60, 60)
        else:
            color = (100, 100, 100)
        pygame.draw.rect(screen, color, w)
        pygame.draw.rect(screen, (150, 150, 150), w, 2)
        
    # Draw Target
    color_target = (0, 255, 0) if target_pos[2] < 50 else (0, 255, 255)
    pygame.draw.circle(screen, color_target, (int(target_pos[0]), int(target_pos[1])), 10)
    pygame.draw.circle(screen, (255, 255, 255), (int(target_pos[0]), int(target_pos[1])), 10, 2)
    
    # Draw Agent
    if agent.active:
        ax, ay = int(agent.position[0]), int(agent.position[1])
        
        # Color based on altitude
        if agent.position[2] > BUILDING_HEIGHT:
            color_agent = (200, 255, 255)  # Cyan = high
        else:
            color_agent = (255, 200, 50)   # Orange = low
            
        pygame.draw.circle(screen, color_agent, (ax, ay), 12)
        pygame.draw.circle(screen, (255, 255, 255), (ax, ay), 12, 2)
        
        # Draw Force Vector (Debug)
        if debug_mode and hasattr(nav_debug, 'last_forces'):
            # Individual force components
            forces = nav_debug.last_forces
            colors = {
                'attract': (0, 255, 0),    # Green
                'repulse': (255, 0, 0),    # Red
                'slide': (255, 255, 0),    # Yellow
                'avoid': (255, 0, 255),    # Magenta
            }
            scale = 0.03
            for name, f in forces.items():
                if name in colors and np.linalg.norm(f) > 1:
                    end_pos = (ax + f[0] * scale, ay + f[1] * scale)
                    pygame.draw.line(screen, colors[name], (ax, ay), end_pos, 2)
        else:
            # Just total force
            end_pos = (ax + force[0] * 0.05, ay + force[1] * 0.05)
            pygame.draw.line(screen, (255, 0, 0), (ax, ay), end_pos, 2)
    else:
        # Agent destroyed marker
        ax, ay = int(agent.position[0]), int(agent.position[1])
        pygame.draw.line(screen, (255, 0, 0), (ax - 15, ay - 15), (ax + 15, ay + 15), 3)
        pygame.draw.line(screen, (255, 0, 0), (ax - 15, ay + 15), (ax + 15, ay - 15), 3)
        
    # Draw Missiles
    for m in missiles:
        mx_pos, my_pos = int(m.position[0]), int(m.position[1])
        pygame.draw.circle(screen, (255, 50, 50), (mx_pos, my_pos), 5)
        # Trail
        trail_end = m.position - m.velocity * 0.05
        pygame.draw.line(screen, (255, 100, 100), 
                        (mx_pos, my_pos), 
                        (int(trail_end[0]), int(trail_end[1])), 2)
    
    # HUD
    alt_str = f"Alt: {agent.position[2]:.1f}"
    vel_str = f"Vel: {np.linalg.norm(agent.velocity):.1f}"
    missile_str = f"Missiles: {len(missiles)}"
    debug_str = "DEBUG ON" if debug_mode else ""
    
    status_color = (200, 255, 200) if agent.active else (255, 100, 100)
    
    texts = [
        (f"Navigator Test | {alt_str} | {vel_str} | {missile_str}", (10, 10)),
        ("Controls: LClick=Target, RClick=High Alt, Space=Missile, R=Reset, D=Debug", (10, 30)),
        (debug_str, (10, 50)),
    ]
    
    for text, pos in texts:
        if text:
            surf = font.render(text, True, status_color)
            screen.blit(surf, pos)
    
    # Debug legend
    if debug_mode:
        legend = [
            ("Green = Attract", (0, 255, 0)),
            ("Red = Repulse", (255, 0, 0)),
            ("Yellow = Slide", (255, 255, 0)),
            ("Magenta = Avoid", (255, 0, 255)),
        ]
        for i, (label, color) in enumerate(legend):
            pygame.draw.rect(screen, color, (SCREEN_SIZE - 150, 10 + i * 20, 15, 15))
            surf = font.render(label, True, (200, 200, 200))
            screen.blit(surf, (SCREEN_SIZE - 130, 10 + i * 20))
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
print("Test complete.")
