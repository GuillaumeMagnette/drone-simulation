import pygame
import numpy as np
from agent import Agent

pygame.init()
debug_font = pygame.font.SysFont("Arial", 16)
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("AI Sandbox - Press 'E' to switch modes")

player = Agent(400, 300, 32)
clock = pygame.time.Clock()

enemies = [
    pygame.Rect(100, 100, 32, 32),
    pygame.Rect(600, 150, 32, 32),
    pygame.Rect(200, 500, 32, 32),
    pygame.Rect(700, 450, 32, 32)
]

running = True
while running:
    dt = clock.tick(60) / 1000.0

    # --- EVENT HANDLING (The Main Change) ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # Check for a key being pressed down ONCE
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_e:
                # Tell the agent to switch its internal state
                player.switch_mode()

    # Get continuous key presses for movement
    keys = pygame.key.get_pressed()
    player_direction_vector = np.array([0.0, 0.0]) # Renamed for clarity
    if keys[pygame.K_w]: player_direction_vector[1] -= 1
    if keys[pygame.K_s]: player_direction_vector[1] += 1
    if keys[pygame.K_a]: player_direction_vector[0] -= 1
    if keys[pygame.K_d]: player_direction_vector[0] += 1

    mouse_pos = np.array(pygame.mouse.get_pos())
    player.aim(mouse_pos)

    # --- UPDATE ---
    # The agent's FSM will decide whether to use the player_direction_vector
    # or to generate its own.
    player.update(dt, player_direction_vector, enemies)

    # --- DRAWING ---
    screen.fill((0, 0, 0))
    player.draw(screen)
    player.draw_awareness(screen, enemies, debug_font)
    pygame.display.flip()

pygame.quit()