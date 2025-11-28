import pygame
import numpy as np
from agent import Agent
from grid import Grid
from projectile import Projectile

pygame.init()
debug_font = pygame.font.SysFont("Arial", 16)

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Phase 2: Physics & Control Test")

# --- CONFIGURATION ---
GRID_SIZE = 20 
SHOW_VISUALS = True
FIXED_DT = 0.016 

grid = Grid(GRID_SIZE, SCREEN_WIDTH)
agent_size = SCREEN_WIDTH // GRID_SIZE
clock = pygame.time.Clock()

# --- GAME VARIABLES ---
# Mass=1.5, Speed=350, Force=2000 gives a "snappy" drone feel
player = Agent(200, 200, agent_size) 
dummy_enemy = pygame.Rect(600, 600, 50, 50)
dummy_enemies = [dummy_enemy] 
projectiles = []
shoot_timer = 0.0
SHOOT_INTERVAL = 2.0 

def reset_game_state(player, dummy_enemy, grid):
    """
    Resets positions and clears physics momentum.
    """
    walls_rects = grid.get_obstacle_rects()

    # 1. Randomize Player Position
    valid_pos = False
    while not valid_pos:
        rand_x = np.random.randint(50, SCREEN_WIDTH - 50)
        rand_y = np.random.randint(50, SCREEN_HEIGHT - 50)
        test_rect = pygame.Rect(rand_x, rand_y, player.size, player.size)
        if test_rect.collidelist(walls_rects) == -1:
            valid_pos = True
            player.position = np.array([float(rand_x), float(rand_y)])
            player.rect.center = player.position
            # --- CRITICAL: KILL MOMENTUM ---
            player.velocity[:] = 0
            player.acceleration[:] = 0

    # 2. Randomize Enemy
    valid_enemy = False
    while not valid_enemy:
        rand_ex = np.random.randint(50, SCREEN_WIDTH - 50)
        rand_ey = np.random.randint(50, SCREEN_HEIGHT - 50)
        dummy_enemy.topleft = (rand_ex, rand_ey)
        if (dummy_enemy.collidelist(walls_rects) == -1 and 
            not dummy_enemy.colliderect(player.rect)):
            valid_enemy = True

    # 3. Clean Grid Visuals
    for row in grid.grid:
        for node in row:
            node.reset_visuals()
            node.update_neighbors(grid.grid)
    
    player.path = [] # Agent will recalculate path automatically in update()

    return player, [dummy_enemy], [], 0.0

# --- INITIAL START ---
player, dummy_enemies, projectiles, shoot_timer = reset_game_state(player, dummy_enemies[0], grid)

running = True
while running:
    # --- 1. TIME CONTROL ---
    if SHOW_VISUALS:
        dt = clock.tick(60) / 1000.0
        # Cap dt to prevent physics explosions during lag spikes
        if dt > 0.05: dt = 0.05 
    else:
        dt = FIXED_DT
    
    walls = grid.get_obstacle_rects()

    # --- 2. INPUT HANDLING ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        # Mouse: Add Walls
        if pygame.mouse.get_pressed()[0]: 
            pos = pygame.mouse.get_pos()
            node = grid.get_node_from_pos(pos)
            if node: node.make_obstacle()
        
        # Mouse Right Click: Clear Walls
        if pygame.mouse.get_pressed()[2]:
            pos = pygame.mouse.get_pos()
            node = grid.get_node_from_pos(pos)
            if node: 
                node.reset()
                node.reset_visuals()

        # Keyboard Controls
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_t:
                SHOW_VISUALS = not SHOW_VISUALS
            
            if event.key == pygame.K_r: 
                # Manual Reset
                player, dummy_enemies, projectiles, shoot_timer = reset_game_state(player, dummy_enemies[0], grid)

            if event.key == pygame.K_c: 
                # Clear Map
                grid = Grid(GRID_SIZE, SCREEN_WIDTH)
                player, dummy_enemies, projectiles, shoot_timer = reset_game_state(player, dummy_enemies[0], grid)

    # --- 3. SHOOTING LOGIC (Hostile Environment) ---
    shoot_timer -= dt
    if shoot_timer <= 0:
        shoot_timer = SHOOT_INTERVAL
        if len(dummy_enemies) > 0:
            dist_to_player = np.linalg.norm(np.array(dummy_enemies[0].center) - player.position)
            # Enemies shoot if you are somewhat far (prevents point-blank spam)
            if dist_to_player > 100:
                new_bullet = Projectile(dummy_enemies[0].center, player.rect.center)
                projectiles.append(new_bullet)

    for bullet in projectiles[:]: 
        bullet.update(dt, walls)
        if not bullet.active:
            projectiles.remove(bullet)

    # --- 4. PHYSICS & AI UPDATE ---
    # This is where the magic happens. 
    # The agent now self-manages: Sensing -> Planning -> Physics -> Collision
    player.update(dt, walls, dummy_enemies, grid, projectiles)

    # Mouse Aim Debug (Optional: override aim for visuals)
    # mouse_pos = np.array(pygame.mouse.get_pos())
    # player.aim(mouse_pos)

    # --- 5. TERMINAL CHECKS (Game Over) ---
    hit_bullet = False
    hit_enemy = False

    # Bullet Collision
    for bullet in projectiles:
        if player.rect.colliderect(bullet.rect):
            hit_bullet = True
            print("Status: HIT BY PROJECTILE")
            player, dummy_enemies, projectiles, shoot_timer = reset_game_state(player, dummy_enemies[0], grid)
            break
            
    # Enemy Collision (Goal Reached)
    if not hit_bullet:
        for enemy in dummy_enemies:
            if player.rect.colliderect(enemy):
                hit_enemy = True
                print("Status: TARGET REACHED")
                player, dummy_enemies, projectiles, shoot_timer = reset_game_state(player, dummy_enemies[0], grid)
                break

    # --- 6. DRAWING ---
    if SHOW_VISUALS:
        screen.fill((255, 255, 255))
        
        # Draw Grid (Walls)
        grid.draw(screen)
        
        # Draw Path Debug (Optional: Show nodes the agent is following)
        if player.path:
            for node in player.path:
                node.make_path() # Sets color to Blue
        
        # Draw Awareness (FOV, Lines)
        player.draw_awareness(screen, dummy_enemies, debug_font)
        
        # Draw Enemies
        pygame.draw.rect(screen, (255, 0, 0), dummy_enemies[0])
        
        # Draw Projectiles
        for bullet in projectiles:
            bullet.draw(screen)
        
        # Draw Lidar
        player.draw_lidar(screen, walls) # <--- ADD THIS
        # Draw Player
        player.draw(screen)
        
        # Draw Physics Stats
        stats_text = f"Vel: {np.linalg.norm(player.velocity):.1f} | State: {'PANIC' if player.color == (255,0,0) else 'NAV'}"
        text_surf = debug_font.render(stats_text, True, (0, 0, 0))
        screen.blit(text_surf, (10, 10))

        pygame.display.flip()

pygame.quit()