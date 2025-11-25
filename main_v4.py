import pygame
import numpy as np
from agent import Agent
from grid import Grid
from algorithm import a_star_algorithm
from projectile import Projectile

pygame.init()
debug_font = pygame.font.SysFont("Arial", 16)

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("A* Hybrid Agent - Training Mode")

# --- CONFIGURATION ---
GRID_SIZE = 20 
SHOW_VISUALS = True
FIXED_DT = 0.016 # Turbo mode step

grid = Grid(GRID_SIZE, SCREEN_WIDTH)
agent_size = SCREEN_WIDTH // GRID_SIZE
clock = pygame.time.Clock()

# --- GAME VARIABLES ---
player = Agent(200, 200, agent_size) 
dummy_enemy = pygame.Rect(600, 600, 50, 50)
dummy_enemies = [dummy_enemy] 
projectiles = []
shoot_timer = 0.0
SHOOT_INTERVAL = 2.0 

# --- REPATHING TIMER ---
repath_timer = 0.0
REPATH_INTERVAL = 0.2 # Recalculate path every 0.2 seconds

# --- ACTION REPEATING ---
frames_to_skip = 4
frame_count = 0
current_action = 0

def reset_game_state(player, dummy_enemy, grid):
    """
    Resets positions randomly and Calculates Initial Path.
    Returns: player, dummy_enemies (list), projectiles, shoot_timer
    """
    walls_rects = grid.get_obstacle_rects()

    # 1. Randomize Player
    valid_pos = False
    while not valid_pos:
        rand_x = np.random.randint(50, SCREEN_WIDTH - 50)
        rand_y = np.random.randint(50, SCREEN_HEIGHT - 50)
        test_rect = pygame.Rect(rand_x, rand_y, player.size, player.size)
        if test_rect.collidelist(walls_rects) == -1:
            valid_pos = True
            player.position = np.array([float(rand_x), float(rand_y)])
            player.rect.center = player.position
            player.velocity = np.array([0.0, 0.0])

    # 2. Randomize Enemy
    valid_enemy = False
    while not valid_enemy:
        rand_ex = np.random.randint(50, SCREEN_WIDTH - 50)
        rand_ey = np.random.randint(50, SCREEN_HEIGHT - 50)
        dummy_enemy.topleft = (rand_ex, rand_ey)
        if (dummy_enemy.collidelist(walls_rects) == -1 and 
            not dummy_enemy.colliderect(player.rect)):
            valid_enemy = True

    # 3. Initial A* Path
    for row in grid.grid:
        for node in row:
            node.reset_visuals()
            node.update_neighbors(grid.grid)

    start_node = grid.get_node_from_pos(player.rect.center)
    end_node = grid.get_node_from_pos(dummy_enemy.center)
    
    path = a_star_algorithm(lambda: None, grid, start_node, end_node)
    player.path = path if path else []

    # --- NEW: REMOVE THE FOG ---
    # A* paints nodes Pink/Green. We want to erase that,
    # keeping only the BLUE path nodes.
    for row in grid.grid:
        for node in row:
            # If it's not the path and not a wall, make it white again
            if node not in path and not node.is_obstacle:
                node.color = (255, 255, 255)
    # ---------------------------
    
    # FIX: Return ALL 4 values to match the unpacking at the call site
    # Note: We wrap dummy_enemy in a list [] because the main loop expects a list
    return player, [dummy_enemy], [], 0.0


# --- INITIAL START ---
player, dummy_enemies, projectiles, shoot_timer = reset_game_state(player, dummy_enemies[0], grid)

running = True
while running:
    # --- 1. TIME CONTROL ---
    if SHOW_VISUALS:
        dt = clock.tick(60) / 1000.0
    else:
        dt = FIXED_DT
    
    # --- 2. GLOBAL WALLS DEFINITION ---
    # Defined at top of loop so everyone can use it
    walls = grid.get_obstacle_rects()

    # --- EVENT HANDLING ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        # Mouse: Add Walls
        if pygame.mouse.get_pressed()[0]: 
            pos = pygame.mouse.get_pos()
            node = grid.get_node_from_pos(pos)
            if node: node.make_obstacle()

        # Keyboard
        if event.type == pygame.KEYDOWN:
            # Turbo Toggle
            if event.key == pygame.K_t:
                SHOW_VISUALS = not SHOW_VISUALS
                print(f"Visuals: {SHOW_VISUALS}")
                if SHOW_VISUALS: clock.tick() # Fix Time Spike

            # Inference Mode Toggle (No Randomness)
            if event.key == pygame.K_i:
                if player.epsilon > 0:
                    print("--- INFERENCE MODE ON (0% Random) ---")
                    player.saved_epsilon = player.epsilon
                    player.epsilon = 0.0
                else:
                    print("--- TRAINING RESUMED ---")
                    player.epsilon = getattr(player, 'saved_epsilon', 1.0)

            if event.key == pygame.K_c: 
                grid = Grid(GRID_SIZE, SCREEN_WIDTH)
                # Reset game to clear path visuals
                player, dummy_enemies, projectiles, shoot_timer = reset_game_state(player, dummy_enemies[0], grid)

    # --- 3. DYNAMIC REPATHING ---
    # Re-calculate path every 0.2 seconds to adapt to dodging
    repath_timer -= dt
    if repath_timer <= 0:
        repath_timer = REPATH_INTERVAL
        
        # Optional: Clear visuals to keep screen clean
        for row in grid.grid:
            for node in row:
                node.reset_visuals() 
                node.update_neighbors(grid.grid)
        
        start_node = grid.get_node_from_pos(player.rect.center)
        end_node = grid.get_node_from_pos(dummy_enemies[0].center)
        path = a_star_algorithm(lambda: None, grid, start_node, end_node)
        if path:
            player.path = path
             #--- SMART PRUNING (The Fix) ---
            # Only prune the start node if we are "past" it relative to the next node.
            if len(player.path) >= 2:
                p0 = player.path[0]
                p1 = player.path[1]
                
                # If the path says "Start at current node", check if we should skip it
                if p0 == start_node:
                    # Vector from P0 to P1 (The direction of the path)
                    # Note: Node.x/y are grid indices, need to convert to pixels for direction? 
                    # Actually simpler: Grid indices are fine for direction vector (e.g. 1, 0)
                    # BUT we need relative position of player to P0 center in pixels.
                    
                    # Path Vector (e.g. Right = (1, 0))
                    path_vec = np.array([p1.x - p0.x, p1.y - p0.y])
                    
                    # Player Vector relative to P0 Center
                    p0_center = np.array([p0.x + agent_size/2, p0.y + agent_size/2])
                    player_vec = player.position - p0_center
                    
                    # Dot Product: Are we on the "far side" of P0 in the direction of P1?
                    # If Dot > 0, we are past the center, moving towards P1. -> PRUNE.
                    # If Dot < 0, we are approaching center (or cornering). -> KEEP.
                    dot_prod = np.dot(path_vec, player_vec)
                    
                    if dot_prod > 0:
                        player.path.pop(0)

            # Visual Cleanup
            for row in grid.grid:
                for node in row:
                    if node not in path and not node.is_obstacle:
                        node.color = (255, 255, 255)

    # --- 4. SHOOTING & PROJECTILES ---
    shoot_timer -= dt
    if shoot_timer <= 0:
        shoot_timer = SHOOT_INTERVAL
        if len(dummy_enemies) > 0:
            # DISTANCE CHECK
            dist_to_player = np.linalg.norm(np.array(dummy_enemies[0].center) - player.position)
            
            # Only shoot if player is NOT in "Kill Range" (e.g., > 100px)
            if dist_to_player > 100:
                new_bullet = Projectile(dummy_enemies[0].center, player.rect.center)
                projectiles.append(new_bullet)

    for bullet in projectiles[:]: 
        bullet.update(dt, walls)
        if not bullet.active:
            projectiles.remove(bullet)

    # --- 5. RL LOOP START ---
    
    # A. Observe State
    current_state, reflex_action = player.get_relative_state(dummy_enemies, projectiles, walls)

    old_danger_dist = player.danger_dist
    
    # B. Calculate "Old Distance" for Reward
    # Critical: Use the Lookahead Target, NOT the enemy, for reward alignment
    target_pos_for_reward = dummy_enemies[0].center
    if player.path:
        # NEW: Always reward getting closer to the IMMEDIATE next step.
        # This ensures we reward the agent for clearing the corner.
        t_node = player.path[0] 
        target_pos_for_reward = (t_node.x + player.size//2, t_node.y + player.size//2)

    old_vector = np.array(target_pos_for_reward) - player.position
    old_dist = np.linalg.norm(old_vector)

    # C. Choose Action (Frame Skipping)
    action = 0
    
    if reflex_action is not None:
        # --- OVERRIDE ACTIVE ---
        # The Math Brain takes control. 
        # We ignore the Q-Table. We ignore Epsilon.
        action = reflex_action
        
        # Optional: We can still "Learn" from this!
        # It's called "Imitation Learning". The agent observes the Reflex saving its life.
    else:
        # --- NORMAL AI ACTIVE ---
        # Use Q-Learning to find the path (very basic RL here, learn to follow A* path by choosing between right left up down)
        if frame_count % frames_to_skip == 0:
            action = player.choose_action(current_state)
            current_action = action
        else:
            action = current_action

    frame_count += 1
    
    # D. Act
    player_direction_vector = player.move_discrete(action)
    
    # E. Physics Update
    # Aim debug visual
    mouse_pos = np.array(pygame.mouse.get_pos())
    player.aim(mouse_pos)
    
    player.update(dt, player_direction_vector, walls, dummy_enemies, grid)

    # --- 6. RL LOOP END ---
    
    # F. Observe New State
    next_state, _ = player.get_relative_state(dummy_enemies, projectiles, walls)
    
    # G. Calculate "New Distance" to SAME Lookahead Target
    new_vector = np.array(target_pos_for_reward) - player.position
    new_dist = np.linalg.norm(new_vector)

    # --- NEW: Capture NEW Danger Distance ---
    new_danger_dist = player.danger_dist
    # ----------------------------------------
    
    # H. Calculate Reward
    reward = -1 # Time penalty

    if reflex_action is not None:
        # If we are in panic mode, reward is just about survival
        # We don't punish hard, we just exist.
        reward = 0
    else:
        # --- SAFE MODE ---
        if new_dist < old_dist:
            reward += 2  
        else:
            reward -= 2

    # I. Terminal Checks
    done = False
    hit_wall = False
    hit_bullet = False
    hit_enemy = False

    # Screen & Grid Wall Check
    if (player.rect.left <= 0 or player.rect.right >= SCREEN_WIDTH or
        player.rect.top <= 0 or player.rect.bottom >= SCREEN_HEIGHT):
        hit_wall = True
    
    if not hit_wall:
        for wall in walls:
            if player.rect.colliderect(wall):
                hit_wall = True
                break
    
     # FIX D: Walls are Sticky, Not Lava.
    if hit_wall:
        #reward = -1 # Small penalty (annoying bump)
        #print("HIT WALL EDGE")
        # done = True <--- CRITICAL: DELETE OR COMMENT THIS OUT.
        # Let the physics engine slide the agent along the wall.
        pass

    # Bullet Check
    for bullet in projectiles:
        if player.rect.colliderect(bullet.rect):
            hit_bullet = True
            reward = -100
            done = True
            break
            
    # Enemy Check
    for enemy in dummy_enemies:
        if player.rect.colliderect(enemy):
            hit_enemy = True
            reward = 100 
            done = True
            break
            
    # J. Learn (Only on decision frames)
    if reflex_action is None and frame_count % frames_to_skip == 0:
        player.learn(current_state, action, reward, next_state)

    # --- 7. RESET & DECAY ---
    if done:
        # Per-Episode Decay
        if player.epsilon > player.epsilon_min:
            player.epsilon *= player.epsilon_decay

        if hit_bullet: print(f"End: SHOT. Eps: {player.epsilon:.3f}")
        elif hit_enemy: print(f"End: VICTORY. Eps: {player.epsilon:.3f}")
        elif hit_wall: print(f"End: WALL. Eps: {player.epsilon:.3f}")
        
        # Reset Game (Keep Brain)
        player, dummy_enemies, projectiles, shoot_timer = reset_game_state(player, dummy_enemies[0], grid)

    # --- 8. DRAWING ---
    if SHOW_VISUALS:
        screen.fill((255, 255, 255))
        grid.draw(screen)
        player.draw_awareness(screen, dummy_enemies, debug_font)
        pygame.draw.rect(screen, (255, 0, 0), dummy_enemies[0])
        for bullet in projectiles:
            bullet.draw(screen)
        player.draw(screen)
        
        # Draw Path (Optional Debug)
        # if player.path:
        #     for node in player.path:
        #         node.make_path()

        pygame.display.flip()

pygame.quit()