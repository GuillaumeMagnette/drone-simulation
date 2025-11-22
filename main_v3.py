import pygame
import numpy as np
from agent import Agent
from grid import Grid
from algorithm import a_star_algorithm
from projectile import Projectile # <-- IMPORT THIS

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

clock = pygame.time.Clock()

# Variables for manual A* testing (Right Click + Spacebar)
start_node = None
end_node = None

# --- SETUP AGENT ---
player = Agent(200, 200, agent_size) 

# --- SETUP DUMMY ENEMY ---
# A red square for the agent to chase automatically
dummy_enemy = pygame.Rect(600, 600, 50, 50)
dummy_enemies = [dummy_enemy] 

# --- PROJECTILE SETUP ---
projectiles = []
shoot_timer = 0.0
SHOOT_INTERVAL = 2.0 # Enemy shoots every 2 seconds

def reset_game():
    """
    Creates a fresh state for the episode.
    Returns: player, dummy_enemies, projectiles, shoot_timer
    """
    # 1. Create a new Agent at the start position
    new_player = Agent(200, 200, agent_size) 

    # 2. Reset Enemy (Maybe randomize position later?)
    new_enemy = pygame.Rect(600, 600, 50, 50)
    new_dummy_enemies = [new_enemy] 

    # 3. Clear Projectiles
    new_projectiles = []
    
    # 4. Reset Timer
    new_timer = 0.0
    
    return new_player, new_dummy_enemies, new_projectiles, new_timer


def reset_game_state(player, dummy_enemy):
    """
    Resets positions but KEEPS the Agent object (and its Brain).
    """
    # Reset Player Position
    player.position = np.array([200.0, 200.0], dtype=float)
    player.rect.center = player.position
    player.velocity = np.array([0.0, 0.0])
    
    # Reset Enemy Position (Optional: Randomize later)
    dummy_enemy.topleft = (600, 600)
    
    # Clear Projectiles
    return [], 0.0 # Returns empty list and 0 timer


# --- CONFIGURATION ---
SHOW_VISUALS = True
FIXED_DT = 0.016 # Simulation time step for Turbo Mode (approx 60 FPS physics)


# --- INITIALIZE GAME ---
# We call the function once to start the first game
player, dummy_enemies, projectiles, shoot_timer = reset_game()

# --- ACTION REPEATING VARS ---
frames_to_skip = 4
frame_count = 0
current_action = 0 # Store the last chosen action

running = True
while running:
    # --- 1. TIME CONTROL ---
    if SHOW_VISUALS:
        dt = clock.tick(60) / 1000.0
    else:
        dt = FIXED_DT
    
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
                
        # --- KEYBOARD INPUT ---
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_i:
                # Toggle Inference Mode
                if player.epsilon > 0:
                    print("--- INFERENCE MODE ON (No Randomness) ---")
                    player.saved_epsilon = player.epsilon # Remember current value
                    player.epsilon = 0.0
                else:
                    print("--- TRAINING MODE RESUMED ---")
                    # Restore the epsilon we had before
                    if hasattr(player, 'saved_epsilon'):
                        player.epsilon = player.saved_epsilon
                    else:
                        player.epsilon = 1.0 # Fallback
            if event.key == pygame.K_t:
                SHOW_VISUALS = not SHOW_VISUALS
                print(f"Visuals: {SHOW_VISUALS}")

                if SHOW_VISUALS:
                    # FIX: Tick the clock once immediately to discard the 
                    # huge chunk of time that passed while in Turbo mode.
                    clock.tick() 
                # Reset visual state but keep brain
                projectiles, shoot_timer = reset_game_state(player, dummy_enemies[0])

            if event.key == pygame.K_e:
                player.switch_mode()
                print(f"Switched Mode to: {player.state}")

            if event.key == pygame.K_c: 
                start_node = None
                end_node = None
                grid = Grid(GRID_SIZE, SCREEN_WIDTH)

    # --- 1. ENEMY SHOOTING LOGIC  ---
    shoot_timer -= dt
    if shoot_timer <= 0:
        shoot_timer = SHOOT_INTERVAL
        if len(dummy_enemies) > 0:
            new_bullet = Projectile(dummy_enemies[0].center, player.rect.center)
            projectiles.append(new_bullet)

    # --- 2. UPDATE PROJECTILES  ---
    for bullet in projectiles[:]: 
        bullet.update(dt)
        if not screen.get_rect().collidepoint(bullet.position):
            projectiles.remove(bullet)

    # --- RL LOOP START ---
    
    # 1. Observe Current State
    current_state = player.get_relative_state(dummy_enemies, projectiles)
    
    # --- Calculate Old Distance (For Shapping) ---
    old_vector = np.array(dummy_enemies[0].center) - player.position
    old_dist = np.linalg.norm(old_vector)

    # 2. Choose Action (WITH REPEATING)
    if frame_count % frames_to_skip == 0:
        # Only ask the brain every 4th frame
        action = player.choose_action(current_state)
        current_action = action # Remember it
    else:
        # Otherwise, keep doing what we decided last time
        action = current_action
        
    frame_count += 1
    
    # 3. Act (Move)
    player_direction_vector = player.move_discrete(action)
    
    # Physics Update
    # We get the painted walls so the physics engine can try to slide against them
    # Even if we decide to kill the agent on touch, the physics update happens first
    walls = grid.get_obstacle_rects()
    
    # Update Aim
    mouse_pos = np.array(pygame.mouse.get_pos())
    player.aim(mouse_pos)
    
    player.update(dt, player_direction_vector, walls, dummy_enemies, grid)

    # --- RL LOOP END ---
    
    # 4. Observe New State
    next_state = player.get_relative_state(dummy_enemies, projectiles)
    
    # --- Calculate Rewards ---
    reward = -1 # Living Penalty
    
    # Distance Shaping
    new_vector = np.array(dummy_enemies[0].center) - player.position
    new_dist = np.linalg.norm(new_vector)
    
    # Danger Logic (From Agent.py)
    if player.is_in_danger_zone:
        reward -= 5 
    else:
        # Hot/Cold Game
        if new_dist < old_dist:
            reward += 2  
        else:
            reward -= 2 

    # --- TERMINAL CHECKS ---
    done = False
    hit_wall = False # Reset flag
    hit_bullet = False
    hit_enemy = False

    # 1. Check Screen Boundaries (Implied Walls)
    if (player.rect.left <= 0 or player.rect.right >= SCREEN_WIDTH or
        player.rect.top <= 0 or player.rect.bottom >= SCREEN_HEIGHT):
        hit_wall = True

    # 2. Check Grid Obstacles (Painted Walls)
    # We iterate through the list of wall rects we got earlier
    if not hit_wall: # If we haven't hit a screen edge yet
        for wall in walls:
            if player.rect.colliderect(wall):
                hit_wall = True
                break
    
    # Apply Wall Penalty
    if hit_wall:
        reward = -20
        done = True

    # 3. Bullet Collision
    for bullet in projectiles:
        if player.rect.colliderect(bullet.rect):
            hit_bullet = True
            reward = -100
            done = True
            break
            
    # 4. Target Reached
    for enemy in dummy_enemies:
        if player.rect.colliderect(enemy):
            hit_enemy = True
            reward = 100 
            done = True
            break
            
    # 5. LEARN!
    # Only learn on the frames where we actually made a decision
    if frame_count % frames_to_skip == 0:
        player.learn(current_state, action, reward, next_state)

    # --- RESET AND DECAY ---
    if done:
        # --- DECAY EPSILON PER EPISODE ---
        if player.epsilon > player.epsilon_min:
            player.epsilon *= player.epsilon_decay
        # ---------------------------------

        if hit_bullet: print(f"Episode End: SHOT. Epsilon: {player.epsilon:.3f}")
        elif hit_enemy: print(f"Episode End: VICTORY. Epsilon: {player.epsilon:.3f}")
        elif hit_wall: print(f"Episode End: WALL CRASH. Epsilon: {player.epsilon:.3f}")
        
        # Soft Reset
        projectiles, shoot_timer = reset_game_state(player, dummy_enemies[0])

    # --- DRAWING ---
    if SHOW_VISUALS:
        screen.fill((255, 255, 255))
        grid.draw(screen)
        player.draw_awareness(screen, dummy_enemies, debug_font)
        pygame.draw.rect(screen, (255, 0, 0), dummy_enemies[0])
        for bullet in projectiles:
            bullet.draw(screen)
        player.draw(screen)
        pygame.display.flip()

pygame.quit()