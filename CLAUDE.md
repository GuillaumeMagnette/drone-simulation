# CLAUDE.md - Drone Simulation Repository Guide

**Last Updated**: 2025-12-03
**Purpose**: Guide for AI assistants working with this reinforcement learning drone simulation codebase

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Core Components](#core-components)
4. [Development Workflows](#development-workflows)
5. [Code Conventions](#code-conventions)
6. [Configuration Patterns](#configuration-patterns)
7. [Testing & Validation](#testing--validation)
8. [Critical Implementation Details](#critical-implementation-details)
9. [Common Tasks](#common-tasks)
10. [Troubleshooting](#troubleshooting)

---

## Project Overview

### Mission
Train autonomous drone agents using reinforcement learning (PPO) to:
- Navigate through complex urban environments
- Evade guided missile interceptors
- Reach target objectives efficiently
- Operate in multi-agent swarm configurations

### Technology Stack
- **RL Framework**: Stable Baselines 3 (PPO algorithm)
- **Environment**: Gymnasium (OpenAI Gym successor)
- **Physics**: Custom 2D Newtonian physics engine
- **Pathfinding**: A* algorithm with costmap-based safety margins
- **Visualization**: Pygame for 2D rendering
- **Parallelization**: SubprocVecEnv for multi-environment training

### Current Status
Based on latest commit (0338111):
- ✅ Basic navigation in sparse wall environments - **MASTERED**
- ✅ Evasion in clear areas (arena mode) - **MASTERED**
- ⚠️ Combined navigation + evasion - **IN PROGRESS** (challenging)
- ⚠️ Multi-agent swarm coordination - **EXPERIMENTAL** (drones still colliding)
- ❌ Turret projectile dodging - **FAILED** (deprecated)

---

## Repository Structure

```
drone-simulation/
├── Core RL Environment
│   ├── drone_env.py              # Main Gymnasium environment (36-dim obs space)
│   └── drone_env_patched.py      # Optimized version with logging control
│
├── Physics & Entities
│   ├── agent.py                  # Drone physics entity (21KB, CRITICAL)
│   ├── agent_skeleton.py         # Template/reference implementation
│   ├── interceptor.py            # Heat-seeking guided missile (600 px/s)
│   ├── projectile.py             # Simple bullet physics
│   └── turret.py                 # Stationary turret (ARCHIVED)
│
├── Navigation System
│   ├── algorithm.py              # A* pathfinding implementation
│   ├── grid.py                   # Grid system for A* (80x80 cells)
│   └── node.py                   # A* node with g/h/f costs
│
├── Training Scripts (Priority Order)
│   ├── train_turbo.py            # ⭐ RECOMMENDED: 8-core curriculum training
│   ├── train_fast.py             # ⭐ RECOMMENDED: 30 parallel environments
│   ├── train.py                  # Basic training (slow, visual feedback)
│   └── main_v[2-5].py            # Archived training experiments
│
├── Testing & Validation
│   ├── test_viewer_enhanced.py   # ⭐ LATEST: Interactive + headless testing
│   ├── test_viewer.py            # Interactive visual validation
│   ├── test.py                   # Simple test script
│   └── test_swarm.py             # Swarm behavior testing
│
├── Multi-Agent (Experimental)
│   ├── multi_agent_env.py        # Swarm environment (4 drones, shared brain)
│   └── swarm.py                  # Follower drone implementation
│
├── Utilities
│   ├── visuals.py                # Drawing utilities (13KB)
│   └── main.py                   # Simple pygame demo
│
└── Models
    └── models/PPO_Tactical3/
        └── drone_tactical.zip    # Pre-trained PPO model (197KB)
```

### File Status Legend
- **ACTIVE**: Currently used in main workflows
- **CORE**: Critical component, rarely modified
- **RECOMMENDED**: Best practice for task
- **EXPERIMENTAL**: Under development, unstable
- **ARCHIVED**: Deprecated or historical reference

---

## Core Components

### 1. Gymnasium Environment (`drone_env.py` / `drone_env_patched.py`)

**Purpose**: Implements the RL training loop, reward shaping, and observation construction.

#### Observation Space (36 dimensions)
```python
[0-7]   Lidar (8 rays)           # Normalized wall distances [0-1]
[8-9]   Velocity (vx, vy)        # Normalized by max_speed
[10-11] GPS Direction (dx, dy)   # Unit vector to waypoint/target
[12-19] Threat Sectors (8)       # Interceptor distances [0-1]
[20-27] Projectile Sectors (8)   # Bullet distances [0-1] (reserved)
[28-35] Neighbor Sectors (8)     # Friendly drone distances [0-1] (MARL)
```

**Key Point**: This is a **MARL-ready** architecture. Sectors [28-35] enable multi-agent cooperation.

#### Action Space
```python
action = [fx, fy]  # Continuous force vector in range [-1, 1]
```
Applied force is scaled by `max_force = 2500.0` in agent physics.

#### Reward Function
```python
reward = 0

# 1. Progress reward (dense shaping)
progress = last_distance - current_distance
reward += progress * 0.15

# 2. Time penalty (encourages speed)
reward -= 0.1

# 3. Threat proximity penalty
if interceptor_nearby:
    reward -= threat_penalty  # Scaled by distance

# 4. Stall penalty (moving too slowly)
if speed < threshold:
    reward -= (0.5 to 1.0)

# 5. Wall scraping penalty
if near_wall_corner:
    reward -= 0.5

# 6. Predictive crash penalty
if on_collision_course:
    reward -= 2.0 * speed

# 7. Terminal rewards (sparse)
if crashed_or_caught:
    reward -= 50
elif reached_target:
    reward += 100
```

**Philosophy**: Dense shaping + sparse terminals. Encourages safe, fast routes.

#### Map Types
- `arena`: Empty space (pure evasion testing)
- `sparse`: Scattered obstacles (navigation + evasion)
- `urban`: City-like grid (complex navigation)

#### Key Methods
```python
reset(options={'map_type': 'sparse', 'num_interceptors': 2})
  → Generates map, spawns entities, computes A* path
  → Returns: observation (36,), info dict

step(action)
  → Applies force, updates physics, checks collisions
  → Returns: observation, reward, terminated, truncated, info
```

### 2. Physics System (`agent.py`)

**Class Hierarchy**: `PhysicsEntity` (base) → `Agent` (drone)

#### Physics Constants (CRITICAL - DO NOT CHANGE WITHOUT TESTING)
```python
mass = 1.0
max_speed = 360.0        # px/s
max_force = 2500.0       # Newtons
friction = 0.88          # ⚠️ TUNED VALUE - enables "juking"
```

**Friction Note**: Value of 0.88 is critical for evasive maneuvers. Too high = slippery, too low = sluggish.

#### Physics Integration Loop
```python
def update_physics(self, dt):
    # F = ma
    self.velocity += self.acceleration * dt

    # Terminal velocity clamping
    speed = np.linalg.norm(self.velocity)
    if speed > self.max_speed:
        self.velocity = (self.velocity / speed) * self.max_speed

    # Air resistance
    self.velocity *= self.friction

    # Position update
    self.position += self.velocity * dt

    # Reset acceleration for next frame
    self.acceleration[:] = 0
```

#### Sensor Methods
```python
cast_rays(walls, num_rays=8)
  → Returns: [8] normalized distances to walls
  → Implementation: Raycasting with 200px max range

get_sector_readings(entities, radius, num_sectors=8)
  → Returns: [8] minimum distances in angular sectors
  → Usage: Threat detection, neighbor detection

_has_line_of_sight(target, walls)
  → Returns: bool (can see target through walls?)
  → Usage: Filters observations in urban environments
```

**Sensor Pattern**: All sensors return normalized [0-1] values for stable RL training.

### 3. Pathfinding System (`algorithm.py`, `grid.py`, `node.py`)

#### A* Implementation
```python
a_star(grid, start_node, end_node)
  → Returns: List of waypoints (node positions)
  → Heuristic: Euclidean distance
  → Movement: 4-directional (no diagonals to prevent corner clipping)
```

#### Grid System
- **Dimensions**: 80x80 grid (10px per cell)
- **Screen Size**: 800x800 pixels
- **Node Properties**: `g_cost`, `h_cost`, `f_cost`, `parent`

#### Costmap Safety Margins
```python
# Cost assignment
open_space = 1.0
near_wall = 10.0  # 2-cell (20px) padding around obstacles
```

**Purpose**: A* strongly prefers paths away from walls, reducing collision risk.

#### Integration with Environment
```python
# Recalculation frequency
self.repath_interval = 20  # steps

# Usage
if self.step_count % self.repath_interval == 0:
    self.waypoint = self._compute_next_waypoint()
```

**Design Decision**: Only used in `sparse` and `urban` maps. Arena mode uses direct line-to-target.

### 4. Interceptor System (`interceptor.py`)

**Behavior**: Heat-seeking guided missile

#### Physics (Intentionally Challenging)
```python
max_speed = 600.0        # FASTER than drone (360.0)
max_force = 1500.0       # Lower turn rate than drone
friction = 1.0           # No drag - constant momentum
```

**Balance**: Faster but less agile. Drone can "juke" around it.

#### Pursuit Algorithm
```python
def update(self, target_pos, dt):
    desired = target_pos - self.position
    desired = normalize(desired) * self.max_speed
    steer = desired - self.velocity
    self.apply_force(clamp(steer, self.max_force))
```

**Pure Pursuit**: Always accelerates directly toward target (no predictive leading).

#### Spawning Logic
```python
# In drone_env.py
if time_since_last_spawn > 1.0 and len(interceptors) < 5:
    spawn_interceptor_near_target()
```

**Design**: Continuous pressure, max 5 active missiles.

### 5. Multi-Agent Environment (`multi_agent_env.py`)

**Purpose**: Validation tool for swarm behavior with shared PPO brain.

#### Configuration
```python
num_drones = 4           # Default
num_interceptors = 2     # Default
map_type = 'sparse'      # Default
```

#### Observation Construction
Each drone gets same 36-dim observation as single-agent:
- Lidar → walls
- Threat sectors → interceptors
- **Neighbor sectors [28-35] → other friendly drones**

#### Collision Rules
```python
# Drone-to-drone collision
if drone1.rect.colliderect(drone2.rect):
    drone1.alive = False
    drone2.alive = False
    stats['friendly_fire'] += 1
```

**Current Issue**: Drones still crossing paths. Hardcoded anti-collision attempts failed.

---

## Development Workflows

### Training Workflow (Recommended)

#### Option 1: Fast Parallel Training (`train_fast.py`)
```bash
python train_fast.py
```

**Features**:
- 30 parallel environments (SubprocVecEnv)
- 100K steps per iteration
- Headless (no visualization)
- True multiprocessing speedup (5-10x)

**Configuration**:
```python
# Edit these variables at top of file
NUM_ENVS = 30
TOTAL_TIMESTEPS = 100_000
MODEL_NAME = "PPO_Tactical3"
STAGE = 2  # Curriculum stage
```

#### Option 2: Curriculum Training (`train_turbo.py`)
```bash
python train_turbo.py
```

**Features**:
- 8 parallel environments (DummyVecEnv - sequential but safer)
- Curriculum stages 0-7
- Performance callback tracking

**Curriculum Stages**:
```python
0: Navigation Basics      # sparse, 0 interceptors
1: Evasion Basics         # arena, 1 interceptor
2: Terrain Evasion        # sparse, 1 interceptor
3: Multi-Threat Terrain   # sparse, 2 interceptors
4: Urban Navigation       # urban, 0 interceptors
5: Urban Evasion          # urban, 1 interceptor
6: Urban Warfare          # urban, 2 interceptors
7: Swarm Basics           # arena, 1 interceptor, multi-agent
```

**How to Advance**:
1. Monitor success rate in console output
2. When success rate >60-70% for 3-5 iterations, increase `CURRICULUM_STAGE`
3. Restart training script
4. Model saves automatically every iteration

#### Option 3: Visual Training (`train.py`)
```bash
python train.py
```

**Use Case**: Initial debugging, understanding agent behavior.
**Warning**: 10x slower than parallel training. Not recommended for long training runs.

### Testing Workflow (Recommended)

#### Interactive Testing (`test_viewer_enhanced.py`)
```bash
python test_viewer_enhanced.py
```

**Controls**:
```
[1-6]   Switch scenarios (preset configs)
[R]     Reset episode
[V]     Toggle visual rendering (headless mode)
[+]     Speed up (2x, 4x, 8x steps per frame)
[-]     Slow down
[ESC]   Quit

Scenario Mapping:
1: Arena + 1 interceptor
2: Sparse + 1 interceptor
3: Sparse + 2 interceptors
4: Urban + 0 interceptors
5: Urban + 1 interceptor
6: Urban + 2 interceptors
```

**Headless Mode**: Press [V] to disable rendering → 10x faster performance evaluation.

**Tracked Metrics**:
- Episodes completed
- Success rate (%)
- Termination breakdown (Success / Crash / Caught / Timeout)

#### Swarm Testing (`multi_agent_env.py`)
```bash
python multi_agent_env.py
```

**Use Case**: Validate multi-agent behavior with trained single-agent brain.
**Warning**: Experimental. Collision avoidance not working yet.

---

## Code Conventions

### 1. Vector Operations

**Always use numpy arrays for positions/velocities**:
```python
# ✅ Correct
position = np.array([x, y], dtype=float)
velocity = np.array([0.0, 0.0])

# ❌ Wrong
position = [x, y]  # Lists don't support vectorized ops
```

**Distance calculations**:
```python
# ✅ Correct
distance = np.linalg.norm(point1 - point2)

# Normalization
if distance > 0:
    direction = vector / distance
```

### 2. Observation Normalization

**All observations must be normalized to [-1, 1] or [0, 1]**:
```python
# Distances
normalized_dist = distance / max_distance

# Velocities
normalized_vel = velocity / max_speed

# Angles (if used)
normalized_angle = angle / np.pi  # Assuming angle in [-π, π]
```

**Rationale**: Neural networks train faster with normalized inputs.

### 3. Reward Shaping

**Follow the established pattern**:
```python
reward = 0

# 1. Dense shaping (frequent, small rewards)
reward += progress_toward_goal * weight

# 2. Penalties (negative rewards for bad behavior)
if bad_state:
    reward -= penalty

# 3. Sparse terminals (large, infrequent rewards)
if episode_complete:
    reward += large_terminal_reward
```

**Anti-patterns**:
- ❌ Only sparse rewards (agent won't learn)
- ❌ Only dense rewards (agent gets "lazy")
- ❌ Unnormalized rewards (unstable training)

### 4. Entity Lifecycle Pattern

**All entities should follow this structure**:
```python
class Entity:
    def __init__(self, x, y):
        self.position = np.array([x, y], dtype=float)
        self.alive = True
        self.rect = pygame.Rect(x, y, width, height)

    def update(self, dt):
        if not self.alive:
            return
        # Update logic...
        self._check_death_conditions()

    def draw(self, screen):
        if not self.alive:
            return
        pygame.draw.circle(screen, color, self.position.astype(int), radius)
```

### 5. Gymnasium Environment Conventions

**Reset signature**:
```python
def reset(self, seed=None, options=None):
    super().reset(seed=seed)

    # Extract options
    map_type = options.get('map_type', 'sparse') if options else 'sparse'
    num_interceptors = options.get('num_interceptors', 1) if options else 1

    # Reset logic...

    observation = self._get_observation()
    info = {}
    return observation, info
```

**Step signature**:
```python
def step(self, action):
    # Apply action
    # Update physics
    # Check collisions

    observation = self._get_observation()
    reward = self._compute_reward()
    terminated = self._check_terminal_conditions()
    truncated = (self.step_count >= self.max_steps)
    info = {}

    return observation, reward, terminated, truncated, info
```

### 6. Pygame Rendering Conventions

**Drawing order (back to front)**:
```python
def render(self):
    # 1. Background
    screen.fill(BLACK)

    # 2. Grid/walls
    for wall in walls:
        pygame.draw.rect(screen, GRAY, wall)

    # 3. Path (if debugging)
    for i in range(len(path) - 1):
        pygame.draw.line(screen, BLUE, path[i], path[i+1], 2)

    # 4. Entities (back to front)
    for projectile in projectiles:
        projectile.draw(screen)
    for interceptor in interceptors:
        interceptor.draw(screen)
    agent.draw(screen)

    # 5. Target
    pygame.draw.circle(screen, GREEN, target, radius)

    # 6. UI overlay
    draw_text(screen, f"Reward: {reward:.2f}", (10, 10))

    pygame.display.flip()
```

---

## Configuration Patterns

### 1. PPO Hyperparameters

**Location**: `train_fast.py`, `train_turbo.py`

```python
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,      # Step size for gradient descent
    n_steps=512,             # Rollout length (collect 512 transitions)
    batch_size=64,           # Mini-batch size for updates
    n_epochs=10,             # Passes over collected data
    gamma=0.99,              # Discount factor (future reward weight)
    gae_lambda=0.95,         # Generalized advantage estimation
    clip_range=0.2,          # PPO trust region clip ε
    ent_coef=0.01,           # Exploration bonus
    verbose=1
)
```

**When to modify**:
- `learning_rate`: Increase if training too slow, decrease if unstable
- `n_steps`: Increase for longer episodes, decrease for short episodes
- `batch_size`: Increase with more compute, decrease if OOM
- `ent_coef`: Increase for more exploration, decrease for exploitation

### 2. Environment Parameters

**Location**: `drone_env.py` `__init__` method

```python
# World dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
PIXEL_SIZE = 10              # Grid cell size
ROWS = 80                    # 800 / 10 = 80 cells

# Episode settings
self.max_steps = 2000        # Episode timeout
self.repath_interval = 20    # A* recalculation frequency

# Spawning
self.missile_interval = 1.0  # Seconds between interceptor spawns
self.max_missiles = 5        # Max active interceptors
```

### 3. Physics Tuning

**Location**: `agent.py`

```python
# Agent (Drone)
self.mass = 1.0
self.max_speed = 360.0       # px/s
self.max_force = 2500.0      # N
self.friction = 0.88         # ⚠️ CRITICAL VALUE

# Interceptor
self.mass = 1.0
self.max_speed = 600.0       # Faster than drone
self.max_force = 1500.0      # Lower turn rate
self.friction = 1.0          # No drag
```

**Testing Protocol for Physics Changes**:
1. Modify value in `agent.py`
2. Run `python test_viewer_enhanced.py`
3. Press [1] for arena mode (pure evasion test)
4. Observe agent behavior for 10+ episodes
5. Check success rate - should be >50% in arena

### 4. Curriculum Stage Configuration

**Location**: `train_turbo.py`

```python
CURRICULUM_STAGE = 2  # Change this to progress

CURRICULUM_STAGES = {
    0: {
        "name": "Navigation Basics",
        "map_type": "sparse",
        "num_interceptors": 0,
        "success_threshold": 0.7
    },
    1: {
        "name": "Evasion Basics",
        "map_type": "arena",
        "num_interceptors": 1,
        "success_threshold": 0.6
    },
    # ... etc
}
```

**Advancement Protocol**:
1. Monitor console output for success rate
2. When success rate >threshold for 3-5 consecutive iterations
3. Increment `CURRICULUM_STAGE`
4. Restart training (model loads automatically)

---

## Testing & Validation

### Pre-Training Validation Checklist

Before starting long training runs, validate:

```bash
# 1. Environment can reset without errors
python -c "
from drone_env_patched import DroneEnv
env = DroneEnv()
obs, info = env.reset()
print(f'Observation shape: {obs.shape}')
assert obs.shape == (36,), 'Wrong obs shape!'
print('✅ Reset works')
"

# 2. Environment can step without errors
python -c "
from drone_env_patched import DroneEnv
import numpy as np
env = DroneEnv()
obs, _ = env.reset()
for i in range(100):
    action = np.array([0.5, 0.5])  # Constant force
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
print('✅ Step works')
"

# 3. Parallel environments work
python -c "
from stable_baselines3.common.vec_env import SubprocVecEnv
from drone_env_patched import DroneEnv
envs = SubprocVecEnv([lambda: DroneEnv() for _ in range(4)])
obs = envs.reset()
print(f'Parallel obs shape: {obs.shape}')
assert obs.shape == (4, 36), 'Wrong parallel obs shape!'
print('✅ Parallel works')
"
```

### Post-Training Validation Protocol

After training iterations, validate model performance:

```bash
# 1. Load model and test in various scenarios
python test_viewer_enhanced.py

# 2. Press [V] to enable headless mode
# 3. Press [1-6] to cycle through scenarios
# 4. Let run for 100+ episodes per scenario
# 5. Record success rates

# Expected success rates (well-trained model):
# Scenario 1 (arena + 1 interceptor):     70-90%
# Scenario 2 (sparse + 1 interceptor):    50-70%
# Scenario 3 (sparse + 2 interceptors):   30-50%
# Scenario 4 (urban + 0 interceptors):    80-95%
# Scenario 5 (urban + 1 interceptor):     40-60%
# Scenario 6 (urban + 2 interceptors):    20-40%
```

### Performance Profiling

**Identify bottlenecks**:
```python
import cProfile
import pstats

# Profile training loop
cProfile.run('model.learn(10000)', 'training_profile.stats')

# Analyze
stats = pstats.Stats('training_profile.stats')
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

**Common bottlenecks**:
- A* pathfinding (if called too frequently)
- Raycasting for lidar (if too many rays)
- Pygame rendering (disable in training)
- Collision detection (optimize with spatial hashing if needed)

---

## Critical Implementation Details

### 1. Threading Configuration

**Why it matters**: Numpy/MKL create thread pools that conflict with SubprocVecEnv multiprocessing.

**Location**: Top of `train_fast.py`
```python
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
```

**Rule**: Always set these BEFORE importing numpy/torch.

### 2. Friction Value (0.88)

**Why it matters**: Enables evasive "juking" maneuvers.

**Effect of changes**:
- `friction = 1.0`: No friction, drone drifts endlessly
- `friction = 0.95`: Slippery, hard to stop
- `friction = 0.88`: ✅ Current value - instant stops possible
- `friction = 0.80`: Too much friction, sluggish

**Testing**: After changing friction, test in arena mode (scenario 1) for pure evasion behavior.

### 3. A* Costmap Weights

**Why it matters**: Prevents wall-hugging behavior.

**Current values**:
```python
open_space = 1.0
near_wall = 10.0  # 2-cell (20px) padding
```

**Effect of changes**:
- `near_wall = 1.0`: Agent cuts corners, frequent crashes
- `near_wall = 5.0`: Slightly safer paths
- `near_wall = 10.0`: ✅ Current value - wide safety margins
- `near_wall = 100.0`: Paths TOO conservative, inefficient

### 4. Observation Normalization Ranges

**Critical for stable training**. Current normalization:

```python
# Lidar: [0, 1] (0=touching wall, 1=max range 200px)
lidar = distance / 200.0

# Velocity: [-1, 1] (normalized by max_speed=360)
velocity_norm = velocity / 360.0

# GPS: [-1, 1] (unit vector)
gps = direction / np.linalg.norm(direction)

# Sectors: [0, 1] (normalized by radius=400px)
sector = 1.0 - (distance / 400.0)
```

**If you add new observations**: Always normalize to [-1, 1] or [0, 1].

### 5. Episode Termination Logic

**Two types of episode endings**:

```python
# terminated = True (early ending)
if crashed or caught or reached_target:
    terminated = True
    # Triggers terminal reward (-50 or +100)

# truncated = True (timeout)
if step_count >= max_steps:
    truncated = True
    # No terminal reward, episode just ends
```

**Important**: `terminated` and `truncated` are handled differently by PPO. `truncated` episodes don't get terminal rewards.

### 6. Model Saving Convention

**Current pattern**:
```python
# Save path format
models/[MODEL_NAME]/drone_tactical.zip

# Example
models/PPO_Tactical3/drone_tactical.zip
```

**Loading**:
```python
from stable_baselines3 import PPO
model = PPO.load("models/PPO_Tactical3/drone_tactical")
# Note: No .zip extension when loading!
```

**Auto-save**: Training scripts save every iteration automatically.

---

## Common Tasks

### Task 1: Add a New Observation to the Observation Space

**Example**: Add "remaining fuel" observation.

1. **Modify observation space size** in `drone_env.py`:
```python
# In __init__
self.observation_space = spaces.Box(
    low=-1.0, high=1.0, shape=(37,), dtype=np.float32  # Was 36
)
```

2. **Add fuel tracking**:
```python
# In reset()
self.fuel = 1.0  # Start full

# In step()
self.fuel -= 0.001  # Deplete each step
```

3. **Modify observation construction**:
```python
# In _get_observation()
obs = np.concatenate([
    lidar,              # 8
    velocity_norm,      # 2
    gps,                # 2
    threat_sectors,     # 8
    projectile_sectors, # 8
    neighbor_sectors,   # 8
    [self.fuel]         # 1 (NEW)
]).astype(np.float32)  # Total: 37
```

4. **Test**:
```bash
python -c "
from drone_env_patched import DroneEnv
env = DroneEnv()
obs, _ = env.reset()
print(f'New obs shape: {obs.shape}')
assert obs.shape == (37,), 'Wrong shape!'
print(f'Fuel value: {obs[36]}')
"
```

5. **Retrain**: Existing models won't work with new obs space. Start training from scratch.

### Task 2: Modify the Reward Function

**Example**: Add penalty for high speed near walls.

```python
# In _compute_reward() method of drone_env.py

# Existing reward calculation...
reward += progress * 0.15
reward -= 0.1

# NEW: Speed penalty near walls
if self.near_wall:
    speed = np.linalg.norm(self.drone.velocity)
    speed_penalty = (speed / self.drone.max_speed) * -0.5
    reward += speed_penalty

return reward
```

**Testing Protocol**:
1. Train for 1-2 iterations (100K steps)
2. Observe if agent slows down near walls
3. Check if success rate improves or degrades
4. Adjust penalty weight if needed

### Task 3: Add a New Map Type

**Example**: "maze" map with narrow corridors.

1. **Define map generation** in `drone_env.py`:
```python
def _generate_maze_map(self):
    """Generate maze-like map with narrow corridors."""
    walls = []

    # Create border
    walls.extend(self._create_border_walls())

    # Create maze grid (15px wide corridors)
    for x in range(0, 800, 100):
        for y in range(0, 800, 100):
            if np.random.rand() > 0.5:
                wall = pygame.Rect(x, y, 80, 15)
                walls.append(wall)
            else:
                wall = pygame.Rect(x, y, 15, 80)
                walls.append(wall)

    return walls
```

2. **Register in map generation logic**:
```python
def generate_map(self, map_type):
    if map_type == 'arena':
        return self._generate_arena_map()
    elif map_type == 'sparse':
        return self._generate_sparse_map()
    elif map_type == 'urban':
        return self._generate_urban_map()
    elif map_type == 'maze':  # NEW
        return self._generate_maze_map()
    else:
        raise ValueError(f"Unknown map type: {map_type}")
```

3. **Add to curriculum** in `train_turbo.py`:
```python
CURRICULUM_STAGES = {
    # ... existing stages
    8: {
        "name": "Maze Navigation",
        "map_type": "maze",
        "num_interceptors": 0,
        "success_threshold": 0.7
    }
}
```

4. **Test generation**:
```bash
python -c "
from drone_env_patched import DroneEnv
env = DroneEnv()
obs, _ = env.reset(options={'map_type': 'maze'})
print('✅ Maze generation works')
"
```

### Task 4: Debug Why Agent Gets Stuck in Corners

**Diagnostic steps**:

1. **Enable path visualization** in `test_viewer.py`:
```python
# In render loop, add:
if hasattr(env.unwrapped, 'path'):
    for i in range(len(env.unwrapped.path) - 1):
        pygame.draw.line(screen, BLUE,
                         env.unwrapped.path[i],
                         env.unwrapped.path[i+1], 2)
```

2. **Print agent state** when stuck:
```python
# In drone_env.py step()
if self._is_stuck():  # Define this method
    print(f"STUCK: pos={self.drone.position}, vel={self.drone.velocity}")
    print(f"Waypoint: {self.waypoint}, target: {self.target}")
    print(f"Last action: {self.last_action}")
```

3. **Check A* path quality**:
```python
# In drone_env.py after pathfinding
if len(self.path) < 2:
    print("WARNING: A* returned empty or single-node path")
```

4. **Common causes**:
- Costmap weights too high → path goes through walls
- A* fails to find path → agent has no waypoint
- Friction too low → agent can't stop at waypoint
- Reward function doesn't penalize stalling enough

### Task 5: Export Training Metrics to TensorBoard

1. **Install tensorboard**:
```bash
pip install tensorboard
```

2. **Modify training script** (`train_fast.py`):
```python
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

# Before training loop
log_dir = f"./logs/{MODEL_NAME}/"
os.makedirs(log_dir, exist_ok=True)

# Configure logger
logger = configure(log_dir, ["stdout", "tensorboard"])
model.set_logger(logger)

# Train with logging
model.learn(total_timesteps=TOTAL_TIMESTEPS)
```

3. **View logs**:
```bash
tensorboard --logdir=./logs/PPO_Tactical3/
# Open browser to http://localhost:6006
```

**Useful metrics to track**:
- `rollout/ep_rew_mean`: Average episode reward
- `rollout/ep_len_mean`: Average episode length
- `train/policy_loss`: Policy gradient loss
- `train/value_loss`: Value function loss

---

## Troubleshooting

### Issue 1: Training is Very Slow

**Symptoms**: <1000 FPS (should be 5000-10000 FPS with parallel envs).

**Diagnosis**:
```bash
# Check if parallel envs are actually running
python -c "
from stable_baselines3.common.vec_env import SubprocVecEnv
from drone_env_patched import DroneEnv
import time

envs = SubprocVecEnv([lambda: DroneEnv() for _ in range(8)])
envs.reset()

start = time.time()
for _ in range(1000):
    actions = envs.action_space.sample()
    envs.step(actions)
elapsed = time.time() - start
print(f'FPS: {8000 / elapsed:.0f}')
"
```

**Solutions**:
- ✅ Set threading env vars (see Critical Implementation Details #1)
- ✅ Use `SubprocVecEnv` not `DummyVecEnv` for true parallelism
- ✅ Disable rendering (`render_mode=None`)
- ✅ Reduce A* frequency (`repath_interval = 50` instead of 20)
- ✅ Use fewer rays in lidar (6 instead of 8)

### Issue 2: Agent Crashes Into Walls Constantly

**Symptoms**: Success rate <10% in any scenario.

**Diagnosis**:
1. Check if A* is running: Print `self.path` in environment
2. Check if waypoints are being followed: Visualize path in test_viewer
3. Check if lidar is working: Print `obs[0:8]` - should see small values near walls

**Solutions**:
- If A* not running → Check `map_type` in reset options
- If path goes through walls → Increase costmap weights (10.0 → 20.0)
- If lidar all 1.0 → Bug in raycasting, check `cast_rays()` method
- If waypoints correct but not following → Increase training time (100K → 500K steps)

### Issue 3: Agent Reaches Target But Episode Doesn't End

**Symptoms**: Agent sits on target, reward stops increasing, episode doesn't terminate.

**Diagnosis**:
```python
# In drone_env.py step() method, add:
dist_to_target = np.linalg.norm(self.drone.position - self.target)
print(f"Distance to target: {dist_to_target}")

# Check termination condition
if dist_to_target < 20:  # Should be your threshold
    print("Should terminate now!")
```

**Solution**:
```python
# In _check_terminal_conditions()
if np.linalg.norm(self.drone.position - self.target) < 20:
    self.success = True
    return True
return False
```

Make sure this method is called in `step()` before returning.

### Issue 4: Model Doesn't Load / Version Mismatch

**Error**: `TypeError: __init__() got unexpected keyword argument...`

**Cause**: Stable-Baselines3 version mismatch between training and loading.

**Solution**:
```bash
# Check SB3 version
pip show stable-baselines3

# Reinstall specific version (match training environment)
pip install stable-baselines3==2.1.0  # Or whatever version was used
```

**Prevention**: Pin versions in `requirements.txt`.

### Issue 5: Interceptors Don't Spawn

**Symptoms**: Agent reaches target easily, no evasion behavior learned.

**Diagnosis**:
```python
# In drone_env.py step() method:
print(f"Num interceptors: {len(self.interceptors)}")
print(f"Time since last spawn: {self.time_since_missile_spawn}")
```

**Solution**:
```python
# Check spawning logic in step()
self.time_since_missile_spawn += self.dt

if (self.time_since_missile_spawn >= self.missile_interval and
    len(self.interceptors) < self.max_missiles and
    self.num_interceptors > 0):  # Check this!

    self._spawn_interceptor()
    self.time_since_missile_spawn = 0
```

Make sure `self.num_interceptors` is set in `reset()` from options.

### Issue 6: GPU Not Being Used

**Note**: This project uses **CPU-based training** by default (PPO is not GPU-bottlenecked).

**If you want GPU acceleration**:
```python
# In training script
from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy",
    env,
    device="cuda",  # or "cpu"
    # ... other params
)
```

**Check GPU usage**:
```bash
nvidia-smi  # Should show python process if GPU active
```

**Note**: GPU helps only with larger networks (>256 hidden units) or very large batch sizes.

---

## Additional Resources

### File Locations Quick Reference

```
Training:
  Fast parallel  → train_fast.py
  Curriculum     → train_turbo.py
  Visual debug   → train.py

Testing:
  Interactive    → test_viewer_enhanced.py
  Simple         → test_viewer.py
  Swarm          → multi_agent_env.py

Core Components:
  Environment    → drone_env_patched.py
  Physics        → agent.py
  Pathfinding    → algorithm.py
  Threats        → interceptor.py

Utilities:
  Drawing        → visuals.py
  Grid system    → grid.py, node.py
```

### Key Constants Reference

```python
# World
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
GRID_SIZE = 80 (10px per cell)

# Physics
DRONE_MAX_SPEED = 360 px/s
DRONE_MAX_FORCE = 2500 N
DRONE_FRICTION = 0.88
INTERCEPTOR_MAX_SPEED = 600 px/s

# Training
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_N_STEPS = 512
DEFAULT_BATCH_SIZE = 64
TOTAL_TIMESTEPS = 100,000 per iteration

# Environment
MAX_EPISODE_STEPS = 2000
REPATH_INTERVAL = 20 steps
MISSILE_SPAWN_INTERVAL = 1.0 seconds
```

### Architecture Decision Log

**Why PPO instead of SAC/TD3?**
- On-policy algorithm, easier to debug
- Works well with discrete-time physics
- Trust region prevents catastrophic forgetting
- Proven to work for navigation tasks

**Why 36-dimensional observation space?**
- Modular design (8-dim sectors)
- MARL-ready (neighbor sectors pre-allocated)
- Markovian (includes velocity for momentum info)
- Normalized for stable training

**Why A* instead of pure RL navigation?**
- Hybrid approach: A* for global planning, RL for local control
- Reduces exploration burden on RL agent
- Enables fast convergence in complex maps
- Can be disabled for pure RL experimentation (arena mode)

**Why 0.88 friction?**
- Empirically tested values: 1.0 (too slippery) → 0.9 (better) → 0.88 (optimal)
- Enables instant direction changes for evasion
- Balances realism with agent capability

**Why SubprocVecEnv instead of DummyVecEnv?**
- True multiprocessing (each env in separate process)
- 5-10x speedup on multi-core systems
- Worth the IPC overhead for long training runs

---

## Contributing Guidelines (For AI Assistants)

When making changes to this codebase:

1. **Test before committing**: Always run basic tests to verify functionality.
   ```bash
   python -c "from drone_env_patched import DroneEnv; env = DroneEnv(); env.reset()"
   ```

2. **Update CLAUDE.md**: If adding new features/conventions, document them here.

3. **Preserve critical values**: Don't change friction (0.88), costmap weights (10.0), or normalization ranges without explicit request and testing.

4. **Follow naming conventions**:
   - Files: `snake_case.py`
   - Classes: `PascalCase`
   - Functions/methods: `snake_case`
   - Constants: `UPPER_SNAKE_CASE`

5. **Comment complex logic**: Especially reward shaping, physics integration, and A* modifications.

6. **Version trained models**: When changing obs/action space, save new models with incremented version number.
   ```
   models/PPO_Tactical3/ → models/PPO_Tactical4/
   ```

7. **Log hyperparameter changes**: Add comment in training script with date and rationale.
   ```python
   # 2025-12-03: Increased learning_rate from 3e-4 to 5e-4
   # Rationale: Training stalled after stage 3
   learning_rate = 5e-4
   ```

---

## Version History

**v1.0** (2025-12-03)
- Initial CLAUDE.md creation
- Documented current state after 11 commits
- Current capabilities: Basic nav mastered, evasion mastered, combined in progress

**Known Issues (as of 2025-12-03)**:
- Multi-agent collision avoidance not working
- Combined navigation + evasion challenging (success rate ~40%)
- Swarm drones still crossing paths despite hardcoded avoidance

**Next Steps**:
- Increase training time for stage 2-3 (sparse + interceptors)
- Experiment with reward shaping for smooth paths
- Investigate MARL training (separate brains per drone) vs shared brain

---

**END OF CLAUDE.md**
