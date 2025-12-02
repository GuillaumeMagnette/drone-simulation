# CLAUDE.md - Drone Simulation RL Project Guide

**Last Updated**: 2025-12-02
**Current Branch**: `claude/claude-md-miolcr314lrlde1p-01XKxYbSUXXobYoGXghhDTYG`
**Latest Commit**: `2cb5473` - Better A* and better reward shaping to finally master navigation

---

## Project Overview

This is a **Reinforcement Learning training environment for tactical drone navigation and evasion**. The project trains autonomous drones to:
- Navigate to target objectives through complex environments (arena, sparse obstacles, urban maps)
- Evade intelligent enemy interceptors using physics-based movement
- Exploit terrain to break pursuit while avoiding wall crashes
- Eventually coordinate with swarms of other drones (MARL-ready)

### Key Innovation
Combines classical pathfinding (A* algorithm) with deep RL (PPO) to create a hybrid control system where the agent learns tactical behavior while benefiting from structured navigation guidance.

---

## Technology Stack

| Category | Technology | Purpose |
|----------|-----------|----------|
| **RL Framework** | Stable-Baselines3 | PPO agent implementation |
| **Environment** | Gymnasium | Standard RL environment API |
| **Simulation** | Pygame | Graphics, physics, collision detection |
| **Computation** | NumPy | Vector math, physics calculations |
| **Pathfinding** | Custom A* | Grid-based navigation with weighted costmaps |
| **Parallelization** | SubprocVecEnv | 20 parallel training environments |

**Dependencies** (no requirements.txt - inferred from imports):
```
stable-baselines3>=2.0
gymnasium>=0.28
pygame>=2.0
numpy>=1.20
```

---

## Codebase Structure

### Core Architecture Files

```
drone-simulation/
├── drone_env.py          # Main Gymnasium environment (36-input obs, 2D continuous action)
├── agent.py              # Physics-based drone with sensing (lidar, sectors, GPS)
├── interceptor.py        # Enemy pursuit drone with predictive targeting
├── grid.py               # 2D grid world with obstacle tracking
├── algorithm.py          # A* pathfinding with costmap weighting
├── node.py               # Grid node class for A* (g/h/f costs)
├── projectile.py         # Bullet physics (linear trajectory)
├── turret.py             # Stationary gun turrets
├── swarm.py              # FollowerDrone for multi-agent formations
└── multi_agent_env.py    # Multi-agent swarm environment
```

### Training & Testing

```
├── train_turbo.py        # Main training loop with curriculum (7 stages)
├── train.py              # Simple single-env training with visualization
└── test_viewer.py        # Interactive test harness for trained models
```

### Legacy/Exploration Files (Functional but Educational)

```
├── main.py               # V1: Manual control sandbox
├── main_v2.py            # V2: A* + Q-learning
├── main_v3.py            # V3: Hybrid physics + reflex
├── main_v4.py            # V4: Full physics, autonomous planning
├── main_v5.py            # V5: Streamlined physics FSM
├── agent_skeleton.py     # Educational template with TODOs
├── test.py               # Early testing script
└── test_swarm.py         # Swarm behavior tests
```

### Model Artifacts

```
├── drone_ppo_physics.zip # Trained model checkpoint (151KB)
├── models/               # Training checkpoints directory
└── logs/                 # TensorBoard logs directory
```

---

## Key Components Deep Dive

### 1. DroneEnv (drone_env.py)

**Purpose**: Gymnasium-compliant RL environment
**Screen Size**: 800×800 pixels
**Grid Resolution**: 10px cells (80×80 grid)
**Physics Timestep**: 0.016s (60 FPS)
**Max Episode Length**: 2000 steps

**Observation Space** (36 inputs, normalized [-1, 1]):
```python
[0-7]    Lidar Rays        # 8 sectors, wall distances
[8-9]    Velocity          # Current velocity vector
[10-11]  GPS Vector        # Direction to next waypoint
[12-19]  Threat Sectors    # 8 sectors, interceptor distances
[20-27]  Projectiles       # 8 sectors, bullet distances (reserved)
[28-35]  Neighbors         # 8 sectors, friendly drone distances (MARL)
```

**Action Space**: Continuous Box(2) in range [-1, 1]
- Maps to force vector: `action × max_force`
- Applied as Newtonian force: `F = ma`

**Reward Shaping**:
```python
# Progress reward
reward += (last_dist - current_dist) * 0.15

# Time penalty (encourage efficiency)
reward -= 0.1

# Threat proximity penalty
reward -= threat_intensity * 0.3  # 0-0.3 based on distance

# Stall penalty (prevent infinite hovering)
if speed < 5.0:
    reward -= 0.5 to 1.0

# Wall scrape penalty (near-miss warning)
if near_wall:
    reward -= 0.5

# Predictive crash penalty
if headed_toward_wall:
    reward -= 2.0 * (speed / max_speed)

# Terminal rewards
if crashed or caught:
    reward = -50
if reached_goal:
    reward = +100
```

**Map Types** (configured via `reset(options={...})`):
- `'arena'`: Empty field (pure evasion)
- `'sparse'`: Scattered obstacles (terrain exploitation)
- `'urban'`: Grid-based city with roads and blocks

### 2. Agent (agent.py)

**Physics Parameters**:
```python
max_speed = 360.0      # pixels/second
max_force = 2500.0     # Newtons (arbitrary units)
friction = 0.88        # Velocity decay (high = committed movement)
```

**Sensing System**:
- **Lidar**: 8-ray raycasting (cardinal + diagonal directions)
- **Sector Detection**: 8-sector awareness for threats/neighbors/projectiles
- **GPS**: Vector to next A* waypoint
- **Collision Prediction**: Look-ahead crash detection

**State Machines** (Visual Debug Colors):
- **Green**: Navigation mode (following A* path)
- **Red**: Panic mode (threatened by interceptor)
- **Orange**: Attack mode (unused in current version)

**A* Integration**:
- Recalculates path every 20 steps
- Uses weighted costmap (safety cushion around walls)
- Follows waypoints sequentially

### 3. Interceptor (interceptor.py)

**Purpose**: Enemy pursuit drone
**Detection Range**: 600 pixels
**Speed Advantage**: 450 px/s (faster than agent's 360 px/s)
**Friction**: 0.95 (more agile than agent)

**AI Strategy**:
1. Predict target future position (0.3s ahead)
2. Steer toward predicted point
3. Use kinematic targeting for intercept

### 4. Training (train_turbo.py)

**Curriculum Stages** (7-stage progression):

| Stage | Map | Interceptors | Goal |
|-------|-----|--------------|------|
| 0 | Sparse | 0 | Navigation baseline |
| 1 | Arena | 1 | Learn "red = danger" |
| 2 | Sparse | 1 | Terrain exploitation |
| 3 | Sparse | 2 | Multi-threat awareness |
| 4 | Urban | 0 | Urban navigation |
| 5 | Urban | 1 | Urban evasion |
| 6 | Urban | 2 | Mission-ready ops |
| 7 | Arena | 1 + swarm | Multi-agent |

**Progression Strategy**: Train each stage until >60% success rate, then advance.

**PPO Configuration** (train_turbo.py:50-60):
```python
CURRICULUM_STAGE = 0      # <-- Change this to progress
LEARNING_RATE = 0.0005    # 3× default (faster learning)
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
NUM_CPU = 20              # Parallel environments
TIMESTEPS_PER_ITER = 50000
```

**Training Outputs**:
- `training_log.csv`: Episode statistics (map, rewards, termination reason)
- `models/PPO_Tactical3/`: Model checkpoints
- `logs/`: TensorBoard training curves

---

## Development Workflows

### Starting a Training Session

1. **Choose curriculum stage** (edit `train_turbo.py:50`):
   ```python
   CURRICULUM_STAGE = 0  # 0-7
   ```

2. **Configure map/threats** (edit stage definitions in `train_turbo.py:60-80`):
   ```python
   MAP_TYPE = 'arena'  # or 'sparse', 'urban'
   NUM_INTERCEPTORS = 1  # 0-3
   ```

3. **Run training**:
   ```bash
   python train_turbo.py
   ```

4. **Monitor progress**:
   - Watch console output (success rate, rewards)
   - Check `training_log.csv` for episode details
   - TensorBoard: `tensorboard --logdir logs/`

5. **Advance stage** when success rate >60%

### Testing a Trained Model

```bash
python test_viewer.py
```

**Interactive Controls**:
- Space: Pause/Resume
- R: Reset episode
- M: Cycle map types (arena → sparse → urban)
- I: Toggle interceptors (0 → 1 → 2)
- ESC: Quit

### Modifying Physics

Edit `agent.py` constants (lines 15-20):
```python
max_speed = 360.0      # Affects max velocity
max_force = 2500.0     # Affects acceleration
friction = 0.88        # Affects turning ability (lower = more agile)
```

**Important**: Changing physics may require retraining from Stage 0.

### Adding New Map Types

1. Edit `drone_env.py:_generate_world()` method
2. Add new map type string (e.g., `'forest'`)
3. Implement obstacle generation logic
4. Update curriculum in `train_turbo.py` if needed

---

## Code Conventions & Patterns

### Architecture Patterns

1. **Physics-First Design**: All motion uses `F=ma` with Euler integration
   ```python
   acceleration = force / mass
   velocity += acceleration * dt
   velocity *= friction  # Apply drag
   position += velocity * dt
   ```

2. **Three-Layer Control Hierarchy**:
   ```
   Layer 1 (Reflexive):  Bullet dodging (emergency override)
   Layer 2 (Tactical):   A* navigation (medium-term planning)
   Layer 3 (Strategic):  RL policy (learns overall behavior)
   ```

3. **Sensor Fusion**: Observations combine multiple modalities
   - Lidar (geometric)
   - GPS (navigation)
   - Threat sectors (tactical)
   - Velocity (state)

### Collision Philosophy

- **Walls**: Lethal (episode terminates, -50 penalty)
- **Interceptors**: Lethal (both die, -50 penalty)
- **Boundaries**: Same as walls
- **Predictive**: Penalize before impact (look-ahead detection)

### Naming Conventions

- `PhysicsEntity`: Base class for moving objects (agent, interceptor)
- `Agent`: Player-controlled or RL-controlled drone
- `DroneEnv`: Gymnasium environment wrapper
- `Grid`: World representation
- `Node`: A* pathfinding node

### Code Style

- **Comments**: Extensive "Phase N" markings show development evolution
- **Type Hints**: Minimal (NumPy arrays implicit float64)
- **Magic Numbers**: Values like `friction=0.88` are "empirically tuned"
- **No Config Files**: All parameters hardcoded (Python constants)
- **State Colors**: Debug visualization via color-coded states

---

## Common Tasks for AI Assistants

### Task 1: Tune Reward Function

**File**: `drone_env.py`
**Method**: `_calculate_reward()`
**Line**: ~350-400

**Considerations**:
- Balance progress vs. safety
- Avoid reward hacking (e.g., circling near goal)
- Test across all curriculum stages

### Task 2: Adjust Physics Parameters

**File**: `agent.py`
**Lines**: 15-20

**Tuning Guide**:
- `max_speed`: Higher = faster but harder to control
- `max_force`: Higher = more acceleration
- `friction`: Lower = more agile, higher = committed movement
- **Remember**: Interceptor physics in `interceptor.py` must stay competitive

### Task 3: Modify Observation Space

**Files**:
- `agent.py`: `get_observation()` method (~line 200)
- `drone_env.py`: `observation_space` definition (~line 50)

**Steps**:
1. Add new sensor reading in `agent.py`
2. Update observation array concatenation
3. Update `observation_space` dimensions in `drone_env.py`
4. Retrain from Stage 0

### Task 4: Add New Curriculum Stage

**File**: `train_turbo.py`
**Section**: Stage definitions (~line 60-80)

**Template**:
```python
elif CURRICULUM_STAGE == 8:
    MAP_TYPE = 'urban'
    NUM_INTERCEPTORS = 3
    DESCRIPTION = "Extreme Urban Warfare"
```

### Task 5: Implement Multi-Agent Training

**Current Status**: Infrastructure exists but not integrated
**Files to Modify**:
- `multi_agent_env.py`: Already implemented
- `train_turbo.py`: Add multi-agent training loop
- Use `swarm.py` for formation behavior

**Key Challenge**: Stable-Baselines3 is single-agent; may need custom training loop or switch to RLlib/PettingZoo.

### Task 6: Add Projectile Dodging

**Current Status**: Observation space has projectile sectors [20-27] but not used
**Files**:
- `projectile.py`: Physics already implemented
- `turret.py`: Shooting logic exists
- `drone_env.py`: Integrate turrets into environment
- `agent.py`: Use projectile sectors in decision-making

**Integration Steps**:
1. Add turrets to world generation in `drone_env.py`
2. Update turrets each step
3. Verify projectile sectors populate correctly
4. Retrain with new threat type

---

## Important Patterns & Gotchas

### Physics Integration

**Euler Integration** (agent.py:~250):
```python
def update(self, dt):
    # Apply forces
    acceleration = self.applied_force / self.mass
    self.velocity += acceleration * dt

    # Apply friction
    self.velocity *= self.friction

    # Clamp speed
    speed = np.linalg.norm(self.velocity)
    if speed > self.max_speed:
        self.velocity = (self.velocity / speed) * self.max_speed

    # Update position
    self.position += self.velocity * dt
```

**Critical**: Order matters (force → velocity → friction → clamp → position)

### A* Pathfinding

**Cost Function** (algorithm.py:~40):
```python
# Manhattan distance heuristic
h_cost = abs(current.x - target.x) + abs(current.y - target.y)

# Actual cost includes weight (costmap penalty)
g_cost = parent.g_cost + 1 + current.weight
```

**Costmap**: `grid.py` generates safety cushions around obstacles (higher weight near walls)

### Observation Normalization

**Critical**: All observations must be in [-1, 1] for neural network stability

```python
# Distance normalization
normalized_distance = 1.0 - (distance / max_distance)  # 0=far, 1=near

# Velocity normalization
normalized_velocity = velocity / max_speed  # [-1, 1]

# GPS normalization
normalized_gps = gps_vector / np.linalg.norm(gps_vector)  # unit vector
```

### Parallel Training Gotchas

**train_turbo.py** uses `SubprocVecEnv` with 20 CPUs:
- Each env runs in separate process
- Must use `if __name__ == "__main__"` guard (already present)
- Pygame rendering disabled in training (set `render_mode=None`)
- Use `test_viewer.py` for visualization

---

## Testing & Validation

### Unit Tests

**Available**:
- `test.py`: Early basic tests
- `test_swarm.py`: Swarm behavior validation
- `test_viewer.py`: Visual model evaluation

**Missing** (opportunities for contribution):
- Physics accuracy tests (collision detection)
- A* pathfinding correctness
- Observation space bounds checking
- Reward function unit tests

### Model Evaluation

**Metrics to Track**:
1. **Success Rate**: % of episodes reaching goal
2. **Average Reward**: Episode return
3. **Survival Time**: Steps before termination
4. **Termination Reasons**: Wall crash vs. interceptor vs. timeout
5. **Path Efficiency**: Steps taken vs. optimal path length

**Validation Process**:
1. Train for 50k timesteps
2. Test across all map types
3. Verify >60% success rate before advancing curriculum
4. Check `training_log.csv` for termination reason distribution

---

## What NOT to Do

### Don't

1. **Change physics during mid-training**: Invalidates learned policy
2. **Skip curriculum stages**: Agent needs progressive difficulty
3. **Remove safety penalties**: Will learn reckless behavior
4. **Train without parallel envs**: Too slow (use NUM_CPU=20)
5. **Ignore reward hacking**: Watch for unintended optimal strategies
6. **Modify observation space without retraining**: Dimension mismatch
7. **Push to non-Claude branches**: Must use `claude/*` branch naming
8. **Use `git push --force` to main**: Destructive, forbidden
9. **Commit secrets**: No API keys, credentials
10. **Add emojis to code**: Only if user explicitly requests

### Architecture Anti-Patterns

- **Don't add random exploration**: PPO handles exploration internally
- **Don't overtune hyperparameters**: Default PPO settings work well
- **Don't remove A* guidance**: Pure RL navigation takes too long to learn
- **Don't make interceptors too fast**: Agent must have escape chance

---

## Git Workflow

### Current Branch
```bash
claude/claude-md-miolcr314lrlde1p-01XKxYbSUXXobYoGXghhDTYG
```

### Commit Guidelines

**Good Commit Messages** (from history):
```
✓ "Better A* and better reward shaping to finally master navigation"
✓ "Working A* + RL with real physics and swarm of followers"
✓ "tested in a multiagent env where several agents using the same PPO trained brain"
```

**Template**:
```
<Summary of what works now>

Details:
- Specific changes made
- Key insights or discoveries
- Known issues remaining
```

### Pushing Changes

```bash
# Stage changes
git add <files>

# Commit with descriptive message
git commit -m "Your message"

# Push to Claude branch (with retry logic)
git push -u origin claude/claude-md-miolcr314lrlde1p-01XKxYbSUXXobYoGXghhDTYG
```

**Important**: Branch name must start with `claude/` and match session ID or push will fail with 403.

### Pull Request Guidelines

**When creating PR**:
1. Test across all curriculum stages
2. Document hyperparameter changes
3. Include training curves (TensorBoard screenshots)
4. Report success rates before/after
5. Describe architectural changes clearly

---

## Quick Reference

### File Quick Lookup

| Task | File | Key Lines |
|------|------|-----------|
| Modify reward | `drone_env.py` | 350-400 |
| Tune physics | `agent.py` | 15-20 |
| Change observation | `agent.py` | ~200 |
| Add curriculum stage | `train_turbo.py` | 60-80 |
| Adjust A* cost | `algorithm.py` | ~40 |
| Modify interceptor AI | `interceptor.py` | 80-120 |
| Change map generation | `drone_env.py` | 150-250 |

### Training Quick Start

```bash
# Stage 0: Navigation baseline
python train_turbo.py  # CURRICULUM_STAGE = 0

# Test trained model
python test_viewer.py

# Monitor training
tensorboard --logdir logs/
```

### Debugging Tips

1. **Agent not moving**: Check `max_force` and `action` magnitude
2. **Crashes into walls**: Increase wall scrape penalty
3. **Circles near goal**: Add stall penalty or increase time penalty
4. **Ignores interceptors**: Increase threat proximity penalty
5. **Training unstable**: Reduce `LEARNING_RATE` or increase `NUM_CPU`
6. **A* path fails**: Check costmap generation in `grid.py`

---

## Future Development Roadmap

### Planned Features (from code TODOs)

1. **Projectile Dodging**: Integrate turrets into training curriculum
2. **Multi-Agent Training**: Use `multi_agent_env.py` for swarm learning
3. **Dynamic Objectives**: Moving targets instead of static waypoints
4. **Hierarchical RL**: Separate policies for navigation vs. evasion
5. **Curriculum Auto-Advancement**: Detect success rate automatically

### Research Opportunities

1. **Compare A* vs. Learned Navigation**: Train without A* guidance
2. **Transfer Learning**: Test trained brain in new map types
3. **Opponent Modeling**: Predict interceptor behavior
4. **Formation Flying**: Coordinate multiple drones in swarm
5. **Sparse Rewards**: Remove distance shaping, use only terminal rewards

---

## Resources & Documentation

### Key Papers/Concepts

- **PPO**: Proximal Policy Optimization (Schulman et al., 2017)
- **Curriculum Learning**: Progressive difficulty training
- **A* Pathfinding**: Grid-based optimal path search
- **Physics-Based RL**: Continuous control with Newtonian dynamics

### External Documentation

- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- Gymnasium: https://gymnasium.farama.org/
- Pygame: https://www.pygame.org/docs/

### Internal Documentation

- `train_turbo.py`: Lines 1-34 (curriculum philosophy)
- `drone_env.py`: Docstrings explain observation/action spaces
- `agent.py`: Comments explain state machine transitions

---

## Contact & Support

**For AI Assistants**:
- This file is your primary reference
- Read code comments for implementation details
- Check commit history for context on design decisions
- Use `test_viewer.py` to visually validate changes

**Repository Issues**:
- Report bugs with training logs and model checkpoints
- Include curriculum stage and success rate in bug reports
- Provide TensorBoard curves for performance issues

---

**End of CLAUDE.md**
**Version**: 1.0
**Generated**: 2025-12-02
**Codebase Status**: Production-ready training pipeline with 7-stage curriculum
