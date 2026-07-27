import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class DroneDynamicsEnv(gym.Env):
    """
    Planar 2D Quadrotor Environment based on Crazyflie 2.1 specifications.
    Simulates lateral navigation under strict actuator constraints.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self):
        super(DroneDynamicsEnv, self).__init__()
        
        self.action_space = spaces.Discrete(5) # Hover, Up, Down, Left, Right
        
        # --- THE OBSERVATION ARRAY (Curing the Blindness) ---
        # 0: Drone X (m)     1: Drone Y (m)
        # 2: Vel X (m/s)     3: Vel Y (m/s)
        # 4: Target X (m)    5: Target Y (m)
        # 6: Wall 1 X (m)    7: Wall 1 Gap Y (m)
        # 8: Wall 2 X (m)    9: Wall 2 Gap Y (m)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -10, -10, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([8, 6, 10, 10, 8, 6, 8, 6, 8, 6], dtype=np.float32),
            dtype=np.float32
        )
        
        # --- REALISTIC PHYSICS (The Crazyflie 2.1 Spec) ---
        self.mass = 0.027  # kg (27 grams)
        
        # A Crazyflie max thrust is ~0.57 N (58g). 
        # Hovering requires m*g = 0.26 N. 
        # Leaving ~0.31 N of available lateral thrust margin.
        self.max_lateral_thrust = 0.31  # Newtons 
        self.linear_drag = 0.015        # kg/s (Air resistance)
        self.time_step = 0.1            # Seconds per frame
        
        # --- SCALING ---
        # 1 Meter = 100 Pixels. A 800x600 window represents an 8m x 6m room.
        self.scale = 100.0
        self.wall_width = 0.4  # 40 cm
        self.gap_height = 1.5  # 1.5 meters
        # --- WIND (new) ---
        # How hard the wind pushes the drone. Start at 0.0 so nothing
        # changes yet - this is a dial we'll turn up later.
        self.wind_strength = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Start at (0.5m, 3.0m)
        self.drone_x, self.drone_y = 0.5, 3.0
        self.vel_x, self.vel_y = 0.0, 0.0
        
        # Target at X = 7.5m
        self.target_x = 7.5
        self.target_y = np.random.uniform(1.0, 5.0)
        
        # Wall 1 at X = 2.5m
        self.wall1_x = 2.5
        self.wall1_gap_y = np.random.uniform(1.0, 4.0)
        
        # Wall 2 at X = 5.0m
        self.wall2_x = 5.0
        self.wall2_gap_y = np.random.uniform(1.0, 4.0)
        
        # Initialize distance tracker for dense reward shaping
        self.prev_dist = np.sqrt((self.target_x - self.drone_x)**2 + (self.target_y - self.drone_y)**2)
        
        obs = np.array([self.drone_x, self.drone_y, self.vel_x, self.vel_y, 
                        self.target_x, self.target_y, 
                        self.wall1_x, self.wall1_gap_y, 
                        self.wall2_x, self.wall2_gap_y], dtype=np.float32)
        return obs, {}

    def step(self, action):
        # 1. Map Action to Thrust (Newtons)
        force_x, force_y = 0.0, 0.0
        if action == 1:   force_y = -self.max_lateral_thrust
        elif action == 2: force_y = self.max_lateral_thrust
        elif action == 3: force_x = -self.max_lateral_thrust
        elif action == 4: force_x = self.max_lateral_thrust
        # --- ADD WIND (new) ---
        # An extra push, added every moment, with some randomness
        # so it feels like real gusts rather than a steady fan.
        gust = self.wind_strength * self.np_random.uniform(-1.0, 1.0)
        force_y += gust   
        # 2. QUADROTOR KINEMATICS (F = ma), sub-stepped to prevent wall-ghosting
        # ------------------------------------------------------------------
        # BUG: a single 0.1s step let a fast drone move further than a wall's
        # 0.4m thickness in one go, so its position was never sampled while
        # actually inside the wall's x-slice and no collision was detected.
        # FIX: split this step into smaller sub-steps sized so the drone can
        # never move more than half a wall's thickness per sub-step, and run
        # the collision check after every sub-step instead of only once.
        max_speed = (self.max_lateral_thrust + self.wind_strength) / self.linear_drag
        max_substep_disp = self.wall_width * 0.5
        n_substeps = max(1, int(np.ceil(max_speed * self.time_step / max_substep_disp)))
        dt = self.time_step / n_substeps

        # 3. COLLISION LOGIC (checked every sub-step)
        crashed = False
        for _ in range(n_substeps):
            # Calculate acceleration: a = (F_thrust - F_drag) / mass
            accel_x = (force_x - (self.linear_drag * self.vel_x)) / self.mass
            accel_y = (force_y - (self.linear_drag * self.vel_y)) / self.mass

            # Update Velocity (v = v + at)
            self.vel_x += accel_x * dt
            self.vel_y += accel_y * dt

            # Update Position (x = x + vt)
            self.drone_x += self.vel_x * dt
            self.drone_y += self.vel_y * dt

            if self.drone_x < 0 or self.drone_x > 8.0 or self.drone_y < 0 or self.drone_y > 6.0:
                crashed = True

            if self.wall1_x <= self.drone_x <= self.wall1_x + self.wall_width:
                if not (self.wall1_gap_y <= self.drone_y <= self.wall1_gap_y + self.gap_height):
                    crashed = True

            if self.wall2_x <= self.drone_x <= self.wall2_x + self.wall_width:
                if not (self.wall2_gap_y <= self.drone_y <= self.wall2_gap_y + self.gap_height):
                    crashed = True

            if crashed:
                break

        # 4. TRUE DENSE REWARD SHAPING
        dist_to_target = np.sqrt((self.target_x - self.drone_x)**2 + (self.target_y - self.drone_y)**2)
        
        # Progress Reward: Positive if it got closer, negative if it flew backward
        progress = self.prev_dist - dist_to_target
        self.prev_dist = dist_to_target
        
        reward = -0.01 + (progress * 10.0)  # Time penalty + Progress multiplier
        terminated = False
        
        if dist_to_target < 0.2:  # Reached within 20 cm
            reward += 100.0
            terminated = True
        elif crashed:
            reward -= 100.0
            terminated = True
            
        obs = np.array([self.drone_x, self.drone_y, self.vel_x, self.vel_y, 
                        self.target_x, self.target_y, 
                        self.wall1_x, self.wall1_gap_y, 
                        self.wall2_x, self.wall2_gap_y], dtype=np.float32)
        
        return obs, reward, terminated, False, {}

    def render(self):
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255)) 
        s = self.scale  # Multiplier to convert meters back to pixels

        # Draw Target
        pygame.draw.rect(self.screen, (0, 255, 0), (self.target_x*s, self.target_y*s, 0.4*s, 0.4*s))

        # Draw Walls
        pygame.draw.rect(self.screen, (255, 0, 0), (self.wall1_x*s, 0, self.wall_width*s, self.wall1_gap_y*s))
        pygame.draw.rect(self.screen, (255, 0, 0), (self.wall1_x*s, (self.wall1_gap_y + self.gap_height)*s, self.wall_width*s, 600))
        pygame.draw.rect(self.screen, (255, 0, 0), (self.wall2_x*s, 0, self.wall_width*s, self.wall2_gap_y*s))
        pygame.draw.rect(self.screen, (255, 0, 0), (self.wall2_x*s, (self.wall2_gap_y + self.gap_height)*s, self.wall_width*s, 600))

        # Draw Drone
        pygame.draw.circle(self.screen, (0, 0, 255), (int(self.drone_x*s), int(self.drone_y*s)), 10)

        pygame.display.flip()
        self.clock.tick(15)
        # ==========================================
# ==========================================
# TEST LOOP: Drone flies toward target, with wind
# ==========================================
# ==========================================
# APF PILOT: the "magnetic guide"
# ==========================================
import numpy as np

def apf_action(obs, gap_height=1.5, wall_width=0.4):
    """
    Works out which way the magnets are pushing/pulling, and returns
    the single best action to move that way.
    """
    drone_x, drone_y = obs[0], obs[1]
    target_x, target_y = obs[4], obs[5]
    walls = [(obs[6], obs[7]), (obs[8], obs[9])]   # (wall_x, gap_y)

    # ---- 1. ATTRACTION: the target pulls the drone toward it ----
    pull_x = target_x - drone_x
    pull_y = target_y - drone_y
    # keep the pull a steady size so it doesn't overpower everything far away
    dist = np.sqrt(pull_x**2 + pull_y**2) + 1e-6
    pull_x, pull_y = pull_x / dist, pull_y / dist

    # ---- 2. REPULSION: walls push the drone away ----
    push_x, push_y = 0.0, 0.0
    influence = 1.5          # walls only matter within 1.5 m
    strength = 2.0           # how hard they shove

    for wall_x, gap_y in walls:
        wall_centre = wall_x + wall_width / 2
        gap_centre = gap_y + gap_height / 2
        gap_low, gap_high = gap_y, gap_y + gap_height

        x_gap = wall_centre - drone_x        # how far ahead the wall is
        if abs(x_gap) < influence:
            closeness = (influence - abs(x_gap)) / influence   # 0 far, 1 touching

            if gap_low < drone_y < gap_high:
                # Lined up with the opening - the wall barely bothers us.
                pass
            else:
                # Lined up with SOLID wall. Two things happen:
                # (a) the wall shoves us back the way we came
                push_x -= np.sign(x_gap) * strength * closeness
                # (b) we get pulled sideways toward the opening
                to_gap = gap_centre - drone_y
                push_y += np.sign(to_gap) * strength * closeness * 1.5

    # ---- 3. REPULSION from the edges of the room ----
    edge = 0.6
    if drone_y < edge:            push_y += strength * (edge - drone_y) / edge
    if drone_y > 6.0 - edge:      push_y -= strength * (drone_y - (6.0 - edge)) / edge
    if drone_x < edge:            push_x += strength * (edge - drone_x) / edge
    if drone_x > 8.0 - edge:      push_x -= strength * (drone_x - (8.0 - edge)) / edge

    # ---- 4. ADD IT ALL UP (now with braking) ----
    vel_x, vel_y = obs[2], obs[3]
    damping = 0.35          # how much it fights its own momentum

    total_x = pull_x + push_x - damping * vel_x
    total_y = pull_y + push_y - damping * vel_y

    # ---- 5. Turn that direction into one of our 5 actions ----
    if abs(total_x) < 0.05 and abs(total_y) < 0.05:
        return 0                                  # forces cancelled out -> STUCK
    if abs(total_x) > abs(total_y):
        return 4 if total_x > 0 else 3            # right / left
    else:
        return 2 if total_y > 0 else 1            # down / up


def run_episode(env, action_fn, render=False, max_steps=400):
    """Fly one episode using action_fn(obs) -> action. Returns (outcome, steps_taken)."""
    obs, info = env.reset()
    stuck_counter = 0

    for step_number in range(max_steps):
        action = action_fn(obs)

        # watch for the drone freezing in place (a pilot getting stuck)
        speed = np.sqrt(obs[2]**2 + obs[3]**2)
        stuck_counter = stuck_counter + 1 if speed < 0.15 else 0
        if stuck_counter > 40:
            return "STUCK", step_number

        obs, reward, terminated, truncated, info = env.step(action)
        if render:
            env.render()

        if terminated:
            return ("REACHED" if reward > 0 else "CRASHED"), step_number

    return "TIMED_OUT", max_steps


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fly the APF pilot in the drone nav environment.")
    parser.add_argument("--watch", action="store_true",
                         help="Watch a single flight with rendering on (default: run a headless batch tally).")
    parser.add_argument("--episodes", type=int, default=50,
                         help="Number of episodes to run in batch mode (default: 50).")
    parser.add_argument("--wind", type=float, default=0.3,
                         help="Wind strength (default: 0.3).")
    args = parser.parse_args()

    env = DroneDynamicsEnv()
    env.wind_strength = args.wind
    apf_pilot = lambda obs: apf_action(obs, env.gap_height, env.wall_width)

    if args.watch:
        # --- Single visual flight (the original behaviour) ---
        print("APF pilot flying... (wind =", env.wind_strength, ")")
        outcome, steps = run_episode(env, apf_pilot, render=True)
        messages = {
            "REACHED": f"REACHED the target! (after {steps} steps)",
            "CRASHED": f"CRASHED. (after {steps} steps)",
            "STUCK": f"STUCK! The drone froze in place at step {steps}.",
            "TIMED_OUT": "Ran out of time without reaching the target.",
        }
        print(messages[outcome])
    else:
        # --- Headless batch run with a tally at the end ---
        print(f"Running {args.episodes} episodes with the APF pilot (rendering off, wind = {env.wind_strength})...")

        tally = {"REACHED": 0, "CRASHED": 0, "STUCK": 0, "TIMED_OUT": 0}
        reached_steps = []

        for ep in range(args.episodes):
            outcome, steps = run_episode(env, apf_pilot, render=False)
            tally[outcome] += 1
            if outcome == "REACHED":
                reached_steps.append(steps)

        avg_steps = (sum(reached_steps) / len(reached_steps)) if reached_steps else None

        print(f"\n--- Tally over {args.episodes} episodes ---")
        print(f"REACHED:   {tally['REACHED']}")
        print(f"CRASHED:   {tally['CRASHED']}")
        print(f"STUCK:     {tally['STUCK']}")
        print(f"TIMED OUT: {tally['TIMED_OUT']}")
        if avg_steps is not None:
            print(f"Average steps for successful runs: {avg_steps:.1f}")
        else:
            print("Average steps for successful runs: N/A (no successful runs)")

    env.close()