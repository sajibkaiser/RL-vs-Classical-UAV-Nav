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
            
        # 2. QUADROTOR KINEMATICS (F = ma)
        # Calculate acceleration: a = (F_thrust - F_drag) / mass
        accel_x = (force_x - (self.linear_drag * self.vel_x)) / self.mass
        accel_y = (force_y - (self.linear_drag * self.vel_y)) / self.mass
        
        # Update Velocity (v = v + at)
        self.vel_x += accel_x * self.time_step
        self.vel_y += accel_y * self.time_step
        
        # Update Position (x = x + vt)
        self.drone_x += self.vel_x * self.time_step
        self.drone_y += self.vel_y * self.time_step
        
        # 3. COLLISION LOGIC
        crashed = False
        if self.drone_x < 0 or self.drone_x > 8.0 or self.drone_y < 0 or self.drone_y > 6.0:
            crashed = True
            
        if self.wall1_x <= self.drone_x <= self.wall1_x + self.wall_width:
            if not (self.wall1_gap_y <= self.drone_y <= self.wall1_gap_y + self.gap_height):
                crashed = True

        if self.wall2_x <= self.drone_x <= self.wall2_x + self.wall_width:
            if not (self.wall2_gap_y <= self.drone_y <= self.wall2_gap_y + self.gap_height):
                crashed = True
        
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
        self.clock.tick(self.metadata["render_fps"])