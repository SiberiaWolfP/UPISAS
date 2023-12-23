from time import sleep
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import math
from stable_baselines3.common.env_checker import check_env
from matplotlib import pyplot as plt
import torch


DEFAULT_NUM_GATEWAYS = 5
DEFAULT_THRESHOLD = -50

class DingNetEnv(gym.Env):

    metadata = {'render_modes': ['console']}

    def __init__(self, render_mode: Optional[str]=None, num_gateways=DEFAULT_NUM_GATEWAYS,
                threshold=DEFAULT_THRESHOLD, verbose=2):
        super(DingNetEnv, self).__init__()

        self.map_size = 3010  # 3km in 1m resolution
        self.num_gateways = num_gateways
        self.threshold = threshold
        self.steps = 0
        self.verbose = verbose
        self.mote = None

        # Grid types with path loss and shadow fading values
        self.grid_types = {'Forest': (3, 1.5), 'City': (1, 2), 'Plain': (2, 1.5)}
        # self.grid_types = {'City': (1, 2)}

        self.action_space = spaces.Discrete(101)  # from -50 to 50
        self.observation_space = spaces.Tuple((
            spaces.Box(low=-130, high=30, shape=(), dtype=np.float32),                                          # Signal strength
            spaces.Box(low=0, high=self.map_size * 1.5, shape=(), dtype=np.float32),                            # Distance to gateway
            spaces.Box(low=-50, high=50, shape=(), dtype=np.float32),                                           # Transmission power
            spaces.Box(low=0, high=self.map_size, dtype=np.float32),                                            # Position x
            spaces.Box(low=0, high=self.map_size, dtype=np.float32),                                            # Position y
            spaces.Box(low=0, high=self.map_size, dtype=np.float32),                                            # Closest Gateway position x
            spaces.Box(low=0, high=self.map_size, dtype=np.float32),                                            # Closest Gateway position y
            spaces.Box(low=1, high=4, shape=(), dtype=np.float32),                                              # Path loss
            spaces.Box(low=1, high=3, shape=(), dtype=np.float32)                                               # Shadow fading
        ))

        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.map_refresh = True
        # Initialize or reset the environment state
        if options is not None:
            self.num_gateways = options.get('num_gateways') if 'num_gateways' in options else DEFAULT_NUM_GATEWAYS
            self.threshold = options.get('threshold') if 'threshold' in options else DEFAULT_THRESHOLD
            self.verbose = options.get('verbose') if 'verbose' in options else 1
            self.map_refresh = options.get('map_refresh') if 'map_refresh' in options else True
        self.mote = {'position': self._random_position(), 
                    'power': random.randint(-15, 15),
                    'direction': random.uniform(0, 2 * math.pi)}
        self.gateways = [{'position': self._random_position()} for _ in range(self.num_gateways)]
        if self.map_refresh:
            self.map_grid = self._generate_grid()
        self.steps = 0
        return self._normalize(self._observe()), {}

    def step(self, action):
        self.steps += 1
        # Implement the logic for one step in the environment
        terminated = False
        truncated = False

        # Update the transmission power of the motes
        self.mote['power'] = action - 50

        self._move_mote()

        observation = self._observe()

        # Check termination conditions
        if observation[0] >= self.threshold + 80 or observation[0] <= self.threshold - 80:
            terminated = True
        
        # Check truncation conditions
        if self.steps >= 2000:
            truncated = True

        reward = self._calculate_reward(observation[0], self.threshold)
        return self._normalize(observation), reward, terminated, truncated, {}  # False for 'done' since the episode doesn't end in this step
    


    def render(self, mode='console'):
        # Implement the visualization
        if mode == 'console':
            if self.verbose == 0:
                return
            elif self.verbose == 1:
                signal_strengths = []
                observation = self._observe()
                signal_strengths.append(observation[0])
                print(signal_strengths)
            elif self.verbose == 2:
                print('=' * 20)
                print('Frame: {}'.format(self.steps))
                observation = self._observe()
                print('Signal strength: {}'.format(observation[0]))
                print('Distance to gateway: {}'.format(observation[1]))
                print('Transmission power: {}'.format(observation[2]))
                # print('Position: {}, {}'.format(observation[3], observation[4]))
                print('Path loss: {}'.format(observation[3]))
                print('Shadow fading: {}'.format(observation[4]))
                print('=' * 20)


    def _random_position(self):
        return random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)

    def _generate_grid(self):
        # Generate the grid with different types
        grid_types = list(self.grid_types.keys())
        grid = []

        # Each 1km x 1km subgrid will have the same type
        subgrid_size = 430  # 1km in 1m resolution

        for _ in range(self.map_size // subgrid_size):
            row = []
            for _ in range(self.map_size // subgrid_size):
                # Assign a random grid type to each subgrid
                grid_type = random.choice(grid_types)
                row.append(grid_type)
            # Repeat the row to fill the subgrid
            for _ in range(subgrid_size):
                grid.append(list(row))

        # Repeat each element in the row to fill the subgrid
        for i in range(len(grid)):
            grid[i] = [elem for elem in grid[i] for _ in range(subgrid_size)]

        return grid

    def _move_mote(self):
        # Update positions of mote
        # Move 10m in each step
        dx = int(10 * math.cos(self.mote['direction']))  # Change in x
        dy = int(10 * math.sin(self.mote['direction']))  # Change in y

        new_x = self.mote['position'][0] + dx
        new_y = self.mote['position'][1] + dy

        # Check for boundary collision and adjust direction if necessary
        if not (0 <= new_x < self.map_size and 0 <= new_y < self.map_size):
            self.mote['direction'] = self._randomize_direction_within_bounds(self.mote['position'])
            dx = int(10 * math.cos(self.mote['direction']))
            dy = int(10 * math.sin(self.mote['direction']))
            new_x = self.mote['position'][0] + dx
            new_y = self.mote['position'][1] + dy

        self.mote['position'] = (new_x, new_y)

    def _randomize_direction_within_bounds(self, position):
        while True:
            new_direction = random.uniform(0, 2 * math.pi)
            dx = int(10 * math.cos(new_direction))
            dy = int(10 * math.sin(new_direction))
            new_x = position[0] + dx
            new_y = position[1] + dy
            if 0 <= new_x < self.map_size and 0 <= new_y < self.map_size:
                return new_direction  # Return a valid direction that keeps the mote within bounds

    def _observe(self):
        # Create the observation based on the current state
        closest_gateway, min_distance = self._find_closest_gateway(self.mote['position'])
        grid_type = self.map_grid[self.mote['position'][0]][self.mote['position'][1]]
        path_loss, shadow_fading = self.grid_types[grid_type]
        signal_strength = self._compute_signal_strength(self.mote['position'], closest_gateway['position'], self.mote['power'])
        return np.array([signal_strength, min_distance, self.mote['power'], self.mote['position'][0], self.mote['position'][1],
                         closest_gateway['position'][0], closest_gateway['position'][1], path_loss, shadow_fading], dtype=np.float32)
    
    def _calculate_reward(self, signal_strength, threshold):
        return -abs(signal_strength - threshold)


    def _compute_signal_strength(self, sender_position, receiver_position, initial_power):
        xPos, yPos = receiver_position
        senderX, senderY = sender_position
        transmissionPower = initial_power

        while transmissionPower > -300:
            xDist = abs(xPos - senderX)
            yDist = abs(yPos - senderY)
            if xDist + yDist == 0:
                break  # Receiver reached

            xDir = np.sign(xPos - senderX)
            yDir = np.sign(yPos - senderY)
            path_loss, shadow_fading = self.grid_types[self.map_grid[xPos][yPos]]

            if xDist + yDist > 1:
                if xDist > 2 * yDist or yDist > 2 * xDist:
                    transmissionPower -= 10 * path_loss * (np.log10(xDist + yDist) - np.log10(xDist + yDist - 1))
                    xPos -= xDir if xDist > 2 * yDist else 0
                    yPos -= yDir if yDist > 2 * xDist else 0
                else:
                    transmissionPower -= 10 * path_loss * (np.log10(xDist + yDist) - np.log10(xDist + yDist - np.sqrt(2)))
                    xPos -= xDir
                    yPos -= yDir
            elif xDist + yDist == 1:
                xPos -= xDir if xDist > yDist else 0
                yPos -= yDir if yDist > xDist else 0


        shadow_fading = shadow_fading if shadow_fading else self.grid_types[self.map_grid[xPos][yPos]][1]
        return transmissionPower - np.random.normal() * shadow_fading
        # return transmissionPower

    def _find_closest_gateway(self, mote_position):
        # Find the closest gateway and its distance from the mote
        closest_gateway = None
        min_distance = float('inf')
        for gateway in self.gateways:
            distance = self._calculate_distance(mote_position, gateway['position'])
            if distance < min_distance:
                min_distance = distance
                closest_gateway = gateway
        return closest_gateway, min_distance

    def _calculate_distance(self, pos1, pos2):
        # Calculate the Euclidean distance between two points
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _normalize(self, observations):
        # Normalize the observations
        for i in range(len(observations)):
            observations[i] = 2 * ((observations[i] - self.observation_space[i].low) / (self.observation_space[i].high - self.observation_space[i].low)) - 1
        return observations
    
signal_strengths = []
power_levels = []
distance_to_gateway = []


def plot_returns(show_result=False):
    signal_strengths_t = torch.tensor(signal_strengths, dtype=torch.float)
    power_levels_t = torch.tensor(power_levels, dtype=torch.float)
    distance_to_gateway_t = torch.tensor(distance_to_gateway, dtype=torch.float)

    fig = plt.figure(1, figsize=(6, 6))
    if show_result:
        fig.suptitle('Result')
    else:
        fig.clf()
        fig.suptitle('Training...')

    axs = fig.subplots(3, 1)

    # Plot signal strengths
    axs[0].set_xlabel('TimeStep')
    axs[0].set_ylabel('Signal Strength')
    axs[0].plot(signal_strengths_t.numpy())

    # Plot power levels
    axs[1].set_xlabel('TimeStep')
    axs[1].set_ylabel('Power Level')
    axs[1].plot(power_levels_t.numpy())

    # Plot distance to gateway
    axs[2].set_xlabel('TimeStep')
    axs[2].set_ylabel('Distance to Gateway')
    axs[2].plot(distance_to_gateway_t.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

if __name__ == '__main__':
    plt.ion()

    env = DingNetEnv(verbose=2, num_gateways=5)
    for i in range(1000000):
        action = 2
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        signal_strengths.append(observation[0])
        power_levels.append(observation[2])
        distance_to_gateway.append(observation[1])
        plot_returns()
