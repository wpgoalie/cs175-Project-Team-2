import gymnasium as gym
from snakeGameCheese import snakeGameCheese
from typing import Optional
import numpy as np

# NOTE FOR FUTURE: gymnasium prefers float32?

class snakeRLEnvironment(gym.Env):
    def __init__(self, length_of_grid_x = 30, length_of_grid_y = 24):
        self.length_of_grid_x = length_of_grid_x
        self.length_of_grid_y = length_of_grid_y

        self.metadata = {"render_modes": []}
        self.render_mode = None
    
        self.game = snakeGameCheese(
            grid_size=np.array([self.length_of_grid_x, self.length_of_grid_y]),
            debug=False,
            draw=False
        )
    
        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32) # the apple
    
        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([self.length_of_grid_x - 1, self.length_of_grid_y - 1]),
                    dtype=np.int32
                ),
                "target": gym.spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([self.length_of_grid_x - 1, self.length_of_grid_y - 1]),
                    dtype=np.int32
                ),
            }
        )
    
        self.action_space = gym.spaces.Discrete(4)
    
        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: "UP",   # move up
            1: "DOWN",  # move down
            2: "LEFT",  # move left
            3: "RIGHT" # move right
        }

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        while True:
            try:
                # Randomly place the agent anywhere on grid
                self._agent_location = np.array([
                    self.np_random.integers(0, self.length_of_grid_x),
                    self.np_random.integers(0, self.length_of_grid_y),
                ], dtype=np.int32)
                # Randomly place target in a different location than agent
                while True:
                    self._target_location = np.array([
                        self.np_random.integers(0, self.length_of_grid_x),
                        self.np_random.integers(0, self.length_of_grid_y),
                    ], dtype=np.int32)
                    if not np.array_equal(self._target_location, self._agent_location):
                        break
                self.game = snakeGameCheese(
                    grid_size=np.array([self.length_of_grid_x, self.length_of_grid_y]),
                    fruit_position=self._target_location,
                    snake_position=self._agent_location,
                    debug=False,
                    draw=False
                )
                break  # success, exit loop
        
            except ValueError:
                continue  # retry random positions

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-2 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-2) to a movement direction
        direction = self._action_to_direction[action]
        prev_score = self.game.score
        
        self.game.step_function(direction)

        # Convert pixel positions back to grid positions
        self._agent_location = self.game.snake_position // self.game.cell_size
        self._target_location = self.game.fruit_position // self.game.cell_size

        # Check if agent reached the target or died
        terminated = self.game.dead

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency (NOTE FOR FUTURE: negative reward if dies?)
        if self.game.score > prev_score:
            reward = 1
        else: 
            reward = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
