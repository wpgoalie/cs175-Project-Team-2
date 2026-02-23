import gymnasium as gym
from snakeGameCheese import snakeGameCheese
from typing import Optional
import numpy as np
import pygame
import math

# NOTE FOR FUTURE: gymnasium prefers float32?

class snakeRLEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, length_of_grid_x = 30, length_of_grid_y = 24, render_mode = None):
        self.length_of_grid_x = length_of_grid_x
        self.length_of_grid_y = length_of_grid_y

        self.metadata = {"render_modes": []}
        self.render_mode = render_mode
        self.window = None
        self.clock = None
    
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
                "danger": gym.spaces.Box(
                    low = 0,
                    high = 1,
                    shape=(3,),
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
        dangers = self._get_dangers()
        danger_arr = np.array([
            dangers["UP"],
            dangers["DOWN"],
            dangers["LEFT"],
            dangers["RIGHT"]
        ], dtype=np.int32)
        
        return {"agent": self._agent_location, "target": self._target_location, "danger": danger_arr}

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
    def _get_dangers(self):
        # should detect if next move results in collision with itself or a wall, returns dictionary of where dangers are based on direction
        head_x, head_y = self.game.snake_position
        step = self.game.cell_size

        potential_positions = {"UP": (head_x, head_y - step), 
                               "DOWN": (head_x, head_y + step), 
                               "LEFT": (head_x - step, head_y), 
                               "RIGHT": (head_x + step, head_y),}
        dangers = {}

        for direction, (nx, ny) in potential_positions.items():
            danger = 0
            # boundary/wall check
            if nx < 0 or nx >= self.game.size_x or ny < 0 or ny >= self.game.size_y:
                danger = 1
            # body collision check
            for segment in self.game.snake_body[1:]:
                if segment[2] == self.game.active_body_key:
                    if nx == segment[0] and ny == segment[1]:
                        danger = 1
                        break
                        
            dangers[direction] = danger

        return dangers
        
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
        prev_distance = math.sqrt((self.game.fruit_position[0] - self.game.snake_position[0]) ** 2 + (self.game.fruit_position[1] - self.game.snake_position[1]) ** 2)
        
        self.game.step_function(direction)

        # Convert pixel positions back to grid positions
        self._agent_location = self.game.snake_position // self.game.cell_size
        self._target_location = self.game.fruit_position // self.game.cell_size

        # Check if agent reached the target or died
        terminated = self.game.dead

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        current_distance = math.sqrt((self.game.fruit_position[0] - self.game.snake_position[0]) ** 2 + (self.game.fruit_position[1] - self.game.snake_position[1]) ** 2)

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency (NOTE FOR FUTURE: negative reward if dies?)
        if self.game.score > prev_score:
            reward = 1
        else: 
            if terminated:
                reward = -1
            else:
                reward = abs(prev_distance - current_distance) * -0.1
                reward = reward - 0.01

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        # if self.window is None and self.render_mode == "human":
        #     pygame.init()
        #     pygame.display.init()
            
        #     self.window = pygame.display.set_mode(
        #         (self.window_size, self.window_size)
        #     )
        
        # if self.clock is None and self.render_mode == "human":
        #     self.clock = pygame.time.Clock()

        # snakeCanvas = pygame.Surface((self.game.size_x, self.game.size_y))
        # snakeCanvas.fill((255, 255, 255))
        # pix_square_size = self.game.cell_size

        # # Draws the apple
        # pygame.draw.rect(
        #     snakeCanvas,
        #     (255, 0, 0),
        #     pygame.Rect(
        #         pix_square_size * self._target_location[::-1],
        #         (pix_square_size, pix_square_size),
        #     ),
        # )

        # # Draws the snake head only (we might need to store the entire snake body in the agent location?)
        # pygame.draw.circle(
        #     snakeCanvas,
        #     (0, 0, 255),
        #     (self._agent_location[::-1] + 0.5) * pix_square_size,
        #     pix_square_size / 3,
        # )

        # if self.render_mode == "human":
        #     # The following line copies our drawings from `canvas` to the visible window
        #     self.window.blit(snakeCanvas, snakeCanvas.get_rect())
        #     pygame.event.pump()
        #     pygame.display.update()

        #     # We need to ensure that human-rendering occurs at the predefined framerate.
        #     # The following line will automatically add a delay to keep the framerate stable.
        #     self.clock.tick(self.metadata["render_fps"])
        # else:  # rgb_array
        #     return np.transpose(
        #         np.array(pygame.surfarray.pixels3d(snakeCanvas)), axes=(1, 0, 2)
        #     )
        # pygame.init()
        # pygame.display.set_caption('Snake Game with Cheese Variation')
        # self.game.game_screen = pygame.display.set_mode((self.game.size_x, self.game.size_y))

        # self.game.game_screen.fill('green')
        # # draw head regardless of its assigned visibility
        # head = self.game.snake_body[0]
        # pygame.draw.rect(self.game.game_screen, 'cyan', pygame.Rect(head[0], head[1], self.game.cell_size, self.game.cell_size))
        # # draw body parts alternating by body key
        # for pos in self.game.snake_body[1:]:
        #     if pos[2] == self.game.active_body_key: # actual body, not skipped part
        #         pygame.draw.rect(self.game.game_screen, 'blue',
        #                         pygame.Rect(pos[0], pos[1], self.game.cell_size, self.game.cell_size))
        
        # pygame.draw.rect(self.game.game_screen, 'red', pygame.Rect(
        #     self.game.fruit_position[0], self.game.fruit_position[1], self.game.cell_size, self.game.cell_size))

        # self.game.score_display('white', 'times new roman', 30)
        if self.window is None:
            self.game.fps = pygame.time.Clock()
            pygame.init()
            pygame.display.set_caption('Snake Game with Cheese Variation')
            self.window = pygame.display.set_mode((self.game.size_x, self.game.size_y))

        self.window.fill('green')
        # draw head regardless of its assigned visibility
        head = self.game.snake_body[0]
        pygame.draw.rect(self.window, 'cyan', pygame.Rect(head[0], head[1], self.game.cell_size, self.game.cell_size))
        # draw body parts alternating by body key
        for pos in self.game.snake_body[1:]:
            if pos[2] == self.game.active_body_key: # actual body, not skipped part
                pygame.draw.rect(self.window, 'blue',
                                pygame.Rect(pos[0], pos[1], self.game.cell_size, self.game.cell_size))
        
        pygame.draw.rect(self.window, 'red', pygame.Rect(
            self.game.fruit_position[0], self.game.fruit_position[1], self.game.cell_size, self.game.cell_size))

        # displaying score continuously
        # self.game.score_display('white', 'times new roman', 30)
    
        # Refresh game screen
        pygame.display.update()
        

        return np.transpose(np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2))

    

