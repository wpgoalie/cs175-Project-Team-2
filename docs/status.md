---
layout: default
title:  Status
---
## Project Summary:

Our project focuses on a specific variation of the classic Snake game called the "cheese" variation, which consists of the snake having every other tile of its body be non-collidable. This means that the snake agent could turn and maneuver through itself if the body tile the agent travels through is invisible, which is very useful in cases where the snake agent cuts itself off into a boundary. From the current point of the snake head, the snake agent could decide to turn its body left or right relative to its current direction. Collecting an apple increases the length of the snake after a small buffer, allowing the snake agent to have time to react to the new environmental change. Our project consists of an application-driven PPO algorithm, where the snake agent is aware of its body position, the fruit position, and if it took a dangerous move. In terms of priority, the main goal of the snake agent is to not only maximize its strength, but to also efficiently choose its moves in order to get closer and closer to the apple.

## Approach

This is a test message.

## Evaluation

This is a test message.

## Remaining Goals and Challenges

Our goals for the rest of the quarter are still in line with the goals we set at the beginning of this project. As a baseline goal, we wanted our agent to achieve a score that is considered above average for an average human player, which was 43-46 on a 9x10 board, 125-128 on a 15x17 board, and 250-253 on a 23x24 board. We believe that our baseline and realistic goals are still relevant and still within reach of completion. 
Right now, our model isn't currently hitting our baseline goal, but the changes we have made to the parameters, environment, and reward system have contributed to a rapidly increasing score. Some examples of the some changes we made were a basic reward system with a postiive and negative point system based on whether our snake find an apple versus hitting itself or the board boundary. We also provided more parameters, particularly a basic danger detection function, that will inform the agent where dangers of termination are in its next move, and more information about where the snake's body is on the board as opposed to having information on just eh snake's head and the fruit currently on the board.
Right now we do think that our current implementation is on track and think that more evaluation is needed to gain better results.
Given what we experienced so far, some challenges we are anticipating is adding more to our implementation. We already have plans to make the reward system more precise by adding more minor reward updates to tiles based on pathing and location relative to fruits. We also plan emphasize the negative rewards on termination conditions (the snake hitting itself or the wall) so that in future trainings, the agent will make decisions taking longer paths to avoid all negative terminations and eventaully find more fruits instead of just hitting itself. We think that these changes will signifiicantly improve our agent's performance and we believe that there is a decent amount of time to meet our goals.

## Resources Used

- [Gymnasium Custom Environment Documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/) used for creating a custom Snake game RL Environment
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) used for implementation on a PPO algorithm setup
- [OpenAI Spinning Up Reinforcement Learning Introductory Overview](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) used to understand the role of observation spaces and how they work with chosen actions during the training process
- [Article on Reinforcement Learning in Snake](https://xiaoyang-rebecca.github.io/posts/2025/01/rl-snake/) used to understand the typical environment setup used to train an agent on the Snake game
- [GeeksforGeeks Snake Pygame Implementation Tutorial](https://www.geeksforgeeks.org/python/snake-game-in-python-using-pygame-module/) used for setting up the base code of the classic Snake game using the Pygame library
- [Pygame Documentation](https://www.pygame.org/docs/) used to add on additional features to the base code of the Snake game, including the "snake" variation

