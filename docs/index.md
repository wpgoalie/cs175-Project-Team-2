---
layout: default
title:  Home
---
## Overview
![Cheese Variation Snake Game](https://image.idntimes.com/post/20230803/snake-cheese-7d76f2d0dbc8e7b90df4c082feb7561b-8356234e322a6af150b7b460d98f82e1.png)
Using the classic Snake game as our environment, we train snake agents to learn effective strategies for collecting fruit while safely navigating the board. We implement a “cheese” variation of the game in which alternating segments of the snake’s body are empty, allowing the snake to move through itself and creating a less restrictive movement space. The objective of the agents is to maximize their score by efficiently reaching fruits while avoiding termination conditions such as colliding with the wall or with itself. The environment provides the agents with information about nearby dangers, fruit locations, and body positioning, enabling informed decision-making that balances short-term rewards with long-term survival.

#### Source code: [https://github.com/wpgoalie/Expert-Snake](https://github.com/wpgoalie/Expert-Snake)

### Interesting Progress Screenshots
 <img alt="image" src="https://github.com/user-attachments/assets/e58779a9-877d-4986-90f9-ca5087cdc9a7" height="300" />
 <img alt="image" src="https://github.com/user-attachments/assets/29d62335-32ab-46f1-842f-fb641414545b" height="300"/>


### Online Resources:
- [Gymnasium Custom Environment Documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
- [Stable-Baselines3 PPO Documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [GeeksforGeeks Basic Snake Pygame Implementation](https://www.geeksforgeeks.org/python/snake-game-in-python-using-pygame-module/)
- [Pygame Documentation](https://www.pygame.org/docs/)
- [OpenAI Spinning Up Reinforcement Learning Introductory Overview](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- [Article on Reinforcement Learning in Snake](https://xiaoyang-rebecca.github.io/posts/2025/01/rl-snake/)

### Reports:

- [Proposal](proposal.html)
- [Status](status.html)
- [Final](final.html)
