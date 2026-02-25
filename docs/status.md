---
layout: default
title:  Status
---
## Project Summary:

Our project focuses on a specific variation of the classic Snake game called the "cheese" variation, which consists of the snake having every other tile of its body be non-collidable. This means that the snake agent could turn and maneuver through itself if the body tile the agent travels through is invisible, which is very useful in cases where the snake agent cuts itself off into a boundary. From the current point of the snake head, the snake agent could decide to turn its body left or right relative to its current direction. Collecting an apple increases the length of the snake after a small buffer, allowing the snake agent to have time to react to the new environmental change. Our project consists of an application-driven PPO algorithm, where the snake agent is aware of its body position, the fruit position, and if it took a dangerous move. In terms of priority, the main goal of the snake agent is to not only maximize its strength, but to also efficiently choose its moves in order to get closer and closer to the apple.

## Approach

This is a test message.
- basic reward system with a postiive and negative point system based on whether our snake find an apple versus hitting itself or the board boundary. 
- provided more parameters, particularly a basic danger detection function, that will inform the agent where dangers of termination are in its next move, and more information about where the snake's body is on the board as opposed to having information on just eh snake's head and the fruit currently on the board.

In terms of our algorithmic approach, our group decided to go with using the Proximal Policy Optimization (PPO) algorithm in order to train our snake agent. This algorithm works well with discrete action spaces, where the dicrete action space for our project consists of: turning left, turning right, turning up, and turning down. We intially started training with 25k-50k timesteps in order to test and see how our setup was doing. However, once we were content with our environment setup, we used 1 million timesteps for our latest training process. We specifically use the clip version of the PPO algorihtm, where policies are updated using:

$$
{E}_{(s,a)∼p\overline{\theta}}[L\frac{\theta}{\theta}(s,a)]
$$

where L is given by:

$$
L\frac{\theta}{\theta}(s,a)=min(\rho\frac{\theta}{\theta}(a|s){A}_{\overline{\theta}}(s,a), {{A}_{\overline{\theta}}(s,a)}+{|\epsilon{A}_{\overline{\theta}}(s,a)|}
$$

### Observations

For our observation space, we originally had only the snake head and the fruit as an observation. However, we noticed that the agent wasn't taking into account its body, resulting in the snake agent constantly running into itself during training. On top of this, the snake also had no way of knowing if it was nearing a boundary or not. This is why our group decided to add an observation for both the rest of the snake body and whether or not the current action the agent took was a "dangerous" move that led it to a boundary or not.

### Rewards

For our reward system, our original reward system was:
- **+1** if the score increased after the chosen action
- **-1** if the game terminated (snake agent ran into a wall or itself)
- **-0.01** for every action the snake took

However, with this reward system, the snake agent never initially got to the apples, so it would always try to run into a wall to end its episode earlier, resulting in a less negative reward total than if they had continued exploring. This is why on top of these rewards, we added another reward mechanic, which was based on the Euclidean distance between the snake head and the fruit:
- **+0.1** if the chosen action resulted in getting closer to the fruit
- **-0.1** if the chosen action resulted in getting farther from the fruit

This additional approach helps the agent know that the fruit is the main goal of the training process, resulting in the snake agent focusing its vision on the fruit rather than finding a fast way to stop its depletion of the reward in its current episode run.

## Evaluation



## Remaining Goals and Challenges

Our goals for the rest of the quarter are still in line with the ones outlined in our proposal, and plan to achieve our realistic goal, which was to maximize the score to be as close to the upper limit as possible. This means that we hope to achieve a score of 85 apples on a 9x10 board, 250 apples on a 15x17 board, and 499 apples on a 23x24 board. However, as mentioned before, we recognize that fruit spawning conditions will impact our maximum, and so we do not expect to achieve this exact score but instead attempt to push the average toward these values as much as possible. If time allows, we would also like to consider adding other modifications, such as bombs that deduct points from the snake's score and other fruits that offer a variety of points, so the snake has to pick and choose which ones to pursue.

As a baseline, we wanted our agent to achieve a score that is considered above average for an average human player, which was 43–46 on a 9x10 board, 125–128 on a 15x17 board, and 250–253 on a 23x24 board. Right now, our model is not hitting our baseline goal, but with the various changes to the parameters, environment, and reward system outlined in our approach rapidly improving our score, we believe that our current implementation is on track, although more evaluation is needed to gain better results. Specifically, we want to tune our reward system even further by granting a higher reward when the snake eats fruit, since we noticed in some episodes that the snake would circle the fruit instead of eating it and finding the next one because this resulted in a less negative score. Additionally, we are considering giving the snake a penalty that is proportional to the distance its head is from the fruit in order to encourage it to take shorter paths. We also plan to increase the penalty for death to prevent the snake from choosing to die earlier in the game to receive a less negative score. Another thing we would like to do is normalize our values (coordinates, reward/penalty, etc.), since we read that doing so improves convergence and stability.

However, given our progress so far, one challenge we are anticipating is how to best polish our reward system. While we do have plans for changes that will help our snake agent progress further, we are concerned about how to best assign values to our penalties and rewards relative to each other to induce the best behavior from the snake agent. Right now, we observe that giving too little of a reward or penalty can encourage undesirable behavior (i.e., intentionally dying earlier or not taking the shortest path), but we are not yet sure how to scale these values appropriately so that the snake will be at its most efficient. Moreover, we are worried that our tuned values might fluctuate with board size.

## Resources Used

- [Gymnasium Custom Environment Documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/) used for creating a custom Snake game RL Environment
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) used for implementation on a PPO algorithm setup
- [OpenAI Spinning Up Reinforcement Learning Introductory Overview](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) used to understand the role of observation spaces and how they work with chosen actions during the training process
- [Article on Reinforcement Learning in Snake](https://xiaoyang-rebecca.github.io/posts/2025/01/rl-snake/) used to understand the typical environment setup used to train an agent on the Snake game
- [GeeksforGeeks Snake Pygame Implementation Tutorial](https://www.geeksforgeeks.org/python/snake-game-in-python-using-pygame-module/) used for setting up the base code of the classic Snake game using the Pygame library
- [Pygame Documentation](https://www.pygame.org/docs/) used to add on additional features to the base code of the Snake game, including the "snake" variation

