---
layout: default
title:  Status
---

<iframe 
  width="560" 
  height="315" 
  src="https://youtu.be/1oSYQwpLjNw" 
  title="YouTube video player" 
  frameborder="0" 
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
  allowfullscreen>
</iframe>

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
L\frac{\theta}{\theta}(s,a)=min(\rho\frac{\theta}{\theta}(a|s)A_{\overline{\theta}}(s,a), {A_{\overline{\theta}}(s,a)}+{|\epsilon A_{\overline{\theta}}(s,a)|}
$$

### Observations

For our observation space, we originally had only the snake head and the fruit as an observation. However, we noticed that the agent wasn't taking into account its body, resulting in the snake agent constantly running into itself during training. On top of this, the snake also had no way of knowing if it was nearing a boundary or not. This is why our group decided to add an observation for both the rest of the snake body and whether or not the current action the agent took was a "dangerous" move that led it to a boundary or not.

### Rewards

For our reward system, our original reward system was:
- **+1** if the score increased after the chosen action
- **-1** if the game terminated (snake agent ran into a wall or itself)
- **-0.01** for every action the snake took

<figure>
<video width="320" height="240" controls>
  <source src="./images/eval-episode-64.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
    <figcaption>Example of a snake rushing towards the wall instead of the fruit in order to end the run.</figcaption>
</figure>

However, with this reward system, the snake agent never initially got to the apples, so it would always try to run into a wall to end its episode earlier, resulting in a less negative reward total than if they had continued exploring. This is why on top of these rewards, we added another reward mechanic, which was based on the Euclidean distance between the snake head and the fruit:
- **+0.1** if the chosen action resulted in getting closer to the fruit
- **-0.1** if the chosen action resulted in getting farther from the fruit

This additional approach helps the agent know that the fruit is the main goal of the training process, resulting in the snake agent focusing its vision on the fruit rather than finding a fast way to stop its depletion of the reward in its current episode run.

## Evaluation

After retrieving the results of our 1 million timestep training process, we noticed significant improvement between the first number of timesteps and the final number of timesteps:

First Timesteps:
```
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 22.5        |
|    ep_rew_mean          | -1.17       |
| time/                   |             |
|    fps                  | 56          |
|    iterations           | 2           |
|    time_elapsed         | 72          |
|    total_timesteps      | 4096        |
| train/                  |             |
|    approx_kl            | 0.013247951 |
|    clip_fraction        | 0.153       |
|    clip_range           | 0.2         |
|    entropy_loss         | -1.38       |
|    explained_variance   | -0.583      |
|    learning_rate        | 0.0003      |
|    loss                 | 0.0353      |
|    n_updates            | 10          |
|    policy_gradient_loss | -0.0196     |
|    value_loss           | 0.238       |
-----------------------------------------
```

Final Timesteps:
```
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 276         |
|    ep_rew_mean          | 33.6        |
| time/                   |             |
|    fps                  | 155         |
|    iterations           | 489         |
|    time_elapsed         | 6433        |
|    total_timesteps      | 1001472     |
| train/                  |             |
|    approx_kl            | 0.015203483 |
|    clip_fraction        | 0.109       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.374      |
|    explained_variance   | -0.146      |
|    learning_rate        | 0.0003      |
|    loss                 | 0.651       |
|    n_updates            | 4880        |
|    policy_gradient_loss | -0.0097     |
|    value_loss           | 2.06        |
-----------------------------------------
```

In terms of the reward mean, at the beginning training process point of 4096 timesteps, the snake agent was getting to either 0 or 1 apples, but the amount of steps it was taking was dropping the reward, resulting in a negative average reward of -1.17. In terms of timestep length, the snake was still training on what was a good or bad move, resulting in many runs in these initial timesteps where the snake ran into either itself or into the boundary walls, with an average timestep length of 22.5. At the end of the training process at 1 million timesteps, the average timestep length was 276, where the snake had already understood that running into itself or the boundary walls is not a good choice, as it lost lots of reward points throughout this training process. On top of this, the average reward in this point in the training process was 33.6, where the snake was getting on average around 30-40 apples each run, with the number of actions taken reducing this total reward. The snake at the end of the training process learned that the apples were the main source of increasing their reward, so the snake agent started to focus on getting to the apple instead of making the least amount of actions.

After deployment of our trained model, we evaluated the results through an evaluation metric table:  

![Evaluation Metrics](/images/eval_metrics.jpg) 

The reward representing the total reward of the episode, the length showing the total timesteps taken in the episode, the score showing the number of apples collected in the episode, and the average reward per step taken all show how our snake was able to maneuver through the environment and successfully collect multiple apples for the most part while remaining alive by avoiding boundaries and running into itself. The total average of these results is shown:  

![Summary of Evaluation Metrics](/images/summary_eval_metrics.jpg) 

Using these results, we could see that the snake agent had successfully learned how to collect apples optimally by using the holes in its body. We were able to get an average score of around 20, where many of the runs had a score between 10 and 50. However, we still plan to improve these results by hypertuning the reward system. One specific section we plan to hypertune is how we reward points based on the distance between the snake's head and the apple, as currently, there is a set reward given based on if the snake moved closer or farther away. We plan to normalize this distance difference and make this added reward more variable for more accurate training results.

For visual reference, below is one of our best training runs:

<figure>
<video width="320" height="240" controls>
  <source src="./images/eval-episode-0.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
    <figcaption>One of our training runs where the snake agent achieved 50 apples.</figcaption>
</figure>

This clip specifically shows how the snake agent efficiently cuts corners to get to the apple quickly. On top of this, it is shown that the snake has learned that it is able to go through the invisible body tiles of its body, where it learned this through the training process as there is no reward given nor taken for doing his. However, reward is taken for running into the visible parts of the body, which is most likely how the snake agent figured this unique mechanic out.

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

