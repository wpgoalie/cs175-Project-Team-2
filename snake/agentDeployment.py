import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from rlEnvironment import snakeRLEnvironment

def main():
    env = snakeRLEnvironment(render_mode = "rgb_array")
    model = PPO.load('data/models/ppo_snake')

    obs, info = env.reset()

    n_eval_episodes = 50 
    all_rewards = []
    all_lengths = []
    all_scores = []

    for ep in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, score, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            ep_reward += reward
            ep_length += 1
            env.render() 

        all_scores.append(env.score())
        all_rewards.append(ep_reward)
        all_lengths.append(ep_length)

    print(f"Average Reward over {n_eval_episodes} episodes: {np.mean(all_rewards):.2f}")
    print(f"Average Episode Length over {n_eval_episodes} episodes: {np.mean(all_lengths):.1f}")

    env.close()


if __name__ == '__main__':
    main()