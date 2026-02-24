import gymnasium as gym
from stable_baselines3 import PPO
from rlEnvironment import snakeRLEnvironment

def main():
    newEnvironment = snakeRLEnvironment(render_mode = "human")
    model = PPO.load('ppo_snake')

    obs, info = newEnvironment.reset()

    while True:
        action, _states = model.predict(obs)
    
        observation, reward, terminated, truncated, info = newEnvironment.step(action.item())
        newEnvironment.render()

if __name__ == '__main__':
    # set_environment = snakeRLEnvironment()
    # obs, info = set_environment.reset()
    
    # for i in range(50):
    #     current_action = set_environment.action_space.sample()
    #     print("CURRENT ACTION", current_action)
    #     observation, reward, terminated, truncated, info = set_environment.step(current_action)
    #     print("REWARD", reward)

    #     if terminated:
    #         print("RESET")
    #         obs, info = set_environment.reset()
    main()