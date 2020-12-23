import gym
from gym import wrappers
from datetime import datetime
from Agent import Agent

env = gym.make('CartPole-v1')

agent = Agent(env)


def play():
    observation = env.reset()
    for _ in range(1000):
        action = agent.act(observation)
        observation, _, done, _ = env.step(action)
        # update the model
        reward = -1*((observation[2]*100)**2)
        agent.update_policy(observation, action, reward)
        if done:
            break


# Train agent
for _ in range(5000):
    play()

# Play test round
env = wrappers.Monitor(env, './../../Downloads/gym/CartPole/'+str(datetime.now()))
play()

env.close()