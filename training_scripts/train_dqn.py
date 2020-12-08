import gym
import slimevolleygym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

SEED = 2020

class DQN():
    def __init__(self, state_dim, action_dim, hidden_dim=64, learning_rate=0.05):
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
                        torch.nn.Linear(state_dim, hidden_dim),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(hidden_dim, hidden_dim*2),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(hidden_dim*2, action_dim)
                    )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def predict(self, state):
        with torch.no_grad():
            return self.model(torch.Tensor(state))

    def update(self, state, y):
        y_pred = self.model(torch.Tensor(state))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # loss = self.criterion(prediction, Variable(target, requires_grad=True))
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

    def print_weights(self):
        for param in self.model.parameters():
            print(param.data)

ɣ = 1
MAX_EPISODES = 200
N_STEPS = 1000

LR_BEGIN = 0.01
LR_END = 0.01

EPS_BEGIN = 0.1
EPS_END = 0.05

env = gym.make("CartPole-v1")
env.seed(SEED)

def get_action_from_policy(model, state, t):
    if t <= N_STEPS:
        ε = EPS_BEGIN - t*(EPS_BEGIN - EPS_END)/N_STEPS
    else:
        ε = EPS_END

    if np.random.uniform() <= ε:
        action = env.action_space.sample()
    else:
        action = torch.argmax(model.predict(state)).item()

    return action


if __name__ == "__main__":
    replay_buffer = deque(maxlen=50)

    t = 0
    plot_episodes = []
    plot_rewards = []

    print(env.observation_space.shape[0])
    model = DQN(env.observation_space.shape[0], env.action_space.n, 50, 0.001)

    for episode in range(MAX_EPISODES):
        current_state = env.reset()

        total_reward = 0
        done = False

        while not done:            
            action = get_action_from_policy(model, current_state, t)

            next_state, reward, done, info = env.step(action)

            replay_buffer.append(current_state)
            replay_state = replay_buffer[np.random.randint(0, high=len(replay_buffer))]


            q_values = model.predict(current_state)
            q_values_next = model.predict(next_state)

            target = q_values.tolist()
            target[action] = reward + ɣ*torch.max(q_values_next).item()

            if done:
                target[action] = reward
                model.update(current_state, target)
                break

            model.update(current_state, target)

            current_state = next_state
            total_reward += reward

            t += 1
            env.render()

        print("Episode number: " + str(episode) + "; Total Reward: " + str(total_reward) + "; t: " + str(t))
        plot_rewards.append(total_reward)
        plot_episodes.append(episode)

    env.close()
    plt.plot(plot_episodes, plot_rewards)
    plt.show()
