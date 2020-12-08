import gym
import slimevolleygym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

ɣ = 1
target_update_freq = 1
N_STEPS = 1000

MAX_EPISODES = 1000

SEED = 2020
LR_BEGIN = 0.01
LR_END = 0.01

EPS_BEGIN = 0.1
EPS_END = 0.05

env = gym.make("CartPole-v1")
env.seed(SEED)

def get_action_from_policy(policy_weights, state, ε):
    if t <= N_STEPS:
        ε = EPS_BEGIN - t*(EPS_BEGIN - EPS_END)/N_STEPS
    else:
        ε = EPS_END

    if np.random.uniform() >= ε:
        i = np.argmax(np.matmul(policy_weights.T, state))
    else:
        i = np.random.randint(0,2)
    out = i
    return out

class ReplayBuffer():
    def __init__(self, len):
        self.replay_buffer = deque(maxlen=len)

    def push(self, state, reward, done, info):
        self.replay_buffer.append((state, reward, done, info))

    def get_gradient(self, observation, state_next, w):
        state_exp, reward, done, info = observation
        q_values = np.matmul(w.T, state_exp)
        q_values_next = np.matmul(w.T, state_next)

        state_exp = np.reshape(state_exp, (4,1))

        target = q_values.copy()
        if not done:
            target[action] = reward + ɣ*np.max(q_values_next)
        else:
            target[action] = reward

        loss = target - q_values
        loss = np.reshape(loss, (1, 2))
        gradient = 𝛼 * np.matmul(state_exp, loss)

        norm = np.linalg.norm(gradient)
        if norm > 10:
            gradient *= 10 / norm;

        return gradient

    def replay(self, size, state_next, w):
        length = len(self.replay_buffer)

        if length < size:
            batch = random.sample(list(self.replay_buffer), int(len(self.replay_buffer)/2))
        else:
            batch = random.sample(list(self.replay_buffer), size)

        if len(batch) == 0:
            gradient = self.get_gradient(self.replay_buffer[-1], state_next, w)
            w += gradient
        else:
            for obs in batch:
                gradient = self.get_gradient(obs, state_next, w)
                w += gradient

        return w

    def clear(self):
        self.replay_buffer.clear()


if __name__ == "__main__":
    w_π = np.zeros((4,2))
    w = np.zeros((4,2))
    w_target = np.random.rand(4,2)
    t = 0
    𝛼 = LR_BEGIN
    ε = EPS_BEGIN

    replay_buffer = ReplayBuffer(50)

    episodes = 0
    plot_episodes = []
    plot_rewards = []
    for episode in range(MAX_EPISODES):
        state_current = env.reset()
        total_reward = 0
        done = False

        # replay_buffer.clear()
        while not done:
            t += 1

            action = get_action_from_policy(w_π, state_current, ε)
            state_next, reward, done, info = env.step(action)

            replay_buffer.push(state_current, reward, done, info)
            w = replay_buffer.replay(20, state_next, w)

            w_π = w.copy()

            state_current = state_next
            total_reward += reward

            env.render()

        print("Episode number: " + str(episode) + "; Total Reward: " + str(total_reward) + "; t: " + str(t))
        plot_rewards.append(total_reward)
        plot_episodes.append(episode)

    env.close()
    plt.plot(plot_episodes, plot_rewards)
    plt.show()
