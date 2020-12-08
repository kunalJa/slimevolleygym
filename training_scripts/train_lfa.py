import gym
import slimevolleygym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


ɣ = 1
target_update_freq = 1
N_STEPS = 1000

MAX_EPISODES = 500

SEED = 2020
LR_BEGIN = 0.01
LR_END = 0.01

EPS_BEGIN = 0.1
EPS_END = 0.05


# env = gym.make("SlimeVolley-v0")
env = gym.make("CartPole-v1")
env.seed(SEED)

# action_space = {0: [1,0,0], 1: [0,1,0], 2: [0,0,1],
#                 3: [1,0,1], 4: [0,1,1], 5: [0,0,0]}

# action_space = {0: [0, 1], 1: [1, 0]}

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


if __name__ == "__main__":
    # w_π = np.random.rand(4, 2) # original policy we are using
    # w = np.random.rand(4,2) # shit there are way more than 3 actions...

    w_π = np.zeros((4,2))
    w = np.zeros((4,2))
    w_target = np.random.rand(4,2)
    t = 0
    𝛼 = LR_BEGIN
    ε = EPS_BEGIN

    replay_buffer = deque(maxlen=50)

    episodes = 0
    plot_episodes = []
    plot_rewards = []
    for episode in range(MAX_EPISODES):
        state_current = env.reset()
        total_reward = 0
        done = False
        while not done:
            t += 1

            action = get_action_from_policy(w_π, state_current, ε)
            state_next, reward, done, info = env.step(action)

            replay_buffer.append(state_current)
            state_exp = replay_buffer[np.random.randint(0, high=len(replay_buffer))]

            state_exp = np.reshape(state_exp, (4, 1))

            q_values = np.matmul(w.T, state_exp)
            q_values_next = np.matmul(w.T, state_next)

            target = q_values.copy()

            if not done:
                target[action] = reward + ɣ*np.max(q_values_next)
            else:
                target[action] = reward

            loss = target - q_values

            loss = np.reshape(loss, (1, 2))
            state_current = np.reshape(state_current, (4,1))

            gradient = 𝛼 * np.matmul(state_exp, loss)

            norm = np.linalg.norm(gradient)

            if norm > 10:
                gradient *= 10 / norm;

            # print(w)
            w += gradient
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
