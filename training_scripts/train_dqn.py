import gym
import slimevolleygym
import numpy as np

from collections import deque


ɣ = 0.5
target_update_freq = 1000
N_STEPS = 100000
SEED = 2020
LR_BEGIN = 1
EPS_BEGIN = 1
LR_END = 0.1
EPS_END = 0.1


env = gym.make("SlimeVolley-v0")
env.seed(SEED)



def get_action_from_policy(policy_weights, state, ε):
    if np.random.uniform() >= ε:
        i = np.argmax(np.matmul(policy_weights.T, state))
    else:
        i = np.random.randint(0,3)
    out = np.zeros(3)
    out[i] = 1
    return out



if __name__ == "__main__":
    w_π = np.zeros((12, 3)) # original policy we are using
    w = np.zeros((12,3)) # shit there are way more than 3 actions...
    w_target = np.zeros((12,3))
    t = 0
    𝛼 = LR_BEGIN
    ε = EPS_BEGIN

    replay_buffer = deque(maxlen=50)

    while t < N_STEPS:
        state_current = env.reset()
        w_π = w.copy()
        done = False
        while not done:
            action = get_action_from_policy(w_π, state_current, ε)
            state_next, reward, done, info = env.step(action)

            replay_buffer.append(state_current)
            state_exp = replay_buffer[np.random.randint(0, high=len(replay_buffer))]


            td_target_q = reward + ɣ*np.amax(np.matmul(w_target.T, state_next))
            loss = td_target_q - np.matmul(w.T, state_exp)
            loss = np.reshape(loss, (1,3))
            state_exp = np.reshape(state_exp, (12,1))
            gradient = np.matmul(state_exp, 𝛼 * loss)
            # clipped_gradient = np.clip(gradient, -10, 10)
            norm = np.linalg.norm(gradient)
            if norm > 10:
                gradient *= 10/norm
            w += gradient


            t += 1
            if t % target_update_freq == 0:
                w_target = w.copy()

            ε -= (EPS_BEGIN - EPS_END)/N_STEPS # Stanford uses linear annealment for 1M steps
            𝛼 -= (LR_BEGIN - LR_END)/N_STEPS

            if t == N_STEPS:
                break

            if t % 3 == 0:
                print(w.tolist())
                print()
                env.render()
