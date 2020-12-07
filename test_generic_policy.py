import gym
import slimevolleygym
from slimevolleygym.randomplay import Model

env = gym.make("SlimeVolley-v0")

obs = env.reset()
done = False
total_reward = 0

policy_right = Model()

while not done:
  action = policy_right.predict(obs)
  obs, reward, done, info = env.step(action)
  total_reward += reward
  env.render()

print("score:", total_reward)
