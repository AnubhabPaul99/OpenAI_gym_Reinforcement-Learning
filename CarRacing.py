import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os

ppo_path = os.path.join('Training', 'Saved Models', 'PPO_Driving_model')



#env = gym.make('CarRacing-v0')
env = DummyVecEnv([lambda:gym.make('CarRacing-v0')])

model = PPO.load(ppo_path, env=env)
score = evaluate_policy(model, env, n_eval_episodes=5, render=True,deterministic=False)
print(score)
#obs = env.reset()
#frames = 500
#model.learn(total_timesteps=1000)
# while 1:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
    #frames -= 1
# for i in range(10):
#     obs = env.reset()
#     done = False
#     score = 0
#     action = env.action_space.sample()
#     while not done:
#         env.render()
#
#         obs, reward, done, info = env.step(action)
#         score += reward
#         print('episode: {}   score: {}'.format(i+1, score))
env.close()
