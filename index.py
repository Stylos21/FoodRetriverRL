from stable_baselines.common import make_vec_env
from SnakeENV import Board
from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy
import pygame
print("a")
# model = DQN(MlpPolicy, env, verbose=1, learning_rate=0.0001).learn(total_timesteps=25000)
# model.save("qwergjtheoigras")
# del model
model = DQN.load("snek")
env = Board(model)
obs = env.reset()
boardInit = False
# while True:
#     if not boardInit:
#         env.init_board()
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     print(obs)
#     env.render()
