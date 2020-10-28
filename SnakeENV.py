import gym
import numpy as np
import pygame
import random
from math import sqrt

SIZE = 500
BLOB_SIZE = 25


class Blob:
    def __init__(self, x, y, color, size=(50, 50)):
        self.x = x
        self.y = y
        self.size = size
        self.color = color

    def get_distance(self, blob):
        return sqrt((blob.x - self.x) ** 2 + (blob.y - self.y)**2)

    def check_for_collision(self, blob):
        if self.get_distance(blob) < 35:
            return True
        else:
            return False


class Player(Blob):
    def __init__(self, board, x=0, y=0, color=(0, 255, 0)):
        super(Player, self).__init__(x, y, color)
        self.score = 0
        self.board = board
        self.state = False

    def move(self):
        if self.state == 0 and self.x > 0:
            self.x -= 10
        elif self.state == 1 and self.x < 500:
            self.x += 10
        elif self.state == 2 and self.y > 0:
            self.y -= 10
        elif self.state == 3 and self.y < 500:
            self.y += 10

        self.board.render()


class Food(Blob):
    def __init__(self, x=random.randint(BLOB_SIZE, SIZE - BLOB_SIZE), y=random.randint(BLOB_SIZE, SIZE - BLOB_SIZE),
                 color=(255, 0, 0)):
        super(Food, self).__init__(x, y, color)

    def reinit_position(self):
        self.x = random.randint(BLOB_SIZE, SIZE - BLOB_SIZE)
        self.y = random.randint(BLOB_SIZE,  SIZE - BLOB_SIZE)


class Board(gym.Env):
    metadata = {'render.modes': ["console"]}

    def __init__(self, model):
        self.dimensions = (SIZE, SIZE)
        self.player = Player(self)
        self.food = Food()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=500, shape=(5,), dtype=np.float32)
        self.model = model
        pygame.init()
        # pygame.display.update()
        pygame.display.set_caption("Snake AI using Reinforcement Learning")
        self.board = pygame.display.set_mode((500, 500))
        notDone = False
        obs = self.reset()

        while not notDone:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    notDone = True
                    break
            action, _states = model.predict(obs)
            obs, rewards, dones, info = self.step(action)
            print(obs)

    def step(self, action):
        done = False
        reward = 0
        if action == 0:
            self.player.state = 0
            self.player.move()

        elif action == 1:
            self.player.state = 1
            self.player.move()

        elif action == 2:
            self.player.state = 2
            self.player.move()

        elif action == 3:
            self.player.state = 3
            self.player.move()

        if self.player.check_for_collision(self.food):
            self.player.score += 1
            reward = 10
            done = True
            self.food.reinit_position()
            self.reset()
            self.render()
            done = False

        elif 50 > self.player.get_distance(self.food) > 25:
            reward = 5

        elif 150 > self.player.get_distance(self.food) > 75:
            reward = 1

        if self.food.x - self.player.x > 0 and action == 1 or self.food.x - self.player.x < 0 and action == 0 or self.food.y - self.player.y > 0 and action == 3 or self.food.y - self.player.y < 0 and action == 2:
            reward += 3
        else:
            reward -= 3

        info = {"score": self.player.score}
        # print(self.player.x, self.player.y, self.food.x, self.food.y, f"Reward: {reward} Score: {self.player.score}")
        return np.array([self.player.x, self.player.y, self.food.x, self.food.y, self.player.get_distance(self.food)]).astype(np.float32), reward, done, info

    def render(self, mode="human"):
        # self.update_score()
        self.board.fill((0, 0, 0))
        pygame.draw.rect(self.board, self.player.color, [self.player.x, self.player.y, BLOB_SIZE, BLOB_SIZE])
        pygame.display.update()
        pygame.draw.rect(self.board, self.food.color, [self.food.x, self.food.y, BLOB_SIZE, BLOB_SIZE])
        pygame.display.update()
        font = pygame.font.SysFont("Comic Sans MS", 24)
        surface = font.render(f"Score: {self.player.score}", True, (255, 255, 255))
        self.board.blit(surface, (250, 250))
        pygame.display.update()

    def reset(self):
        # self.player.score = 0
        self.player.x = 0
        self.player.y = 0
        self.render()
        return np.array([self.player.x, self.player.y, self.food.x, self.food.y, self.player.get_distance(self.food)])
