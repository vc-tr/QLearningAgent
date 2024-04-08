import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
from collections import deque
import time
import random
import os
from PIL import Image
import cv2

# Personalized constants
MY_DISCOUNT = 0.99
MY_REPLAY_MEMORY_SIZE = 50_000
MY_MIN_REPLAY_MEMORY_SIZE = 1_000
MY_MINIBATCH_SIZE = 64
MY_UPDATE_TARGET_EVERY = 5
MY_MODEL_NAME = '2x256_custom'
MY_MIN_REWARD = -200
MY_MEMORY_FRACTION = 0.20
MY_EPISODES = 20_000
MY_EPSILON = 1
MY_EPSILON_DECAY = 0.99975
MY_MIN_EPSILON = 0.001
MY_AGGREGATE_STATS_EVERY = 50
MY_SHOW_PREVIEW = False

# Environment settings
SIZE = 10

class CustomBlob:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        # other choices...

    def move(self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1


class CustomBlobEnv:
    SIZE = 10
    RETURN_IMAGES = True
    MOVE_PENALTY = 1
    ENEMY_PENALTY = 300
    FOOD_REWARD = 25
    ACTION_SPACE_SIZE = 9
    PLAYER_N = 1
    FOOD_N = 2
    ENEMY_N = 3
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}

    def reset(self):
        self.player = CustomBlob(self.SIZE)
        self.food = CustomBlob(self.SIZE)
        self.enemy = CustomBlob(self.SIZE)
        self.episode_step = 0
        observation = np.array(self.get_image())
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)
        new_observation = np.array(self.get_image())
        reward = -self.MOVE_PENALTY
        done = False
        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.food:
            reward = self.FOOD_REWARD
        if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or self.episode_step >= 200:
            done = True
        return new_observation, reward, done

    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]
        img = Image.fromarray(env, 'RGB')
        return img


env = CustomBlobEnv()

# Personalized stats
ep_rewards = [-200]
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

class CustomDQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=MY_REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256, (3, 3), input_shape=env.RETURN_IMAGES))
        # more layers...
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MY_MIN_REPLAY_MEMORY_SIZE:
            return
        minibatch = random.sample(self.replay_memory, MY_MINIBATCH_SIZE)
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)
        X = []
        y = []
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + MY_DISCOUNT * max_future_q
            else:
                new_q = reward
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            X.append(current_state)
            y.append(current_qs)
        self.model.fit(np.array(X)/255, np.array(y), batch_size=MY_MINIBATCH_SIZE, verbose=0, shuffle=False)

        if terminal_state:
            self.target_update_counter += 1
        if self.target_update_counter > MY_UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

agent = CustomDQNAgent()

# Training loop
for episode in range(1, MY_EPISODES + 1):
    agent.tensorboard.step = episode
    episode_reward = 0
    step = 1
    current_state = env.reset()
    done = False
    while not done:
        if np.random.random() > MY_EPSILON:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)
        new_state, reward, done = env.step(action)
        episode_reward += reward
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)
        current_state = new_state
        step += 1
    ep_rewards.append(episode_reward)
    if not episode % MY_AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-MY_AGGREGATE_STATS_EVERY:])/len(ep_rewards[-MY_AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-MY_AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-MY_AGGREGATE_STATS_EVERY:])
        if min_reward >= MY_MIN_REWARD:
            agent.model.save(f'{MY_MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
    if MY_EPSILON > MY_MIN_EPSILON:
        MY_EPSILON *= MY_EPSILON_DECAY
        MY_EPSILON = max(MY_MIN_EPSILON, MY_EPSILON)

