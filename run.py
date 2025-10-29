from DQN import DDQNAgent, DQNAgent, RepeatActionAndMaxFrame, StackFrames, PreprocessFrame
import collections
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import ale_py
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

def make_env(env_name, shape=(84,84,1), repeat=4, clip_rewards=False,
             no_ops=0, fire_first=False):
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)

    return env

###################
# Notebook Code
###################
env = make_env('ALE/Pong-v5')
best_score = -np.inf
load_checkpoint = False
n_games = 10

agent = DDQNAgent(gamma=0.99, epsilon=1, lr=0.0001,
                 input_dims=(env.observation_space.shape),
                 n_actions=env.action_space.n, mem_size=20000, eps_min=0.1,
                 batch_size=32, replace=1000, eps_dec=1e-5,
                 chkpt_dir='models/', algo='DDQNAgent',
                 env_name='Pong')

if load_checkpoint:
  agent.load_models()

fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
            + str(n_games) + 'games'
figure_file = 'plots/' + fname + '.png'
n_steps = 0
scores, eps_history, steps_array = [], [], []

for i in tqdm(range(n_games)):
  observation, info = env.reset()

  score = 0
  done = False
  while not done:
    action = agent.choose_action(observation)
    observation_, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    score += reward

    if not load_checkpoint:
      agent.store_transition(observation, action,
                                     reward, observation_, done)
      agent.learn()
    observation = observation_
    n_steps += 1
  scores.append(score)
  steps_array.append(n_steps)

  avg_score = np.mean(scores[-100:])
  print('episode: ', i,'score: ', score,
             ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
            'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

  if avg_score > best_score:
    if not load_checkpoint:
      agent.save_models()
    best_score = avg_score

  eps_history.append(agent.epsilon)

x = [i+1 for i in range(len(scores))]
plot_learning_curve(steps_array, scores, eps_history, figure_file)
