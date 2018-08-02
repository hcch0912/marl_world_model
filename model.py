import numpy as np
import random

import json
import sys

from two_player.pong import PongGame
import time
import argparse
from PIL import Image
from vae.vae import ConvVAE
from rnn.rnn import hps_sample, MDNRNN, rnn_init_state, rnn_next_state, rnn_output, rnn_output_size



def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for stochastic games")
    parser.add_argument("--agent", type =str, default = 'DDPG', help = "agent type")
    parser.add_argument("--adv_agent", type = str, default = "script", help = "adv agent type")
    parser.add_argument("--game", type=str, default="Pong-2p-v0", help="name of the  env")
    parser.add_argument("--timestep", type= int, default= 2, help ="the time step of act_trajectory")
    parser.add_argument("--iteration", type=int, default=60000, help="number of episodes")
    parser.add_argument("--seed", type = int, default = 10, help = "random seed")
    parser.add_argument("--epoch", type = int, default = 10, help = "training epoch")
    parser.add_argument("--episodes", type = int, default = 100000, help = "episodes")
    parser.add_argument("--steps", type = int, default = 50, help ="steps in one episode" )
    parser.add_argument("--batch_size", type = int, default = 1000, help = "set the batch_size")
    parser.add_argument("--max_episode_len", type = int, default = 50, help = "max episode length")
    parser.add_argument("--warm_up_steps", type = int, default = 1000, help = "set the warm up steps")
    parser.add_argument("--lr", type = float, default = 0.0001, help = "learning rate")
    parser.add_argument("--gamma", type = float, default = 0.99, help = "discount rate")
    parser.add_argument("--kl_tolerance", type = float, default = 0.5, help = "dk divergence tolerance")
    parser.add_argument("--data_dir", type = str,default = "./image_data")
    parser.add_argument("--model_save_path", type = str,default = "./tf_vae", help= "model save path")
    parser.add_argument("--z_size", type = int, default = 32, help = "z size")
    parser.add_argument("--render_mode", type = bool, default = False, help = "render mode")
    parser.add_argument("--use_model", type = bool, default = False, help = "use model")
    parser.add_argument("--model_path", type = str, default = "", help = "load model path")
    parser.add_argument("--recording_mode", type = bool, default = True, help = "training model")
    return parser.parse_args()


render_mode = True

# controls whether we concatenate (z, c, h), etc for features used for car.
MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3 # extra hidden later
MODE_ZH = 4

EXP_MODE = MODE_ZH

def make_model(load_model=True):
  # can be extended in the future.
  model = Model(load_model=load_model)
  return model

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
  return np.maximum(x, 0)

def clip(x, lo=-2, hi=2):
  return np.minimum(np.maximum(x, lo), hi)

def passthru(x):
  return x

def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)

def sample(p):
  return np.argmax(np.random.multinomial(1, p))

class Model:
  ''' simple one layer model for car racing '''
  def __init__(self, load_model=True):
    self.env_name = "Pong-2p-v0"
    self.vae = ConvVAE(batch_size=1, gpu_mode=False, is_training=False, reuse=True)

    self.rnn = MDNRNN(hps_sample, gpu_mode=False, reuse=True)

    if load_model:
      self.vae.load_json('vae/vae.json')
      self.rnn.load_json('rnn/rnn.json')

    self.state = rnn_init_state(self.rnn)
    self.rnn_mode = True

    self.input_size = rnn_output_size(EXP_MODE)
    self.z_size = 32

    if EXP_MODE == MODE_Z_HIDDEN: # one hidden layer
      self.hidden_size = 40
      self.weight_hidden = np.random.randn(self.input_size, self.hidden_size)
      self.bias_hidden = np.random.randn(self.hidden_size)
      self.weight_output = np.random.randn(self.hidden_size, 2)
      self.bias_output = np.random.randn(3)
      self.param_count = ((self.input_size+1)*self.hidden_size) + (self.hidden_size*2+2)
    else:
      self.weight = np.random.randn(self.input_size, 2)
      self.bias = np.random.randn(2)
      self.param_count = (self.input_size)*2+2

    self.render_mode = False

  def make_env(self, seed=-1, render_mode=False, full_episode=False):
    self.render_mode = render_mode
    self.env = PongGame()

  def reset(self):
    self.state = rnn_init_state(self.rnn)

  def encode_obs(self, obs):
    # convert raw obs to z, mu, logvar
    result = np.copy(obs).astype(np.float)/255.0
    result = result.reshape(1, 64, 64, 3)
    mu, logvar = self.vae.encode_mu_logvar(result)
    mu = mu[0]
    logvar = logvar[0]
    s = logvar.shape
    z = mu + np.exp(logvar/2.0) * np.random.randn(*s)
    return z, mu, logvar

  def get_action(self, z):
    h = rnn_output(self.state, z, EXP_MODE)

    '''
    action = np.dot(h, self.weight) + self.bias
    action[0] = np.tanh(action[0])
    action[1] = sigmoid(action[1])
    action[2] = clip(np.tanh(action[2]))
    '''
    if EXP_MODE == MODE_Z_HIDDEN: # one hidden layer
      h = np.tanh(np.dot(h, self.weight_hidden) + self.bias_hidden)
      action = np.tanh(np.dot(h, self.weight_output) + self.bias_output)
    else:
      action = np.tanh(np.dot(h, self.weight) + self.bias)
    

    action[0] = clip(action[0])
    action[1] = clip(action[1])

    self.state = rnn_next_state(self.rnn, z, action, self.state)

    return action

  def set_model_params(self, model_params):
    if EXP_MODE == MODE_Z_HIDDEN: # one hidden layer
      params = np.array(model_params)
      cut_off = (self.input_size+1)*self.hidden_size
      params_1 = params[:cut_off]
      params_2 = params[cut_off:]
      self.bias_hidden = params_1[:self.hidden_size]
      self.weight_hidden = params_1[self.hidden_size:].reshape(self.input_size, self.hidden_size)
      self.bias_output = params_2[:2]
      self.weight_output = params_2[2:].reshape(self.hidden_size, 2)
    else:
      self.bias = np.array(model_params[:2])
      self.weight = np.array(model_params[2:]).reshape(self.input_size, 2)

  def load_model(self, filename):
    with open(filename) as f:    
      data = json.load(f)
    print('loading file %s' % (filename))
    self.data = data
    model_params = np.array(data[0]) # assuming other stuff is in data
    self.set_model_params(model_params)

  def get_random_model_params(self, stdev=0.1):
    #return np.random.randn(self.param_count)*stdev
    return np.random.standard_cauchy(self.param_count)*stdev # spice things up

  def init_random_model_params(self, stdev=0.1):
    params = self.get_random_model_params(stdev=stdev)
    self.set_model_params(params)
    vae_params = self.vae.get_random_model_params(stdev=stdev)
    self.vae.set_model_params(vae_params)
    rnn_params = self.rnn.get_random_model_params(stdev=stdev)
    self.rnn.set_model_params(rnn_params)

def simulate(model, arglist, train_mode=False, render_mode=False, num_episode=5, seed=-1, max_len=-1):

  reward_list = []
  t_list = []

  max_episode_length = 1000

  penalize_turning = False

  if train_mode and max_len > 0:
    max_episode_length = max_len

  if (seed >= 0):
    random.seed(seed)
    np.random.seed(seed)
    model.env.seed(seed)

  for episode in range(num_episode):

    model.reset()

    obs = model.env.reset()
    # obs = Image.fromarray(obs)
    obs = Image.fromarray(obs)
    obs = obs.resize((64,64),Image.ANTIALIAS)
    total_reward = 0.0

    random_generated_int = np.random.randint(2**31-1)

    filename = "./record/"+str(random_generated_int)+".npz"
    recording_mu = []
    recording_logvar = []
    recording_action = []
    recording_reward = [0]

    for t in range(max_episode_length):

      if render_mode:
        model.env.render("human")
      else:
        model.env.render('rgb_array')

      z, mu, logvar = model.encode_obs(obs)
      action = model.get_action(z)

      recording_mu.append(mu)
      recording_logvar.append(logvar)
      recording_action.append(action)

      #train player1 
      obs, rewards, [act1, act2], done, info = model.env.step([action[0], 'script'])

      extra_reward = 0.0 # penalize for turning too frequently
      if train_mode and penalize_turning:
        extra_reward -= np.abs(action[0])/10.0
        rewards[0] += extra_reward

      recording_reward.append(rewards[0])

      #if (render_mode):
      #  print("action", action, "step reward", reward)

      total_reward += rewards[0]

      if done:
        break

    #for recording:
     # obs = Image.fromarray(obs)
    obs = Image.fromarray(obs)
    obs = obs.resize((64,64),Image.ANTIALIAS)
    z, mu, logvar = model.encode_obs(obs)
    action = model.get_action(z)
    recording_mu.append(mu)
    recording_logvar.append(logvar)
    recording_action.append(action)

    recording_mu = np.array(recording_mu, dtype=np.float16)
    recording_logvar = np.array(recording_logvar, dtype=np.float16)
    recording_action = np.array(recording_action, dtype=np.float16)
    recording_reward = np.array(recording_reward, dtype=np.float16)

    if not render_mode:
      if arglist.recording_mode:
        np.savez_compressed(filename, mu=recording_mu, logvar=recording_logvar, action=recording_action, reward=recording_reward)

    if render_mode:
      print("total reward", total_reward, "timesteps", t)
    reward_list.append(total_reward)
    t_list.append(t)

  return reward_list, t_list

def main():
  arglist = parse_args()

  if (arglist.use_model):
    model = make_model()
    print('model size', model.param_count)
    model.make_env(render_mode=arglist.render_mode)
    model.load_model(arglist.model_path)
  else:
    model = make_model(load_model=False)
    print('model size', model.param_count)
    model.make_env(render_mode=arglist.render_mode)
    model.init_random_model_params(stdev=np.random.rand()*0.01)

  N_episode = arglist.episodes
  if arglist.render_mode:
    N_episode = 1
  reward_list = []
  for i in range(N_episode):

    reward, steps_taken = simulate(model, arglist,
      train_mode=False, render_mode=arglist.render_mode, num_episode=1)
    if arglist.render_mode:
      print("terminal reward", reward, "average steps taken", np.mean(steps_taken)+1)
    else:
      # print(reward[0])
      pass
    reward_list.append(reward[0])
  if not arglist.render_mode:
    print("seed", arglist.seed, "average_reward", np.mean(reward_list), "stdev", np.std(reward_list))

if __name__ == "__main__":
  main()