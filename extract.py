
import numpy as np
import random
import os
import gym

from model import make_model
from PIL import Image
import argparse



def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for stochastic games")
    parser.add_argument("--agent", type =str, default = 'DDPG', help = "agent type")
    parser.add_argument("--adv_agent", type = str, default = "script", help = "adv agent type")
    parser.add_argument("--game", type=str, default="Pong-2p-v0", help="name of the  env")
    parser.add_argument("--timestep", type= int, default= 2, help ="the time step of act_trajectory")
    parser.add_argument("--iteration", type=int, default=60000, help="number of episodes")
    parser.add_argument("--seed", type = int, default = 10, help = "random seed")
    parser.add_argument("--epoch", type = int, default = 10, help = "training epoch")
    parser.add_argument("--episodes", type = int, default = 100, help = "episodes")
    parser.add_argument("--steps", type = int, default = 50, help ="steps in one episode" )
    parser.add_argument("--batch_size", type = int, default = 64, help = "set the batch_size")
    parser.add_argument("--max_episode_len", type = int, default = 50, help = "max episode length")
    parser.add_argument("--warm_up_steps", type = int, default = 1000, help = "set the warm up steps")
    parser.add_argument("--lr", type = float, default = 0.0001, help = "learning rate")
    parser.add_argument("--gamma", type = float, default = 0.99, help = "discount rate")
    parser.add_argument("--kl_tolerance", type = float, default = 0.5, help = "dk divergence tolerance")
    parser.add_argument("--data_dir", type = str,default = "./record")
    parser.add_argument("--model_save_path", type = str,default = "./tf_rnn/", help= "model save path")
    parser.add_argument("--z_size", type = int, default = 32, help = "z size")
    parser.add_argument("--initial_z_save_path", type = str, default = "tf_initial_z", help = "intial_z")
    parser.add_argument("--render_mode",type = bool, default = False, help = "render or not")
    parser.add_argument("--max_frames", type = int, default = 1000, help = "max frames to store")
    parser.add_argument("--max_trials", type = int, default = 200, help = "use this to extract one trial")
    return parser.parse_args()


if __name__ == '__main__':


    arglist = parse_args()
    if not os.path.exists(arglist.data_dir):
        os.makedirs(arglist.data_dir)

    model = make_model(load_model=False)

    total_frames = 0
    model.make_env(render_mode=arglist.render_mode, full_episode=True)
    for trial in range(arglist.max_trials): # 200 trials per worker
      try:
        random_generated_int = random.randint(0, 2**31-1)
        filename = arglist.data_dir+"/"+str(random_generated_int)+".npz"
        recording_obs = []
        recording_action = []

        np.random.seed(random_generated_int)
        model.env.seed(random_generated_int)

        # random policy
        model.init_random_model_params(stdev=np.random.rand()*0.01)

        model.reset()
        obs = model.env.reset() # pixels

        for frame in range(arglist.max_frames):
          if arglist.render_mode:
            model.env.render("human")
          else:
            model.env.render("rgb_array")
          obs = Image.fromarray(obs)
          obs = obs.resize((64,64),Image.ANTIALIAS)
          recording_obs.append(np.array(obs))

          z, mu, logvar = model.encode_obs(obs)
          action = model.get_action(z)

          recording_action.append(action)
          obs, rewards, [act1, act2], goals, win = model.env.step(action)

          if win:
            break

        total_frames += (frame+1)
        print("dead at", frame+1, "total recorded frames for this worker", total_frames)
        recording_obs = np.array(recording_obs, dtype=np.uint8)
        recording_action = np.array(recording_action, dtype=np.float16)
        np.savez_compressed(filename, obs=recording_obs, action=recording_action)
      except gym.error.Error:
        print("stupid gym error, life goes on")
        model.env.close()
        model.make_env(render_mode=arglist.render_mode)
        continue
    model.env.close()