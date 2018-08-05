from two_player.pong import PongGame
from prey_predator.env import PreyPredatorEnv



def make_env(env_name,competitive,  seed=-1, render_mode=False, full_episode=False):
    if env_name == "Pong-2p-v0":
      return PongGame(competitive = competitive)
    if env_name == "prey_predator":
      return PreyPredatorEnv()  
