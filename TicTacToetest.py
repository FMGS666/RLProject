from src.Environments.tictactoe import *




env = raw_env(render_mode="human")


env.step((0, 0)) 
env.step((1, 1)) 
env.step((0, 1))  
env.step((1, 2)) 
env.step((0, 2))  

env.render()
