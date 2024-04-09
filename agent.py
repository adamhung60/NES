# A script to demonstrate training an RL agent with my Evolution Strategies implementation - Adam Hung

import gymnasium as gym
from NES import NES

# Try me with your favorite benchmark or custom environment! Just remember to tune your hyperparameters.
env_id = "InvertedPendulum-v4"
def train(its):
    env = gym.make(env_id, render_mode="rgb_array")

    # These hyperparameters are very important
    # I usually try to use the largest stddev and learning_rate possible that don't destabilize training
    # Fine tuning can always be done with smaller lambda and stddev by rerunning the program
    # lambda is maybe less important, but shouldn't be unnecessarily large
    model = NES(env, 
               lamb = 1000, 
               learning_rate = 0.005,
               stddev = 0.1,
               #state_dict_path = "ant_net.pt"
               )
    model.learn(iterations = its, evaluation_episodes = 1)

def test(eps):
    env = gym.make(env_id, render_mode="human")
    model = NES(env = env) 
    model.test(eps, path = "net.pt")

if __name__ == '__main__':
    #train(100)
    test(10)