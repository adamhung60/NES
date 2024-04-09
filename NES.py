# Natural Evolution Strategies Algorithm for Reinforcement Learning - Adam Hung
# Sorry if I don't follow coding conventions or my code isn't perfect, I'm not a coder :)

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class NESNet(nn.Module):
    def __init__(self, input_features, output_features, h1_nodes, h2_nodes, state_dict = None):
        super().__init__()
        self.fc1 = nn.Linear(input_features, h1_nodes)   
        self.fc2 = nn.Linear(h1_nodes, h2_nodes)  
        self.out = nn.Linear(h2_nodes, output_features)  
        if state_dict:
            self.load_state_dict(state_dict)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)   
        return x

class NES(nn.Module):
    def __init__(self, env, lamb=1, learning_rate = 0.001, stddev = 0.01, h1_nodes = 64, h2_nodes = 64, state_dict_path = None, rseed = None):
        super().__init__()

        # Set random seeds, if desired for reproducibility
        if rseed:
            np.random.seed(rseed)
            torch.manual_seed(rseed)   

        # Gym Env params
        self.env = env

        # Initialize Network
        self.h1_nodes = h1_nodes
        self.h2_nodes = h2_nodes
        self.input_features = env.observation_space.shape[0]
        self.output_features = env.action_space.shape[0]
        # If an existing neural net is supplied, load it into our model. Otherwise, a new one is generated
        if state_dict_path: state_dict = torch.load(state_dict_path)
        else: state_dict = None
        self.net = NESNet(input_features = self.input_features, 
                         output_features = self.output_features, 
                         h1_nodes = self.h1_nodes, 
                         h2_nodes = self.h2_nodes,
                         state_dict = state_dict)
        # prevents pytorch from performing unnecessary calculations
        for param in self.net.parameters():
            param.requires_grad = False

        # ES params
        self.lamb = lamb
        self.learning_rate = learning_rate
        self.stddev = stddev

        # Progress vars
        self.iterations = 0
        # Just a pointer to self.net, not its own network object, so be adjusting its values
        self.best_net = self.net
        self.best_eval = None
    
    # Trains model by iterating over generations and updating policy
    def learn(self, iterations = 1, evaluation_episodes = 1):
        # again just make sure no gradient calculations are being made
        with torch.no_grad():
            parent = self.net
            while self.iterations < iterations:
                self.iterations += 1
                # generate new population
                population = self.reproduce(parent)
                # evaluate everyone
                evaluations = [self.evaluate(child, evaluation_episodes) for child in population]
                # print some useful info
                print("iteration number: ", self.iterations, ", median reward of current population: " , np.median(evaluations)/evaluation_episodes)
                # normalize evaluations to keep update step sizes more consistent
                normalized_evaluations = (evaluations - np.mean(evaluations)) / (np.std(evaluations) + 1e-8)
                # update parameters. param points to actual parameter object, so updates to param affect the real object
                for name, param in parent.named_parameters():   
                    update = torch.zeros_like(param)
                    for index, child in enumerate(population):
                        child_param = child.state_dict()[name]
                        # recalculate the noise we applied to each child (makes things simpler, but not very efficient)
                        noise = child_param - param 
                        # for each data point, increment our weighted sum by (direction perturbed) * (how well this performed)
                        update += noise * normalized_evaluations[index]
                    # scale this by our learning rate and apply it to the parameters to create the parent of the next generation
                    param += self.learning_rate * update

            self.env.close()

    # Evaluate a policy in the environment by returning the reward gathered over evaluation_episodes
    def evaluate(self, child, evaluation_episodes):
        evaluation = 0
        for _ in range(evaluation_episodes):
            state = self.env.reset()[0]
            state = torch.tensor(state, dtype=torch.float32) 
            terminated = False
            truncated = False
            while(not terminated and not truncated):
                action = child.forward(state)
                state, reward, terminated, truncated, _ = self.env.step(action.detach().cpu().numpy())
                state = torch.tensor(state, dtype=torch.float32)
                evaluation += reward
        # check if any members of the population exceeded the best policy we've seen so far
        if self.best_eval is None or evaluation > self.best_eval:
            self.best_eval = evaluation
            self.best_net = child
            print("New best eval: ", self.best_eval/evaluation_episodes)
            self.save()
        return evaluation
    
    # Returns a new population of policies generated randomly around a parent
    def reproduce(self, parent):
        population = list()
        while len(population) < self.lamb:
            # we copy over all of the parameters of the parent, but apply some noise to each one as we go
            child_params = {name: param + torch.randn(param.size()) * self.stddev for name, param in parent.named_parameters()}
            child = NESNet(input_features=self.input_features, output_features=self.output_features, 
                            h1_nodes=self.h1_nodes, h2_nodes=self.h2_nodes, state_dict=child_params)
            population.extend([child])
        return population

    def save(self, path = "net.pt"):
        torch.save(self.best_net.state_dict(), path)
    
    # Primarily for visualization of the policy
    def test(self, episodes = 1, path = "net.pt"):

        self.net = NESNet(input_features = self.input_features, 
                         output_features = self.output_features, 
                         h1_nodes = self.h1_nodes, 
                         h2_nodes = self.h2_nodes)
        self.net.load_state_dict(torch.load(path))

        for _ in range(episodes):
            state = self.env.reset()[0]
            state = torch.tensor(state, dtype=torch.float32) 
            terminated = False
            truncated = False
            while(not terminated and not truncated):
                action = self.net.forward(state)
                state, reward, terminated, truncated, _ = self.env.step(action.detach().cpu().numpy())
                state = torch.tensor(state, dtype=torch.float32)
        
        self.env.close()