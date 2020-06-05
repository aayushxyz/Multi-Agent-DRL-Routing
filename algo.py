import networkx as nx
from env import RoutingEnv
from agent import Agent
import numpy as np
import torch

# hyperparameters
hidden_size = 256
learning_rate = 3e-4

# Constants
GAMMA = 0.90
num_steps = 300
max_episodes = 100
packets = 100

"""
This is the graph to be used for simulation
"""
G = nx.Graph()
edges = [(9, 8), (8, 7), (9, 6), (8, 5), (6, 5), (7, 4),
         (5, 4), (6, 3), (5, 2), (3, 2), (4, 1), (2, 1)]
G.add_edges_from(edges)
#nx.draw_spectral(G, with_labels=True)
# plt.show(G)

env = RoutingEnv(G)

all_rewards = []
all_lenghts = []

for episode in range(max_episodes):

    #print("Starting the episode ", episode, sep=" ")
    env.reset()
    #env.render()

    done = False
    while not done:
        for node in G.nodes():
            observation, emptyQ, packet = env.getState(node)
            if emptyQ:
                continue

            value, policy_dist = env.agents[node].forward(observation)
            dist = policy_dist.detach().numpy()

            action = np.random.choice(len(env.neighbours[node]), p=np.squeeze(dist))

            env.step(node,action,packet,observation)

        done = env.run()
        #env.render()

    rew = 0
    episode_length = 0
    for node in G.nodes():
        #update actor critic
        try:
            rew += sum(env.rewards[node])
            episode_length += len(env.rewards[node])
            values = torch.FloatTensor(env.critic_value[node])
            Qvals = torch.FloatTensor(env.rewards[node])
            log_probs = torch.stack(env.log_probs[node])

            advantage = Qvals - values
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = 0.5 * advantage.pow(2).mean()
            loss = actor_loss + critic_loss + 0.001 * env.entropy_term[node]

            env.optimizers[node].zero_grad()
            loss.backward()
            env.optimizers[node].step()
        except:
            print()

    all_rewards.append(rew)
    all_lenghts.append(episode_length)

print("Rewards: ", all_rewards)
print("Lengths: ", all_lenghts)

import matplotlib.pyplot as plt
plt.plot(range(len(all_rewards)-1), all_rewards[1:])
plt.show()
