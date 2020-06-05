import random
from collections import deque
from networkx.algorithms.shortest_paths.unweighted import single_source_shortest_path as sssp
from agent import Agent
import numpy as np
import torch

GAMMA = 0.90

class Packet():
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination
        self.hops = 0
        self.states = []
        self.queuetime = []
        self.nodes = [source]
        self.actions = []
        self.rewards = []

class RoutingEnv():
    def __init__(self, graph):
        """
        It initializes the environemt for the routing agents.
        The functionalities include:
        1. Read the graph and set the nodes and links.
        2. Calculate the shortest paths between nodes.

        The state contains:
            - Packet Destination
            - Number of packets
            - Number of packets in neighbours queue
            - Previous 3 actions
        """
        self.packets = 100
        self.transmitreward = self.packets/2
        self.learning_rate = 3e-4

        self.graph = graph
        self.neighbours = {}
        for node in self.graph.nodes():
            self.neighbours[node] =  [n for n in self.graph.neighbors(node)]

        #Initialize nodes queue, channels and agents
        self.queues = {}
        self.channels = {}
        self.agents = {}
        self.previousActions = {}
        self.optimizers = {}
        self.entropy_term = {}
        self.log_probs = {}
        self.critic_value = {}
        self.rewards = {}

        for node in self.graph.nodes():
            neighbors = len(list(self.graph.neighbors(node)))
            inputs = 5 + neighbors
            self.agents[node] = Agent(inputs, neighbors)
            self.optimizers[node] = torch.optim.Adam(self.agents[node].parameters(),lr=self.learning_rate)


        #Calculate forwarding table
        self.forwardingTable = {}
        for node in self.graph.nodes():
            self.forwardingTable[node] = {}
            shortest_p = sssp(self.graph, node)
            for other_node in shortest_p:
                if other_node != node:
                    self.forwardingTable[node][other_node] = shortest_p[other_node][1]


    def reset(self):
        """
        It resets the environment.
        The functionalities include:
        1. Inserting the initial 100 packets in the queue.

        """
        self.queues = {}
        self.channels = {}
        self.previousActions = {}
        self.entropy_term = {}
        self.log_probs = {}
        self.critic_value = {}
        self.rewards = {}

        for node in self.graph.nodes():
            self.queues[node] = deque()
            self.channels[node] = {}
            self.entropy_term[node] = 0
            self.log_probs[node] = []
            self.critic_value[node] = []
            self.rewards[node] = []

            for n in self.graph.neighbors(node):
                self.channels[node][n] = deque()

        nodes = [n for n in self.graph.nodes()]
        while self.packets > 0:
            source = random.choice(nodes)
            destination = random.choice(nodes)
            if source == destination:
                continue
            pkt = Packet(source, destination)
            pkt.queuetime.append(len(self.queues[source]))
            self.queues[source].append(pkt)
            self.packets = self.packets - 1

        for node in nodes:
            ngbrs = list(self.graph.neighbors(node))
            ngbrs = ngbrs*2
            self.previousActions[node] = ngbrs[:3]

        self.packets = 50


    def step(self, node, action, packet, observation):
        """
        It fowards the packet in the link.
        The functionalities include:
        1. Forwarding the packet to the destination via action.
        2. Updates the information in the packet.
        3. If packet reaches the destination, then updates replay buffers.
        """
        packet.actions.append(action)
        action = self.neighbours[node][action]

        """
        This is for the forwarding forwardingTable
        """
        #action = self.forwardingTable[node][packet.destination]


        packet.states.append(observation)
        packet.nodes.append(action)
        packet.hops = packet.hops + 1
        self.previousActions[node].append(action)

        if action == packet.destination:

            reward = np.zeros(packet.hops)
            values = np.zeros(packet.hops)
            policy = [[]]*packet.hops

            reward[packet.hops - 1] = int(-1*self.transmitreward)
            for t in reversed(range(packet.hops-1)):
                reward[t] = int(reward[t+1]*GAMMA - self.transmitreward - packet.queuetime[t+1])
            for i in range(packet.hops):
                a,policy[i] = self.agents[packet.nodes[i]].forward(packet.states[i])
                values[i] = a.detach().numpy()[0,0]
            for i in range(packet.hops-1):
                self.rewards[packet.nodes[i]].append(reward[i])
                self.critic_value[packet.nodes[i]].append(values[i])
                log_prob = torch.log(policy[i].squeeze(0)[packet.actions[i]])
                self.log_probs[packet.nodes[i]].append(log_prob)
                entropy = -np.sum(np.mean(policy[i].detach().numpy()) * np.log(policy[i].detach().numpy()))
                self.entropy_term[packet.nodes[i]] += entropy

        else:
            self.channels[node][action].append(packet)



    def run(self):
        """
        Send the packets from channel to queue
        """
        done = True
        for node in self.graph.nodes():
            for n in self.graph.neighbors(node):
                try:
                    packet = self.channels[node][n].popleft()
                    packet.queuetime.append(len(self.queues[n]))
                    self.queues[n].append(packet)
                    done = False
                except:
                    pass
        return done

    def render(self):
        """
        It prints the information on the screen, as needed.
        """
        for node in self.graph.nodes():
            print(len(self.queues[node]), end=" ")
        print("\n")

    def getState(self, node):
        """
        Returns the state of the node, containing:
            - Packet Destination
            - Number of packets
            - Number of packets in neighbours queue
            - Previous 3 actions
        """
        try:
            packet = self.queues[node].popleft()
        except:
            return {}, True, None
        neighborLengths = [len(self.queues[n]) for n in self.graph.neighbors(node)]
        state = [packet.destination, len(self.queues[node]), *neighborLengths, *self.previousActions[node][-3:]]
        state = np.array(state)
        return state, False, packet
