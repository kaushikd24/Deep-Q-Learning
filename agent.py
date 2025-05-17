import random
import torch
import torch.nn as nn
import torch.optim as optiom
import numpy
import matplotlib.pyplot as plt
import torch.nn.functional as F

class DQNAgent:
    def __init__(self, model, optimizer, replay_buffer, num_actions, device, gamma = 0.99, epsilon_start = 1.0, epsilon_end = 0.1, epsilon_decay=1000000):
        self.model = model
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.num_actions = num_actions
        self.device = device
        
        self.gamma = gamma
        
        self.epsilon = epsilon_start
        self.epsilon_end  = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.step_count  = 0
        
    def select_action(self, state):
        self.step_count +=1
        self.epsilon = max(self.epsilon_end, self.epsilon - (1/self.epsilon_decay))
        
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions-1)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state)
            return q_values.argmax(dim=1).item()
        
    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        
    def learn(self, batch_size):
        if len(self.replay_buffer)<batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        #q-val
        q_values = self.model(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.model(next_states)
            target_q = rewards + self.gamma*next_q*(1-dones)
            
        #loss 
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()