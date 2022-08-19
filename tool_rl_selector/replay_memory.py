import random
import numpy as np
import torch
from collections import deque, namedtuple


class ReplayMemory:
    def __init__(self, capacity=50000):       
        self.memory_size = capacity
        self.memory = deque(maxlen=self.memory_size)
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminal', 'action_one'))
        self.position = 0 # replay memory position indicater
        self._available = False

    def push(self, state, action, reward, next_state, terminal, action_one):
        """save transition """
        if len(self.memory) < self.memory_size:
            self.memory.append(None)

        state = torch.FloatTensor(state)
        reward = torch.FloatTensor([reward])
        action = torch.FloatTensor(action)
        action_one = torch.LongTensor(action_one)
        if next_state is not None:
            next_state = torch.FloatTensor(next_state)

        transition = self.Transition(state=state, action=action, reward=reward, next_state=next_state, terminal=terminal, action_one=action_one)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.memory_size
    
    def sample(self, batch_size):
        # batch x transition
        transitions = random.sample(self.memory, batch_size)
        return self.Transition(*(zip(*transitions)))

    def __len__(self):
        return len(self.memory)

    def is_available(self, batch_size):
        if self._available:
            return True

        if len(self.memory) > batch_size:
            self._available = True
        return self._available


class PriReplayMemory:
    def __init__(self, capacity=50000, prob_alpha=0.6):       
        self.memory_size = capacity
        self.memory = deque(maxlen=self.memory_size)
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminal', 'action_one'))
        self.position = 0 # replay memory position indicater
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self._available = False
        self.prob_alpha = prob_alpha
        
    def push(self, state, action, reward, next_state, terminal, action_one):
        """save transition """
        if len(self.memory) < self.memory_size:
            self.memory.append(None)

        state = torch.FloatTensor(state)
        reward = torch.FloatTensor([reward])
        action = torch.FloatTensor(action)
        action_one = torch.LongTensor(action_one)
        if next_state is not None:
            next_state = torch.FloatTensor(next_state)

        transition = self.Transition(state=state, action=action, reward=reward, next_state=next_state, terminal=terminal, action_one=action_one)

        max_prio = self.priorities.max() if len(self.memory)>1 else 1.0
        self.memory[self.position] = transition
        self.priorities[self.position] = max_prio

        self.position = (self.position + 1) % self.memory_size
    
    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.memory_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]

        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        transitions = [self.memory[idx] for idx in indices]
        
        total    = len(self.memory)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)

        return self.Transition(*(zip(*transitions))), indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.memory)

    def is_available(self, batch_size):
        if self._available:
            return True

        if len(self.memory) > batch_size:
            self._available = True
        return self._available
