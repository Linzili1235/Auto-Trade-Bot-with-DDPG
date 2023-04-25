import numpy as np
import torch


# REPLAY BUFFER:{obs, act, obs_, reward, done}
class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e4)):
        self.max_size = max_size
        self.cur = 0
        self.size = 0

        self.states = np.zeros((max_size, state_dim))
        self.actions = np.zeros((max_size, action_dim))
        self.next_states = np.zeros((max_size, state_dim))
        self.rewards = np.zeros((max_size, 1))
        self.dones = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    # STORE THE DATA
    def add(self, state, action, next_state, reward, done):
        self.states[self.cur] = state
        self.actions[self.cur] = action
        self.next_states[self.cur] = next_state
        self.rewards[self.cur] = reward
        self.dones[self.cur] = done

        # MOVE THE CUR
        self.cur = (self.cur + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    
    # SAMPLING
    def sample(self, batch):
        ids = np.random.randint(0, self.size, size=batch)

        return (
            torch.tensor(self.states[ids], dtype=torch.float32, device=self.device),
            torch.tensor(self.actions[ids], dtype=torch.float32, device=self.device),
            torch.tensor(self.next_states[ids], dtype=torch.float32, device=self.device),
            torch.tensor(self.rewards[ids], dtype=torch.float32, device=self.device),
            torch.tensor(self.dones[ids], dtype=torch.float32, device=self.device)
        )

    
