import random


class ReplayBuffer:
    def __init__(self, buffer_size = 10000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add_experience(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def get_buffer_size(self):
        return len(self.buffer)

class MABuffer:
    def __init__(self, buffer_size = 10000):
        self.buffer = {}
        self.buffer_size = buffer_size

    def add_experience(self, experience, agent_name):
        if agent_name not in self.buffer:
            self.buffer[agent_name] = []
        self.buffer[agent_name].append(experience)
        if len(self.buffer[agent_name]) > self.buffer_size:
            self.buffer[agent_name].pop(0)

    def sample_batch(self, batch_size):
        batch = []
        if len(self.buffer) == 0:
            return batch
        size_for_each_agent = max(1, batch_size // len(self.buffer))
        for agent_name in self.buffer:
            to_sample = min(len(self.buffer[agent_name]), size_for_each_agent)
            batch += random.sample(self.buffer[agent_name], to_sample)
        return batch
    
    def get_full_batch(self):
        batch = []
        for agent_name in self.buffer:
            batch += self.buffer[agent_name]
        self.buffer = {}
        return batch
    
    def pop_agent_buffer(self, agent_name):
        agent_buffer = self.buffer[agent_name]
        del self.buffer[agent_name]
        return agent_buffer

    def get_agent_buffer(self, agent_name, batch_size):
        return random.sample(self.buffer[agent_name], batch_size)


    def get_buffer_size(self):
        return len(self.buffer)
    
    
class RewardRecorder:
    def __init__(self):
        self.rewards = {}

    def add_reward(self, agent_name, reward):
        if agent_name not in self.rewards:
            self.rewards[agent_name] = reward
        else:
            self.rewards[agent_name] += reward
            
    def get_rewards(self):
        result = self.rewards.copy()
        self.rewards = {}
        return result