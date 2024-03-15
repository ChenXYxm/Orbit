import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add_experience(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample_batch(self, batch_size):
        batch = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

# Example usage:
# Assuming state has shape (2, 50, 50)

# Create a replay buffer with a buffer size of 1000
buffer_size = 1000
replay_buffer = ReplayBuffer(buffer_size)

# Add experiences to the buffer
state = np.random.random((2, 50, 50))
action = 1
reward = 0.5
next_state = np.random.random((2, 50, 50))
done = False

replay_buffer.add_experience(state, action, reward, next_state, done)

# Sample a batch from the buffer
batch_size = 32
sampled_batch = replay_buffer.sample_batch(batch_size)
print(sampled_batch)