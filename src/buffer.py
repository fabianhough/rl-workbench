import numpy as np



class ReplayBuffer():
    def __init__(self, buffer_len, observ_space, action_space):

        # Total size of buffer
        self.buffer_len = buffer_len

        # Critical dims of buffers
        self.observ_space = observ_space
        self.action_space = action_space

        # Pre-allocated buffers
        self.observs = np.zeros((self.buffer_len, self.observ_space.shape[0])) # Only valid for flat
        self.actions = np.zeros((self.buffer_len, self.action_space.n)) # Only valid for discrete
        self.rewards = np.zeros((self.buffer_len))
        self.next_observs = np.zeros((self.buffer_len, *self.observ_space))
        self.dones = np.zeros((self.buffer_len))

        # Pointer for buffer position; used to determine next entry to overwrite
        self.pos = 0
        self.full = False


    def reset(self):
        self.pos = 0
        self.full = False


    def add(self, observ, action, reward, next_observ, done):
        # Overwriting entries at location
        self.observs[self.pos] = observ
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_observs[self.pos] = next_observ
        self.dones[self.pos] = done

        # Incrementing pointer, and resetting when full
        self.pos += 1
        if self.pos >= self.buffer_len:
            self.pos = 0
            self.full = True


    def sample(self, batch_size):
        upper_bound = self.buffer_len if self.full else self.pos
        batch_idxs = np.random.randint(0, upper_bound, size=batch_size)
        return (
            self.observs[batch_idxs],
            self.actions[batch_idxs],
            self.rewards[batch_idxs],
            self.next_observs[batch_idxs],
            self.dones[batch_idxs]
        )


