
from enum import Enum
from collections import deque
import numpy as np
from gymnasium.spaces import Box, Discrete



class ReplayBuffer():
    def __init__(self,
        buffer_len: int,
        observ_space: Box,
        action_space: Discrete | Box,
    ):
        '''
        '''

        assert isinstance(observ_space, Box)
        assert isinstance(action_space, (Discrete, Box))

        # Total size of buffer
        self.buffer_len = buffer_len

        # Pre-allocated buffers
        self.observs = np.zeros((self.buffer_len, *observ_space.shape), dtype=observ_space.dtype)
        if isinstance(action_space, Discrete):
            self.actions = np.zeros((self.buffer_len,), dtype=action_space.dtype)
        else:
            self.actions = np.zeros((self.buffer_len, action_space.shape[0]), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_len,), dtype=np.float32)
        self.next_observs = np.zeros((self.buffer_len, *observ_space.shape), dtype=observ_space.dtype)
        self.dones = np.zeros((self.buffer_len,), dtype=np.float32)

        # Pointer for buffer position; used to determine next entry to overwrite
        self.pos = 0
        self._full = False


    def reset(self) -> None:
        '''
        Resets the buffer and effectively empties it out
        '''
        self.pos = 0
        self._full = False

    @property
    def full(self) -> bool:
        '''
        Boolean property to show if buffer is full/buffer_len

        Returns:
            (bool): Boolean flag for buffer fill
        '''
        return self._full

    @property
    def length(self) -> int:
        '''
        Integer property for fill length of buffer

        Returns:
            (int):  Length of buffer
        '''
        return self.pos if not self._full else self.buffer_len

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
            self._full = True


    def sample(self, batch_size):
        upper_bound = self.buffer_len if self._full else self.pos
        batch_idxs = np.random.randint(0, upper_bound, size=batch_size)
        return (
            self.observs[batch_idxs],
            self.actions[batch_idxs],
            self.rewards[batch_idxs],
            self.next_observs[batch_idxs],
            self.dones[batch_idxs]
        )


class RolloutBuffer():
    def __init__(self):
        '''
        '''

        self.observs = []
        self.actions = []
        self.rewards = []
        self.next_observs = []
        self.dones = []

    def add(self, observ, action, reward, next_observ, done):
        self.observs.append(observ)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observs.append(next_observ)
        self.dones.append(done)

    def reset(self):
        self.observs = []
        self.actions = []
        self.rewards = []
        self.next_observs = []
        self.dones = []

    def sample(self):
        return (
            np.array(self.observs),
            np.array(self.actions),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.next_observs),
            np.array(self.dones, dtype=np.float32)
        )


class NStepBuffer():
    def __init__(self, n: int):
        self.n = n

        self.observs = deque(maxlen=n)
        self.actions = deque(maxlen=n)
        self.rewards = deque(maxlen=n)
        self.next_observs = deque(maxlen=n)
        self.dones = deque(maxlen=n)

    @property
    def full(self):
        return len(self.observs) == self.n

    def add(self, observ, action, reward, next_observ, done):
        self.observs.append(observ)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observs.append(next_observ)
        self.dones.append(done)

    def reset(self):
        self.observs = deque(maxlen=n)
        self.actions = deque(maxlen=n)
        self.rewards = deque(maxlen=n)
        self.next_observs = deque(maxlen=n)
        self.dones = deque(maxlen=n)

    def sample(self):
        return (
            np.array(self.observs),
            np.array(self.actions),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.next_observs),
            np.array(self.dones, dtype=np.float32)
        )

