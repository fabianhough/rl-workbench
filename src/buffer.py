
from enum import Enum
from collections import deque
from abc import ABC, abstractmethod

import numpy as np
from gymnasium.spaces import Box, Discrete


class Buffer(ABC):

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def add(self, observ, action, reward, next_observ, done, value) -> None: ...

    @abstractmethod
    def sample(self) -> tuple: ...

    @abstractmethod
    def ready(self) -> bool: ...



class ReplayBuffer(Buffer):
    def __init__(self,
        buffer_len: int,
        sample_size: int,
        observ_space: Box,
        action_space: Discrete | Box,
    ):
        '''
        '''

        assert isinstance(observ_space, Box)
        assert isinstance(action_space, (Discrete, Box))

        # Total size of buffer
        self.buffer_len = buffer_len
        self.sample_size = sample_size

        # Pre-allocated buffers
        self.observs = np.zeros((self.buffer_len, *observ_space.shape), dtype=observ_space.dtype)
        if isinstance(action_space, Discrete):
            self.actions = np.zeros((self.buffer_len,), dtype=action_space.dtype)
        else:
            self.actions = np.zeros((self.buffer_len, action_space.shape[0]), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_len,), dtype=np.float32)
        self.values = np.zeros((self.buffer_len,), dtype=np.float32)
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

    def ready(self):
        return True

    @property
    def length(self) -> int:
        '''
        Integer property for fill length of buffer

        Returns:
            (int):  Length of buffer
        '''
        return self.pos if not self._full else self.buffer_len

    def add(self, observ, action, reward, next_observ, done, value):
        # Overwriting entries at location
        self.observs[self.pos] = observ
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_observs[self.pos] = next_observ
        self.dones[self.pos] = done
        self.values[self.pos] = value

        # Incrementing pointer, and resetting when full
        self.pos += 1
        if self.pos >= self.buffer_len:
            self.pos = 0
            self._full = True


    def sample(self):
        upper_bound = self.buffer_len if self._full else self.pos
        sample_idxs = np.random.randint(0, upper_bound, size=self.sample_size)
        return (
            self.observs[sample_idxs],
            self.actions[sample_idxs],
            self.rewards[sample_idxs],
            self.next_observs[sample_idxs],
            self.dones[sample_idxs],
            self.values[sample_idxs]
        )


class RolloutBuffer(Buffer):
    def __init__(self):
        '''
        '''

        self.observs = []
        self.actions = []
        self.rewards = []
        self.next_observs = []
        self.dones = []
        self.values = []

    def ready(self):
        return True

    def add(self, observ, action, reward, next_observ, done, value):
        self.observs.append(observ)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observs.append(next_observ)
        self.dones.append(done)
        self.values.append(value)

    def reset(self):
        self.observs = []
        self.actions = []
        self.rewards = []
        self.next_observs = []
        self.dones = []
        self.values = []

    def sample(self):
        return (
            np.array(self.observs),
            np.array(self.actions),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.next_observs),
            np.array(self.dones, dtype=np.float32),
            np.array(self.values, dtype=np.float32)
        )


class NStepBuffer(Buffer):
    def __init__(self, n: int):
        self.n = n

        self.observs = deque(maxlen=n)
        self.actions = deque(maxlen=n)
        self.rewards = deque(maxlen=n)
        self.next_observs = deque(maxlen=n)
        self.dones = deque(maxlen=n)
        self.values = deque(maxlen=n)

    @property
    def full(self):
        return len(self.observs) == self.n

    def ready(self):
        return self.full

    def add(self, observ, action, reward, next_observ, done, value):
        self.observs.append(observ)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observs.append(next_observ)
        self.dones.append(done)
        self.values.append(value)

    def reset(self):
        self.observs = deque(maxlen=n)
        self.actions = deque(maxlen=n)
        self.rewards = deque(maxlen=n)
        self.next_observs = deque(maxlen=n)
        self.dones = deque(maxlen=n)
        self.values = deque(maxlen=n)

    def sample(self):
        return (
            np.array(self.observs),
            np.array(self.actions),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.next_observs),
            np.array(self.dones, dtype=np.float32),
            np.array(self.values, dtype=np.float32)
        )

