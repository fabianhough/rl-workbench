
from enum import Enum
import numpy as np
from gymnasium.spaces import Box, Discrete


class BufferSampleMode(Enum):
    RANDOM = 'random'
    FULL = 'full'


class Buffer():
    def __init__(self,
        buffer_len: int,
        observ_space: Box,
        action_space: Discrete | Box,
        buffer_sample_mode: BufferSampleMode=BufferSampleMode.RANDOM
    ):
        '''
        Replay Buffer:  Random Sample, Any buffer_len
        n-step Buffer:  Full Sample, buffer_len == n
        Batch Buffer:   Full Sample, buffer_len == batch_size
        '''

        assert isinstance(observ_space, Box)
        assert isinstance(action_space, (Discrete, Box))

        # Total size of buffer
        self.buffer_len = buffer_len
        self.buffer_sample_mode = buffer_sample_mode

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
        if self.buffer_sample_mode == BufferSampleMode.RANDOM:
            upper_bound = self.buffer_len if self._full else self.pos
            batch_idxs = np.random.randint(0, upper_bound, size=batch_size)
            return (
                self.observs[batch_idxs],
                self.actions[batch_idxs],
                self.rewards[batch_idxs],
                self.next_observs[batch_idxs],
                self.dones[batch_idxs]
            )
        else:
            return (
                self.observs,
                self.actions,
                self.rewards,
                self.next_observs,
                self.dones
            )


class ReplayBuffer(Buffer):
    def __init__(self, buffer_len, observ_space, action_space):
        super().__init__(
            buffer_len=buffer_len,
            observ_space=observ_space,
            action_space=action_space,
            buffer_sample_mode=BufferSampleMode.RANDOM
        )
