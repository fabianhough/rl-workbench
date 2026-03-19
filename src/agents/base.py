from abc import ABC, abstractmethod



class Agent(ABC):
    
    @abstractmethod
    def act(self, observ, **kwargs): ...

    @abstractmethod
    def train(self, sample): ...

    @abstractmethod
    def post_episode(self): ...
