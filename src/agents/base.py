from abc import ABC, abstractmethod



class Agent(ABC):
    
    @abstractmethod
    def act(observ, **kwargs): ...

    @abstractmethod
    def train(sample): ...


