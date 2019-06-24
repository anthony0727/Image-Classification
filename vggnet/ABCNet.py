from abc import ABC, abstractmethod


class Network(ABC):
    @abstractmethod
    def _attach_layers(self):
        pass

    @abstractmethod
    def _attach_loss(self):
        pass

    @abstractmethod
    def _attach_metric(self):
        pass

    @abstractmethod
    def _attach_summary(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self, optimizer):
        pass
