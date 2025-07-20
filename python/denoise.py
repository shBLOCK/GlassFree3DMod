from collections import deque
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np

class Denoiser(ABC):
    @abstractmethod
    def add(self, sample, duration_secs: float):
        """Adds a new sample to the denoiser"""
        pass

    @abstractmethod
    def get_denoised(self):
        """Returns the current denoised sample"""
        pass

class SimpleDenoiser:
    """
    A simple denoiser that takes in samples one at a time and performs exponential moving average.
    """
    def __init__(self, decay_per_sec: float):
        assert 0.0 < decay_per_sec < 1.0, "`decay` value must be between 0.0 and 1.0"
        self.decay_per_sec = decay_per_sec
        self.last_denoised_value = None

    def add(self, sample, duration_secs: float):
        decay = self.decay_per_sec ** duration_secs
        if self.last_denoised_value == None:
            self.last_denoised_value = sample
        else:
            self.last_denoised_value = self.last_denoised_value * decay + sample * (1 - decay)
    
    def get_denoised(self):
        return self.last_denoised_value