from collections import deque
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np

from spatium import *

class Denoiser[T](ABC):
    @abstractmethod
    def add(self, sample: T):
        """Adds a new sample to the denoiser"""
        pass

    @abstractmethod
    def advance(self, duration_secs: float):
        """Advance the denoiser by a given amount of time"""
        pass

    @abstractmethod
    def get_denoised(self) -> Optional[T]:
        """Returns the current denoised sample"""
        pass

class SimpleDenoiser[T](Denoiser[T]):
    """
    A simple denoiser that takes in samples one at a time and performs exponential moving average.
    """
    def __init__(self, decay_per_sec: float):
        assert 0.0 < decay_per_sec < 1.0, "`decay` value must be between 0.0 and 1.0"
        self.decay_per_sec = decay_per_sec
        self.last_denoised_value: Optional[T] = None
        self.last_sample: Optional[T] = None

    def add(self, sample: T):
        self.last_sample = sample
    
    def advance(self, duration_secs: float):
        decay = self.decay_per_sec ** duration_secs
        if self.last_denoised_value == None:
            self.last_denoised_value = self.last_sample
        else:
            self.last_denoised_value = self.last_denoised_value * decay + self.last_sample * (1 - decay)
    
    def get_denoised(self) -> Optional[T]:
        return self.last_denoised_value

class KalmanFilterDenoiser3D(Denoiser[Vec3]):
    """
    A simple Kalman filter denoiser.

    The denoiser records the point's position and velocity, and assumes that the observation is the same as
    the inherent state (or: the observation matrix is identity matrix).

    Notations:
     - x, y, z: the 3 components of position vector
     - u, v, w: the 3 components of velocity vector
    """
    def __init__(self, pred_noise_cov: np.ndarray, observation_noise_cov: np.ndarray, init_pos: Vec3 = Vec3(0.0), init_vel: Vec3 = Vec3(0.0)):
        self.state = np.concatenate([np.array(init_pos), np.array(init_vel)])
        self.pred_noise_cov = pred_noise_cov
        self.observation_noise_cov = observation_noise_cov
        self.covariance_mat = np.zeros((6, 6))   # TODO: initialize covariance matrix
        self.last_sample: Optional[Vec3] = None
    
    def transformation_mat(self, dt: float) -> np.ndarray:
        return np.array([
            [1., 0., 0., dt, 0., 0.],
            [0., 1., 0., 0., dt, 0.],
            [0., 0., 1., 0., 0., dt],
            [0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 1.],
        ])
    
    def observation_mat(self) -> np.ndarray:
        """Projects the state (x, y, z, u, v, w) to the observation (x, y, z)"""
        return np.array([
            [1., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.],
        ])
    
    def inverse_observation_mat(self) -> np.ndarray:
        return np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
        ])
    
    def kf_predict(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        A = self.transformation_mat(dt)
        pred_state = A @ self.state    # No control term
        pred_cov = A @ self.covariance_mat @ A.T + self.pred_noise_cov
        return (pred_state, pred_cov)
    
    def kf_next(self, pred_state: np.ndarray, pred_cov: np.ndarray, observed_pos: Vec3):
        observation = np.array(observed_pos)
        H = self.observation_mat()
        Hinv = self.inverse_observation_mat()
        K = pred_cov @ Hinv @ np.linalg.inv(H @ pred_cov @ Hinv + self.observation_noise_cov)
        new_state = pred_state + K @ (observation - H @ pred_state)
        new_cov = pred_cov - K @ H @ pred_cov
        self.state = new_state
        self.covariance_mat = new_cov

    def add(self, sample: Vec3):
        self.last_sample = sample

    def add_with_cov(self, sample: Vec3, observation_cov: np.ndarray):
        self.last_sample = sample
        if observation_cov.shape == (3, 3):
            self.observation_noise_cov = observation_cov
        else:
            raise f"Invalid shape for observation covariance matrix: {observation_cov.shape}"

    def advance(self, duration_secs: float):
        pred_state, pred_cov = self.kf_predict(duration_secs)
        if self.last_sample != None:
            self.kf_next(pred_state, pred_cov, self.last_sample)
            self.last_sample = None
        else:
            self.state = pred_state
            self.covariance_mat = pred_cov
    
    def get_denoised(self):
        return Vec3(self.state[0], self.state[1], self.state[2])

class PyKalmanDenoiser3D(Denoiser[Vec3]):
    def __init__(self, transition_covariance: float, observation_covariance: float):
        import pykalman
        dt = 1
        F = np.block([
            [np.eye(3), dt * np.eye(3)],
            [np.zeros((3, 3)), np.eye(3)],
        ])
        H = np.hstack([np.eye(3), np.zeros((3, 3))])
        self.state_mean = np.zeros(6)
        self.state_cov = np.eye(6)
        observation_covariance_mat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 8.0]]) * observation_covariance
        self.filter = pykalman.KalmanFilter(
            transition_matrices=F,
            observation_matrices=H,
            transition_covariance=np.eye(6) * transition_covariance,
            observation_covariance=observation_covariance_mat
        )

    def add(self, sample: Vec3):
        self.state_mean, self.state_cov = self.filter.filter_update(
            filtered_state_mean=self.state_mean,
            filtered_state_covariance=self.state_cov,
            observation=sample
        )

    def advance(self, duration_secs: float):
        self.state_mean, self.state_cov = self.filter.filter_update(
            filtered_state_mean=self.state_mean,
            filtered_state_covariance=self.state_cov,
            observation=None
        )

    def get_denoised(self) -> Optional[Vec3]:
        return Vec3(*self.state_mean[:3])
