# coding: utf-8

import torch
import numpy as np
from pykalman import KalmanFilter


class KalmanFilterLimit:
    def __init__(self, process_variance=1e-5, measurement_variance=1e-2):
        # Initialize the Kalman filter
        self.kf = KalmanFilter(
            transition_matrices=[1],  # For 1D values, state transition matrix is 1
            observation_matrices=[1],  # Observation matrix is 1 for direct observations
            initial_state_mean=0,  # Can be set to the first value or zero
            initial_state_covariance=1,  # Initial error covariance
            transition_covariance=process_variance,  # Variance in the process
            observation_covariance=measurement_variance  # Variance in the measurements
        )
        self.state_mean = 0
        self.state_covariance = 1

    def update(self, x, shape=None, device='cpu'):
        # Update the Kalman filter with the new value
        self.state_mean, self.state_covariance = self.kf.filter_update(
            filtered_state_mean=self.state_mean,
            filtered_state_covariance=self.state_covariance,
            observation=x.reshape(-1)
        )

        # Return the filtered value as a tensor in the desired shape
        filtered_tensor = torch.tensor(self.state_mean, dtype=torch.float32, device=device)
        if shape is not None:
            return filtered_tensor.reshape(shape[-2:])  # Reshape if required
        return filtered_tensor  # Return the filtered value directly if no shape provided


def smooth(x_d_lst, shape, device, observation_variance=3e-7, process_variance=1e-5):
    x_d_lst_reshape = [x.reshape(-1) for x in x_d_lst]
    x_d_stacked = np.vstack(x_d_lst_reshape)
    kf = KalmanFilter(
        initial_state_mean=x_d_stacked[0],
        n_dim_obs=x_d_stacked.shape[1],
        transition_covariance=process_variance * np.eye(x_d_stacked.shape[1]),
        observation_covariance=observation_variance * np.eye(x_d_stacked.shape[1])
    )
    smoothed_state_means, _ = kf.smooth(x_d_stacked)
    x_d_lst_smooth = [torch.tensor(state_mean.reshape(shape[-2:]), dtype=torch.float32, device=device) for state_mean in smoothed_state_means]
    return x_d_lst_smooth
