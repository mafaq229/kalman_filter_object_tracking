"""Kalman Filter implementation for object tracking."""

import numpy as np


class KalmanFilter:
    """A Kalman filter tracker for object tracking."""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initialize the Kalman Filter.

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        # Initial state vector [x, y, vx, vy]
        self.state = np.array([init_x, init_y, 0., 0.])
        
        # Initial covariance matrix
        initial_covariance = 1
        self.covariance = np.eye(4) * initial_covariance
        
        # State transition matrix
        dt = 1
        self.Dt = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.Mt = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise matrix
        self.Q = Q
        
        # Measurement noise matrix
        self.R = R

    def predict(self):
        """Predict the next state using the Kalman Filter prediction step."""
        # Predict state
        self.state = np.matmul(self.Dt, self.state)
        
        # Predict covariance
        self.covariance = np.matmul(
            self.Dt,
            np.matmul(self.covariance, self.Dt.T)
        ) + self.Q

    def correct(self, meas_x, meas_y):
        """Correct the state estimate using measurements.

        Args:
            meas_x (float): Measured x position.
            meas_y (float): Measured y position.
        """
        # Calculate Kalman Gain
        S = np.matmul(
            self.Mt,
            np.matmul(self.covariance, self.Mt.T)
        ) + self.R
        
        Kt = np.matmul(
            self.covariance,
            np.matmul(self.Mt.T, np.linalg.inv(S))
        )
        
        # Update state
        Yt = np.array([meas_x, meas_y])
        residual_m = Yt - np.matmul(self.Mt, self.state)
        self.state = self.state + np.matmul(Kt, residual_m)
        
        # Update covariance
        I = np.identity(len(self.state))
        self.covariance = np.matmul(
            (I - np.matmul(Kt, self.Mt)),
            self.covariance
        )

    def process(self, measurement_x, measurement_y):
        """Process a new measurement and return the updated state estimate.

        Args:
            measurement_x (float): Measured x position.
            measurement_y (float): Measured y position.

        Returns:
            tuple: (x, y) position estimate.
        """
        self.predict()
        self.correct(measurement_x, measurement_y)
        return self.state[0], self.state[1] 