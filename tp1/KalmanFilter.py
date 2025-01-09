import numpy as np

class KalmanFilter:
    def __init__(self, dt: float, u_x: float, u_y:float, std_acc: float, x_std_meas: float, y_std_meas: float):

        self.dt = dt
        self.u = np.array([[u_x], [u_y]])
        
        self.xk = np.array([0] * 4) # State vector [x, y, vx, vy]
        self.A = np.eye(4) # State transition matrix
        self.A[0][2] = self.A[1][3] = self.dt

        self.B = np.array([[0.5 * self.dt ** 2, 0], [0, 0.5 * self.dt ** 2], [self.dt, 0], [0, self.dt]]) # Control matrix

        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        self.Q = np.array([
            [0.25 * self.dt ** 4, 0, 0.5 * self.dt ** 3, 0], 
            [0, 0.25 * self.dt ** 4, 0, 0.5 * self.dt ** 3], 
            [0.5 * self.dt ** 3, 0, self.dt ** 2, 0], 
            [0, 0.5 * self.dt ** 3, 0, self.dt ** 2]
            ]) * std_acc ** 2
        

        self.R = np.array([
            [x_std_meas ** 2, 0],
            [0, y_std_meas ** 2]
            ]) 
        
        self.P = np.eye(4) # Covariance matrix

    def predict(self):
        # Update time state
        self.xk = self.A @ self.xk + self.B @ self.u

        # Calculate error covariance
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, zk: np.array):

        # Compute Kalman gain
        Sk = self.H @ self.P @ self.H.T + self.R
        Kk = self.P @ self.H.T @ np.linalg.inv(Sk)

        # Update state estimate
        self.xk = self.xk + Kk @ (zk - self.H @ self.xk)
        self.P = (np.eye(4) - Kk @ self.H) @ self.P


    
