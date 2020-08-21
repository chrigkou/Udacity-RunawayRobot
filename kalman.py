import numpy as np
from matrix import matrix


class KalmanFilter():
    def __init__(self, sigma):
        self.F = np.array('''
                        1. 0. 1. 0.;
                        0. 1. 0. 1.;
                        0. 0. 1. 0.;
                        0. 0. 0. 1.
                             ''')
        self.H = np.array('''
                        1. 0. 0. 0.;
                        0. 1. 0. 0.''')

        self.motion = np.array('0. 0. 0. 0.').T
        self.Q = np.array(np.eye(4))
        self.P = np.array(np.eye(4)) * 1000  # initial uncertainty

    def update(self, x, measurement):
        # Update Step in Kalman Filter.
        y = np.array(measurement).T - self.H * x
        S = self.H * self.P * self.H.T + self.R
        K = self.P * self.H.T * S.I  # Kalman gain

        x = x + K * y
        I = np.array(np.eye(self.F.shape[0]))  # identity matrix
        P = (I - K * self.H) * self.P
        return x, P

    def predict(self, x, motion):
        x = self.F * x + motion
        P = self.F * self.P * self.F.T + self.Q
        return x, P


class ExtendedKalman():
    def __init__(self, sigma):
        self.x = matrix([
                           [0.],  # x
                           [0.],  # y
                           [0.],  # heading
                           [0.],  # rotation
                           [0.]])  # distance

        self.p = matrix([
            [999, 0, 0, 0, 0],
            [0, 999, 0, 0, 0],
            [0, 0, 999, 0, 0],
            [0, 0, 0, 999, 0],
            [0, 0, 0, 0, 999]
        ])
        self.h = matrix([[1., 0., 0., 0., 0.],
                        [0., 1., 0., 0., 0.]])

        self.r = matrix([
            [sigma, 0],
            [0, sigma]
        ])
        self.f = matrix([[]])
        self.f.zero(5, 5)
        # use q if there is motion uncertainty (not needed in this problem)
        # self.q = self.f * self.f.transpose()
        # for i in range(self.q.dimx):
        #     for j in range(self.q.dimy):
        #        self.q.value[i][j] = 0.001 * self.q.value[i][j]

    def predict(self):

        x = self.x.value[0][0]
        y = self.x.value[1][0]
        h = self.x.value[2][0]
        r = self.x.value[3][0]
        d = self.x.value[4][0]
        self.f = matrix([
                [1, 0, -d*np.sin(r + h), -d*np.sin(r + h), np.cos(r + h)],
                [0, 1, d*np.cos(r + h), d*np.cos(r + h), np.sin(r + h)],
                [0, 0,                         1,                   1, 0],
                [0, 0,                         0,                   1, 0],
                [0, 0,           0, 0, 1]])

        new_x = x + d * np.cos(h + r)
        new_y = y + d * np.sin(h + r)
        new_h = h + r
        new_r = r
        new_d = d

        self.x = matrix([
            [new_x], [new_y], [new_h], [new_r], [new_d]])
        self.p = self.f * self.p * self.f.transpose()  # + self.q
        pred_x = float(self.x.value[0][0])
        pred_y = float(self.x.value[1][0])
        return pred_x, pred_y

    def update(self, measurement):
        z = matrix([
            [measurement[0]],
            [measurement[1]]
        ])
        y = z - self.h * self.x

        s = self.h * self.p * self.h.transpose() + self.r
        k = self.p * self.h.transpose() * s.inverse()

        self.x = self.x + k * y
        i = matrix([[]])
        i.identity(5)
        self.p = (i - k * self.h) * self.p

