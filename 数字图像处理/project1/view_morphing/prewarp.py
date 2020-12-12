import numpy as np


def get_epipoles(F):
    eigval0, eigvec0 = np.linalg.eig(F)
    eigval1, eigvec1 = np.linalg.eig(F.T)

    e0 = eigvec0[:, np.argmin(np.abs(eigval0))]
    e1 = eigvec1[:, np.argmin(np.abs(eigval1))]

    return np.real(e0), np.real(e1)


def rotation_matrix(u, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    t = 1 - np.cos(theta)
    x = u[0]
    y = u[1]
    return np.array([[t*x*x + c, t*x*y, s*y],
                    [t*x*y, t*y*y + c, -s*x],
                    [-s*y, s*x, c]], dtype=np.float32)


def get_prewarp(F, use_T=False):
    e0, e1 = get_epipoles(F)

    d0 = np.array([-e0[1], e0[0], 0], dtype=np.float32)

    Fd0 = F.dot(d0)
    d1 = np.array([-Fd0[1], Fd0[0], 0], dtype=np.float32)

    theta0 = np.arctan(e0[2]/(d0[1]*e0[0] - d0[0]*e0[1]))
    theta1 = np.arctan(e1[2]/(d1[1]*e1[0] - d1[0]*e1[1]))

    R_d0_theta0 = rotation_matrix(d0, theta0)
    R_d1_theta1 = rotation_matrix(d1, theta1)

    new_e0 = R_d0_theta0.dot(e0)
    new_e1 = R_d1_theta1.dot(e1)

    phi0 = -np.arctan(new_e0[1]/new_e0[0])
    phi1 = -np.arctan(new_e1[1]/new_e1[0])

    R_phi0 = np.array([[np.cos(phi0), -np.sin(phi0), 0],
                       [np.sin(phi0), np.cos(phi0), 0],
                       [0, 0, 1]], dtype=np.float32)
    R_phi1 = np.array([[np.cos(phi1), -np.sin(phi1), 0],
                       [np.sin(phi1), np.cos(phi1), 0],
                       [0, 0, 1]], dtype=np.float32)

    H0 = R_phi0.dot(R_d0_theta0)
    H1 = R_phi1.dot(R_d1_theta1)

    if use_T:
        new_F = R_phi0 * R_d0_theta0 * F * R_d1_theta1 * R_d1_theta1

        a = new_F[1,2]
        b = new_F[2,1]
        c = new_F[2,2]

        T = np.array([[0, 0, 0],
                    [0,-a,-c],
                    [0, 0, b]])

        H0 = R_phi0.dot(R_d0_theta0)
        H1 = (T.dot(R_phi1)).dot(R_d1_theta1)

    return H0, H1
