import numpy as np
from scipy.optimize import bisect
from functools import partial


def convert(x, y, z):
    x, y, z = x / 1000, y / 1000, z / 1000
    pos = np.abs(np.array([x, y, z]))  # position vector in cartesian
    R = np.sqrt(pos[0] ** 2 + pos[1] ** 2)  # radial component of position (polar)
    X, Y, Z = pos

    # Quartic equations
    def quartic(P0, P1, P3, P4, x):
        return P0 + P1 * x + P3 * x ** 3 + P4 * x ** 4

    # Model of Earth (WGS84 model)
    a = 6378.1370  # [m]
    b = 6356.752314245  # [m]

    # Determination of longitude
    if pos[1] <= pos[0]:
        long = 2 * np.arctan(Y / (X + R))

    else:
        long = 2 * np.arctan(X / (Y + R))

    # Find z and r with bisection method
    A = a / b
    U = (a ** 2 - b ** 2) / b

    B = b / a
    V = (a ** 2 - b ** 2) / a

    if Z >= (b / a) ** 2 * R:
        if R == 0 or Z == 0:
            r, z = (0, b)
        else:
            P0 = -Z
            P1 = 2 * (A * R - U)
            P3 = 2 * (A * R + U)
            P4 = Z
            P_freeze = partial(quartic, P0, P1, P3, P4)
            P_root = bisect(P_freeze, 0, 1)

            r = a * (1 - P_root ** 2) / (1 + P_root ** 2)
            z = b * (2 * P_root) / (1 + P_root ** 2)

    elif Z < (b / a) ** 2 * R:

        P0 = -R
        P1 = 2 * (B * Z + V)
        P3 = 2 * (B * Z - V)
        P4 = R
        P_freeze = partial(quartic, P0, P1, P3, P4)
        P_root = bisect(P_freeze, 0, 1)

        r = a * (2 * P_root) / (1 + P_root ** 2)
        z = b * (1 - P_root ** 2) / (1 + P_root ** 2) * np.sign(z)

    # Determination of lattitude
    if z <= (b / a) ** 2 * r:
        lat = np.arctan(z / ((b / a) ** 2 * r))
    else:
        lat = (np.pi / 2) - np.arctan(((b / a) ** 2 * r) / z)

    # Correct sign

    if x < 0 and y > 0:
        long += np.pi / 2

    if x < 0 and y < 0:
        long += np.pi

    if x > 0 and y < 0:
        long += (3 / 2) * np.pi

    if long > np.pi:
        long = -(2*np.pi-long)
    # Geodetic height
    delta_r = R - r
    delta_z = Z - z

    if np.abs(delta_z) <= np.abs(delta_r):
        altitude = delta_r * np.sqrt(1 + (delta_z / delta_r) ** 2)

    elif np.abs(delta_z) > np.abs(delta_r):
        altitude = delta_z * np.sqrt(1 + (delta_r / delta_z) ** 2)
    return (lat, long, altitude * 1000)


def jacobian(lat, long):
    jacobian = np.array([
    [-np.sin(lat) * np.cos(long), -np.sin(lat) * np.sin(long),  np.cos(lat)],  # North
    [-np.sin(long),                np.cos(long),                0],            # East
    [ np.cos(lat) * np.cos(long),  np.cos(lat) * np.sin(long),  np.sin(lat)]   # Up
])
    return jacobian


def error(conversion, E):
    J = jacobian(conversion[0], conversion[1])
    J_T = np.transpose(J)
    trans_matrix = np.array(J @ E @ J_T)
    return trans_matrix


def getGeodetic(x, y, z, E):
    conversion = convert(x, y, z)
    err = error(conversion, E)
    return conversion, err





