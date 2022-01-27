"""transform utils based on mujoco api

- quaternion convention is (w, x, y, z)
"""

import math
from mujoco_py import functions
import numpy as np

IDENTITY_QUATERNION = np.array([1.0, 0, 0, 0])
EPS = np.finfo(float).eps * 4.0

# ============= Transform Utils ============= 

def quat_multiply(q1, q2):
    res = np.zeros(4)
    functions.mju_mulQuat(res, q1, q2)
    return res


def mat2quat(R):
    q = np.zeros(4)
    functions.mju_mat2Quat(q, R.flatten())
    return q


def quat2mat(q):
    R = np.zeros(9)
    functions.mju_quat2Mat(R, q)
    return R.reshape((3, 3))


def axisangle2quat(vec):
    # Grab angle
    angle = np.linalg.norm(vec)

    # handle zero-rotation case
    if math.isclose(angle, 0.0):
        return IDENTITY_QUATERNION

    # make sure that axis is a unit vector
    axis = vec / angle

    q = np.zeros(4)
    functions.mju_axisAngle2Quat(q, axis, angle)
    return q


def quat2axisangle(quat):
    den = np.sqrt(1.0 - quat[0] * quat[0])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[1:] * 2.0 * math.acos(quat[0])) / den


def axis_angle(r, ref):
    """modify angle axis based on its distance to ref (because there are many angle axis corresponds to one rotation)

    Args:
        r (np.array(3)): the source angle axis vector
        ref (np.array(3)): the reference vector

    Returns:
        np.array(3): new axis angle
    """
    if r.dot(ref) < 0:
        angle = np.linalg.norm(r)
        axis = r / angle
        angle = angle - 2 * np.pi
        r = axis * angle
    return r


def rotate_quat(q, vec):
    res = np.array(q)
    functions.mju_quatIntegrate(res, vec, 1)
    return res


def random_quaternion(q, theta):
    """Generate a random quaternion a theta rad from the current quaternion
    source: https://math.stackexchange.com/questions/3448216/uniformly-sample-orientations-quaternions-k-degrees-from-a-given-orientation

    Args:
        q (np.array): current quaternion
        theta (float): angle in radian
    """
    w = np.sqrt((1 + np.cos(theta)) / 2)
    r = np.sqrt((1 - np.cos(theta)) / 2)
    u = np.random.uniform(0, 1, size=2)
    z = 2 * u[0] - 1
    x = r * np.sqrt(1 - z ** 2) * np.cos(2 * np.pi * u[1])
    y = r * np.sqrt(1 - z ** 2) * np.sin(2 * np.pi * u[1])
    z = r * z
    q0 = np.array([w, x, y, z])
    return quat_multiply(q0, q)


def quat_error(q1, q2):
    """Calculate distance q2 - q1 between q1 and q2"""
    dtype = q1.dtype
    neg_q1 = np.zeros(4, dtype=dtype)
    err_rot_quat = np.zeros(4, dtype=dtype)
    err_rot = np.zeros(3, dtype=dtype)

    if q1.dot(q2) < 0:
        q1 = -np.array(q1)

    functions.mju_negQuat(neg_q1, q1)
    functions.mju_mulQuat(err_rot_quat, q2, neg_q1)
    functions.mju_quat2Vel(err_rot, err_rot_quat, 1)
    return err_rot


def multiply_pose(pos1, quat1, pos2, quat2):
    """useful for coordinate transformation. For example, if we want to transform
    pose of point A in frame B to frame C, set
        pos1 = pos_B_in_C, quat1 = quat_B_in_C
        pos2 = pos_A_in_B, quat2 = quat_A_in_B

    """
    p_res = np.zeros(3)
    q_res = np.zeros(4)
    functions.mju_mulPose(p_res, q_res, pos1, quat1, pos2, quat2)
    return p_res, q_res


def pose_inverse(pos, quat):
    pinv = np.zeros(3)
    qinv = np.zeros(4)
    functions.mju_negPose(pinv, qinv, pos, quat)
    return pinv, qinv


def quat_slerp(quat0, quat1, fraction, shortestpath=True):
    """
    Return spherical linear interpolation between two quaternions.

    E.g.:
    >>> q0 = random_quat()
    >>> q1 = random_quat()
    >>> q = quat_slerp(q0, q1, 0.0)
    >>> np.allclose(q, q0)
    True

    >>> q = quat_slerp(q0, q1, 1.0)
    >>> np.allclose(q, q1)
    True

    >>> q = quat_slerp(q0, q1, 0.5)
    >>> angle = math.acos(np.dot(q0, q))
    >>> np.allclose(2.0, math.acos(np.dot(q0, q1)) / angle) or \
        np.allclose(2.0, math.acos(-np.dot(q0, q1)) / angle)
    True

    Args:
        quat0 (np.array): (w,x,y,z) quaternion startpoint
        quat1 (np.array): (w,x,y,z) quaternion endpoint
        fraction (float): fraction of interpolation to calculate
        shortestpath (bool): If True, will calculate the shortest path

    Returns:
        np.array: (w,x,y,z) quaternion distance
    """
    q0 = np.array(quat0)
    q1 = np.array(quat1)
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        q1 *= -1.0
    angle = math.acos(np.clip(d, -1, 1))
    if abs(angle) < EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0


def transform_spatial(v1, q21):
    """transform spatial vector from frame 1 to frame 2, given the orientation of frame 1 relative to frame 2

    Args:
        v1 (list[6] or np.array(6)): spatial vector, can be twist or wrench
        q21 (list[4] or np.array(4)): quaternion defining the orientation
    """
    R21 = quat2mat(q21)
    R = np.block([[R21, np.zeros((3, 3))], [np.zeros((3, 3)), R21]])
    return R.dot(v1)


def similarity_transform(A1, q21):
    """Similarity transformation of a matrix from frame 1 to frame 2
    A2 = R21 * A1 * R12
    """
    R21 = quat2mat(q21)
    return R21.dot(A1.dot(R21.T))
