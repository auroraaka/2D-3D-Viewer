import numpy as np

q01 = np.array([1, 0, 0, 0])
q02 = np.array([0, 1, 0, 0])

def quaternion_multiplication(q0, q1):
    v0, w0 = q0[:3], q0[3]
    v1, w1 = q1[:3], q1[3]
    w = w0 * w1 - np.dot(v0, v1)
    v = np.cross(v0, v1) + w0 * v1 + w1 * v0
    return np.array([v[0], v[1], v[2], w])

def quaternion_conjugate(q):
    return np.concatenate((-q[:3], [q[3]]))

def quaternion_division(q0, q1):
    return quaternion_multiplication(q0, quaternion_conjugate(q1))

def slerp(q0, q1, alpha=0.1):
    qr = quaternion_division(q1, q0)
    vr, wr = qr[:3], qr[3]
    if wr < 0:
        qr = -qr
    magnitude_vr = np.linalg.norm(vr)
    thetar = 2 * np.arctan(magnitude_vr  / wr)
    nr = vr / magnitude_vr 
    thetaa = alpha * thetar
    qa = np.concatenate((np.sin(thetaa/2)*nr, [np.cos(thetaa/2)]))
    return quaternion_multiplication(qa, q0)

print('=================================')

print(slerp(q01, q02, .0))
print(slerp(q01, q02, 0.2))
print(slerp(q01, q02, 0.4))
print(slerp(q01, q02, 0.6))
print(slerp(q01, q02, 0.8))
print(slerp(q01, q02, 1.))
