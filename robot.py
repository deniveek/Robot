import numpy as np
from math import sin, cos, pi
from functools import reduce
from copy import deepcopy

h = 1
l = 1
parameters = {1: {'a': 0, 'alpha': pi / 2, 'd': 0, 'theta': 0},
              2: {'a': 0, 'alpha': pi / 2, 'd': h, 'theta': pi / 2},
              3: {'a': l, 'alpha': 0, 'd': 0, 'theta': pi/2}}

position = {1: ['d', 0.1],
            2: ['theta', 0],
            3: ['theta', pi/2]}

threshold = {1: [0, 10],
             2: [-1.3*pi, 1.3*pi],
             3: [-1.3*pi, 1.3*pi]}


class DanHart:
    def __init__(self, par, pos, thresh):
        self.par = par
        self.pos = pos
        self.thresh = thresh

    @staticmethod
    def template(num, pos, par):
        a, alpha = par[num]['a'], par[num]['alpha']
        d = par[num]['d'] + pos[num][1] if 'd' in pos[num] else par[num]['d']
        theta = par[num]['theta'] + pos[num][1] if 'theta' in pos[num] else par[num]['theta']
        T = np.array([[cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
                      [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
                      [0, sin(alpha), cos(alpha), d],
                      [0, 0, 0, 1]])
        return T

    def T(self, pos):
        return dict(enumerate([self.template(i, pos, self.par) for i in self.par.keys()]))

    @staticmethod
    def direct_kin(robot):
        return np.array(list(reduce(lambda a, b: a @ b, robot.values())))

    @property
    def position(self):
        return DanHart.direct_kin(self.T(self.pos))

    @property
    def line(self):
        return self.position[:3, 3]

    @property
    def angle(self):
        return self.position[:3, :3]

    @staticmethod
    def norm(vec):
        return vec @ np.array([[0,-1,0],[0,0,-1],[1,0,0]])

    @staticmethod
    def vec_len(vec):
        return sum([a**2 for a in vec])

    @staticmethod
    def find_p(vec, dest):
        #d_0 = 0.01
        #tmp = []
        #for key in thresh.keys():
            #tmp.append((min([pos[key][1] - thresh[key][0], thresh[key][1] - pos[key][1]])/(thresh[key][1] - thresh[key][0])))
        #d = min(tmp) if min(tmp) != 0 else 0.001
        #p_r = 0.5*50*(1/d-1/d_0)**2 if d <= d_0 else 0
        p_a = 0.5*DanHart.vec_len(vec - dest)
        return p_a

    def gradient(self, pos, dest):
        grad = []
        p_q = DanHart.find_p(DanHart.direct_kin(self.T(pos))[:3, 3], dest)
        delta = (self.thresh[1][1] - self.thresh[1][0]) / 10000
        for key in pos:
            pos_ = deepcopy(pos)
            pos_[key][1] += delta
            grad.append((DanHart.find_p(DanHart.direct_kin(self.T(pos_))[:3, 3], dest) - p_q)/delta)
        return pos[1][1], pos[2][1], p_q, grad

    def descend(self, dest, pos, step=1):
        pos_ = deepcopy(pos)
        p_q, grad = self.gradient(pos_, dest)[2:]
        while p_q > 0.0001 and DanHart.vec_len(grad) > 0.000001:
            for key in pos_.keys():
                pos_[key][1] -= grad[key-1]
            p_q, grad = self.gradient(pos_, dest)[2:]
            #print(f"{pos_} {P}")
        return pos_


res = DanHart(parameters, position, threshold)
#print(res.position)
"""
with open('tst_P.dat', 'w') as tst_1:
    with open('tst_Grad.dat', 'w') as tst_2:
        tst_pos = {1: ['d', 0.1],
                   2: ['theta', 0],
                   3: ['theta', pi/2]}
        delta_d = (res.thresh[1][1] - res.thresh[1][0]) / 50
        delta_theta = (res.thresh[2][1] - res.thresh[2][0]) / 50
        while tst_pos[1][1] < res.thresh[1][1]:
            print(tst_pos[1][1])
            while res.thresh[2][0] <= tst_pos[2][1] < res.thresh[2][1]:
                tst_pos[2][1] += delta_theta
                res_ = res.gradient(tst_pos, [0, -1, 5])
                tst_1.write(f"{res_[0]} {res_[1]} {res_[2]}\n")
                tst_2.write(f"{res_[0]} {res_[1]} {res_[2]} {res_[3][0]} {res_[3][1]} {res_[3][2]}\n")
            tst_1.write('\n')
            tst_2.write('\n')
            tst_pos[2][1] = res.thresh[2][0]
            tst_pos[1][1] += delta_d
"""
new = res.descend([0, -0.5, 9], res.pos)
for key in res.pos.keys():
    res.pos[key][1] = new[key][1]
print(f"{res.pos}\n{res.position[:3, 3]}")

