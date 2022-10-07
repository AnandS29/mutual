import numpy as np

class Dynamics:
    def __init__(self, f, g_u, dt, x0, u0):
        self.dt = dt
        self.x = x0
        self.u = u0
        self.f = f
        self.g_u = g_u

    def step(self, u):
        self.u = u
        self.x = self.x + self.dt * [self.f(self.x) + self.g_u(self.x) @ u]
        return self.x

    def random_sim(self, n_steps, A, b):
        x = self.x
        u = self.u
        for i in range(n_steps):
            u = 0 # TODO: random sample from polytope
            x = self.step(u)
        return x

    # Random sample from polytope
    def random_sample_polytope(self, A, b):
        return None