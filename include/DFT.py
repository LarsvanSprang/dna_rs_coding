import math
import numpy as np

class FieldPower:
    def __init__(self, n, a):
        self.n = n  # Maximum power (order of the field)
        self.a = a  # Element of the field
        self.nm = pow(n, -1, n)  # Inverse of n mod n

        # Precompute powers of a and a^-1
        self.powers_a = [1] * n
        self.powers_am = [1] * n
        self.powers_a[1] = a
        for i in range(2, n):
            self.powers_a[i] = (self.powers_a[i - 1] * a) % n
        
        self.am = pow(a, -1, n)
        self.powers_am[1] = self.am
        for i in range(2, n):
            self.powers_am[i] = (self.powers_am[i - 1] * self.am) % n
        
        print("fieldpower: lookup tables generated.")

    def copow(self, powe):
        """ Returns a^powe if powe >= 0, or a^(-powe) otherwise. """
        if powe >= 0:
            return self.powers_a[powe % self.n]
        else:
            return self.powers_am[(-powe) % self.n]

def dft_primitive(xv, fp, prefac=1):
    """ Performs a primitive DFT. """
    Xv = np.zeros(len(xv), dtype=complex)
    for j in range(len(Xv)):
        for i in range(len(xv)):
            if prefac < 0:
                Xv[j] += fp.nm * xv[i] * fp.copow(-i * j)
            else:
                Xv[j] += xv[i] * fp.copow(i * j)
    return Xv

def fft(xv, fp, prefac, P, Q):
    """ FFT implementation using FieldPower class. """
    assert Q * P == len(xv)
    Xv = np.zeros(len(xv), dtype=complex)
    Y = np.zeros((Q, P), dtype=complex)

    # Compute y_p^(r)
    for r in range(Q):
        for p in range(P):
            for q in range(Q):
                Y[r][p] += xv[P * q + p] * fp.copow(prefac * P * q * r)
            Y[r][p] *= fp.copow(prefac * p * r)
    
    # Compute A_{Qs+r}
    for r in range(Q):
        for s in range(P):
            Xv[Q * s + r] = 0
            for p in range(P):
                Xv[Q * s + r] += Y[r][p] * fp.copow(prefac * Q * s * p)
            if prefac == -1:
                Xv[Q * s + r] *= fp.nm
    
    return Xv

class DFT_FFT:
    def __init__(self, n, a, P, Q):
        self.fp = FieldPower(n, a)
        self.P = P
        self.Q = Q

    def dft(self, xv):
        return fft(xv, self.fp, prefac=1, P=self.P, Q=self.Q)

    def idft(self, Xv):
        return fft(Xv, self.fp, prefac=-1, P=self.P, Q=self.Q)

class DFT_LA:
    def __init__(self, n, a):
        self.a = a  # The Fourier kernel
        self.n = n
        self.nm = pow(n, -1, n)
        self.powers_a = [1] * (n * n + 1)
        self.powers_am = [1] * (n * n + 1)

        self.powers_a[1] = a
        for i in range(2, len(self.powers_a)):
            self.powers_a[i] = (self.powers_a[i - 1] * a) % n
        
        self.am = pow(a, -1, n)
        self.powers_am[1] = self.am
        for i in range(2, len(self.powers_am)):
            self.powers_am[i] = (self.powers_am[i - 1] * self.am) % n
        
        print("lookup tables generated.")

    def dft(self, xv):
        Xv = np.zeros(len(xv), dtype=complex)
        for j in range(len(Xv)):
            for i in range(len(xv)):
                Xv[j] += xv[i] * self.powers_a[i * j]
        return Xv

    def idft(self, Xv):
        xv = np.zeros(len(Xv), dtype=complex)
        for j in range(len(xv)):
            for i in range(len(Xv)):
                xv[j] += self.nm * Xv[i] * self.powers_am[i * j]
        return xv

class DFT_PRIM:
    def __init__(self, n, a):
        self.a = a
        self.n = n
        self.nm = pow(n, -1, n)

    def dft(self, xv):
        Xv = np.zeros(len(xv), dtype=complex)
        for j in range(len(Xv)):
            for i in range(len(xv)):
                Xv[j] += xv[i] * pow(self.a, i * j)
        return Xv

    def idft(self, Xv):
        xv = np.zeros(len(Xv), dtype=complex)
        for j in range(len(xv)):
            for i in range(len(Xv)):
                xv[j] += self.nm * Xv[i] * pow(self.a, i * j)
        return xv

def dft(xv, a):
    n = len(xv)
    powers = [1] * (n * n + 1)
    powers[1] = a
    for i in range(2, len(powers)):
        powers[i] = (powers[i - 1] * a) % n

    Xv = np.zeros(len(xv), dtype=complex)
    for j in range(len(Xv)):
        for i in range(len(xv)):
            Xv[j] += xv[i] * powers[i * j]
    return Xv

def idft(xv, Xv, a):
    n = len(xv)
    nm = pow(n, -1, n)
    am = pow(a, -1, n)

    powers = [1] * (n * n + 1)
    powers[1] = am
    for i in range(2, len(powers)):
        powers[i] = (powers[i - 1] * am) % n

    for j in range(len(Xv)):
        for i in range(len(Xv)):
            xv[j] += nm * Xv[i] * powers[i * j]
    return xv
