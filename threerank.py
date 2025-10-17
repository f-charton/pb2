from typing import Optional
import numpy as np
import math
from collections import Counter
from environment import DataPoint
from sympy import primerange, factorint, isprime
from numpy import lcm



def legendre(a, p):
    """ return the legendre symbol (a/p) """
    q = pow(a, (p - 1) // 2, p)
    return -1 if q == p - 1 else q

def nonresidue(p):
    """ returns the least nonresidue modulo an odd prime p"""
    assert p%2 == 1
    if p % 8 in [3,5]: return 2
    elif p % 12 in [5,7]: return 3
    elif p % 5 in [2,3]: return 5
    n = 7
    while legendre(n, p) != -1:
        n += 2
    return n

def sqrtmod(a, p):
    """ returns the least square root of a modulo p in [0,p-1], or -1 if none exist """
    a = a % p
    if a == 0: return 0
    elif a == 1: return 1
    elif legendre(a, p) != 1: return -1
    elif p % 4 == 3:
        x = pow(a, (p + 1) // 4, p)
        return x if 2*x < p else p - x
    # write p = 2^e*s with s odd
    s = p - 1; e = 0
    while s % 2 == 0:
        s //= 2; e += 1
    # now run Tonelli-Shanks using nonresidue s
    x = pow(a, (s + 1) // 2, p)
    b = pow(a, s, p)
    g = pow(nonresidue(p), s, p)
    r = e
    while True:
        t = b
        m = 0
        while t != 1:
            t = (t * t) % p
            m += 1
        if m == 0: return x if 2*x < p else p - x # canonicalize
        gs = pow(g, 2 ** (r - m - 1), p)
        g = (gs * gs) % p
        x = (x * gs) % p
        b = (b * g) % p
        r = m

def xgcd(a, b):
    """
    Returns (d, x, y) such that d = gcd(a, b) and ax + by = d.
    """
    if a == 0: return b, 0, 1
    d, y, x = xgcd(b % a, a)
    return d, x - (b // a) * y, y

def is_discriminant(D):
    """ Checks if -D is a negative discriminant. """
    if D <= 0: return False
    return D%4 in [0,3]


def is_mostly_squarefree(n,B):
    """ Checks if a positive integer n is square-free is divisible by the square of a small prime. """
    assert n > 0
    if n == 1: return True
    elif n % 4 == 0: return False
    elif n % 2 == 0: n //= 2
    # now n >= 3 is odd
    for p in primerange(3,B):
        if n %(p*p) == 0: return False
    return True

def is_squarefree(n,B=100):
    """ Checks if a positive integer n is square-free. """
    assert n > 0
    if n == 1: return True
    elif n % 4 == 0: return False
    elif n % 2 == 0: n //= 2
    for p in primerange(3,B):
        if n %(p*p) == 0: return False
    a = factorint(n)
    return all([a[p] == 1 for p in a])

def is_fundamental(D):
    """ Checks if -D is a negative fundamental discriminant (too slow for large D). """
    if D <= 0: return False
    elif D % 4 == 3: return is_squarefree(D)
    elif D % 4 == 0:
        k = D // 4
        return k%4 in [1,2] and is_squarefree(k)
    return False

def identity(D):
    if D%4 == 0: return 1, 0, D // 4
    else: return 1, 1, (D + 1) // 4

def reduce(a, b, c, D):
    """
    Reduces a positive definite binary quadratic form (a, b, c) of fundamental discriminant -D
    A form is reduced if |b| <= a <= c and if |b| = a or a = c, then b >= 0.
    """
    while True:
        if a > c:
            a, c = c, a
            b = -b
            continue
        if abs(b) > a:
            r = b % (2 * a)
            if r > a:
                r -= 2 * a
            q = (b - r) // (2 * a)
            c = c - q * b + q * q * a
            b = r
            continue
        if (abs(b) == a or a == c) and b < 0:
            b = -b
            continue
        return a, b, c

def compose(f1, f2, D):
    """
    Composes two primitive positive definite forms of fundamental discriminant -D
    Returns the reduced result (unique representative in cl(-D))
    """
    a1, b1, c1 = f1
    a2, b2, c2 = f2
    if a1 == 1: return reduce(a2, b2, c2, D)
    if a2 == 1: return reduce(a1, b1, c1, D)
    s = (b1 + b2) // 2
    d1, u1, v1 = xgcd(a1, a2)
    d, u2, v2 = xgcd(d1, s)
    a3 = (a1 * a2) // (d * d)
    b3 = (u2 * u1 * a1 * b2 + u2 * v1 * a2 * b1 + v2 * (b1 * b2 - D) // 2) // d
    b3_reduced = b3 % (2 * a3)
    c3 = (b3_reduced * b3_reduced + D) // (4 * a3)
    return reduce(a3, b3_reduced, c3, D)

def power_form(f, n, D):
    """ returns the nth power of f in cl(-D) using binary exponentiation """
    if n==1: return f
    g = identity(D)
    h = f
    while n > 0:
        if n % 2 == 1: g = compose(g, h, D)
        h = compose(h, h, D)
        n //= 2
    return g

def primeform(p, D):
    """ returns form (p,b,c) corresponding to a prime ideal of norm p or (0,0,0) if none exists """
    if p==2:
        if (-D) % 16 == 0: return 0, 0, 0 # avoid 2 when it divides the conductor
        if (-D) % 8 == 0: return 2, 0, D // 8
        elif (-D) % 8 == 1: return 2, 1, (D+1) // 8
        elif (-D) % 8 == 4: return 2, 2, (D+4) // 8
        else: return 0, 0, 0
    if D % (p*p) == 0: return 0, 0, 0 # avoid p dividing the conductor
    b = sqrtmod(-D, p)
    if b < 0: return 0, 0, 0
    if D%2 == 1:
        b = b if b%2 == 1 else p-b # make b odd if D is odd
    else:
        b = b if b%2 == 0 else p-b # make b even if D is even
    return p, b, (b*b + D) // (4*p)

def primeforms(D):
    """
    Returns a set of reduced primeforms that generate the class group (conditional on ERH)
    Uses Bach's bound (see Theorem 4 and comments on page 376 regarding the quadratic case in
    https://www.ams.org/journals/mcom/1990-55-191/S0025-5718-1990-1023756-8/S0025-5718-1990-1023756-8.pdf)
    """
    P = [primeform(p, D) for p in primerange(2,math.floor(6*math.log(D)**2+1)) if p==2 or D%(p*p) != 0]
    return {reduce(*f,D) for f in P if f[0] != 0}

def naive_order(f,D):
    """ computes the order of the form f in cl(-D) using brute force (for testing) """
    g = f; n = 1
    while g[0] != 1:
        g = compose(g,f,D)
        n += 1
    return n

def bsgs_order(f,D,s=2):
    """ computes the order of the form f in cl(-D) using a triangular BSGS search ()"""
    if f[0] == 1:
        return 1
    if f[1] == 0:
        return 2
    T = { identity(D):0, f:1 }
    a = f
    for i in range(2,s+1):
        a = compose(f,a,D)
        if a[0] == 1:
            return i
        T[a] = i
    j = 0
    b = compose(a,a,D)
    t = 2*s
    while True:
        if b in T:
            return t-T[b]
        a = compose(a,f,D); j += 1;   # baby step
        T[a] = j+s
        b = compose(b,a,D); t += j+s  # giant step

def three_part(n):
    """ returns v_3(n) and n/3^v_3(n) """
    e = 0
    while True:
        m = n // 3
        if 3*m != n: return e, n
        n, e = m, e+1

def triple(f,D):
    return compose(f,compose(f,f,D),D)

def group_order(G,D):
    """ returns the order of the subgroup of cl(-D) generated by the forms in G """
    S = {identity(D)}
    for g in G:
        h = g
        T = [f for f in S]
        while not h in S:
            T += [compose(f,h,D) for f in S]
            h = compose(g,h,D)
        S = {f for f in T}
    return len(S)

def get_three_rank(D):
    """ computes the 3-rank of cl(-d) """
    # we don't require -D to be fundamental (too costly to check)
    if not is_fundamental(D): return -1
    F = primeforms(D)
    # reduce primeforms to generators of the 3-sylow
    e = 1 # lower bound on the prime-to-3 part of the group exponent
    G = []
    for f in F:
        g = power_form(f,e,D)
        n = bsgs_order(g,D)
        v3, n = three_part(n)
        e = lcm(e,n)
        if v3 > 0: G.append(power_form(g,n,D))
    if len(G) <= 1: return len(G)
    # rather than using https://arxiv.org/abs/0809.3413
    # we will simply enumerate the 3-Sylow (which we expect to be small) and its cube
    n = group_order(G,D)
    G = [triple(f,D) for f in G]
    G = [g for g in G if g[0] > 1]
    m = group_order(G,D)
    e,_ = three_part(n//m)
    return e

def encode_threeranks(d,base=10, reverse=False) -> list[str]:
    """
    Encode the data as a list of tokens containing the ap and the value of the discriminant
    """
    lst = []
    for s in d.ap:
        lst.append(str(s))
    v = d.val
    w = []
    while v > 0: #@Francois I change d in v, please check I'm correct and revert otherwise.
        w.append(str(v%base))
        v=v//base
    if reverse:
        return lst + w
    else:
        return lst + w[::-1]


class GroupClass(DataPoint):
    """
    Main object class representing a group class. Contains:
     val: (absolute value of the) discriminant
     ap: list of the NB_AP first legendre symbols
     score: 3-rank
    """
    NB_AP=20
    primes = [3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73]
    def __init__(self,args):
      super().__init__(args)
      assert len(self.primes) >= self.NB_AP
      if args.val >= 0:
         self.val = args.val
      else:
          self.val = np.random.randint(1, args.max_int)
      self.ap = [-2]*self.NB_AP
      self.score = -1

    def calc_features(self):
      for i in range(self.NB_AP): #TODO
          self.ap[i] = legendre(self.val,self.primes[i]) + 1
    
    def calc_score(self):
        self.score=get_three_rank(self.val)

    @classmethod  
    def from_string(cls,data):
        dat,sc = data.split('\t')
        d = cls(int(dat))
        d.score = int(sc)
        d.calc_features()
        return d

    def encode(self,base:int=10, reverse=False) -> list[str]:
        return encode_threeranks(self,base,reverse)

    def decode(self,lst, base=10, reverse=False)-> Optional["GroupClass"]:
      """
      Decode a list of tokens to return a datapoint with the corresponding discriminant. Note: only reads the determinant and do not return the ap
      """
      if len(lst) <= GroupClass.NB_AP + 1:
          return None
      lst = lst[GroupClass.NB_AP:]
      val=0
      if reverse:
          try:
              for d in lst[::-1]:
                  v = int(d)
                  if v<0 or v>=base:
                      return None
                  val = val*base + v
          except:
              return None
      else:
          try:
              for d in lst:
                  v = int(d)
                  if v<0 or v>=base:
                      return None
                  val = val*base + v
          except:
              return None
      return GroupClass(val)