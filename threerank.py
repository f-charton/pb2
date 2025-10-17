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



# @njit
# def extended_gcd(a, b):
#   if a == 0:
#     return b, 0, 1
#   g, y, x = extended_gcd(b % a, a)
#   return g, x - (b // a) * y, y


# @njit
# def reduce_form(a, b, c, D):
#   while True:
#     if a > c:
#       a, c = c, a
#       b = -b
#       continue
#     if abs(b) > a:
#       r = b % (2 * a)
#       if r > a:
#         r -= 2 * a
#       q = (b - r) // (2 * a)
#       c = c - q * b + q * q * a
#       b = r
#       continue
#     if (abs(b) == a or a == c) and b < 0:
#       b = -b
#       continue
#     return a, b, c


# @njit
# def compose_forms(form1, form2, D):
#   a1, b1, c1 = form1
#   a2, b2, c2 = form2
#   if a1 == 1:
#     return reduce_form(a2, b2, c2, D)
#   if a2 == 1:
#     return reduce_form(a1, b1, c1, D)
#   s = (b1 + b2) // 2
#   d1, u1, v1 = extended_gcd(a1, a2)
#   d, u2, v2 = extended_gcd(d1, s)
#   u_comp = u2 * u1
#   v_comp = u2 * v1
#   w_comp = v2
#   a3 = (a1 * a2) // (d * d)
#   b3 = (u_comp * a1 * b2 + v_comp * a2 * b1 + w_comp * (b1 * b2 - D) // 2) // d
#   b3_reduced = b3 % (2 * a3)
#   c3 = (b3_reduced * b3_reduced + D) // (4 * a3)
#   return reduce_form(a3, b3_reduced, c3, D)


# @njit
# def power_form(form, n, D, identity_form):
#   res = identity_form
#   base = form
#   while n > 0:
#     if n % 2 == 1:
#       res = compose_forms(res, base, D)
#     base = compose_forms(base, base, D)
#     n //= 2
#   return res


# @njit
# def get_divisors(n):
#   divs_list = []
#   for i in range(1, int(math.sqrt(n)) + 1):
#     if n % i == 0:
#       divs_list.append(i)
#       if i * i != n:
#         divs_list.append(n // i)
#   return sorted(divs_list)


# def get_prime_factorization(n):  # This is not JIT-compiled
#   factors = {}
#   d = 2
#   temp = n
#   while d * d <= temp:
#     while (temp % d) == 0:
#       factors[d] = factors.get(d, 0) + 1
#       temp //= d
#     d += 1
#   if temp > 1:
#     factors[temp] = factors.get(temp, 0) + 1
#   return factors

# def is_square_free(n):
#     """Checks if a positive integer n is square-free."""
#     if n <= 0: return False
#     if n % 4 == 0: return False
#     limit = int(math.sqrt(n)) + 1
#     for i in range(2, limit):
#         if n % (i * i) == 0:
#             return False
#     return True


# @njit
# def _get_forms_and_orders_jit(D):
#   reduced_forms = []
#   limit_b = int(math.sqrt(D / 3.0))
#   for b in range(-limit_b, limit_b + 1):
#     if (b * b + D) % 4 != 0:
#       continue
#     ac = (b * b + D) // 4
#     limit_a = int(math.sqrt(ac))
#     for a in range(max(1, abs(b)), limit_a + 1):
#       if ac % a == 0:
#         c = ac // a
#         if a <= c:
#           if (abs(b) == a or a == c) and b < 0:
#             continue
#           reduced_forms.append((a, b, c))
#   reduced_forms.sort()
#   h = len(reduced_forms)
#   if h == 0:
#     # Numba requires consistent return types. We return a list of tuples and an empty list of integers.
#     # An empty list for orders is created like this to help Numba's type inference.
#     orders = [0]
#     return reduced_forms, orders[:0]
#   identity_form = reduced_forms[0]
#   divisors_of_h = get_divisors(h)
#   orders = []
#   for form in reduced_forms:
#     found_order = False
#     for d in divisors_of_h:
#       if power_form(form, d, D, identity_form) == identity_form:
#         orders.append(d)
#         found_order = True
#         break
#     if not found_order:
#       return reduced_forms, orders
#   return reduced_forms, orders


# def is_fundamental_discriminant(D):
#   """Checks if -D is a negative fundamental discriminant."""
#   if D <= 0:
#     return False
#   # Case 1: -D = 1 (mod 4) => D = 3 (mod 4)
#   if D % 4 == 3:
#     return is_square_free(D)
#   # Case 2: -D = 4k, k = 2,3 (mod 4)
#   if D % 4 == 0:
#     k = D // 4
#     if k % 4 == 1 or k % 4 == 2:  # Corresponds to -D/4 = 3,2 (mod 4)
#       return is_square_free(k)
#   return False


# def get_class_group_info(D):
#   reduced_forms, orders = _get_forms_and_orders_jit(D)
#   h = len(reduced_forms)
#   if h == 0:
#     return 0, []
#   if len(orders) != h:
#     return h, [-1]
#   if h == 1:
#     return 1, []
#   elementary_divisors = {}
#   h_factors = get_prime_factorization(h)
#   print("First done")
#   for p, total_exponent in h_factors.items():
#     print("one more done")
#     R = [0] * (total_exponent + 1)
#     for k in range(1, total_exponent + 1):
#       pk = p**k
#       num_elements = sum(1 for o in orders if pk % o == 0)
#       if num_elements > 0:
#         math.log_val = math.math.log(num_elements, p)
#         if abs(math.log_val - round(math.log_val)) > 1e-9:
#           return h, [-2]
#         R[k] = int(round(math.log_val))
#     c = [0] * (total_exponent + 1)
#     for k in range(1, total_exponent + 1):
#       c[k] = R[k] - R[k - 1]
#     e = [0] * (total_exponent + 1)
#     for k in range(1, total_exponent + 1):
#       e[k] = c[k] - (c[k + 1] if k < total_exponent else 0)
#     p_divs = []
#     for k in range(1, total_exponent + 1):
#       p_divs.extend([p**k] * e[k])
#     elementary_divisors[p] = p_divs
#   inv_factors = []
#   max_len = (
#       max(len(v) for v in elementary_divisors.values())
#       if elementary_divisors
#       else 0
#   )
#   for p in elementary_divisors:
#     elementary_divisors[p].sort(reverse=True)
#     elementary_divisors[p].extend([1] * (max_len - len(elementary_divisors[p])))
#   for i in range(max_len):
#     factor = 1
#     for p in elementary_divisors:
#       factor *= elementary_divisors[p][i]
#     inv_factors.append(factor)
#   inv_factors = sorted([f for f in inv_factors if f > 1])
#   return h, inv_factors


# def get_three_rank(D):
#   """Computes the 3-rank. This is a wrapper around the other functions."""
#   if not is_fundamental_discriminant(D):
#     return -1
#   h, inv_factors = get_class_group_info(D)
#   if not inv_factors and h > 1:
#     return -1  # Error code
#   return sum(1 for factor in inv_factors if factor % 3 == 0)

# def legendre_symbol(a: int, p: int) -> int:
#     """
#     Return (a/p) for odd prime p, in {-1,0,1}.
#     Uses Euler's criterion. Assumes p is an odd prime.
#     """
#     a %= p
#     if a == 0:
#         return 0
#     r = pow(a, (p - 1) // 2, p)   # r ∈ {1, p-1}
#     return 1 if r == 1 else -1

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