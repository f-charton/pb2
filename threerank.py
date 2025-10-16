from typing import Optional
from numba import njit
import numpy as np
import math
from collections import Counter
from environment import DataPoint





@njit
def extended_gcd(a, b):
  if a == 0:
    return b, 0, 1
  g, y, x = extended_gcd(b % a, a)
  return g, x - (b // a) * y, y


@njit
def reduce_form(a, b, c, D):
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


@njit
def compose_forms(form1, form2, D):
  a1, b1, c1 = form1
  a2, b2, c2 = form2
  if a1 == 1:
    return reduce_form(a2, b2, c2, D)
  if a2 == 1:
    return reduce_form(a1, b1, c1, D)
  s = (b1 + b2) // 2
  d1, u1, v1 = extended_gcd(a1, a2)
  d, u2, v2 = extended_gcd(d1, s)
  u_comp = u2 * u1
  v_comp = u2 * v1
  w_comp = v2
  a3 = (a1 * a2) // (d * d)
  b3 = (u_comp * a1 * b2 + v_comp * a2 * b1 + w_comp * (b1 * b2 - D) // 2) // d
  b3_reduced = b3 % (2 * a3)
  c3 = (b3_reduced * b3_reduced + D) // (4 * a3)
  return reduce_form(a3, b3_reduced, c3, D)


@njit
def power_form(form, n, D, identity_form):
  res = identity_form
  base = form
  while n > 0:
    if n % 2 == 1:
      res = compose_forms(res, base, D)
    base = compose_forms(base, base, D)
    n //= 2
  return res


@njit
def get_divisors(n):
  divs_list = []
  for i in range(1, int(math.sqrt(n)) + 1):
    if n % i == 0:
      divs_list.append(i)
      if i * i != n:
        divs_list.append(n // i)
  return sorted(divs_list)


def get_prime_factorization(n):  # This is not JIT-compiled
  factors = {}
  d = 2
  temp = n
  while d * d <= temp:
    while (temp % d) == 0:
      factors[d] = factors.get(d, 0) + 1
      temp //= d
    d += 1
  if temp > 1:
    factors[temp] = factors.get(temp, 0) + 1
  return factors

def is_square_free(n):
    """Checks if a positive integer n is square-free."""
    if n <= 0: return False
    if n % 4 == 0: return False
    limit = int(math.sqrt(n)) + 1
    for i in range(2, limit):
        if n % (i * i) == 0:
            return False
    return True


@njit
def _get_forms_and_orders_jit(D):
  reduced_forms = []
  limit_b = int(math.sqrt(D / 3.0))
  for b in range(-limit_b, limit_b + 1):
    if (b * b + D) % 4 != 0:
      continue
    ac = (b * b + D) // 4
    limit_a = int(math.sqrt(ac))
    for a in range(max(1, abs(b)), limit_a + 1):
      if ac % a == 0:
        c = ac // a
        if a <= c:
          if (abs(b) == a or a == c) and b < 0:
            continue
          reduced_forms.append((a, b, c))
  reduced_forms.sort()
  h = len(reduced_forms)
  if h == 0:
    # Numba requires consistent return types. We return a list of tuples and an empty list of integers.
    # An empty list for orders is created like this to help Numba's type inference.
    orders = [0]
    return reduced_forms, orders[:0]
  identity_form = reduced_forms[0]
  divisors_of_h = get_divisors(h)
  orders = []
  for form in reduced_forms:
    found_order = False
    for d in divisors_of_h:
      if power_form(form, d, D, identity_form) == identity_form:
        orders.append(d)
        found_order = True
        break
    if not found_order:
      return reduced_forms, orders
  return reduced_forms, orders


def is_fundamental_discriminant(D):
  """Checks if -D is a negative fundamental discriminant."""
  if D <= 0:
    return False
  # Case 1: -D = 1 (mod 4) => D = 3 (mod 4)
  if D % 4 == 3:
    return is_square_free(D)
  # Case 2: -D = 4k, k = 2,3 (mod 4)
  if D % 4 == 0:
    k = D // 4
    if k % 4 == 1 or k % 4 == 2:  # Corresponds to -D/4 = 3,2 (mod 4)
      return is_square_free(k)
  return False


def get_class_group_info(D):
  reduced_forms, orders = _get_forms_and_orders_jit(D)
  h = len(reduced_forms)
  if h == 0:
    return 0, []
  if len(orders) != h:
    return h, [-1]
  if h == 1:
    return 1, []
  elementary_divisors = {}
  h_factors = get_prime_factorization(h)
  for p, total_exponent in h_factors.items():
    R = [0] * (total_exponent + 1)
    for k in range(1, total_exponent + 1):
      pk = p**k
      num_elements = sum(1 for o in orders if pk % o == 0)
      if num_elements > 0:
        log_val = math.log(num_elements, p)
        if abs(log_val - round(log_val)) > 1e-9:
          return h, [-2]
        R[k] = int(round(log_val))
    c = [0] * (total_exponent + 1)
    for k in range(1, total_exponent + 1):
      c[k] = R[k] - R[k - 1]
    e = [0] * (total_exponent + 1)
    for k in range(1, total_exponent + 1):
      e[k] = c[k] - (c[k + 1] if k < total_exponent else 0)
    p_divs = []
    for k in range(1, total_exponent + 1):
      p_divs.extend([p**k] * e[k])
    elementary_divisors[p] = p_divs
  inv_factors = []
  max_len = (
      max(len(v) for v in elementary_divisors.values())
      if elementary_divisors
      else 0
  )
  for p in elementary_divisors:
    elementary_divisors[p].sort(reverse=True)
    elementary_divisors[p].extend([1] * (max_len - len(elementary_divisors[p])))
  for i in range(max_len):
    factor = 1
    for p in elementary_divisors:
      factor *= elementary_divisors[p][i]
    inv_factors.append(factor)
  inv_factors = sorted([f for f in inv_factors if f > 1])
  return h, inv_factors


def get_three_rank(D):
  """Computes the 3-rank. This is a wrapper around the other functions."""
  if not is_fundamental_discriminant(D):
    return -1
  h, inv_factors = get_class_group_info(D)
  if not inv_factors and h > 1:
    return -1  # Error code
  return sum(1 for factor in inv_factors if factor % 3 == 0)

def legendre_symbol(a: int, p: int) -> int:
    """
    Return (a/p) for odd prime p, in {-1,0,1}.
    Uses Euler's criterion. Assumes p is an odd prime.
    """
    a %= p
    if a == 0:
        return 0
    r = pow(a, (p - 1) // 2, p)   # r ∈ {1, p-1}
    return 1 if r == 1 else -1

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
        w.push_back(str(v%base))
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
          self.ap[i] = legendre_symbol(self.val,self.primes[i]) + 1
    
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