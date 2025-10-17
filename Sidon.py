from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from environment import DataPoint
import random
import math
import bisect


class SidonSetDataPoint(DataPoint):
    """
    Candidate Sidon set. The candidate is a Sidon set if the number of collisions is zero.
    """

    def __init__(self, params):
        super().__init__(params)
        self.N: int = int(params.N)
        self.M: int = int(params.M)
        self.hard : bool = params.hard

        seed = params.seed
        if seed is not None:
            random.seed(seed)

        self.steps: int = int(params.sidon_steps)

        #Probabilities of the different moves for the local search

        self.insert_prob: float = float(params.insert_prob)
        self.delete_prob: float = float(params.delete_prob)
        self.shift_prob: float  = float(params.shift_prob)

        total_prob = self.insert_prob + self.delete_prob + self.shift_prob
        if total_prob <= 0:
            self.insert_prob, self.delete_prob, self.shift_prob = 0.35, 0.10, 0.55
        else:
            self.insert_prob /= total_prob
            self.delete_prob /= total_prob
            self.shift_prob  /= total_prob


        self.temp: float = float(params.temp0)
        self.temp_decay: float = float(params.temp_decay)

        # Initialize candidate
        self.init_method = params.init_method or "random_greedy"

        if isinstance(params.val,list):
            self.val = sorted(set(int(x) for x in params.val if 0 <= int(x) <= self.N))
        else:
            if params.init_k > 0:
                target_k = int(params.init_k)
            elif params.init_k == 0:
                target_k = max(1, int(math.sqrt(self.N)))
            else:
                target_k = None
            if self.init_method == "random_greedy":
                self.val = self._seed_random_greedy(target_k)
            elif self.init_method == "mian_chowla":
                self.val = self._seed_mian_chowla()
            else:
                self.val = self._seed_evenly_spaced(int(params.init_k), jitter=bool(params.jitter_init))

        # Using the fact that a set is Sidon is all the positive differences between to elements a-x are distincts. We keep in memory the differences
        self.diffs_count = [0] * (self.N + 1)
        self._build_diffs()
        # Initial score/features
        self.score = self.calc_score()
        self.features = self.calc_features()




    def calc_score(self):
        """
        Compute score.
        """
        collisions = self._count_collisions()
        if self.hard and collisions > 0:
            self.score = -1
            return
        #TODO Currently the soft score penalizes the collision, but it is not exactly a Sidon set so one could create "invalid" instead by returning -1 if collisions > 0
        self.score = self.M * len(self.val) - collisions


    def calc_features(self):
        """
        Compute optional features to give more signal to the transformer
        """
        pass

    def encode(self, base=10, reverse=False):
        w = []
        for el in self.val:
            v = el
            curr_w = []
            while v > 0: #@Francois I change d in v, please check I'm correct and revert otherwise.
                curr_w.append(str(v%base))
                v=v//base
            w.extend(curr_w)
            w.append("|")
        return w

    def decode(self,lst):
        pass

    def local_search(self) -> None:
        """
        Very basic search: incremental hill-climb with simulated annealing.
        Moves:
          - shift: pick an index i and try a_{i} -> a_{i} \pm 1
          - insert: propose x in [0,N]\val at a low-conflict position
          - delete: remove the mark that yields the worst sum-collision burden
        """
        #TODO could probably be improved using a better known search. This could also be included as an alternative generation method during the loop.
        for _ in range(self.steps):
            r = random.random()
            if r < self.shift_prob:
                self._move_shift()
            elif r < self.shift_prob + self.insert_prob:
                self._move_insert()
            else:
                self._move_delete()
            self.temp *= self.temp_decay

        # finalize score/features from scratch (safety)
        self.score = self.calc_score()
        self.features = self.calc_features()

    # -----------------------
    # Methods specific to the problem
    # -----------------------


    ### Generation methods ###

    def _seed_evenly_spaced(self, k: int, jitter: bool = True) -> List[int]:
        """
        Generate a sequence of k elements evenly spaced in [0,N].
        If jitter is set to True, then randomly tries to move around the elements by +1 / -1 as long as they still belong to the set.
        """
        if k <= 0:
            return []
        if k == 1:
            return [0]
        raw = [round(i * self.N / (k - 1)) for i in range(k)]
        raw = sorted(set(raw))

        # Optional jitter
        if jitter:
            tried = set(raw)
            for idx in range(len(raw)):
                if random.random() < 0.6:
                    delta = random.choice([-1, 1])
                    cand = raw[idx] + delta
                    if 0 <= cand <= self.N and cand not in tried:
                        tried.remove(raw[idx])
                        raw[idx] = cand
                        tried.add(cand)
        return sorted(set(raw))



    def _seed_random_greedy(self, target_k: Optional[int]) -> List[int]:
        """
        Random order through [0..N], uses the fact that a set is Sidon if
        all the positive differences a-x for a>x are distinct (here uses |x-a|)
        """
        order = list(range(self.N + 1))
        random.shuffle(order)
        A: List[int] = []
        used_diff = [False] * (self.N + 1)  #Store the positive differences
        for x in order:
            ok = True
            for a in A:
                d = abs(x - a)
                if d == 0 or used_diff[d]:
                    #either we already have this value or this value will create a collision
                    ok = False
                    break
            if ok:
                for a in A:
                    used_diff[abs(x - a)] = True
                bisect.insort(A, x)
                if target_k is not None and len(A) >= target_k:
                    break
        return A

    def _seed_mian_chowla(self) -> List[int]:
        """
        Use the Mian-Chowla method: deterministic (could be cached) but less variety
        """
        A: List[int] = []
        used_diff = [False] * (self.N + 1)
        for x in range(self.N + 1):
            ok = True
            for a in A:
                d = x - a if x >= a else a - x
                if d == 0 or used_diff[d]:
                    ok = False
                    break
            if ok:
                for a in A:
                    used_diff[abs(x - a)] = True
                A.append(x)
        return A


    ### Methods for computing scores and local search ###

    def _build_diffs(self) -> None:
        """
        Re-compute the diffs count from scratch. Assumes that the set is ordered.
        """
        self.diffs_count = [0] * (self.N + 1)
        v = self.val
        for i, a in enumerate(v):
            for j in range(i+1, len(v)):
                d = v[j] - a
                self.diffs_count[d] += 1

    def _count_collisions(self) -> int:
        """
        Count collisions in the set without multiplicities.
        """
        seen = [0] * (self.N + 1)
        collisions = 0
        v = self.val
        for i, a in enumerate(v):
            for j in range(i+1, len(v)):
                d = v[j] - a
                if seen[d] >= 1:
                    collisions += 1
                seen[d] += 1
        return collisions

    def _current_collisions(self) -> int:
        """
        Compute collisions from diffs with mutiplicities.
        """
        return sum((c - 1) for c in self.diffs_count[1:] if c >= 1)


    def _anneal_accept(self, delta_score: float) -> bool:
        """
        Metropolis acceptance on negative delta_score.
        """
        if self.temp <= 0:
            return False
        return random.random() < math.exp(delta_score / max(1e-9, self.temp))



    ### Move for local search ###

    def _add_element(self, x: int) -> int:
        """
        Insert mark x into self.val; update pair sums incrementally.
        Returns delta_collisions (new_collisions - old_collisions).
        """
        v = self.val
        pos = bisect.bisect_left(v, x)

        delta = 0
        # Compute the collisions potentially created by larger elements than x
        for a in v[:pos]:
            d = x - a
            old = self.diffs_count[d]
            if old >= 1:
                delta += 1
            self.diffs_count[d] = old + 1

        # Compute the collisions potentially created by smaller elements than x
        for b in v[pos:]:
            d = b - x
            old = self.diffs_count[d]
            if old >= 1:
                delta += 1
            self.diffs_count[d] = old + 1

        bisect.insort(v, x)
        return delta

    def _remove_element(self, x: int) -> int:
        """
        Remove element x from the set self.val; update pair sums incrementally.
        Returns difference of collisions delta_collisions = (new_collisions - old_collisions).
        """
        v = self.val
        pos = bisect.bisect_left(v, x)
        if pos >= len(v) or v[pos] != x:
            raise RuntimeError(f"Element {x} not in the set {v}")

        delta = 0
        # Compute the collisions potentially removed by larger elements than x
        for a in v[:pos]:
            d = x - a
            old = self.diffs_count[d]
            if old >= 2: delta -= 1
            self.diffs_count[d] = old - 1

        # Compute the collisions potentially removed by smaller elements than x
        for b in v[pos+1:]:
            d = b - x
            old = self.diffs_count[d]
            if old >= 2: delta -= 1
            self.diffs_count[d] = old - 1

        v.pop(pos)
        return delta

    # def _add_element(self, x: int) -> int:
    #     """
    #     Insert mark x into self.val; update pair sums incrementally.
    #     Returns delta_collisions (new_collisions - old_collisions).
    #     """
    #     delta = 0
    #     v = self.val
    #     # Compute the occurence of the first pair created by x (that is x+x)
    #     s = x + x
    #     old = self.sums_count.get(s, 0)
    #     delta += 1 if old >= 1 else 0  # adding one more creates a collision iff old>=1
    #     self.sums_count[s] = old + 1

    #     for y in v:
    #         s = x + y
    #         old = self.sums_count.get(s, 0)
    #         delta += 1 if old >= 1 else 0
    #         self.sums_count[s] = old + 1
    #     # Insert into val (keep sorted)
    #     bisect.insort(v, x)
    #     return delta

    # def _remove_element(self, x: int) -> int:
    #     """
    #     Remove element x from the set self.val; update pair sums incrementally.
    #     Returns difference of collisions delta_collisions = (new_collisions - old_collisions).
    #     """
    #     delta = 0
    #     v = self.val
    #     # Look at all the pairs to which x contributed, starting by x+x
    #     s = x + x
    #     old = self.sums_count.get(s, 0)
    #     # removing reduces a collision iff old>=2 which meant that there was a collision at value s
    #     delta -= 1 if old >= 2 else 0
    #     newc = old - 1

    #     #Update the dict of collisions
    #     if newc == 0:
    #         self.sums_count.pop(s, None)
    #     else:
    #         self.sums_count[s] = newc

    #     # do the same for all the values s = x+y, could be factorized
    #     for y in v:
    #         if y == x:
    #             continue
    #         s = x + y
    #         old = self.sums_count.get(s, 0)
    #         delta -= 1 if old >= 2 else 0
    #         newc = old - 1
    #         if newc == 0:
    #             self.sums_count.pop(s, None)
    #         else:
    #             self.sums_count[s] = newc

    #     # Remove from val
    #     idx = bisect.bisect_left(v, x)
    #     if idx < len(v) and v[idx] == x:
    #         v.pop(idx)
    #     return delta

    def _shift_element(self, old_x: int, new_x: int) -> int:
        """
        Shift an element of the set old_x -> new_x then compute delta_collisions.
        """
        # Remove old_x, then add new_x
        # Recall that _remove_elements and _add_elements already do the update and return the delta of collisions
        delta = self._remove_element(old_x)
        delta += self._add_element(new_x)
        return delta

    # ---- move proposals ----
    def _move_shift(self) -> None:
        """
        First possibility of move for the local phase: select one element of the sum, shift it by +1 or -1
        and then see if it improves the score. If so, then update the candidate set. If not, leave the set unchanged.
        Optionally: can accept the modification even if it decreases the score using simulated annealing to avoid being stuck in local minima.
        """
        if not self.val:
            raise RuntimeWarning("Impossible to apply move_shift, no val")
        i = random.randrange(len(self.val))
        x = self.val[i]
        direction = random.choice([-1, 1])
        new_x = x + direction
        if not (0 <= new_x <= self.N):
            return None
        if new_x in self.val:
            return None # keep distinct elements of the sequence

        # Evaluate delta score incrementally
        if self.hard:
            # compute delta collisions incrementally.
            # TODO Given that feasible is often false, we could change the functions add and remove elements to raise any collision, currently shifting twice might be costly
            # TODO we could keep sometimes with simulated annealing otherwise the local search will never create a valid example.
            _ = self._shift_element(x, new_x)
            feasible = (self._current_collisions() == 0)
            if feasible:
                # keep; recompute score exactly
                self.score = len(self.val)
            else:
                # revert
                self._shift_element(new_x, x)
            return

        delta_coll = self._shift_element(x, new_x)
        delta_score = - delta_coll  # Note that the size of the sequence is unchanged
        accept = (delta_score >= 0) or self._anneal_accept(delta_score)
        if not accept:
            # revert
            self._shift_element(new_x, x)
        else:
            self.score += delta_score

    def _move_insert(self) -> None:
        """
        Second possibility of move for the local phase: insert an x in the subset
        """
        # pick a candidate location; bias toward low-collision sums
        if len(self.val) >= self.N + 1:
            return
        # Try a few samples and pick the best
        trials = min(8, self.N + 1 - len(self.val))
        best_x = None
        best_delta = None
        for _ in range(trials):
            x = random.randrange(self.N + 1)
            if x in self.val:
                continue
            v = self.val
            pos = bisect.bisect_left(v, x)
            delta = 0
            for a in v[:pos]:
                if self.diffs_count[x - a] >= 1:
                    delta += 1
            for b in v[pos:]:
                if self.diffs_count[b - x] >= 1:
                    delta += 1
            if (best_delta is None) or (delta < best_delta):
                best_delta = delta
                best_x = x
        if best_x is None:
            return

        if self.hard:
            # only insert if it creates no collisions
            if best_delta == 0:
                self._add_element(best_x)
                self.score = len(self.val)
            return

        # soft mode: evaluate acceptance
        delta_coll = self._add_element(best_x)
        delta_score = self.M * 1 - delta_coll
        accept = (delta_score >= 0) or self._anneal_accept(delta_score)
        if not accept:
            # revert
            self._remove_element(best_x)
        else:
            self.score += delta_score

    def _move_delete(self) -> None:
        """
        Third move of the local search: delete an element.
        In hard mode (we don't accept any subset with a collision), should only be used with simulated annealing or to fix examples.
        """
        if not self.val:
            return
        # Choose the mark whose removal most reduces collisions.
        # Score removal by the number of pair-sums involving it that currently collide.
        best_idx = None
        best_gain = None
        v = self.val
        for idx, x in enumerate(v):
            gain = 0
            for a in v[:idx]:
                if self.diffs_count[x - a] >= 2:
                    gain += 1
            for b in v[idx+1:]:
                if self.diffs_count[b - x] >= 2:
                    gain += 1
            if (best_gain is None) or (gain > best_gain):
                best_gain, best_idx = gain, idx

        x = self.val[best_idx]

        if self.hard:
            # if currently infeasible, allow deletion; else avoid
            if self._current_collisions() > 0:
                self._remove_element(x)
                self.score = len(self.val)
            return

        old_collisions = self._current_collisions()
        delta_coll = self._remove_element(x)
        new_collisions = old_collisions + delta_coll
        delta_score = self.M * (-1) - delta_coll
        accept = (delta_score >= 0) or self._anneal_accept(delta_score)
        if not accept:
            # revert
            self._add_element(x)
        else:
            self.score += delta_score



# On pourrait générer des éléments, puis passer à une destructions d'élements jusqu'à ce qu'il n'y ait plus de collision et garder ce qui reste puis filtrer.
# Alternativement on pourrait générer depuis une meilleure seed, mais on risque de sur-spécialiser le modèle.