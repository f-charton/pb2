from __future__ import annotations
from abc import ABC, abstractmethod
from logging import getLogger
from typing import Dict, List, Tuple, Optional
from envs.environment import BaseEnvironment, DataPoint
import random
from collections import Counter
import math
import bisect

from envs.tokenizers import SidonTokenizer
from utils import bool_flag

logger = getLogger()

# Sidon utils

def _parse_baseN(digits, base, limit=None):
    acc = 0
    for v in digits:
        if v < 0 or v >= base:
            raise ValueError(f"Digit {v} out of range for base {base}")
        acc = acc * base + v
    return acc



class SidonSetDataPoint(DataPoint):
    """
    Candidate Sidon set. The candidate is a Sidon set if the number of collisions is zero.
    """

    def __init__(self, val:Optional[List]=None, init=False):
        super().__init__()

        total_prob = self.insert_prob + self.delete_prob + self.shift_prob
        if total_prob <= 0:
            self.insert_prob, self.delete_prob, self.shift_prob = 0.35, 0.10, 0.55
        else:
            self.insert_prob /= total_prob
            self.delete_prob /= total_prob
            self.shift_prob  /= total_prob


        if init:
            if val is not None:
                self.val = sorted(set(int(x) for x in val if 0 <= int(x) <= self.N))
            else:
                if self.init_k > 0:
                    target_k = int(self.init_k)
                elif self.init_k == 0:
                    target_k = max(1, int(math.sqrt(self.N)))
                else:
                    target_k = None
                if self.init_method == "random_greedy":
                    self.val = self._seed_random_greedy(target_k)
                elif self.init_method == "mian_chowla":
                    self.val = self._seed_mian_chowla()
                else:
                    self.val = self._seed_evenly_spaced(int(self.init_k), jitter=bool(self.jitter_init))
                #Sort the set
                self.val = sorted(self.val)

            # Using the fact that a set is Sidon is all the positive differences between to elements a-x are distincts. We keep in memory the differences
            self.diffs_count = [0] * (self.N + 1)
            # Initial score/features
            self.calc_score()
            self.calc_features()
        else:
            assert isinstance(val,list)
            self.val = sorted(set(int(x) for x in val if 0 <= int(x) <= self.N))


    def calc_score(self):
        """
        Compute score.
        """
        collisions = self._count_collisions()
        if self.hard and collisions > 0:
            self.score = -1
            assert len(self.val) > 0
            return
        #TODO Currently the soft score penalizes the collision, but it is not exactly a Sidon set so one could create "invalid" instead by returning -1 if collisions > 0
        self.score = self.M * len(self.val) - collisions


    def calc_features(self):
        """
        Compute optional features to give more signal to the transformer
        """
        pass


    def local_search(self) -> None:
        """
        Very basic search: incremental hill-climb with simulated annealing.
        Moves:
          - shift: pick an index i and try a_{i} -> a_{i} \pm 1
          - insert: propose x in [0,N]\val at a low-conflict position
          - delete: remove the mark that yields the worst sum-collision burden
          NOTE: currently the local search is mainly called to fix elements that have a problem, as a consequence it could be done differently:
          - deleting the elements that provide collisions
          - starting a random greedy from the set provided
        """
        #TODO could probably be improved using a better known search. This could also be included as an alternative generation method during the loop.
        step_left = self.steps
        # print(f"HERE in local search {self.val} and score {self.score}")
        if len(self.val) == 0:
            raise RuntimeError(f"Empty val at the beginning of local search with object {self}")

        while self.score <0 and self.hard and step_left > 0:
            self._move_delete()
            # print(f"HERE after move delete {self.val} and score {self.score}")
            step_left -= 1
        if self.score < 0:
            raise RuntimeError("score negative even after local_search, should not be possible")
        if len(self.val) == 0:
            logger.info(f"Error no val with {self.diffs_count()}")

        # old_score = self.score # debug
        # print("HERE OLD", old_score)
        # self.calc_score()
        # new_score = self.score
        # assert old_score == new_score, new_score

        for _ in range(step_left):
            r = random.random()
            if r < self.shift_prob:
                self._move_shift()
                # score1 = self.score
                # print("shift",self.score)
                # # debug
                # self.calc_score()
                # score2 = self.score
                # assert score2 == score1
                # print("score2",self.score)

            elif r < self.shift_prob + self.insert_prob:
                self._move_insert()
                # score1 = self.score
                # print("insert",self.score)
                # # debug
                # self.calc_score()
                # score2 = self.score
                # assert score2 == score1
                # print("score2",self.score)

            else:
                self._move_delete()
                # score1 = self.score
                # print("delete",self.score)
                # # debug
                # self.calc_score()
                # score2 = self.score
                # assert score2 == score1
                # print("score2",self.score)

            self.temp *= self.temp_decay

        # finalize score/features from scratch
        self.calc_score()
        self.calc_features()
        # print("HERE NEW", self.score)

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
            local = set()
            for a in A:
                d = abs(x - a)
                if d == 0 or d in local or used_diff[d]:
                    #either we already have this value or this value will create a collision
                    ok = False
                    break
                local.add(d)
            if ok:
                for a in A:
                    used_diff[abs(x - a)] = True
                bisect.insort(A, x)
                if target_k is not None and len(A) >= target_k:
                    break
        # print(f"Here generated A {A}")
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
        Count collisions in the set with multiplicities.
        """
        self._build_diffs()
        return self._current_collisions()

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


    def _shift_element(self, old_x: int, new_x: int) -> int:
        """
        Shift an element of the set old_x -> new_x then compute delta_collisions.
        """
        # Remove old_x, then add new_x
        # Recall that _remove_elements and _add_elements already do the update and return the delta of collisions
        delta = self._remove_element(old_x)
        delta += self._add_element(new_x)
        return delta

    def _move_shift(self) -> None:
        """
        First possibility of move for the local phase: select one element of the sum, shift it by +1 or -1
        and then see if it improves the score. If so, then update the candidate set. If not, leave the set unchanged.
        Optionally: can accept the modification even if it decreases the score using simulated annealing to avoid being stuck in local minima.
        """
        if not self.val:
            logger.info("Error, impossible to apply move_shift no val")
            logger.info(f"val is {self.val} with length {len(self.val)}")
            # raise RuntimeWarning("Impossible to apply move_shift, no val")
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
            _ = self._shift_element(old_x=x, new_x=new_x)
            feasible = (self._current_collisions() == 0)
            if feasible:
                # keep; score does not change
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
        Second possibility of move for the local phase: insert an x in the subset.
        In hard mode, we only allow insertions that keep the set collision-free.
        """
        v = self.val

        trials = min(8, self.N + 1 - len(v)) # how many candidate xs we sample
        best_x = None
        best_delta = None
        for _ in range(trials):
            x = random.randrange(self.N + 1)
            if x in v:
                continue

            delta = self._predicted_collision_increase_if_insert_x(x)

            if (best_delta is None) or (delta < best_delta):
                best_delta = delta
                best_x = x
        if best_x is None:
            return

        if self.hard:
            # only insert if it creates no NEW collisions
            if best_delta == 0:
                self._add_element(best_x)
                # after insertion, still collision-free: score is size
                self.score = len(self.val)
            return

        # soft mode (simulated annealing / hill-climb with penalties)
        delta_coll = self._add_element(best_x)
        delta_score = self.M * 1 - delta_coll
        accept = (delta_score >= 0) or self._anneal_accept(delta_score)
        if not accept:
            # revert
            self._remove_element(best_x)
        else:
            self.score += delta_score

    def _predicted_collision_increase_if_insert_x(self, x: int) -> int:
        """
        Exact predicted INCREASE in total collisions if we insert x into self.val.
        Collisions are Σ_d max(0, diffs_count[d] - 1).
        """
        v = self.val
        pos = bisect.bisect_left(v, x)

        # Count how many new pairs (x, ·) produce each difference d
        k = Counter()

        # pairs (x - a) for a < x
        for a in v[:pos]:
            k[x - a] += 1
        # pairs (b - x) for b > x
        for b in v[pos:]:
            k[b - x] += 1

        increase = 0
        for d, kx in k.items():
            c = self.diffs_count[d]
            before = max(0, c - 1)
            after = max(0, c + kx - 1)
            increase += (after - before)

        return increase

    def _predicted_collision_drop(self, idx: int) -> int:
        """
        Predict the number of collisions removed if we remove the idx-th element
        """
        x = self.val[idx]
        used = {}
        drop = 0

        for a in self.val[:idx]:
            d = x - a
            c = self.diffs_count[d] - used.get(d, 0) #this to make sure we don't count twice a collision with a middle point e.g. 1,4,7
            if c >= 2:
                drop += 1
                used[d] = used.get(d, 0) + 1

        for b in self.val[idx+1:]:
            d = b - x
            c = self.diffs_count[d] - used.get(d, 0)
            if c >= 2:
                drop += 1
                used[d] = used.get(d, 0) + 1

        return drop

    def _move_delete(self) -> None:
        """
        Third move of the local search: delete an element.
        In hard mode (we don't accept any subset with a collision), should only be used with simulated annealing or to fix examples.
        """
        if not self.val:
            return
        # Choose the mark whose removal most reduces collisions.
        # Score removal by the number of differences involving it that currently collide.
        best_idx = None
        best_gain = None
        assert isinstance(self.val,list)
        v = self.val


        if self.hard and self.score > 0:
            #Change here to have simulated annealing but keep at least 2 elements.
            return


        for idx, _ in enumerate(v):
            gain = self._predicted_collision_drop(idx)
            if (best_gain is None) or (gain > best_gain):
                best_gain, best_idx = gain, idx

        gain = best_gain
        if gain == 0: # if no collisions then select an element at random
            best_idx = random.randrange(len(v))
        x = self.val[best_idx]

        old_collisions = self._current_collisions()
        # logger.info(f"Delete move old collisions are {old_collisions}")
        if self.hard:
            # if currently invalid, allow deletion; else avoid deleting
            assert old_collisions > 0
            if len(self.val) <= 1:
                logger.info(f"Error in deleted: should not be any collisions {self.val}")
            self._remove_element(x)
            if old_collisions > gain:
                # if, after removing we still expect some collisions
                collisions = self._count_collisions()
                # logger.info(f"Delete move done, new collisions are {collisions}")
                assert collisions > 0, collisions
                self.score = -1
            else:
                # if, after removing we expect no collisions
                collisions = self._count_collisions() #debug only
                # logger.info(f"Delete move done and should not be any anymore, new collisions are {collisions}")
                assert collisions == 0, collisions #debug only
                self.score = len(self.val)
                # debug_score = self.score
                # self.calc_score()
                # assert self.score == debug_score
            return

        old_collisions = self._current_collisions()
        delta_coll = self._remove_element(x)
        delta_score = self.M * (-1) - delta_coll
        accept = (delta_score >= 0) or self._anneal_accept(delta_score)
        if not accept:
            # revert
            self._add_element(x)
        else:
            self.score += delta_score

# On pourrait générer des éléments, puis passer à une destructions d'élements jusqu'à ce qu'il n'y ait plus de collision et garder ce qui reste puis filtrer.
# Alternativement on pourrait générer depuis une meilleure seed, mais on risque de sur-spécialiser le modèle.


    @classmethod
    def _update_class_params(cls,pars):
        cls.N, cls.M, cls.hard, cls.insert_prob, cls.delete_prob, cls.shift_prob, cls.temp, cls.temp_decay, cls.init_method, cls.init_k, cls.jitter_init, cls.steps = pars

    @classmethod
    def _save_class_params(cls):
        return (cls.N, cls.M, cls.hard, cls.insert_prob, cls.delete_prob, cls.shift_prob, cls.temp, cls.temp_decay, cls.init_method, cls.init_k, cls.jitter_init, cls.steps)

    @classmethod
    def _batch_generate_and_score(cls,n, pars=None):
        return super()._batch_generate_and_score(n,pars)



class SidonSetEnvironment(BaseEnvironment):
    data_class = SidonSetDataPoint

    def __init__(self, params):
        super().__init__(params)
        self.data_class.N = int(params.N)
        self.data_class.M = int(params.M)
        self.data_class.hard = params.hard
        self.data_class.jitter_init = params.jitter_init

        # seed = params.seed # Only for debug and not for running: it will generated the same candidate otherwise.
        # if seed is not None:
        #     random.seed(seed)

        self.data_class.steps = int(params.sidon_steps)
        self.data_class.temp = float(params.temp0)
        self.data_class.temp_decay = float(params.temp_decay)
        self.data_class.init_method = params.init_method or "random_greedy"
        self.data_class.init_k = params.init_k

        #Probabilities of the different moves for the local search

        self.data_class.insert_prob = float(params.insert_prob)
        self.data_class.delete_prob = float(params.delete_prob)
        self.data_class.shift_prob  = float(params.shift_prob)

        self.tokenizer = SidonTokenizer(params.N, self.data_class,nosep=params.nosep,base=params.base)
        self.symbols = [str(i) for i in range(params.base)]
        self.symbols.extend(BaseEnvironment.SPECIAL_SYMBOLS)


    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument('--N', type=int, default="500", help='Defines the set {0,....,N} in which the Sidon subset is looked for')
        parser.add_argument('--M', type=int, default="1", help='reward weight for length of Sidon Sets')
        parser.add_argument('--base', type=int, default="10", help='reward weight for length of Sidon Sets')
        parser.add_argument('--hard', type=bool_flag, default="true", help='whether only sidon sets are accepted')
        parser.add_argument('--insert_prob', type=float, default=0.33, help='probability of insert move in the local search')
        parser.add_argument('--delete_prob', type=float, default=0.33, help='probability of delete move in the local search')
        parser.add_argument('--shift_prob', type=float, default=0.33, help='probability of shift move in the local search')
        parser.add_argument('--temp0', type=float, default=0.33, help='temp0 of the local search')
        parser.add_argument('--temp_decay', type=float, default=0.33, help='temp_decay of the local search')
        parser.add_argument('--init_method', type=str, default="random_greedy", help='method of generation')
        parser.add_argument("--init_k", type=int, default=-1, help="by default size of the Sidon set that one tries to construct in the generation")
        parser.add_argument("--jitter_init", type=bool_flag, default="true", help="if generation is evenly spaced, should there be random displacements")
        parser.add_argument('--sidon_steps', type=int, default=2000, help='number of steps in local search')
        parser.add_argument('--nosep', type=bool_flag, default="false", help='separator (for adjacency and double edge)')
