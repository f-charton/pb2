from environment import DataPoint
import math
import random
import numpy as np
from typing import Optional, List, Tuple


class TriangleDataPoint(DataPoint):
    def __init__(self, val, params):
        super().__init__(params)
        self.N: int = params.triangle_N
        self.triangle_hard: bool = params.triangle_hard
        if params.triangle_init_method == "edge_removal":
            self._generate_graph_by_edge_removal()
        elif params.triangle_init_method == "edge_addition":
            self._generate_graph_by_edge_addition()
        self.calc_score()
        self.calc_features()

    def _edge_to_index(self, i: int, j: int) -> int:
        """Convert edge (i,j) to linear index in upper triangular matrix"""
        return i * (2 * self.N - i - 1) // 2 + (j - i - 1)

    def _index_to_edge(self, idx: int) -> Tuple[int, int]:
        """Convert linear index to edge (i,j) in upper triangular matrix"""
        discriminant = (2 * self.N - 1) ** 2 - 8 * idx
        if discriminant < 0:
            i = self.N - 2
        else:
            i = int((2 * self.N - 1 - math.sqrt(discriminant)) // 2)
            i = max(0, min(i, self.N - 2))
        
        Sx = i * (2 * self.N - i - 1) // 2
        j = i + 1 + idx - Sx
        
        return (i, j)

    def _ensure_unique_val(self) -> None:
        """Ensure self.val contains only unique elements and is sorted"""
        self.val = sorted(list(set(self.val)))

    def _generate_graph_by_edge_removal(self) -> List[int]:
        """Generate random edges for initialization"""
        max_edges = self.N * (self.N - 1) // 2
        num_edges = random.randint(0, max_edges)
        all_edges = list(range(max_edges))
        random.shuffle(all_edges)
        self.val = sorted(all_edges[:num_edges])
        self._ensure_unique_val()
        self._create_matrix()
        self._triangles()
        self._remove_triangles_greedily()

    def _generate_graph_by_edge_addition(self) -> List[int]:
        self.val = []
        self._create_matrix()
        self._triangles()
        self._add_edges_greedily()

    def _create_matrix(self):
        self.matrix = np.zeros((self.N, self.N), dtype=int)
        for v in self.val:
            i, j = self._index_to_edge(v)
            self.matrix[i, j] = 1
            self.matrix[j, i] = 1

    def _triangles(self):
        self.triangles = []
        for i in range(self.N):
            for j in range(i+1, self.N):
                if self.matrix[i, j] == 1:
                    for k in range(j+1, self.N):
                        if self.matrix[i, k] == 1 and self.matrix[j, k] == 1:
                            self.triangles.append((i, j, k))

    def calc_score(self):
        if self.triangle_hard and len(self.triangles) > 0:
            self.score = -1
        self.score = len(self.val) - 2 * len(self.triangles)

    def calc_features(self):
        """
        Compute optional features to give more signal to the transformer
        """
        pass
    
    def encode(self, base=10, reverse=False) -> List[str]:
        """Encode the triangle-free graph as a list of tokens"""
        if base == self.N * (self.N - 1) // 2:
            w = list(map(str, self.val))
            w.append("|")
            return w
        w = []
        for el in self.val:
            v = el
            curr_w = []
            while v > 0:
                curr_w.append(str(v % base))
                v = v // base
            w.extend(curr_w)
            w.append("|")
        return w

    def decode(self, lst, base=10, reverse=False) -> Optional["TriangleDataPoint"]:
        """Decode a list of tokens to return a TriangleDataPoint"""
        if base == self.N * (self.N - 1) // 2:
            for i, el in enumerate(lst):
                if el == "|":
                    lst = lst[:i]
                    break
            try:
                result = list(map(int, lst))
            except ValueError as e:
                print(f"Value error in the generation {e}")
                return None
        else:
            sub_lists = []
            current = []
            for item in lst:
                if item == "|":
                    if current:
                        sub_lists.append(current)
                        current = []
                else:
                    current.append(item)
            if current:
                sub_lists.append(current)

            result = []
            try:
                for sub_list in sub_lists:
                    if base <= 36:
                        num_str = ''.join(sub_list)
                        num = int(num_str, base)
                    else:
                        num = 0
                        for el in sub_list:
                            v = int(el)
                            if v < 0 or v >= base:
                                raise ValueError(f"Digit {v} out of range for base {base}")
                            num = num * base + v
                    if num >= self.N * (self.N - 1) // 2:
                        continue
                    result.append(num)
            except ValueError as e:
                print(f"Value error in the generation {e}")
                return None
        
        self.val = sorted(result)
        self._ensure_unique_val()
        self._create_matrix()
        self._triangles()
        self.calc_features()
        self.calc_score()
        return self

    def local_search(self) -> None:
        self._remove_triangles_greedily()
        self._add_edges_greedily()
        self._triangles()
        self.calc_score()

    def _remove_triangles_greedily(self) -> None:
        while self.triangles:
            edge_count = {}
            for (i, j, k) in self.triangles:
                for edge in [(i, j), (j, k), (i, k)]:
                    edge_count[edge] = edge_count.get(edge, 0) + 1
            
            most_frequent_edge = max(edge_count, key=edge_count.get)
            
            i, j = most_frequent_edge
            edge_idx = self._edge_to_index(i, j)
            if edge_idx in self.val:
                self.val.remove(edge_idx)
                self.matrix[i, j] = 0
                self.matrix[j, i] = 0

            self.triangles = [t for t in self.triangles if most_frequent_edge not in [(t[0], t[1]), (t[1], t[2]), (t[0], t[2])]]

    def _add_edges_greedily(self) -> None:
        adjmat2 = self.matrix @ self.matrix
        allowed_edges = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.matrix[i, j] == 0 and adjmat2[i, j] == 0:
                    allowed_edges.append((i, j))
        
        while allowed_edges:
            edge = random.choice(allowed_edges)
            i, j = edge
            
            edge_idx = self._edge_to_index(i, j)
            if edge_idx not in self.val:
                self.val.append(edge_idx)
                self.val.sort()
            self.matrix[i, j] = 1
            self.matrix[j, i] = 1
            
            new_allowed_edges = []
            for (a, b) in allowed_edges:
                if ((a == i and self.matrix[b, j] == 1) or (a == j and self.matrix[b, i] == 1) or
                    (b == i and self.matrix[a, j] == 1) or (b == j and self.matrix[a, i] == 1)):
                    continue
                
                if (a == i and b == j) or (a == j and b == i):
                    continue
                
                new_allowed_edges.append((a, b))
            
            allowed_edges = new_allowed_edges
        
        self._ensure_unique_val()
