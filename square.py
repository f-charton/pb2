from environment import DataPoint
import math
import random
import numpy as np
from itertools import permutations
from typing import Optional, List, Tuple


class SquareDataPoint(DataPoint):
    def __init__(self, val, params):
        super().__init__(params)
        self.N: int = params.square_N
        self.square_hard: bool = params.square_hard
        # if val:
        #     self.val = val
        #     self._ensure_unique_val()
        #     self._create_matrix()
        #     self._squares()

        if params.square_init_method == "edge_removal":
            self._generate_graph_by_edge_removal()
        elif params.square_init_method == "edge_addition":
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
        self._squares()
        self._remove_squares_greedily()

    def _generate_graph_by_edge_addition(self) -> List[int]:
        self.val = []
        self._create_matrix()
        self._squares()
        self._add_edges_greedily()

    def _create_matrix(self):
        self.matrix = np.zeros((self.N, self.N), dtype=int)
        for v in self.val:
            i, j = self._index_to_edge(v)
            self.matrix[i, j] = 1
            self.matrix[j, i] = 1

    def _squares(self):
        row_bits = []
        for i in range(self.N):
            mask = 0
            row = self.matrix[i]
            for j in range(self.N):
                if row[j] == 1:
                    mask |= 1 << j
            row_bits.append(mask)
        
        self.squares = []
        for i in range(self.N):
            bits_i = row_bits[i]
            for j in range(i + 1, self.N):
                common = bits_i & row_bits[j]
                x = common
                while x:
                    lsb_u = x & -x
                    u = lsb_u.bit_length() - 1
                    x ^= lsb_u
                    y = x
                    while y:
                        lsb_v = y & -y
                        v = lsb_v.bit_length() - 1
                        y ^= lsb_v
                        self.squares.append((i, u, j, v))


    def calc_score(self):
        if self.square_hard and len(self.squares) > 0:
            self.score = -1
            return
        self.score = len(self.val) - 6 * len(self.squares)

    def calc_features(self):
        """
        Compute optional features to give more signal to the transformer
        """
        pass
    
    def encode(self, base=10, reverse=False) -> List[str]:
        """Encode the square-free graph as a list of tokens"""
        if base == self.N * (self.N - 1) // 2:
            w = list(map(str, self.val))
            w.append("|")
            return w
        elif base == -2:
            w = []
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    w.append(str(self.matrix[i, j]))
                w.append("&")
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

    def decode(self, lst, base=10, reverse=False) -> Optional["SquareDataPoint"]:
        """Decode a list of tokens to return a SquareDataPoint"""
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
        elif base == -2:
            result = []
            for i, el in enumerate(lst):
                if el == "|":
                    lst = lst[:i]
                    break
            try:
                idx = 0
                jdx = 1
                expected_count = self.N - 1
                count = 0
                for el in lst:
                    if el == "&":
                        if count == expected_count:
                            # Move to next row
                            idx += 1
                            jdx = idx + 1
                            expected_count = self.N - idx - 1
                            count = 0
                        else:
                            return None
                    elif jdx < self.N:
                        if el == "1":
                            result.append(self._edge_to_index(idx, jdx))
                        jdx += 1
                        count += 1
                    else:
                        return None
                if idx != self.N or count != 0:
                    return None
            except (ValueError, IndexError) as e:
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
        self._squares()
        self.calc_features()
        self.calc_score()
        return self

    def local_search(self) -> None:
        self._remove_squares_greedily()
        self._add_edges_greedily()
        self._squares()
        self.calc_score()

    def _get_square_edges(self, vertices: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
        i, j, k, l = vertices
        for v1, v2, v3 in permutations([j, k, l]):
            if (self.matrix[i, v1] == 1 and 
                self.matrix[v1, v2] == 1 and 
                self.matrix[v2, v3] == 1 and 
                self.matrix[v3, i] == 1):
                edges = [(min(i, v1), max(i, v1)), 
                         (min(v1, v2), max(v1, v2)),
                         (min(v2, v3), max(v2, v3)),
                         (min(v3, i), max(v3, i))]
                return edges

    def _remove_squares_greedily(self) -> None:
        while self.squares:
            edge_count = {}
            for square in self.squares:
                edges = self._get_square_edges(square)
                if edges is not None:
                    for edge in edges:
                        edge_count[edge] = edge_count.get(edge, 0) + 1
            
            most_frequent_edge = max(edge_count, key=edge_count.get)
            
            i, j = most_frequent_edge
            edge_idx = self._edge_to_index(i, j)
            if edge_idx in self.val:
                self.val.remove(edge_idx)
                self.matrix[i, j] = 0
                self.matrix[j, i] = 0

            remaining_squares = []
            for square in self.squares:
                edges = self._get_square_edges(square)
                if edges is not None and most_frequent_edge not in edges:
                    remaining_squares.append(square)
            self.squares = remaining_squares

    def _add_edges_greedily(self) -> None:
        adjmat3 = self.matrix @ self.matrix @ self.matrix
        allowed_edges = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.matrix[i, j] == 0 and adjmat3[i, j] == 0:
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
            adjmat3 = self.matrix @ self.matrix @ self.matrix
            for (a, b) in allowed_edges:
                if self.matrix[a, b] == 0 and adjmat3[a, b] == 0:
                    new_allowed_edges.append((a, b))
            
            allowed_edges = new_allowed_edges
        
        self._ensure_unique_val()
