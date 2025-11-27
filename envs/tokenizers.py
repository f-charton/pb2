from abc import ABC, abstractmethod
#from .square import SquareDataPoint

class Tokenizer(ABC):
    """
    Base class for encoders, encodes and decodes matrices
    abstract methods for encoding/decoding numbers
    """
    def __init__(self):
        pass

    @abstractmethod
    def encode(self, val):
        pass
   
    @abstractmethod
    def decode(self, lst):
        pass


class SparseTokenizer(Tokenizer):
    def __init__(self, N, dataclass):
        self.N = N
        self.dataclass = dataclass

    def encode(self, graph):
        w = list(map(str, graph.val))
        w.append("|")
        return w
    
    def decode(self, lst):
        """Decode a list of tokens to return a SquareDataPoint"""
        for i, el in enumerate(lst):
            if el == "|":
                lst = lst[:i]
                break
        try:
            result = list(map(int, lst))
        except ValueError as e:
            # need to check bounds, too
            print(f"Value error in the generation {e}")
            return None
        return self.dataclass(result)


class DenseTokenizer(Tokenizer):
    def __init__(self, N, dataclass):
        self.N = N
        self.dataclass =  dataclass

    def _edge_to_index(self, i: int, j: int) -> int:
        """Convert edge (i,j) to linear index in upper triangular matrix"""
        return i * (2 * self.N - i - 1) // 2 + (j - i - 1)

    def encode(self, graph):
        w = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                w.append(str(graph.matrix[i, j]))
            w.append("&")
        w.append("|")
        return w

    def decode(self, lst):
        """Decode a list of tokens to return a SquareDataPoint"""
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
            
        return self.dataclass(result)

