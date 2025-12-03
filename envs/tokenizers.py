from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
import math
#from .square import SquareDataPoint

class Tokenizer(ABC):
    """
    Base class for encoders, encodes and decodes matrices
    abstract methods for encoding/decoding numbers
    """
    def __init__(self):
        self.dataclass = None

    @abstractmethod
    def encode(self, val):
        pass
   
    @abstractmethod
    def decode(self, lst):
        pass

    def _edge_to_index(self, N, i, j):
        return i * (2 * N - i - 1) // 2 + (j - i - 1)

    def _index_to_edge(self, N, idx):
        discriminant = (2 * N - 1) ** 2 - 8 * idx
        if discriminant < 0:
            i = N - 2
        else:
            i = int((2 * N - 1 - math.sqrt(discriminant)) // 2)
            i = max(0, min(i, N - 2))

        Sx = i * (2 * N - i - 1) // 2
        j = i + 1 + idx - Sx
        return (i, j)

    def decode_batch(self, data, pars=None):
        """
        Worker function for detokenizing a batch of data
        """
        out = []
        if pars is not None:
            self.dataclass._update_class_params(pars)
        for lst in data:
            l = self.decode(lst)
            if l is not None:
                out.append(l)
        return out



class SparseTokenizer(Tokenizer):
    def __init__(self, N, dataclass):
        self.N = N
        self.dataclass = dataclass

    def encode(self, graph):
        w = []
        for i in range(self.N):
            for j in range(i+1, self.N):
                if graph.matrix[i, j] == 1:
                    w.append(str(self._edge_to_index(self.N, i, j)))
        w.append("EOS")
        return w
    
    def decode(self, lst):
        """Decode a list of tokens to return a SquareDataPoint"""
        graph = self.dataclass()
        for i, el in enumerate(lst):
            if el == "EOS":
                lst = lst[:i]
                break
        try:
            for el in lst:
                i, j = self._index_to_edge(self.N, int(el))
                graph.matrix[i, j] = 1
                graph.matrix[j, i] = 1
        except ValueError as e:
            return None
        return graph

    # stupid but needed to please PoolExectutor
    def decode_batch(self, data, pars=None):
        return super().decode_batch(data, pars)



class EdgeTokenizer(Tokenizer):
    def __init__(self, N, dataclass, nosep):
        self.N = N
        self.dataclass = dataclass
        self.nosep = nosep

    def encode(self, graph):
        w = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if graph.matrix[i, j] == 1:
                    w.append(str(i))
                    w.append(str(j))
                    if not self.nosep:
                        w.append("SEP")
        w.append("EOS")
        return w
    
    def decode(self, lst):
        """Decode a list of tokens to return a SquareDataPoint"""
        graph = self.dataclass()
        for i, el in enumerate(lst):
            if el == "EOS":
                lst = lst[:i]
                break
        ll = 2 if self.nosep else 3
        if len(lst) % 2 != 0:
            return None
        try:
            if self.nosep:
                for c in range(0, len(lst), 2):
                    i, j = int(lst[c]), int(lst[c+1])
                    if i >= self.N or j >= self.N:
                        return None
                    graph.matrix[i, j] = 1
                    graph.matrix[j, i] = 1
            else:
                for c in range(0, len(lst), 3):
                    i, j, sep = int(lst[c]), int(lst[c+1]), lst[c+2]
                    if sep != "SEP" or i >= self.N or j >= self.N:
                        return None
                    graph.matrix[i, j] = 1
                    graph.matrix[j, i] = 1
        except ValueError as e:
            return None
        return graph

    # stupid but needed to please PoolExectutor
    def decode_batch(self, data, pars=None):
        return super().decode_batch(data, pars)


class DenseTokenizer(Tokenizer):
    def __init__(self, N, dataclass, nosep):
        self.N = N
        self.dataclass =  dataclass
        self.nosep = nosep

    def encode(self, graph):
        w = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                w.append(str(graph.matrix[i, j]))
            if not self.nosep:
                w.append("SEP")
        w.append("EOS")
        return w

    def decode(self, lst):
        """Decode a list of tokens to return a SquareDataPoint"""
        graph = self.dataclass()
        for i, el in enumerate(lst):
            if el == "EOS":
                lst = lst[:i]
                break
        try:
            idx = 0
            jdx = 1
            expected_count = self.N - 1
            count = 0
            if not self.nosep:
                for el in lst:
                    if el == "SEP":
                        if count != expected_count:
                            return None
                        idx += 1
                        jdx = idx + 1
                        expected_count = self.N - idx - 1
                        count = 0
                    elif jdx >= self.N:
                        return None
                    else:
                        graph.matrix[idx, jdx] = int(el)
                        graph.matrix[jdx, idx] = int(el)
                        jdx += 1
                        count += 1
                if idx != self.N or count != 0:
                    return None
            else:
                for el in lst:
                    if count == expected_count:
                        idx += 1
                        jdx = idx + 1
                        expected_count = self.N - idx - 1
                        count = 0
                    if jdx >= self.N:
                        return None
                    
                    graph.matrix[idx, jdx] = int(el)
                    graph.matrix[jdx, idx] = int(el)
                    jdx += 1
                    count += 1
                if idx != self.N or count != 0:
                    return None

        except ValueError as e:
            return None

        return graph            

    # stupid but needed to please PoolExectutor
    def decode_batch(self, data, pars=None):
        return super().decode_batch(data, pars)
