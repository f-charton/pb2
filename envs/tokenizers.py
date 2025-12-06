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
        if len(lst) % ll != 0:
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
    def __init__(self, N, dataclass, nosep, pow2base=1):
        self.N = N
        self.dataclass =  dataclass
        self.nosep = nosep
        self.pow2base = pow2base

    def encode(self, graph):
        w = []
        count = 1
        val = 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                val += count * graph.matrix[i, j]
                count *= 2
                if count == 2 ** self.pow2base:
                    w.append(str(val))
                    count = 1
                    val = 0
            if not self.nosep:
                w.append("SEP")
        if count > 1:
            w.append(str(val))
        w.append("EOS")
        return w

    def decode(self, lst):
        """Decode a list of tokens to return a SquareDataPoint"""
        graph = self.dataclass()
        for i, el in enumerate(lst):
            if el == "EOS":
                lst = lst[:i]
                break
        if len(lst) != math.ceil(((self.N*(self.N-1))//2)/self.pow2base):
            return None
        try:
            idx = 0
            jdx = 1
            for el in lst[:-1]:
                val = int(el)
                for _ in range(self.pow2base):
                    graph.matrix[idx, jdx] = val%2
                    graph.matrix[jdx, idx] = val%2
                    val //= 2
                    jdx += 1
                    if jdx == self.N:
                        idx +=1
                        jdx = idx + 1
            
            val = int(lst[-1])
            last_count = (self.N*(self.N-1))//2 - self.pow2base * (len(lst) - 1)
            for _ in range(last_count):
                graph.matrix[idx, jdx] = val%2
                graph.matrix[jdx, idx] = val%2
                val //= 2
                jdx += 1
                if jdx == self.N:
                    idx +=1
                    jdx = idx + 1
            if val > 0:
                return None

        except ValueError as e:
            return None

        return graph            

    # stupid but needed to please PoolExectutor
    def decode_batch(self, data, pars=None):
        return super().decode_batch(data, pars)

class SidonTokenizer(Tokenizer):
    def __init__(self, N, dataclass, nosep, base, separator :str = "SEP"):
        self.N = N
        self.dataclass =  dataclass
        self.nosep = nosep
        self.separator = separator
        self.base = base
        print("HERE base", self.base)

    def encode(self, sidonset, reverse=False):
        w = []
        val = sidonset.val
        for el in val:
            v = el
            curr_w = []
            while v > 0:
                curr_w.append(str(v%self.base))
                v=v//self.base
            w.extend(curr_w)
            if not self.nosep:
                w.append(self.separator)
        return w

    def decode(self, lst, reverse=False):
        """
        Create a SidonSetDataPoint from a list
        """
        sub_lists = []
        current = []
        for item in lst:
            if item == self.separator:
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
                if self.base <= 36:
                    #36 is the maximum supported by the int method, suprisingly
                    num_str = ''.join(sub_list)
                    num = int(num_str, self.base)
                else:
                    #fallback to an explicit method
                    num = 0
                    for el in sub_list:
                        v =int(el)
                        if v < 0 or v >= self.base:
                            raise ValueError(f"Digit {v} out of range for self.base {self.base}")
                        num = num * self.base + v
                    print("HERE num", num)
                if num > self.N:
                    print("HERE num pas ouf")
                    # return None #with this option, as soon as the model outputs a number above self.N we discard the full sequence
                    continue #at least for debug this option is a bit softer when the model is at the beginning of training and allows it to only remove the element that shouldn't be there,
                result.append(num)
                print("HERE result",result)
        except ValueError as e:
            print(f"Value error in the generation {e}")
            return None
        val = sorted(result)
        assert len(val) > 0, (sub_lists,lst)
        sidonpoint = self.dataclass(val=val,init=True)
        return sidonpoint

    # stupid but needed to please PoolExectutor
    def decode_batch(self, data, pars=None):
        return super().decode_batch(data, pars)
