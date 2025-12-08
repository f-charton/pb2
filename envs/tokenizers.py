from abc import ABC, abstractmethod
import math
from itertools import product, combinations, permutations
import numpy as np


def iterate_k_times(N, k, symmetric):
    if k == 1:
        yield from range(N)
    else:
        if symmetric:
            yield from combinations(range(N), k)
        else:
            yield from product(range(N), repeat=k)


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

    def decode_batch(self, data, pars=None):
        """
        Worker function for detokenizing a batch of data
        """
        out = []
        if pars is not None:
            self.dataclass._update_class_params(pars)
        for _,lst in enumerate(data):
            # assert len(lst) > 0, (i,data[i], data)
            l = self.decode(lst)
            if l is not None:
                out.append(l)
        return out



class SparseTokenizer(Tokenizer):
    def __init__(self, dataclass, N, k, is_adj_matrix_symmetric, extra_symbols, token_embeddings):
        self.dataclass = dataclass
        self.N = N
        self.k = k
        self.is_adj_matrix_symmetric = is_adj_matrix_symmetric
        self.token_embeddings = token_embeddings
        assert self.token_embeddings == self.k or self.token_embeddings == 1, f"token_embeddings must be 1 or {k}"
        self.stoi, self.itos = {}, {}

        for idx, el in enumerate(iterate_k_times(self.N, self.k // self.token_embeddings, self.is_adj_matrix_symmetric)):
            self.stoi[el] = idx
            self.itos[idx] = el
        for jdx, el in enumerate(extra_symbols):
            self.stoi[el] = idx + jdx + 1
            self.itos[idx + jdx + 1] = el

    def encode(self, graph):
        w = []
        for el in iterate_k_times(self.N, self.k, self.is_adj_matrix_symmetric):
            if graph.matrix[el] == 1:
                if self.token_embeddings == 1:
                    w.append([self.stoi[el]])
                else:
                    w.append([self.stoi[x] for x in el])
        w.append([self.stoi["EOS"] for _ in range(self.token_embeddings)])
        return np.array(w, dtype=np.int32)  # (N_elements, self.token_embeddings)

    def decode(self, ll):
        # ll size (N_elements, self.token_embeddings)
        try:
            graph = self.dataclass()
            for l in ll:
                if self.token_embeddings == 1:
                    el = self.itos[l[0]]
                    if el == "EOS":
                        return graph
                    elif el in ["PAD", "BOS"]:
                        return None
                else:
                    el = tuple([self.itos[x] for x in l])
                    if any(x == "EOS" for x in el):
                        return graph
                    elif any(x in ["PAD", "BOS"] for x in el):
                        return None
                if self.is_adj_matrix_symmetric:
                    if len(set(el)) != len(el):
                        return None
                    for permutation in permutations(el):
                        graph.matrix[permutation] = 1
                else:
                    graph.matrix[el] = 1
        except:
            return None
    # stupid but needed to please PoolExectutor
    def decode_batch(self, data, pars=None):
        return super().decode_batch(data, pars)


class EdgeTokenizer(Tokenizer):
    def __init__(self, dataclass, N, k, is_adj_matrix_symmetric, extra_symbols, nosep):
        self.dataclass = dataclass
        self.N = N
        self.k = k
        self.is_adj_matrix_symmetric = is_adj_matrix_symmetric
        self.nosep = nosep
        self.stoi, self.itos = {}, {}
        self.token_embeddings = 1

        for idx, el in enumerate(range(N)):
            self.stoi[el] = idx
            self.itos[idx] = el
        for jdx, el in enumerate(extra_symbols):
            self.stoi[el] = idx + jdx + 1
            self.itos[idx + jdx + 1] = el

    def encode(self, graph):
        w = []
        for el in iterate_k_times(self.N, self.k, self.is_adj_matrix_symmetric):
            if graph.matrix[el] == 1:
                w.extend([self.stoi[x] for x in el])
                if not self.nosep:
                    w.append(self.stoi["SEP"])
        w.append(self.stoi["EOS"])
        return np.array(w, dtype=np.int32).reshape(-1, 1)  # (N_elements * (k if self.nosep else k + 1) + 1, 1)
    
    def decode(self, ll):
        # ll size (N_elements * (k if self.nosep else k + 1) + 1, 1)
        new_ll = []
        for el in ll:
            el = self.itos[el[0]]
            if el == "EOS":
                break
            new_ll.append(el)
        ll = new_ll
        jump = self.k if self.nosep else self.k + 1
        try:
            size = len(ll)
            graph = self.dataclass()
            for start_idx in range(0, size, jump):
                edge_rep = ll[start_idx : start_idx + jump]
                if not self.nosep:
                    if edge_rep[-1] != "SEP":
                        return graph
                    edge_rep.pop()
                if not all(isinstance(x, int) for x in edge_rep):
                    return graph
                el = tuple(edge_rep)
                if self.is_adj_matrix_symmetric:
                    if len(set(el)) != len(el):
                        return None
                    for permutation in permutations(el):
                        graph.matrix[permutation] = 1
                else:
                    graph.matrix[el] = 1
            return graph
        except:
            return None

    # stupid but needed to please PoolExectutor
    def decode_batch(self, data, pars=None):
        return super().decode_batch(data, pars)


class DenseTokenizer(Tokenizer):
    def __init__(self, dataclass, N, k, is_adj_matrix_symmetric, extra_symbols, nosep, pow2base):
        self.dataclass = dataclass
        self.N = N
        self.k = k
        self.is_adj_matrix_symmetric = is_adj_matrix_symmetric
        self.nosep = nosep
        self.pow2base = pow2base
        self.stoi, self.itos = {}, {}
        self.token_embeddings = 1

        # DenseTokenizer is only for not self.nosep and self.k == 2 or self.nosep
        assert self.nosep or self.k == 2

        if self.nosep:
            self.expected_elements_in_a_decoded_sequence = math.ceil(sum(1 for _ in iterate_k_times(self.N, self.k, self.is_adj_matrix_symmetric)) / self.pow2base)
        else:
            if self.is_adj_matrix_symmetric:
                self.expected_elements_in_a_decoded_sequence = sum(math.ceil(i / self.pow2base) for i in range(self.N)) + self.N
            else:
                self.expected_elements_in_a_decoded_sequence = self.N * math.ceil(self.N / self.pow2base) + self.N

        for idx, el in enumerate(range(2 ** pow2base)):
            self.stoi[el] = idx
            self.itos[idx] = el
        for jdx, el in enumerate(extra_symbols):
            self.stoi[el] = idx + jdx + 1
            self.itos[idx + jdx + 1] = el

    def _pack_bits(self, bits):
        tokens = []
        count = 1
        val = 0
        for bit in bits:
            val += count * bit
            count *= 2
            if count == 2 ** self.pow2base:
                tokens.append(self.stoi[val])
                count = 1
                val = 0
        if count > 1:
            tokens.append(self.stoi[val])
        return tokens

    def _unpack_bits(self, tokens, expected_bits):
        bits = []
        for token in tokens:
            val = token
            for _ in range(self.pow2base):
                bits.append(val % 2)
                val //= 2
                if len(bits) == expected_bits:
                    return bits
        return bits

    def _row_indices(self, row):
        if self.is_adj_matrix_symmetric:
            return range(row + 1, self.N)
        else:
            return range(self.N)

    def encode(self, graph):
        w = []
        if self.nosep:
            bits = (graph.matrix[el] for el in iterate_k_times(self.N, self.k, self.is_adj_matrix_symmetric))
            w = self._pack_bits(bits)
        else:
            for i in range(self.N):
                row_bits = (graph.matrix[i, j] for j in self._row_indices(i))
                w.extend(self._pack_bits(row_bits))
                w.append(self.stoi["SEP"])
        w.append(self.stoi["EOS"])
        return np.array(w, dtype=np.int32).reshape(-1, 1)

    def decode(self, ll):
        new_ll = []
        for el in ll:
            el = self.itos[el[0]]
            if el == "EOS":
                break
            if el in ["PAD", "BOS"]:
                return None
            new_ll.append(el)
        ll = new_ll
        if len(ll) != self.expected_elements_in_a_decoded_sequence:
            return None
        try:
            graph = self.dataclass()
            if self.nosep:
                total_bits = sum(1 for _ in iterate_k_times(self.N, self.k, self.is_adj_matrix_symmetric))
                bits = self._unpack_bits(ll, total_bits)
                if len(bits) != total_bits:
                    return None
                for bit, el in zip(bits, iterate_k_times(self.N, self.k, self.is_adj_matrix_symmetric)):
                    if bit == 1:
                        if self.is_adj_matrix_symmetric:
                            for permutation in permutations(el):
                                graph.matrix[permutation] = 1
                        else:
                            graph.matrix[el] = 1
            else:
                idx = 0
                for i in range(self.N):
                    row_indices = list(self._row_indices(i))
                    num_elements = len(row_indices)
                    tokens_for_row = math.ceil(num_elements / self.pow2base) if num_elements > 0 else 0
                    row_tokens = ll[idx:idx + tokens_for_row]
                    idx += tokens_for_row
                    
                    if not all(isinstance(t, int) for t in row_tokens):
                        return None
                    
                    if num_elements > 0:
                        bits = self._unpack_bits(row_tokens, num_elements)
                        for bit, j in zip(bits, row_indices):
                            if bit == 1:
                                graph.matrix[i, j] = 1
                                if self.is_adj_matrix_symmetric:
                                    graph.matrix[j, i] = 1
                    
                    if idx >= len(ll) or ll[idx] != "SEP":
                        return None
                    idx += 1
            return graph
        except:
            return None
        


class SidonTokenizer(Tokenizer):
    def __init__(self, dataclass, N, nosep, base, extra_symbols, separator = "SEP"):
        self.N = N
        self.dataclass =  dataclass
        self.nosep = nosep
        self.separator = separator
        self.base = base
        self.token_embeddings = 1
        self.stoi, self.itos = {}, {}
        for idx, el in enumerate(range(self.base)):
            self.stoi[el] = idx
            self.itos[idx] = el
        for jdx, el in enumerate(extra_symbols):
            self.stoi[el] = idx + jdx + 1
            self.itos[idx + jdx + 1] = el

    def encode(self, sidonset, reverse=False):
        w = []
        val = sidonset.val
        for el in val:
            v = el
            curr_w = []
            while v > 0:
                curr_w.append(self.stoi[v%self.base])
                v=v//self.base
            w.extend(curr_w)
            if not self.nosep:
                w.append(self.stoi[self.separator])
        w.append(self.stoi["EOS"])
        return np.array(w, dtype=np.int32).reshape(-1, 1)

    def decode(self, lst, reverse=False):
        """
        Create a SidonSetDataPoint from a list
        """
        sub_lists = []
        current = []
        for item in lst:
            item = self.itos[item[0]]
            if item == "EOS":
                break
            if item in ["PAD", "BOS"]:
                return None
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
                    num_str = ''.join(map(str, sub_list))
                    num = int(num_str, self.base)
                else:
                    #fallback to an explicit method
                    num = 0
                    for v in sub_list:
                        if v < 0 or v >= self.base:
                            raise ValueError(f"Digit {v} out of range for self.base {self.base}")
                        num = num * self.base + v
                    # print("HERE num", num)
                if num > self.N:
                    print("HERE num pas ouf")
                    # return None #with this option, as soon as the model outputs a number above self.N we discard the full sequence
                    continue #at least for debug this option is a bit softer when the model is at the beginning of training and allows it to only remove the element that shouldn't be there,
                result.append(num)
                # print("HERE result",result)
        except ValueError as e:
            print(f"Value error in the generation {e}")
            return None
        if len(result) == 0:
            print(f"Empty decoded list for {lst} with sublists {sub_lists}.")
            return None
        val = sorted(result)
        sidonpoint = self.dataclass(val=val,init=True)
        return sidonpoint

    # stupid but needed to please PoolExectutor
    def decode_batch(self, data, pars=None):
        return super().decode_batch(data, pars)
