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
    def __init__(self, dataclass, N, k, is_adj_matrix_symmetric, extra_symbols, token_embeddings, encoding, shuffle_elements=False, nosep=None, encoding_augmentation=None):
        self.dataclass = dataclass
        self.N = N
        self.k = k
        self.is_adj_matrix_symmetric = is_adj_matrix_symmetric

        self.use_tuple_vocab = encoding == "single_integer"
        self.flatten_output = encoding == "sequence_k_tokens"
        self.nosep = nosep
        self.shuffle_elements = shuffle_elements
        self.encoding_augmentation = encoding_augmentation
        assert not self.flatten_output or self.nosep is not None
        
        expected_emb = 1 if encoding in ["single_integer", "sequence_k_tokens"] else k
        self.token_embeddings = token_embeddings
        assert token_embeddings == expected_emb, f"{encoding} requires token_embeddings={expected_emb}"
        self.stoi, self.itos = {}, {}

        if self.use_tuple_vocab:
            for idx, el in enumerate(iterate_k_times(N, k, is_adj_matrix_symmetric)):
                self.stoi[el] = idx
                self.itos[idx] = el
        else:
            for idx in range(N):
                self.stoi[idx] = idx
                self.itos[idx] = idx
        
        for jdx, el in enumerate(extra_symbols):
            self.stoi[el] = idx + jdx + 1
            self.itos[idx + jdx + 1] = el

    def _iter_token_groups(self, ll):
        if not self.flatten_output:
            for row in ll:
                yield list(row)
        else:
            flat = [row[0] for row in ll]
            jump = self.k + (0 if self.nosep else 1)
            for i in range(0, len(flat), jump):
                chunk = flat[i:i + jump]
                if not self.nosep:
                    if len(chunk) == jump and self.itos[chunk[-1]] == "SEP":
                        chunk = chunk[:-1]
                yield chunk

    def encode(self, graph):
        if self.encoding_augmentation:
            matrix = self.encoding_augmentation(graph.matrix)
        else:
            matrix = graph.matrix

        edges = []
        w = []
        for el in iterate_k_times(self.N, self.k, self.is_adj_matrix_symmetric):
            if matrix[el] == 1:
                edges.append(el)

        if self.shuffle_elements:
            indices = np.random.permutation(len(edges))
            edges = [edges[i] for i in indices]

            if self.is_adj_matrix_symmetric and not self.use_tuple_vocab:
                edges = [tuple(np.random.permutation(list(el))) for el in edges]

        w = []
        for el in edges:
            tokens = [self.stoi[el]] if self.use_tuple_vocab else [self.stoi[x] for x in el]
            if self.flatten_output:
                w.extend([[t] for t in tokens])
                if not self.nosep:
                    w.append([self.stoi["SEP"]])
            else:
                w.append(tokens)
        
        w.append([self.stoi["EOS"]] if self.flatten_output else [self.stoi["EOS"]] * self.token_embeddings)
        return np.array(w, dtype=np.int32)
        

    def decode(self, ll):
        try:
            graph = self.dataclass()
            for tokens in self._iter_token_groups(ll):
                if not tokens:
                    continue
                
                el = self.itos[tokens[0]] if self.use_tuple_vocab else tuple(self.itos[t] for t in tokens)
                
                is_eos = el == "EOS" if self.use_tuple_vocab else any(x == "EOS" for x in el)
                if is_eos:
                    return graph

                is_invalid = (el in ["PAD", "BOS"]) if self.use_tuple_vocab else any(x in ["PAD", "BOS"] for x in el)
                if is_invalid:
                    return None
                
                if self.flatten_output and not all(isinstance(x, int) for x in el):
                    return graph
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


class DenseTokenizer(Tokenizer):
    def __init__(self, dataclass, N, k, is_adj_matrix_symmetric, extra_symbols, nosep, pow2base, encoding_augmentation=None):
        self.dataclass = dataclass
        self.N = N
        self.k = k
        self.is_adj_matrix_symmetric = is_adj_matrix_symmetric
        self.nosep = nosep
        self.pow2base = pow2base
        self.encoding_augmentation = encoding_augmentation
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
        if self.encoding_augmentation:
            matrix = self.encoding_augmentation(graph.matrix)
        else:
            matrix = graph.matrix

        w = []
        if self.nosep:
            bits = (matrix[el] for el in iterate_k_times(self.N, self.k, self.is_adj_matrix_symmetric))
            w = self._pack_bits(bits)
        else:
            for i in range(self.N):
                row_bits = (matrix[i, j] for j in self._row_indices(i))
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
