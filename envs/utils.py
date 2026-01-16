import numpy as np
import random


############################################################
# Utils for graph
############################################################

# sort the graph based on the degree of the nodes
def sort_graph_based_on_degree(adj_matrix):
    out_degree = adj_matrix.sum(axis=1)
    in_degree = adj_matrix.sum(axis=0)
    degree = in_degree + out_degree  # so this works for both undirected and directed graphs
    sorted_indices = np.argsort(-degree, kind='stable')
    return adj_matrix[np.ix_(sorted_indices, sorted_indices)]


def random_symmetry_adj_matrix(adj_matrix):
    perm = np.random.permutation(adj_matrix.shape[0])
    return adj_matrix[np.ix_(perm, perm)]  # this is creating a copy already


############################################################
# Utils for square
############################################################
def canonical_form_2d(matrix):
    best = None
    best_matrix = None
    
    # 8 symmetries: 4 rotations × 2 (with/without reflection)
    current = matrix
    for _ in range(4):
        flat = current.flatten().tolist()
        if best is None or flat < best:
            best = flat
            best_matrix = current.copy()
        
        reflected = np.flip(current, axis=1)
        flat = reflected.flatten().tolist()
        if flat < best:
            best = flat
            best_matrix = reflected.copy()
        
        current = np.rot90(current)
    
    return best_matrix


def random_symmetry_2d(matrix):
    k = random.randint(0, 3)
    result = np.rot90(matrix, k)

    if random.randint(0, 1):
        result = np.flip(result, axis=1)
    
    return result.copy()


############################################################
# Utils for half-square (isosceles with symmetry)
############################################################
def canonical_form_2d_symmetric(matrix):
    transposed = matrix.T
    
    flat_original = matrix.flatten().tolist()
    flat_transposed = transposed.flatten().tolist()
    
    if flat_original <= flat_transposed:
        return matrix.copy()
    else:
        return transposed.copy()


def random_symmetry_2d_symmetric(matrix):
    if random.randint(0, 1):
        return matrix.T.copy()
    else:
        return matrix.copy()


############################################################
# Utils for cube
############################################################
def canonical_form_3d(matrix):
    best = None
    best_matrix = None
    
    # 48 symmetries: 6 axis permutations × 8 reflections
    for perm in [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]:
        for flips in range(8):
            transformed = np.transpose(matrix, perm)
            if flips & 1:
                transformed = np.flip(transformed, axis=0)
            if flips & 2:
                transformed = np.flip(transformed, axis=1)
            if flips & 4:
                transformed = np.flip(transformed, axis=2)
            
            flat = transformed.flatten().tolist()
            if best is None or flat < best:
                best = flat
                best_matrix = transformed.copy()
    
    return best_matrix


def random_symmetry_3d(matrix):
    perms = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]
    perm = random.choice(perms)
    
    flips = random.randint(0, 7)
    
    result = np.transpose(matrix, perm)
    
    if flips & 1:
        result = np.flip(result, axis=0)
    if flips & 2:
        result = np.flip(result, axis=1)
    if flips & 4:
        result = np.flip(result, axis=2)
    
    return result.copy()
