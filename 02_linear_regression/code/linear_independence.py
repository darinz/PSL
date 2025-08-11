import numpy as np
import matplotlib.pyplot as plt

def check_linear_independence(vectors):
    """
    Check if a set of vectors is linearly independent
    
    Parameters:
    vectors: list of numpy arrays representing vectors
    
    Returns:
    is_independent: boolean indicating if vectors are linearly independent
    rank: rank of the matrix formed by the vectors
    """
    # Stack vectors into a matrix
    A = np.column_stack(vectors)
    
    # Check rank
    rank = np.linalg.matrix_rank(A)
    is_independent = rank == len(vectors)
    
    return is_independent, rank

def demonstrate_linear_independence():
    """Demonstrate linear independence concepts"""
    
    # Example 1: Linearly independent vectors in R^3
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([0, 0, 1])
    
    independent_vectors = [v1, v2, v3]
    is_indep, rank = check_linear_independence(independent_vectors)
    
    print("=== Linear Independence Example 1 ===")
    print(f"Vectors: {[v.tolist() for v in independent_vectors]}")
    print(f"Linearly independent: {is_indep}")
    print(f"Rank: {rank}")
    
    # Example 2: Linearly dependent vectors
    v4 = np.array([1, 1, 0])
    v5 = np.array([2, 2, 0])  # v5 = 2*v4
    v6 = np.array([0, 0, 1])
    
    dependent_vectors = [v4, v5, v6]
    is_indep, rank = check_linear_independence(dependent_vectors)
    
    print("\n=== Linear Independence Example 2 ===")
    print(f"Vectors: {[v.tolist() for v in dependent_vectors]}")
    print(f"Linearly independent: {is_indep}")
    print(f"Rank: {rank}")
    print(f"Note: v5 = 2*v4, so these vectors are dependent")
    
    return independent_vectors, dependent_vectors

# Run demonstration
independent_vectors, dependent_vectors = demonstrate_linear_independence()
