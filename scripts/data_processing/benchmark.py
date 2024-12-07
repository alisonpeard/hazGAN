# %%
import numpy as np
from numba import njit

def frobenius(test: np.ndarray, template: np.ndarray) -> float:
    """
    Frobenius similarity
    """
    similarity = np.sum(template * test) / (np.linalg.norm(template) * np.linalg.norm(test))
    return similarity

@njit
def frobenius_numba(test: np.ndarray, template: np.ndarray) -> float:
    """
    Numba-accelerated Frobenius similarity
    Best for repeated calls with same array shapes
    """
    return np.sum(template * test) / (np.linalg.norm(template) * np.linalg.norm(test))


def frobenius_numpy_optimized(test: np.ndarray, template: np.ndarray) -> float:
    """
    Numpy-optimized version avoiding repeated norm calculations
    """
    template_norm = np.linalg.norm(template)
    test_norm = np.linalg.norm(test)
    return np.dot(template.ravel(), test.ravel()) / (template_norm * test_norm)

def frobenius_einsum(test: np.ndarray, template: np.ndarray) -> float:
    """
    Using Einstein summation for potentially faster computation
    """
    return np.einsum('ij,ij->', template, test) / (
        np.linalg.norm(template) * np.linalg.norm(test)
    )


class FrobeniusSimilarity:
    def __init__(self, template: np.ndarray):
        self.template = template
        self.template_norm = np.linalg.norm(template)
    
    def calculate(self, test: np.ndarray) -> float:
        """
        Caches template norm for repeated calculations
        """
        test_norm = np.linalg.norm(test)
        return np.sum(self.template * test) / (self.template_norm * test_norm)

import timeit

# Benchmarking function
def benchmark_frobenius(test, template, num_iterations=2_0000):
    # Prepare template for FrobeniusSimilarity class
    similarity_obj = FrobeniusSimilarity(template)
    
    # Functions to benchmark
    funcs = {
        "Original Function": lambda: frobenius(test, template),
        "NumPy Optimized": lambda: frobenius_numpy_optimized(test, template),
        "Numba Accelerated": lambda: frobenius_numba(test, template),
        "FrobeniusSimilarity Class": lambda: similarity_obj.calculate(test)
    }
    
    # Benchmark each function
    results = {}
    for name, func in funcs.items():
        # Verify correctness first
        reference = funcs["Original Function"]()
        current_result = func()
        assert np.isclose(reference, current_result), f"Incorrect result for {name}"
        
        # Time the function
        time = timeit.timeit(func, number=num_iterations)
        results[name] = time
    
    # Print results
    print("Benchmarking Frobenius Similarity Functions:")
    for name, time in sorted(results.items(), key=lambda x: x[1]):
        print(f"{name}: {time:.5f} seconds for {num_iterations} iterations")
    
    return results

# %%
def main():
    # Test with different array sizes
    array_sizes = [10, 20, 60]
    
    for size in array_sizes:
        print(f"\nBenchmarking with {size}x{size} matrices:")
        test = np.random.rand(size, size)
        template = np.random.rand(size, size)
        
        benchmark_frobenius(test, template)

if __name__ == "__main__":
    main()
# %%