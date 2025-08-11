import numpy as np

# Calculate complexity bounds for different models
n_samples = 100
complexities = {
    'Linear': 10,
    'Polynomial (degree 3)': 4,
    'Polynomial (degree 5)': 6,
    'Neural Network': 100
}

for model_name, complexity in complexities.items():
    bound = np.sqrt(complexity / n_samples)
    print(f"{model_name}: Complexity = {complexity}, Bound = {bound:.3f}")
