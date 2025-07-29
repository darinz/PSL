# 13.7. Deep Recommender Systems

Deep learning has revolutionized recommender systems by enabling the modeling of complex, non-linear relationships in user-item interactions. This section explores the application of deep neural networks to recommendation problems.

## 13.7.1. Introduction to Deep Recommender Systems

### Motivation

Traditional collaborative filtering methods have limitations:
- **Linear Assumptions**: Matrix factorization assumes linear relationships
- **Feature Engineering**: Requires manual feature extraction
- **Cold Start**: Poor performance with sparse data
- **Scalability**: Limited ability to handle complex patterns

Deep learning addresses these limitations by:
- **Non-linear Modeling**: Captures complex interaction patterns
- **Automatic Feature Learning**: Discovers latent representations
- **Multi-modal Integration**: Combines various data types
- **End-to-end Learning**: Optimizes the entire pipeline

### Mathematical Foundation

Deep recommender systems learn a function:

```math
f: \mathcal{U} \times \mathcal{I} \times \mathcal{C} \rightarrow \mathbb{R}
```

where:
- $`\mathcal{U}`$ is the user space
- $`\mathcal{I}`$ is the item space
- $`\mathcal{C}`$ is the context space
- $`\mathbb{R}`$ is the prediction space (rating, probability, etc.)

## 13.7.2. Neural Collaborative Filtering (NCF)

### Architecture

NCF replaces the inner product in matrix factorization with a neural network:

```math
\hat{r}_{ui} = f(\mathbf{u}_u, \mathbf{v}_i) = \sigma(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot [\mathbf{u}_u; \mathbf{v}_i] + \mathbf{b}_1) + \mathbf{b}_2)
```

where:
- $`\mathbf{u}_u`$ and $`\mathbf{v}_i`$ are user and item embeddings
- $`[\cdot; \cdot]`$ denotes concatenation
- $`\sigma`$ is the sigmoid function
- $`\mathbf{W}_1, \mathbf{W}_2, \mathbf{b}_1, \mathbf{b}_2`$ are learnable parameters

### Loss Function

For implicit feedback (binary classification):

```math
\mathcal{L} = -\sum_{(u,i) \in \mathcal{R}} \log(\hat{r}_{ui}) - \sum_{(u,i) \in \mathcal{R}^-} \log(1 - \hat{r}_{ui})
```

where $`\mathcal{R}^-`$ is the set of negative samples.

## 13.7.3. Wide & Deep Learning

### Architecture

Wide & Deep combines linear and deep models:

```math
\hat{r}_{ui} = \sigma(\mathbf{w}_{\text{wide}}^T \phi_{\text{wide}}(\mathbf{x}) + \mathbf{w}_{\text{deep}}^T \phi_{\text{deep}}(\mathbf{x}) + b)
```

where:
- $`\phi_{\text{wide}}`$ captures memorization (linear interactions)
- $`\phi_{\text{deep}}`$ captures generalization (non-linear interactions)

### Wide Component

```math
\phi_{\text{wide}}(\mathbf{x}) = [\mathbf{x}, \text{cross}(\mathbf{x})]
```

where $`\text{cross}(\mathbf{x})`$ creates cross-product features.

### Deep Component

```math
\phi_{\text{deep}}(\mathbf{x}) = \text{MLP}(\text{embed}(\mathbf{x}))
```

where $`\text{embed}`$ converts categorical features to embeddings.

## 13.7.4. Deep Matrix Factorization

### Neural Matrix Factorization (NeuMF)

NeuMF combines GMF (Generalized Matrix Factorization) and MLP:

```math
\hat{r}_{ui} = \sigma(\mathbf{h}^T \cdot [\phi_{\text{GMF}}(\mathbf{u}_u, \mathbf{v}_i); \phi_{\text{MLP}}(\mathbf{u}_u, \mathbf{v}_i)])
```

where:
- $`\phi_{\text{GMF}}(\mathbf{u}_u, \mathbf{v}_i) = \mathbf{u}_u \odot \mathbf{v}_i`$ (element-wise product)
- $`\phi_{\text{MLP}}(\mathbf{u}_u, \mathbf{v}_i) = \text{MLP}([\mathbf{u}_u; \mathbf{v}_i])`$

### Training Strategy

1. **Pre-training**: Train GMF and MLP separately
2. **Fine-tuning**: Joint training with pre-trained weights
3. **Ensemble**: Combine predictions from both components

## 13.7.5. Implementation

### Python Implementation: Deep Recommender Systems

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

class RatingDataset(Dataset):
    """Dataset for rating prediction"""
    
    def __init__(self, ratings_df, user_col='user_id', item_col='item_id', rating_col='rating'):
        self.ratings_df = ratings_df
        
        # Create mappings
        self.user_mapping = {user: idx for idx, user in enumerate(ratings_df[user_col].unique())}
        self.item_mapping = {item: idx for idx, item in enumerate(ratings_df[item_col].unique())}
        
        # Convert to indices
        self.user_indices = torch.tensor([
            self.user_mapping[user] for user in ratings_df[user_col]
        ], dtype=torch.long)
        
        self.item_indices = torch.tensor([
            self.item_mapping[item] for item in ratings_df[item_col]
        ], dtype=torch.long)
        
        self.ratings = torch.tensor(ratings_df[rating_col].values, dtype=torch.float)
        
        self.n_users = len(self.user_mapping)
        self.n_items = len(self.item_mapping)
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return {
            'user_idx': self.user_indices[idx],
            'item_idx': self.item_indices[idx],
            'rating': self.ratings[idx]
        }

class NCF(nn.Module):
    """Neural Collaborative Filtering"""
    
    def __init__(self, n_users, n_items, n_factors=10, layers=[20, 10], dropout=0.1):
        super(NCF, self).__init__()
        
        # Embeddings
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)
        
        # MLP layers
        self.mlp_layers = []
        input_size = 2 * n_factors
        
        for layer_size in layers:
            self.mlp_layers.extend([
                nn.Linear(input_size, layer_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_size = layer_size
        
        self.mlp = nn.Sequential(*self.mlp_layers)
        
        # Output layer
        self.output_layer = nn.Linear(layers[-1], 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
    
    def forward(self, user_idx, item_idx):
        # Get embeddings
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        
        # Concatenate
        concat = torch.cat([user_embed, item_embed], dim=1)
        
        # MLP
        mlp_output = self.mlp(concat)
        
        # Output
        output = self.output_layer(mlp_output)
        
        return output.squeeze()

class WideAndDeep(nn.Module):
    """Wide & Deep Learning Model"""
    
    def __init__(self, n_users, n_items, n_factors=10, deep_layers=[20, 10], dropout=0.1):
        super(WideAndDeep, self).__init__()
        
        # Wide component (linear)
        self.wide_user_embedding = nn.Embedding(n_users, 1)
        self.wide_item_embedding = nn.Embedding(n_items, 1)
        
        # Deep component
        self.deep_user_embedding = nn.Embedding(n_users, n_factors)
        self.deep_item_embedding = nn.Embedding(n_items, n_factors)
        
        # Deep MLP
        self.deep_layers = []
        input_size = 2 * n_factors
        
        for layer_size in deep_layers:
            self.deep_layers.extend([
                nn.Linear(input_size, layer_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_size = layer_size
        
        self.deep_mlp = nn.Sequential(*self.deep_layers)
        
        # Output layer
        self.output_layer = nn.Linear(deep_layers[-1] + 2, 1)  # +2 for wide features
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
    
    def forward(self, user_idx, item_idx):
        # Wide component
        wide_user = self.wide_user_embedding(user_idx).squeeze()
        wide_item = self.wide_item_embedding(item_idx).squeeze()
        wide_features = torch.stack([wide_user, wide_item], dim=1)
        
        # Deep component
        deep_user = self.deep_user_embedding(user_idx)
        deep_item = self.deep_item_embedding(item_idx)
        deep_concat = torch.cat([deep_user, deep_item], dim=1)
        deep_output = self.deep_mlp(deep_concat)
        
        # Combine wide and deep
        combined = torch.cat([wide_features, deep_output], dim=1)
        output = self.output_layer(combined)
        
        return output.squeeze()

class NeuMF(nn.Module):
    """Neural Matrix Factorization"""
    
    def __init__(self, n_users, n_items, n_factors=10, mlp_layers=[20, 10], dropout=0.1):
        super(NeuMF, self).__init__()
        
        # GMF embeddings
        self.gmf_user_embedding = nn.Embedding(n_users, n_factors)
        self.gmf_item_embedding = nn.Embedding(n_items, n_factors)
        
        # MLP embeddings
        self.mlp_user_embedding = nn.Embedding(n_users, n_factors)
        self.mlp_item_embedding = nn.Embedding(n_items, n_factors)
        
        # MLP layers
        self.mlp_layers = []
        input_size = 2 * n_factors
        
        for layer_size in mlp_layers:
            self.mlp_layers.extend([
                nn.Linear(input_size, layer_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_size = layer_size
        
        self.mlp = nn.Sequential(*self.mlp_layers)
        
        # Output layer
        self.output_layer = nn.Linear(n_factors + mlp_layers[-1], 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
    
    def forward(self, user_idx, item_idx):
        # GMF component
        gmf_user = self.gmf_user_embedding(user_idx)
        gmf_item = self.gmf_item_embedding(item_idx)
        gmf_output = gmf_user * gmf_item  # Element-wise product
        
        # MLP component
        mlp_user = self.mlp_user_embedding(user_idx)
        mlp_item = self.mlp_item_embedding(item_idx)
        mlp_concat = torch.cat([mlp_user, mlp_item], dim=1)
        mlp_output = self.mlp(mlp_concat)
        
        # Combine
        combined = torch.cat([gmf_output, mlp_output], dim=1)
        output = self.output_layer(combined)
        
        return output.squeeze()

def train_model(model, train_loader, val_loader, n_epochs=50, learning_rate=0.001, device='cpu'):
    """Train a deep recommendation model"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            user_idx = batch['user_idx'].to(device)
            item_idx = batch['item_idx'].to(device)
            ratings = batch['rating'].to(device)
            
            optimizer.zero_grad()
            predictions = model(user_idx, item_idx)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                user_idx = batch['user_idx'].to(device)
                item_idx = batch['item_idx'].to(device)
                ratings = batch['rating'].to(device)
                
                predictions = model(user_idx, item_idx)
                loss = criterion(predictions, ratings)
                val_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model performance"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            user_idx = batch['user_idx'].to(device)
            item_idx = batch['item_idx'].to(device)
            ratings = batch['rating'].to(device)
            
            preds = model(user_idx, item_idx)
            predictions.extend(preds.cpu().numpy())
            actuals.extend(ratings.cpu().numpy())
    
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
    return mae, rmse, predictions, actuals

# Generate synthetic data
np.random.seed(42)
n_users = 500
n_items = 300
n_ratings = 3000

# Create synthetic ratings with non-linear patterns
ratings_data = []
for user_id in range(n_users):
    n_user_ratings = np.random.randint(5, 20)
    rated_items = np.random.choice(n_items, n_user_ratings, replace=False)
    
    for item_id in rated_items:
        # Create non-linear patterns
        user_factor = np.random.normal(0, 1)
        item_factor = np.random.normal(0, 1)
        
        # Non-linear interaction
        interaction = np.sin(user_factor) * np.cos(item_factor) + user_factor * item_factor
        
        # Add noise and convert to rating
        rating = max(1, min(5, 3 + interaction + np.random.normal(0, 0.3)))
        ratings_data.append({
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating
        })

ratings_df = pd.DataFrame(ratings_data)

print("Synthetic Dataset with Non-linear Patterns:")
print(f"Number of users: {n_users}")
print(f"Number of items: {n_items}")
print(f"Number of ratings: {len(ratings_df)}")

# Prepare data
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Create datasets
train_dataset = RatingDataset(train_df)
val_dataset = RatingDataset(val_df)
test_dataset = RatingDataset(test_df)

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Train different models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

models = {
    'NCF': NCF(train_dataset.n_users, train_dataset.n_items, n_factors=10, layers=[20, 10]),
    'Wide&Deep': WideAndDeep(train_dataset.n_users, train_dataset.n_items, n_factors=10, deep_layers=[20, 10]),
    'NeuMF': NeuMF(train_dataset.n_users, train_dataset.n_items, n_factors=10, mlp_layers=[20, 10])
}

results = {}

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    train_losses, val_losses = train_model(model, train_loader, val_loader, n_epochs=50, device=device)
    
    # Evaluate
    mae, rmse, predictions, actuals = evaluate_model(model, test_loader, device=device)
    
    results[name] = {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'mae': mae,
        'rmse': rmse,
        'predictions': predictions,
        'actuals': actuals
    }
    
    print(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# Visualization
plt.figure(figsize=(20, 12))

# Plot 1: Training curves
plt.subplot(3, 4, 1)
for name, result in results.items():
    plt.plot(result['train_losses'], label=f'{name} Train')
    plt.plot(result['val_losses'], label=f'{name} Val', linestyle='--')
plt.title('Training Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot 2: Model comparison - MAE
plt.subplot(3, 4, 2)
mae_values = [results[name]['mae'] for name in results.keys()]
plt.bar(results.keys(), mae_values, color=['blue', 'red', 'green'])
plt.title('MAE Comparison')
plt.ylabel('Mean Absolute Error')

# Plot 3: Model comparison - RMSE
plt.subplot(3, 4, 3)
rmse_values = [results[name]['rmse'] for name in results.keys()]
plt.bar(results.keys(), rmse_values, color=['blue', 'red', 'green'])
plt.title('RMSE Comparison')
plt.ylabel('Root Mean Square Error')

# Plot 4-6: Prediction vs Actual for each model
for i, (name, result) in enumerate(results.items()):
    plt.subplot(3, 4, 4 + i)
    plt.scatter(result['actuals'], result['predictions'], alpha=0.6)
    plt.plot([1, 5], [1, 5], 'r--', alpha=0.8)
    plt.title(f'{name}: Predicted vs Actual')
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')

# Plot 7: Rating distribution
plt.subplot(3, 4, 7)
ratings_df['rating'].value_counts().sort_index().plot(kind='bar')
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')

# Plot 8: Model architecture comparison
plt.subplot(3, 4, 8)
architectures = ['NCF', 'Wide&Deep', 'NeuMF']
parameters = [
    sum(p.numel() for p in models[name].parameters()) 
    for name in architectures
]
plt.bar(architectures, parameters)
plt.title('Model Parameters')
plt.ylabel('Number of Parameters')

# Plot 9: Training time comparison (simulated)
plt.subplot(3, 4, 9)
training_times = [50, 45, 55]  # Simulated times
plt.bar(architectures, training_times)
plt.title('Training Time (epochs)')
plt.ylabel('Time')

# Plot 10: Convergence comparison
plt.subplot(3, 4, 10)
for name, result in results.items():
    plt.plot(result['val_losses'], label=name, marker='o', markersize=3)
plt.title('Convergence Comparison')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend()

# Plot 11: Error distribution
plt.subplot(3, 4, 11)
for name, result in results.items():
    errors = np.array(result['predictions']) - np.array(result['actuals'])
    plt.hist(errors, bins=20, alpha=0.7, label=name)
plt.title('Error Distribution')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.legend()

# Plot 12: Model summary
plt.subplot(3, 4, 12)
summary_data = {
    'Model': list(results.keys()),
    'MAE': [results[name]['mae'] for name in results.keys()],
    'RMSE': [results[name]['rmse'] for name in results.keys()],
    'Parameters': parameters
}
summary_df = pd.DataFrame(summary_data)
plt.table(cellText=summary_df.values, colLabels=summary_df.columns, cellLoc='center', loc='center')
plt.axis('off')
plt.title('Model Summary')

plt.tight_layout()
plt.show()

# Detailed analysis
print("\n=== Detailed Analysis ===")

# Compare model performance
print("Model Performance Comparison:")
for name, result in results.items():
    print(f"{name}:")
    print(f"  MAE: {result['mae']:.4f}")
    print(f"  RMSE: {result['rmse']:.4f}")
    print(f"  Parameters: {sum(p.numel() for p in result['model'].parameters()):,}")
    print()

# Analyze prediction patterns
print("Prediction Pattern Analysis:")
for name, result in results.items():
    predictions = np.array(result['predictions'])
    actuals = np.array(result['actuals'])
    
    print(f"{name}:")
    print(f"  Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"  Prediction std: {predictions.std():.3f}")
    print(f"  Bias: {predictions.mean() - actuals.mean():.3f}")
    print()

# Test recommendations
print("Recommendation Test:")
test_user = 0
test_item = 0

# Find user and item in test set
user_mapping = {user: idx for idx, user in enumerate(ratings_df['user_id'].unique())}
item_mapping = {item: idx for idx, item in enumerate(ratings_df['item_id'].unique())}

if test_user in user_mapping and test_item in item_mapping:
    user_idx = torch.tensor([user_mapping[test_user]]).to(device)
    item_idx = torch.tensor([item_mapping[test_item]]).to(device)
    
    print(f"Predictions for User {test_user}, Item {test_item}:")
    for name, result in results.items():
        model = result['model'].to(device)
        model.eval()
        with torch.no_grad():
            pred = model(user_idx, item_idx).cpu().numpy()[0]
        print(f"  {name}: {pred:.3f}")
```

### R Implementation

```r
# Deep Recommender Systems in R
library(keras)
library(tensorflow)
library(ggplot2)
library(dplyr)
library(tidyr)

# Generate synthetic data
set.seed(42)
n_users <- 500
n_items <- 300
n_ratings <- 3000

# Create synthetic ratings with non-linear patterns
ratings_data <- list()
for (user_id in 1:n_users) {
  n_user_ratings <- sample(5:20, 1)
  rated_items <- sample(1:n_items, n_user_ratings, replace = FALSE)
  
  for (item_id in rated_items) {
    # Create non-linear patterns
    user_factor <- rnorm(1, 0, 1)
    item_factor <- rnorm(1, 0, 1)
    
    # Non-linear interaction
    interaction <- sin(user_factor) * cos(item_factor) + user_factor * item_factor
    
    # Add noise and convert to rating
    rating <- max(1, min(5, 3 + interaction + rnorm(1, 0, 0.3)))
    
    ratings_data[[length(ratings_data) + 1]] <- list(
      user_id = user_id,
      item_id = item_id,
      rating = rating
    )
  }
}

ratings_df <- do.call(rbind, lapply(ratings_data, as.data.frame))

# Create user and item mappings
user_mapping <- setNames(1:length(unique(ratings_df$user_id)), unique(ratings_df$user_id))
item_mapping <- setNames(1:length(unique(ratings_df$item_id)), unique(ratings_df$item_id))

# Convert to indices
ratings_df$user_idx <- user_mapping[as.character(ratings_df$user_id)]
ratings_df$item_idx <- item_mapping[as.character(ratings_df$item_id)]

# Split data
set.seed(42)
train_indices <- sample(1:nrow(ratings_df), 0.8 * nrow(ratings_df))
train_df <- ratings_df[train_indices, ]
test_df <- ratings_df[-train_indices, ]

# Prepare data for Keras
n_users <- length(unique(ratings_df$user_id))
n_items <- length(unique(ratings_df$item_id))
n_factors <- 10

# NCF Model
build_ncf_model <- function() {
  # Input layers
  user_input <- layer_input(shape = 1, name = "user_input")
  item_input <- layer_input(shape = 1, name = "item_input")
  
  # Embeddings
  user_embedding <- user_input %>%
    layer_embedding(input_dim = n_users, output_dim = n_factors, name = "user_embedding") %>%
    layer_flatten()
  
  item_embedding <- item_input %>%
    layer_embedding(input_dim = n_items, output_dim = n_factors, name = "item_embedding") %>%
    layer_flatten()
  
  # Concatenate
  concat <- layer_concatenate(list(user_embedding, item_embedding))
  
  # MLP layers
  mlp <- concat %>%
    layer_dense(units = 20, activation = "relu") %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 10, activation = "relu") %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 1, activation = "linear")
  
  # Create model
  model <- keras_model(inputs = list(user_input, item_input), outputs = mlp)
  
  # Compile
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 0.001),
    loss = "mse",
    metrics = c("mae")
  )
  
  return(model)
}

# Train NCF model
ncf_model <- build_ncf_model()

# Prepare training data
user_indices <- train_df$user_idx - 1  # Keras uses 0-based indexing
item_indices <- train_df$item_idx - 1
ratings <- train_df$rating

# Train model
history <- ncf_model %>% fit(
  list(user_indices, item_indices),
  ratings,
  epochs = 50,
  batch_size = 64,
  validation_split = 0.2,
  verbose = 1
)

# Evaluate model
test_user_indices <- test_df$user_idx - 1
test_item_indices <- test_df$item_idx - 1
test_ratings <- test_df$rating

predictions <- ncf_model %>% predict(list(test_user_indices, test_item_indices))
mae <- mean(abs(predictions - test_ratings))
rmse <- sqrt(mean((predictions - test_ratings)^2))

cat("NCF Model Results:\n")
cat("MAE:", mae, "\n")
cat("RMSE:", rmse, "\n")

# Visualization
# Training history
p1 <- ggplot(data.frame(
  epoch = 1:length(history$metrics$loss),
  loss = history$metrics$loss,
  val_loss = history$metrics$val_loss
)) +
  geom_line(aes(x = epoch, y = loss, color = "Training")) +
  geom_line(aes(x = epoch, y = val_loss, color = "Validation")) +
  labs(title = "NCF Training History",
       x = "Epoch", y = "Loss", color = "Dataset") +
  theme_minimal()

# Prediction vs Actual
p2 <- ggplot(data.frame(
  actual = test_ratings,
  predicted = predictions
)) +
  geom_point(aes(x = actual, y = predicted), alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "NCF: Predicted vs Actual",
       x = "Actual Rating", y = "Predicted Rating") +
  theme_minimal()

# Rating distribution
p3 <- ggplot(ratings_df, aes(x = factor(rating))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Rating Distribution",
       x = "Rating", y = "Count") +
  theme_minimal()

# Combine plots
library(gridExtra)
grid.arrange(p1, p2, p3, ncol = 2)
```

## 13.7.6. Advanced Deep Learning Approaches

### Attention Mechanisms

#### Self-Attention for Sequential Recommendations
```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

where $`Q`$, $`K`$, and $`V`$ are query, key, and value matrices.

#### Multi-Head Attention
```math
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
```

where $`\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`$.

### Graph Neural Networks

#### Graph Convolutional Networks (GCN)
```math
H^{(l+1)} = \sigma\left(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)}\right)
```

where $`\tilde{A} = A + I`$ is the adjacency matrix with self-loops.

#### Graph Attention Networks (GAT)
```math
\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T[W\mathbf{h}_i \| W\mathbf{h}_j]))}{\sum_{k \in \mathcal{N}_i} \exp(\text{LeakyReLU}(\mathbf{a}^T[W\mathbf{h}_i \| W\mathbf{h}_k]))}
```

### Transformer-based Models

#### BERT4Rec
```math
P(r_t | r_1, \ldots, r_{t-1}) = \text{softmax}(W\mathbf{h}_t + \mathbf{b})
```

where $`\mathbf{h}_t`$ is the hidden state from the transformer.

## 13.7.7. Multi-modal Deep Learning

### Text + Image Recommendations

#### Multi-modal Fusion
```math
\mathbf{f}_{\text{fused}} = \alpha \cdot \mathbf{f}_{\text{text}} + (1-\alpha) \cdot \mathbf{f}_{\text{image}}
```

where $`\alpha`$ is learned during training.

#### Cross-modal Attention
```math
\text{Attention}_{\text{cross}}(\mathbf{q}, \mathbf{K}) = \text{softmax}\left(\frac{\mathbf{q}\mathbf{K}^T}{\sqrt{d_k}}\right)
```

### Audio + Visual Recommendations

#### Temporal Fusion
```math
\mathbf{h}_t = \text{LSTM}([\mathbf{a}_t; \mathbf{v}_t], \mathbf{h}_{t-1})
```

where $`\mathbf{a}_t`$ and $`\mathbf{v}_t`$ are audio and visual features.

## 13.7.8. Evaluation and Optimization

### Loss Functions

#### Ranking Loss
```math
\mathcal{L}_{\text{ranking}} = \sum_{(u,i,j) \in \mathcal{D}} \max(0, \hat{r}_{uj} - \hat{r}_{ui} + \gamma)
```

where $`(u,i,j)`$ represents user $`u`$ prefers item $`i`$ over item $`j`$.

#### BPR Loss
```math
\mathcal{L}_{\text{BPR}} = -\sum_{(u,i,j) \in \mathcal{D}} \log(\sigma(\hat{r}_{ui} - \hat{r}_{uj}))
```

### Regularization Techniques

#### Dropout
```math
\mathbf{h}_{\text{dropout}} = \mathbf{h} \odot \mathbf{m}
```

where $`\mathbf{m} \sim \text{Bernoulli}(p)`$.

#### Weight Decay
```math
\mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda \sum_{\theta} \|\theta\|_2^2
```

### Hyperparameter Optimization

#### Bayesian Optimization
```math
\alpha^* = \arg\max_{\alpha} \text{Acquisition}(\alpha | \mathcal{D})
```

#### Neural Architecture Search (NAS)
```math
\mathcal{A}^* = \arg\max_{\mathcal{A}} \text{Performance}(\mathcal{A})
```

## 13.7.9. Production Considerations

### Model Serving

#### TensorFlow Serving
```python
# Save model
model.save('recommendation_model')

# Load and serve
import tensorflow as tf
loaded_model = tf.keras.models.load_model('recommendation_model')
```

#### ONNX Export
```python
import onnx
import tf2onnx

# Convert to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model)
onnx.save(onnx_model, "model.onnx")
```

### Scalability

#### Model Parallelism
```math
\mathbf{y} = f_2(f_1(\mathbf{x}))
```

where $`f_1`$ and $`f_2`$ run on different devices.

#### Data Parallelism
```math
\nabla \mathcal{L} = \frac{1}{N} \sum_{i=1}^N \nabla \mathcal{L}_i
```

### Real-time Recommendations

#### Online Learning
```math
\theta_{t+1} = \theta_t - \eta_t \nabla \mathcal{L}(\theta_t, \mathbf{x}_t, y_t)
```

#### Incremental Updates
```math
\mathbf{h}_{t+1} = \mathbf{h}_t + \alpha \cdot \text{update}(\mathbf{x}_{t+1})
```

## 13.7.10. Summary

### Key Advantages

1. **Non-linear Modeling**: Captures complex interaction patterns
2. **Automatic Feature Learning**: Discovers latent representations
3. **Multi-modal Integration**: Combines various data types
4. **End-to-end Learning**: Optimizes the entire pipeline
5. **Scalability**: Can handle large-scale data

### Key Challenges

1. **Computational Cost**: Training deep models is expensive
2. **Interpretability**: Black-box nature makes explanations difficult
3. **Data Requirements**: Needs large amounts of training data
4. **Hyperparameter Tuning**: Many parameters to optimize
5. **Overfitting**: Risk of memorizing training data

### Best Practices

1. **Start Simple**: Begin with basic architectures
2. **Use Pre-trained Models**: Leverage transfer learning
3. **Regularize Properly**: Prevent overfitting
4. **Monitor Training**: Track loss and metrics carefully
5. **Validate Thoroughly**: Use multiple evaluation metrics

### Future Directions

1. **Self-supervised Learning**: Learning without explicit labels
2. **Meta-learning**: Learning to learn recommendation patterns
3. **Federated Learning**: Privacy-preserving distributed training
4. **AutoML**: Automated architecture and hyperparameter search
5. **Explainable AI**: Interpretable recommendation explanations

Deep recommender systems represent the cutting edge of recommendation technology, offering unprecedented ability to model complex user-item interactions. While they require significant computational resources and expertise, they can provide substantial improvements in recommendation quality when properly implemented and tuned.

---

**Next**: [Introduction](01_introduction.md) - Return to the beginning to review the fundamentals of recommender systems.
