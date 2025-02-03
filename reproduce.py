import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import networkx as nx
from itertools import combinations
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def normalize_adjacency_dense_gpu(A):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    A = A.to(device)  # Move to GPU if available

    # Ensure self-loops
    A = A + torch.eye(A.size(0), device=A.device)

    # Degree vector
    row_sum = torch.sum(A, dim=1)

    # Avoid division by zero by adding a small epsilon
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(1e-10+ row_sum ))
    # Normalize adjacency
    normalized_A = D_inv_sqrt @ A @ D_inv_sqrt

    # Enforce symmetry (optional but helps to handle numerical instability)
    normalized_A = (normalized_A + normalized_A.T) / 2.0

    return normalized_A


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        #self.weight = nn.Parameter(torch.randn(output_dim, input_dim))  # output_dim should be first


    def forward(self, X, A_tilde):
        return A_tilde @ X @ self.weight


class InferenceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(InferenceModel, self).__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2_mu = GCNLayer(hidden_dim, latent_dim)
        self.gcn2_logsigma = GCNLayer(hidden_dim, latent_dim)

    def forward(self, X, A_tilde):
        A_tilde = A_tilde / (A_tilde.sum(dim=1, keepdim=True) + 1e-8)

        H = F.relu(self.gcn1(X, A_tilde))  # Shared first layer
        if torch.isnan(H).any():
            print("NaN detected in H!")

        mu = self.gcn2_mu(H, A_tilde)  # Mean matrix
        log_sigma = self.gcn2_logsigma(H, A_tilde)  # Log-variance matrix
        return mu, log_sigma


class VGAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VGAE, self).__init__()
        self.encoder = InferenceModel(input_dim, hidden_dim, latent_dim)

    def forward(self, X, A_tilde):
        mu, log_sigma = self.encoder(X, A_tilde)
        # Reparameterization trick
        std = torch.exp(0.5 * log_sigma)
        std = torch.clamp(std, min=1e-5, max=10)
        eps = torch.randn_like(std)
        if torch.isnan(std).any():
            print("NaN detected in std!")
        Z = mu + eps * std
        Z = F.normalize(Z, p=2, dim=1)  # Normalize rows of Z to unit length
        #Z = Z / torch.sqrt(torch.tensor(Z.shape[1], dtype=torch.float32, device=Z.device))
        if torch.isnan(mu).any():
            print("NaN detected in mu!")

        if torch.isnan(log_sigma).any():
            print("NaN detected in log_sigma!")

        A_reconstructed = torch.sigmoid(torch.matmul(Z, Z.T))
        return Z, A_reconstructed, mu, log_sigma



class VGAE_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = InferenceModel(input_dim, hidden_dim, latent_dim)

        # MLP Decoder (2-layer perceptron)
        self.decoder = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
            )


    def forward(self, X, A_tilde, edge_index=None):
        mu, log_sigma = self.encoder(X, A_tilde)

        # Reparameterization trick
        std = torch.exp(0.5 * log_sigma)
        std = torch.clamp(std, max=1e5)
        eps = torch.randn_like(std)
        Z = mu + eps * std
        #Z = F.normalize(Z, p=2, dim=1)  # Normalize rows of Z to unit length



        batch_size = 10000  # Adjust based on memory
        num_edges = edge_index.shape[1]
        A_reconstructed_list = []

        for i in range(0, num_edges, batch_size):
            batch_edges = edge_index[:, i : i + batch_size]
            src, dst = batch_edges
            Z_concat = torch.cat([Z[src], Z[dst]], dim=1)
            A_reconstructed_list.append(torch.sigmoid(self.decoder(Z_concat)).squeeze())

        A_reconstructed = torch.cat(A_reconstructed_list)

        return Z, A_reconstructed, mu, log_sigma


class WeightedInnerProductDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(latent_dim))  # Learnable weight vector

    def forward(self, Z):
        Z_weighted = Z * self.weight  # Apply element-wise weight
        A_reconstructed = torch.sigmoid(torch.matmul(Z, Z_weighted.T))  # Full adjacency matrix
        return A_reconstructed


class VGAE_W(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = InferenceModel(input_dim, hidden_dim, latent_dim)
        self.decoder = WeightedInnerProductDecoder(latent_dim)

    def forward(self, X, A_tilde):
        mu, log_sigma = self.encoder(X, A_tilde)

        # Reparameterization trick
        std = torch.exp(0.5 * log_sigma)
        std = torch.clamp(std, max=1e5)
        eps = torch.randn_like(std)
        Z = mu + eps * std
        Z = F.normalize(Z, p=2, dim=1)  # Normalize rows of Z to unit length

        A_reconstructed = self.decoder(Z)  # No need for edge_index now

        return Z, A_reconstructed, mu, log_sigma


def loss_function(A, A_reconstructed, mu, log_sigma):
    # Reconstruction loss (Binary Cross-Entropy)
    epsilon = 1e-7
    A_reconstructed = torch.clamp(A_reconstructed, min=epsilon, max=1 - epsilon)
    recon_loss = F.binary_cross_entropy(A_reconstructed, A, reduction='sum')

    # KL Divergence
    kl_loss = -0.5 * torch.sum(1 + log_sigma.clamp(min=-2, max=2) - mu.clamp(min=-2, max=2).pow(2) - log_sigma.clamp(min=-2, max=2).exp())
    #print(log_sigma)
    return recon_loss + kl_loss


def loss_function_mlp(A, A_reconstructed, mu, log_sigma, edge_index):
    src, dst = edge_index  # Edge indices

    # If A_reconstructed is 1D, select indices correctly
    A_pred = A_reconstructed[torch.arange(edge_index.shape[1])]

    # Get true adjacency values
    A_true = A[src, dst]

    # BCE loss
    recon_loss = F.binary_cross_entropy(A_pred, A_true, reduction='sum')

    # KL Divergence
    kl_loss = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.clamp(max=10).exp())

    return recon_loss + kl_loss

def to_dense_adj_custom(edge_index, batch=None, num_nodes=None):

    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1  # Infer number of nodes if not provided

    if batch is None:
        adj = torch.zeros((1, num_nodes, num_nodes), dtype=torch.float32, device=edge_index.device)
        adj[0, edge_index[0], edge_index[1]] = 1
        return adj
    else:
        num_graphs = batch.max().item() + 1
        max_nodes = torch.bincount(batch).max().item()  # Max nodes per graph
        adj = torch.zeros((num_graphs, max_nodes, max_nodes), dtype=torch.float32, device=edge_index.device)

        for i in range(num_graphs):
            mask = batch[edge_index[0]] == i
            nodes = batch == i
            node_idx = torch.arange(nodes.sum(), device=edge_index.device)
            node_map = torch.full((batch.size(0),), -1, device=edge_index.device)
            node_map[nodes] = node_idx
            adj[i, node_map[edge_index[0, mask]], node_map[edge_index[1, mask]]] = 1

        return adj

from torch_geometric.datasets import CoraFull

cora_dataset = CoraFull(root='GraphDatasets/Cora')
print(f'Dataset: {cora_dataset}:')
print('======================')
print(f'Number of graphs: {len(cora_dataset)}')
print(f'Number of features: {cora_dataset.num_features}')
print(f'Number of classes: {cora_dataset.num_classes}')

# Graph data
data = cora_dataset[0]
X = data.x  # features matrix (N x D)
edge_index = data.edge_index  # Edge list (2 x E)

# Create the adjacency matrix (A)
num_nodes = X.size(0)
A = torch.zeros((num_nodes, num_nodes))

# Convert the edge_index to an adjacency matrix
row, col = edge_index
A[row, col] = 1
A[col, row] = 1  # Since the graph is undirected

# Optionally, add self-loops (diagonal elements set to 1)
A.fill_diagonal_(1)

print("Adjacency matrix (A):", A)
print("Node feature matrix (X):", X)


X.shape

edge_index.shape

from torch_geometric.datasets import Planetoid

cite_dataset = Planetoid(root='GraphDatasets/CiteSeer', name='CiteSeer')
print(f'Dataset: {cite_dataset}:')
print('======================')
print(f'Number of graphs: {len(cite_dataset)}')
print(f'Number of features: {cite_dataset.num_features}')
print(f'Number of classes: {cite_dataset.num_classes}')

# Graph data
data = cite_dataset[0]
X = data.x  # features matrix (N x D)
edge_index = data.edge_index  # Edge list (2 x E)

# Create the adjacency matrix (A)
num_nodes = X.size(0)
A = torch.zeros((num_nodes, num_nodes))

# Convert the edge_index to an adjacency matrix
row, col = edge_index
A[row, col] = 1
A[col, row] = 1  # Since the graph is undirected

# Optionally, add self-loops (diagonal elements set to 1)
A.fill_diagonal_(1)

print("Adjacency matrix (A):", A)
print("Node feature matrix (X):", X)


from torch_geometric.datasets import Planetoid

pub_dataset = Planetoid(root='GraphDatasets/PubMed', name='PubMed')
print(f'Dataset: {pub_dataset}:')
print('======================')
print(f'Number of graphs: {len(pub_dataset)}')
print(f'Number of features: {pub_dataset.num_features}')
print(f'Number of classes: {pub_dataset.num_classes}')

# Graph data
data = pub_dataset[0]
X = data.x  # features matrix (N x D)
edge_index = data.edge_index  # Edge list (2 x E)

# Create the adjacency matrix (A)
num_nodes = X.size(0)
A = torch.zeros((num_nodes, num_nodes))

# Convert the edge_index to an adjacency matrix
row, col = edge_index
A[row, col] = 1
A[col, row] = 1  # Since the graph is undirected

# Optionally, add self-loops (diagonal elements set to 1)
A.fill_diagonal_(1)

print("Adjacency matrix (A):", A)
print("Node feature matrix (X):", X)




def remove_edges_and_sample_optimized(edge_index, num_nodes, test_size=0.1, val_size=0.05):

    # Convert edge_index to a set of edges for faster lookup
    edges = set(map(tuple, edge_index.t().tolist()))

    # Generate all possible node pairs (i, j) for non-edges
    all_pairs = set(combinations(range(num_nodes), 2))
    non_edges = list(all_pairs - edges)

    # Split edges into validation and test sets
    edges = list(edges)
    train_edges, temp_edges = train_test_split(edges, test_size=test_size + val_size, random_state=42)
    val_edges, test_edges = train_test_split(temp_edges, test_size=test_size / (test_size + val_size), random_state=42)

    # Sample non-edges for validation and test sets
    num_val_non_edges = len(val_edges)
    num_test_non_edges = len(test_edges)

    val_non_edges = random.sample(non_edges, num_val_non_edges)
    test_non_edges = random.sample(non_edges, num_test_non_edges)
    # Recreate the training graph without validation and test edges
    train_graph = nx.Graph()
    train_graph.add_edges_from(train_edges)
    train_graph.add_nodes_from(range(num_nodes))  # Add isolated nodes
    train_edge_index = torch.tensor(list(train_graph.edges)).t().contiguous()

    # Recreate training edge_index
    train_edge_index = torch.tensor(train_edges).t().contiguous()

    return train_edge_index, val_edges, test_edges, val_non_edges, test_non_edges,train_graph

# Extract edge_index and number of nodes
edge_index = data.edge_index  # (2, E)
num_nodes = data.num_nodes

# Split edges and sample non-edges
train_edge_index, val_edges, test_edges, val_non_edges, test_non_edges, train_graph = remove_edges_and_sample_optimized(edge_index, num_nodes)

print("Train edge index shape:", train_edge_index.shape)
print("Number of validation edges:", len(val_edges))
print("Number of test edges:", len(test_edges))
print("Number of validation non-edges:", len(val_non_edges))
print("Number of test non-edges:", len(test_non_edges))


A.shape

X.shape

# Extract train graph adjacency matrix
num_n=train_edge_index.max().item() + 1
train_adj_matrix = to_dense_adj_custom(train_edge_index, max_num_nodes=num_nodes)[0]
train_adj_matrix = train_adj_matrix.to(torch.float32)  # Ensure float type for computations

train_adj_matrix = train_adj_matrix.clamp(max=1)

# Normalize adjacency for training graph
A_tilde_train = normalize_adjacency_dense_gpu(train_adj_matrix)

# Initialize model
input_dim = X.shape[1]
hidden_dim = 32
latent_dim = 16
model = VGAE(input_dim, hidden_dim, latent_dim)

model = model.to('cuda')  # Move model to GPU if available
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Extract train graph adjacency matrix
num_n=train_edge_index.max().item() + 1
train_adj_matrix = to_dense_adj_custom(train_edge_index, max_num_nodes=num_n)[0]  # Convert to dense adjacency matrix
train_adj_matrix = train_adj_matrix.to(torch.float32)  # Ensure float type for computations

train_adj_matrix = train_adj_matrix.clamp(max=1)

A_tilde_train = normalize_adjacency_dense_gpu(train_adj_matrix)

# Initialize model
input_dim = X.shape[1]
hidden_dim = 128
latent_dim = 64
model = VGAE_W(input_dim, hidden_dim, latent_dim)

model = model.to('cuda')  # Move model to GPU if available
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 200

X = torch.eye(num_nodes)
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    Z, A_reconstructed, mu, log_sigma = model(X.to('cuda'), A_tilde_train.to('cuda'))

    # Debugging: Check for NaNs in the output of the model
    if torch.isnan(A_reconstructed).sum() > 0 :
        print("NaN  detected in A_reconstructed!")
        break
    if  torch.isinf(A_reconstructed).sum() > 0:
        print(" Inf detected in A_reconstructed!")
        break


    # Compute loss
    loss = loss_function(train_adj_matrix.to('cuda'), A_reconstructed.to('cuda'), mu.to('cuda'), log_sigma.to('cuda'))

    # Check if loss becomes NaN or Inf
    if torch.isnan(loss).sum() > 0 or torch.isinf(loss).sum() > 0:
        print("NaN or Inf detected in loss!")
        break

    # Apply gradient clipping before backward pass
    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Backward pass
    loss.backward()

    # Update parameters
    optimizer.step()

    # Print loss at each epoch
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


import torch
from torch_geometric.utils import to_dense_adj

# Assume edge_index, num_nodes, and remove_edges_and_sample_optimized are defined
# Extract train graph adjacency matrix
num_n=train_edge_index.max().item() + 1
train_adj_matrix = to_dense_adj(train_edge_index, max_num_nodes=num_n)[0]  # Convert to dense adjacency matrix
train_adj_matrix = train_adj_matrix.to(torch.float32)  # Ensure float type for computations
train_adj_matrix = to_dense_adj(train_edge_index, max_num_nodes=num_nodes)[0]
# Convert to SciPy sparse matrix
# Create the adjacency matrix from the edge list (train_edge_index)
#train_adj_matrix = to_dense_adj(train_edge_index, max_num_nodes=num_nodes)[0]

# Enforce symmetry (add the transpose to ensure both directions are captured)
train_adj_matrix = train_adj_matrix + train_adj_matrix.T

# Ensure that the diagonal entries are 1 (self-loops)
train_adj_matrix.fill_diagonal_(1.0)
train_adj_matrix = train_adj_matrix.clamp(max=1)
# Node features
if data.x is not None:
    X = data.x  # Use provided node features
else:
    X = torch.eye(num_nodes)  # Use identity matrix if featureless

# Normalize adjacency for training graph
A_tilde_train = normalize_adjacency_dense_gpu(train_adj_matrix)

# Initialize model
input_dim = X.shape[1]
hidden_dim = 128
latent_dim = 64
model = VGAE_MLP(input_dim, hidden_dim, latent_dim)

model = model.to('cuda')  # Move model to GPU if available
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    Z, A_reconstructed, mu, log_sigma = model(X.to('cuda'), A_tilde_train.to('cuda'), edge_index=edge_index)

    # Clamp log_sigma to prevent extreme values
    log_sigma = torch.clamp(log_sigma, min=-18, max=18)

    # Compute loss
    loss = loss_function_mlp(train_adj_matrix.to('cuda'), A_reconstructed, mu, log_sigma, train_edge_index.to('cuda'))
    loss.backward()

    # Update parameters
    optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")




A_reconstructed = A_reconstructed.detach().cpu()

# Ensure test edges and non-edges are tensors
test_edges = torch.tensor(test_edges, dtype=torch.long)
test_non_edges = torch.tensor(test_non_edges, dtype=torch.long)

# Handle different decoder outputs
if A_reconstructed.dim() == 2:
    # If A_reconstructed is a full adjacency matrix
    test_edge_scores = A_reconstructed[test_edges[:, 0], test_edges[:, 1]].numpy()
    test_non_edge_scores = A_reconstructed[test_non_edges[:, 0], test_non_edges[:, 1]].numpy()
else:
    # If A_reconstructed is a 1D tensor (edge probabilities only)
    test_edge_scores = A_reconstructed[:len(test_edges)].numpy()
    test_non_edge_scores = A_reconstructed[len(test_edges):].numpy()

# Combine scores and create labels
scores = np.concatenate([test_edge_scores, test_non_edge_scores])
labels = np.concatenate([np.ones(len(test_edge_scores)), np.zeros(len(test_non_edge_scores))])



roc_auc = roc_auc_score(labels, scores)
print(f"ROC-AUC Score: {roc_auc}")


ap_score = average_precision_score(labels, scores)
print(f"Average Precision (AP): {ap_score:.4f}")

from scipy.sparse import save_npz, load_npz
import numpy as np
import torch
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split

A = load_npz("combined_adj_small.npz")
X = load_npz("combined_features_matrix.npz")

X = torch.tensor(X.toarray(), dtype=torch.float32)

# Check if the matrix is symmetric
if (A != A.T).nnz == 0:  # If the number of non-zero elements in (A - A.T) is zero
    print("Matrix A is symmetric.")
else:
    print("Matrix A is not symmetric.")

X



def split_edges_and_sample(A, num_samples=None, test_size=0.1, val_size=0.05, random_state=42):
    """
    Efficiently splits edges and samples non-edges.

    Parameters:
    - A: scipy.sparse.coo_matrix (adjacency matrix)
    - num_samples: Number of non-edges to sample (adjust based on graph size)
    - test_size: Proportion of edges/non-edges for testing
    - val_size: Proportion of edges/non-edges for validation
    - random_state: Random seed for reproducibility

    Returns:
    - train_edge_index: Edge list for training
    - val_edges, test_edges: Validation and test edge lists
    - val_non_edges, test_non_edges: Validation and test non-edges
    - train_graph: Sparse matrix representing the training graph
    """
    A_coo = coo_matrix(A)
    edges = np.vstack((A_coo.row, A_coo.col)).T  # Extract edges
    num_nodes = A.shape[0]

    # Convert edges to a set for fast lookup
    existing_edges = set(map(tuple, edges))
    num_samples=len(edges)*0.15
    # Randomly sample non-edges
    np.random.seed(random_state)
    non_edges = set()
    while len(non_edges) < num_samples:
        i = np.random.randint(0, num_nodes)
        j = np.random.randint(0, num_nodes)
        if i != j and (i, j) not in existing_edges and (j, i) not in existing_edges:
            non_edges.add((i, j))
    #print(existing_edges)
    non_edges = np.array(list(non_edges))

    # Split edges into train, validation, and test sets
    train_edges, temp_edges = train_test_split(edges, test_size=(test_size + val_size), random_state=random_state)

    val_edges, test_edges = train_test_split(temp_edges, test_size=(test_size / (test_size + val_size)), random_state=random_state)

    # Split sampled non-edges into validation and test sets
    val_non_edges, test_non_edges = train_test_split(non_edges, test_size=(test_size / (test_size + val_size)), random_state=random_state)

    # Create a training graph (without val/test edges)
    train_graph = coo_matrix(
        (np.ones(len(train_edges)), (train_edges[:, 0], train_edges[:, 1])),
        shape=A.shape
    )
    train_graph = train_graph + train_graph.T  # Ensure symmetry

    # Convert to PyTorch tensors
    train_edge_index = torch.tensor(train_edges.T, dtype=torch.long)
    val_edges = torch.tensor(val_edges, dtype=torch.long)
    test_edges = torch.tensor(test_edges, dtype=torch.long)
    val_non_edges = torch.tensor(val_non_edges, dtype=torch.long)
    test_non_edges = torch.tensor(test_non_edges, dtype=torch.long)

    return train_edge_index, val_edges, test_edges, val_non_edges, test_non_edges, train_graph


#A = A + A.T  # Ensure symmetry for an undirected graph
#A[A > 1] = 1  # Remove duplicate edges

# Split edges and sample non-edges
train_edge_index, val_edges, test_edges, val_non_edges, test_non_edges, train_graph= split_edges_and_sample(A)

print("Train edge index shape:", train_edge_index.shape)
print("Number of validation edges:", len(val_edges))
print("Number of test edges:", len(test_edges))
print("Number of validation non-edges:", len(val_non_edges))
print("Number of test non-edges:", len(test_non_edges))


def normalize_adjacency_sparse_gpu(A):
    """
    Normalize adjacency matrix on GPU using sparse matrices.
    A: Sparse adjacency matrix (torch.sparse.FloatTensor).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    A = A.to(device)  # Move to GPU if available

    # Ensure self-loops (can be done with sparse matrices too)
    #eye = torch.eye(A.size(0), device=A.device).to_sparse()
    #A = A + eye

    # Degree vector (sparse sum)
    row_sum = torch.sum(A, dim=1) # Sparse sum and convert to dense

    # Avoid division by zero by adding a small epsilon
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(row_sum + 1e-10))

    # Normalize adjacency
    normalized_A = D_inv_sqrt @ A @ D_inv_sqrt

    # Enforce symmetry
    normalized_A = (normalized_A + normalized_A.T) / 2.0

    return normalized_A


import torch
#import torch_sparse
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.utils import to_dense_adj

# Assume edge_index, num_nodes, and remove_edges_and_sample_optimized are defined
# Extract train graph adjacency matrix

# Extract train graph adjacency matrix
num_nodes =  max(train_edge_index[0].max(), train_edge_index[1].max()) + 1


train_adj_matrix = to_dense_adj(train_edge_index, max_num_nodes=num_nodes)[0]
# Convert to SciPy sparse matrix


#train_adj_matrix = to_scipy_sparse_matrix(train_edge_index, num_nodes=num_nodes)
# Ensure adjacency matrix is sparse on GPU
train_adj_matrix = train_adj_matrix.to('cuda')  # Move sparse matrix to GPU

# Convert directly to a PyTorch sparse tensor
#train_adj_matrix = torch.tensor(train_adj_matrix.toarray(), dtype=torch.float32)
# Ensure adjacency matrix is sparse on GPU
#train_adj_matrix = train_adj_matrix.to_sparse().to('cuda')  # Move sparse matrix to GPU
#train_adj_matrix = torch.sparse_coo_tensor(indices, values, size=(num_nodes, num_nodes), dtype=torch.float32).to(device)

# Normalize adjacency for training graph
A_tilde_train = normalize_adjacency_dense_gpu(train_adj_matrix.to(torch.float32))  # Normalize

#train_adj_matrix = train_adj_matrix.to('cuda')
#A_tilde_train = normalize_adjacency_dense_gpu(train_adj_matrix)


# Initialize model
input_dim = X.shape[1]
hidden_dim = 32
latent_dim = 16
model = VGAE_MLP(input_dim, hidden_dim, latent_dim)

model = model.to('cuda')  # Move model to GPU if available
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    Z, A_reconstructed, mu, log_sigma = model(X.to('cuda'), A_tilde_train.to('cuda'), edge_index=train_edge_index)

    # Clamp log_sigma to prevent extreme values
    log_sigma = torch.clamp(log_sigma, min=-14, max=14)

    # Compute loss
    loss = loss_function_mlp(train_adj_matrix.to('cuda'), A_reconstructed, mu, log_sigma, train_edge_index.to('cuda'))
    loss.backward()

    # Update parameters
    optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")



A_tilde_train = normalize_adjacency_dense_gpu(train_adj_matrix)
A_tilde_train = A_tilde_train.to(torch.float32)

D = torch.diag(A_tilde_train.sum(dim=1).clamp(min=1).pow(-0.5))
A_tilde_train = D @ A_tilde_train @ D


A_tilde_train = A_tilde_train / A_tilde_train.sum(dim=1, keepdim=True).clamp(min=1)


is_symmetric = torch.allclose(train_adj_matrix, train_adj_matrix.T, atol=1e-6)
print("Is A_tilde_train symmetric?", is_symmetric)

is_symmetric = torch.allclose(A_tilde_train, A_tilde_train.T, atol=1e-6)
print("Is A_tilde_train symmetric?", is_symmetric)


for i in train_adj_matrix[0]:
    if i!=0:
        print(i)

for i in A_tilde_train[0]:
    if i!=0:
        print(i)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Layer: {name} | Weights: {param.data}")


Layer: decoder.weight | Weights: tensor([ 0.7694,  0.7375,  0.7400,  0.8426,  0.6726,  0.5184,  0.8114, -0.0369,
         0.8284,  0.7812,  0.6179,  0.5305,  0.7229,  0.0629,  0.1734,  0.5006]

Layer: decoder.weight | Weights: tensor([0.2511, 0.4997, 0.5713, 0.3977, 0.4057, 0.6872, 0.3339, 0.1781, 0.3881,
        0.8694, 0.3839, 0.3271, 0.4893, 0.7137, 0.7647, 0.5228],

row_sums=A_tilde_train.sum(dim=1)
print("Row sum min:", row_sums.min().item())
print("Row sum max:", row_sums.max().item())
print("Row sum mean:", row_sums.mean().item())


def split_edges_and_sample(A, num_samples=None, test_size=0.1, val_size=0.05, random_state=42):
    """
    Efficiently splits edges and samples non-edges.

    Parameters:
    - A: scipy.sparse.coo_matrix (adjacency matrix)
    - num_samples: Number of non-edges to sample (adjust based on graph size)
    - test_size: Proportion of edges/non-edges for testing
    - val_size: Proportion of edges/non-edges for validation
    - random_state: Random seed for reproducibility

    Returns:
    - train_edge_index: Edge list for training
    - val_edges, test_edges: Validation and test edge lists
    - val_non_edges, test_non_edges: Validation and test non-edges
    - train_graph: Sparse matrix representing the training graph
    """
    A_coo = coo_matrix(A)
    edges = np.vstack((A_coo.row, A_coo.col)).T  # Extract edges
    num_nodes = A.shape[0]

    # Normalize edge representation to avoid both (i, j) and (j, i)
    edges = np.array([tuple(sorted((i, j))) for i, j in edges])
    existing_edges = set(map(tuple, edges))

    # Sample non-edges
    if num_samples is None:
        num_samples = int(len(edges) * 0.15)  # Sample 15% of edges as non-edges by default

    np.random.seed(random_state)
    non_edges = set()
    while len(non_edges) < num_samples:
        i = np.random.randint(0, num_nodes)
        j = np.random.randint(0, num_nodes)
        if i != j:
            edge = tuple(sorted((i, j)))
            if edge not in existing_edges:
                non_edges.add(edge)

    non_edges = np.array(list(non_edges))

    # Split edges into train, validation, and test sets
    train_edges, temp_edges = train_test_split(edges, test_size=(test_size + val_size), random_state=random_state)
    val_edges, test_edges = train_test_split(temp_edges, test_size=(test_size / (test_size + val_size)), random_state=random_state)

    # Split sampled non-edges into validation and test sets
    val_non_edges, test_non_edges = train_test_split(non_edges, test_size=(test_size / (test_size + val_size)), random_state=random_state)

    # Create a training graph (without val/test edges)
    train_graph = coo_matrix(
        (np.ones(len(train_edges)), (train_edges[:, 0], train_edges[:, 1])),
        shape=A.shape
    )
    train_graph = train_graph + train_graph.T  # Ensure symmetry

    # Convert to PyTorch tensors
    train_edge_index = torch.tensor(train_edges.T, dtype=torch.long)
    val_edges = torch.tensor(val_edges, dtype=torch.long)
    test_edges = torch.tensor(test_edges, dtype=torch.long)
    val_non_edges = torch.tensor(val_non_edges, dtype=torch.long)
    test_non_edges = torch.tensor(test_non_edges, dtype=torch.long)

    return train_edge_index, val_edges, test_edges, val_non_edges, test_non_edges, train_graph


!pip install torch_sparse -f https://pytorch-geometric.com/whl/cpu.html



import torch
#import torch_sparse
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.utils import to_dense_adj

# Assume edge_index, num_nodes, and remove_edges_and_sample_optimized are defined
# Extract train graph adjacency matrix

# Extract train graph adjacency matrix
num_nodes = train_edge_index.max().item() + 1

train_adj_matrix = to_dense_adj(train_edge_index, max_num_nodes=num_nodes)[0]
# Convert to SciPy sparse matrix
# Create the adjacency matrix from the edge list (train_edge_index)
#train_adj_matrix = to_dense_adj(train_edge_index, max_num_nodes=num_nodes)[0]

# Enforce symmetry (add the transpose to ensure both directions are captured)
train_adj_matrix = train_adj_matrix + train_adj_matrix.T

# Ensure that the diagonal entries are 1 (self-loops)
train_adj_matrix.fill_diagonal_(1.0)
train_adj_matrix = train_adj_matrix.clamp(max=1)
#train_adj_matrix = to_scipy_sparse_matrix(train_edge_index, num_nodes=num_nodes)

# Convert directly to a PyTorch sparse tensor
#train_adj_matrix = torch.tensor(train_adj_matrix.toarray(), dtype=torch.float32)

# Normalize adjacency for training graph
A_tilde_train = normalize_adjacency_dense_gpu(train_adj_matrix.to(torch.float32))  # Normalize
#D = torch.diag(train_adj_matrix.sum(dim=1).clamp(min=1).pow(-0.5))
#A_tilde_train = D @ train_adj_matrix @ D


# Initialize model
input_dim = X.shape[1]
hidden_dim = 128
latent_dim = 64
model = VGAE(input_dim, hidden_dim, latent_dim)

model = model.to('cuda')  # Move model to GPU if available
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Training OLD
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    Z, A_reconstructed, mu, log_sigma = model(X.to('cuda'), A_tilde_train.to('cuda'))

    # Clamp log_sigma to prevent extreme values
    log_sigma = torch.clamp(log_sigma, min=-18, max=18)

    # Compute loss
    loss = loss_function(train_adj_matrix.to('cuda'), A_reconstructed.to('cuda'), mu, log_sigma)
    loss.backward()

    # Update parameters
    optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


import numpy as np
from sklearn.metrics import roc_auc_score

# Convert the reconstructed adjacency matrix to CPU if necessary
A_reconstructed = A_reconstructed.detach().cpu()

# Ensure test edges and non-edges are tensors
test_edges = torch.tensor(test_edges, dtype=torch.long)
test_non_edges = torch.tensor(test_non_edges, dtype=torch.long)

# Handle different decoder outputs
if A_reconstructed.dim() == 2:
    # If A_reconstructed is a full adjacency matrix
    test_edge_scores = A_reconstructed[test_edges[:, 0], test_edges[:, 1]].numpy()
    test_non_edge_scores = A_reconstructed[test_non_edges[:, 0], test_non_edges[:, 1]].numpy()
else:
    # If A_reconstructed is a 1D tensor (edge probabilities only)
    test_edge_scores = A_reconstructed[:len(test_edges)].numpy()
    test_non_edge_scores = A_reconstructed[len(test_edges):].numpy()

# Combine scores and create labels
scores = np.concatenate([test_edge_scores, test_non_edge_scores])
labels = np.concatenate([np.ones(len(test_edge_scores)), np.zeros(len(test_non_edge_scores))])


from sklearn.metrics import roc_auc_score

# Assuming y_true contains actual edge labels (1 for edges, 0 for non-edges)
# and y_score contains the predicted scores for each pair of nodes
roc_auc = roc_auc_score(labels, scores)
print(f"ROC-AUC Score: {roc_auc}")


from sklearn.metrics import average_precision_score
ap_score = average_precision_score(labels, scores)
print(f"Average Precision (AP): {ap_score:.4f}")

import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    """
    GAE
    """
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, X, A):

        support = torch.matmul(X, self.weight)  # Apply linear transformation
        output = torch.matmul(A, support)  # Aggregate neighbor information
        return output + self.bias  # Add bias term

class GAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GAE, self).__init__()
        # Encoder layers
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, latent_dim)

    def forward(self, X, A_tilde):

        # Graph convolution layer 1
        H = F.relu(self.gcn1(X, A_tilde))  # First GCN layer with ReLU activation

        # Graph convolution layer 2
        Z = self.gcn2(H, A_tilde)  # Second GCN layer for embeddings Z

        # Normalize the embeddings (optional, based on your specific use case)
        Z = F.normalize(Z, p=2, dim=1)  # Normalize each row of Z to have unit length

        # Reconstruct the adjacency matrix
        A_reconstructed = torch.sigmoid(torch.matmul(Z, Z.T))  # Reconstruct the adjacency matrix

        return Z, A_reconstructed


def loss_function(A, A_reconstructed):

    A = A.view(-1)
    A_reconstructed = A_reconstructed.view(-1)

    return F.binary_cross_entropy(A_reconstructed, A)


train_adj_matrix = to_dense_adj_custom(train_edge_index, max_num_nodes=num_nodes)[0]  # Convert to dense adjacency matrix
train_adj_matrix = train_adj_matrix.to(torch.float32)  # Ensure float type for computations

# Node features
if data.x is not None:
    X = data.x  # Use provided node features
else:
    X = torch.eye(num_nodes)  # Use identity matrix if featureless


A_tilde = normalize_adjacency_dense_gpu(train_adj_matrix)

input_dim = X.shape[1]  # Number of features per node
hidden_dim = 32  # Hidden layer size
latent_dim = 16  # Latent space size (embedding dimension)
num_epochs = 200
learning_rate = 0.01
device='cuda'


model = GAE(input_dim, hidden_dim, latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    Z, A_reconstructed = model(X.to(device), train_adj_matrix.to(device))

    # Compute loss
    loss = loss_function(train_adj_matrix.to(device), A_reconstructed)

    # Backward pass
    loss.backward()

    # Update parameters
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")


# Convert the reconstructed adjacency matrix to CPU if necessary
A_reconstructed = A_reconstructed.detach().cpu()

test_edges = torch.tensor(test_edges, dtype=torch.long)
test_non_edges = torch.tensor(test_non_edges, dtype=torch.long)
# Get the scores for test edges and test non-edges
test_edge_scores = A_reconstructed[test_edges[:, 0], test_edges[:, 1]].numpy()
test_non_edge_scores = A_reconstructed[test_non_edges[:, 0], test_non_edges[:, 1]].numpy()

# Combine scores and create labels
scores = np.concatenate([test_edge_scores, test_non_edge_scores])
labels = np.concatenate([np.ones(len(test_edge_scores)), np.zeros(len(test_non_edge_scores))])

# Assuming y_true contains actual edge labels (1 for edges, 0 for non-edges)
# and y_score contains the predicted scores for each pair of nodes
roc_auc = roc_auc_score(labels, scores)
print(f"ROC-AUC Score: {roc_auc}")

ap_score = average_precision_score(labels, scores)
print(f"Average Precision (AP): {ap_score:.4f}")

