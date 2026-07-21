import torch.nn as nn
import torch
from torch_geometric.utils import scatter

# Will need a few classes here: RBF expansion of distances between connected nodes; GNN message-passing block updating node, edge, and global attributes,
# then the full model which predicts the HOMO-LUMO gap from the final global feature vector. Also have a convenience class MLP for update functions.

class GaussianRBF(nn.Module):
    def __init__(self,
                 min_centre: float,
                 max_centre: float,
                 num_centres: float,
                 sigma_scale: float,
    ):
        super().__init__()
        centres = torch.linspace(min_centre,max_centre,num_centres)
        spacing = centres[1] - centres[0]
        sigma = sigma_scale * spacing
        self.register_buffer("centres",centres)
        self.register_buffer("sigma",sigma)
    
    def forward(self, distances: torch.Tensor):
        return torch.exp(-0.5*((distances.unsqueeze(-1) - self.centres)/self.sigma)**2)
    
class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim,output_dim)
        )    
    
    def forward(self, x: torch.Tensor):
        return self.layers(x)

class GNNBlock(nn.Module):
    def __init__(self,
            node_dim: int,
            edge_dim: int,
            global_dim: int,
            mlp_hidden_dim: int
    ):
        super().__init__()
        edge_input_dim = 2 * node_dim + edge_dim + global_dim
        node_input_dim = node_dim + edge_dim + global_dim
        global_input_dim = node_dim + edge_dim + global_dim
        self.edge_update = MLP(edge_input_dim,mlp_hidden_dim,edge_dim)
        self.node_update = MLP(node_input_dim,mlp_hidden_dim,node_dim)
        self.global_update = MLP(global_input_dim,mlp_hidden_dim,global_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.node_norm = nn.LayerNorm(node_dim)
        self.global_norm = nn.LayerNorm(global_dim)
    
    def forward(
            self,
            node_state: torch.Tensor,
            edge_state: torch.Tensor,
            global_state: torch.Tensor,
            edge_index: torch.Tensor,
            node_batch: torch.Tensor
    ):
        source, target = edge_index
        num_nodes = node_state.size(0)
        num_graphs = global_state.size(0)
        edge_batch = node_batch[source]

        # Edge update $\vec{e}^{'}_k = \phi^e\left( \vec{e}_k, \vec{v}_{rk}, \vec{v}_{sk}, \vec{u}\right)$ where \phi is self.edge_update() MLP with two layers c.f. MEGNet has 3
        edge_inputs = torch.cat(
            [
                edge_state,
                node_state[source],
                node_state[target],
                global_state[edge_batch]
            ],
            dim=-1
        )
        edge_update = self.edge_update(edge_inputs)
        edge_state = self.edge_norm(edge_state + edge_update)
        # Aggregate edge updates $\bar{e}_i^{'}=\rho^{e\to v}(E_{i}^{'})$ with sum for \rho as standard e.g in CGCNN. Intensive/extensive doesn't matter here, MEGNet uses mean, would test both with more time.
        incoming_edges = scatter(
            edge_state,
            target,
            dim=0,
            dim_size=num_nodes,
            reduce='sum'
        )
        # Update nodes $\vec{v}^{'}_i = \phi^v\left(\bar{e}_{i}^{'},\vec{v}_i,\vec{u}\right)$ where $\bar{e}$ is the incoming_edges
        node_inputs = torch.cat(
            [
                node_state,
                incoming_edges,
                global_state[node_batch]
            ],
            dim=-1,
        )
        node_update = self.node_update(node_inputs)
        node_state = self.node_norm(node_state + node_update)
        
        # Aggregate all edge updates $\bar{e}^{'}=\rho^{e\to u}(E^{'})$ using mean pooling for intensive property
        pooled_edges = scatter(
            edge_state,
            edge_batch,
            dim=0,
            dim_size = num_graphs,
            reduce='mean'
        )

        # Aggregate all node updates $\bar{v}^{'}=\rho^{v\to u}(V^{'})$ using mean pooling for intensive property
        pooled_nodes = scatter(
            node_state,
            node_batch,
            dim=0,
            dim_size = num_graphs,
            reduce='mean'
        )

        # Update global (graph-level) attributes $\vec{u}^{'}=\phi^{u}(\bar{e}^{'},\bar{v}^{'},\vec{u})$  
        global_inputs = torch.cat([
            global_state,
            pooled_nodes,
            pooled_edges
        ],
        dim=-1,
        )
        global_update = self.global_update(global_inputs)
        global_state = self.global_norm(global_state + global_update)

        return node_state, edge_state, global_state


class GapPredictor(nn.Module):
    def __init__(self,
                 node_feat_indices: list[int],
                 edge_feat_indices: list[int],
                 node_dim: int,
                 edge_dim: int,
                 global_dim: int,
                 block_hidden_dim: int,
                 num_blocks: int,
                 rbf_min: float,
                 rbf_max: float,
                 num_rbf: int,
                 sigma_scale: float,
                 head_hidden_dim: int,
                 node_mean: torch.Tensor | None = None,
                 node_std: torch.Tensor | None = None
                ):
        super().__init__()
        self.register_buffer(
            "node_feat_indices",
            torch.tensor(node_feat_indices, dtype=torch.long),
        )
        self.register_buffer(
            "edge_feat_indices",
            torch.tensor(edge_feat_indices, dtype=torch.long),
        )
        if node_mean is not None and node_std is not None:
            self.register_buffer("node_mean",node_mean)
            self.register_buffer("node_std",node_std)
        else:
            self.node_mean = None
            self.node_std = None
        
        self.rbf = GaussianRBF(
            min_centre = rbf_min,
            max_centre = rbf_max,
            num_centres = num_rbf,
            sigma_scale = sigma_scale
        )
        node_input_dim = len(node_feat_indices)
        edge_input_dim = len(edge_feat_indices) + num_rbf

        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, node_dim),
            nn.SiLU(),
            nn.LayerNorm(node_dim)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, edge_dim),
            nn.SiLU(),
            nn.LayerNorm(edge_dim)
        )
        self.initial_global = nn.Parameter(torch.zeros(1,global_dim)) # parameter so that it's updated
        self.blocks = nn.ModuleList(
            [
                GNNBlock(
                    node_dim = node_dim,
                    edge_dim = edge_dim,
                    global_dim = global_dim,
                    mlp_hidden_dim = block_hidden_dim
                )
                for _ in range(num_blocks)
            ]
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(global_dim, head_hidden_dim),
            nn.SiLU(),
            nn.Linear(head_hidden_dim, head_hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(head_hidden_dim //2, 1)
        )

    def forward(self, data):
        x = data.x[:, self.node_feat_indices]
        if self.node_mean is not None:
            x = (x - self.node_mean) / self.node_std
        node_state = self.node_encoder(x)
        source,target = data.edge_index
        distances = torch.linalg.vector_norm(data.pos[source]-data.pos[target], dim = -1)
        distance_features = self.rbf(distances)
        edge_features = data.edge_attr[:, self.edge_feat_indices]
        edge_inputs = torch.cat(
            [edge_features, distance_features],
            dim=-1
        )
        edge_state = self.edge_encoder(edge_inputs)
        global_state = self.initial_global.expand(data.num_graphs, -1)

        for block in self.blocks:
            node_state, edge_state, global_state = block(
                node_state=node_state,
                edge_state=edge_state,
                global_state=global_state,
                edge_index=data.edge_index,
                node_batch=data.batch,
            )

        prediction = self.prediction_head(global_state)

        return prediction.squeeze(-1)
