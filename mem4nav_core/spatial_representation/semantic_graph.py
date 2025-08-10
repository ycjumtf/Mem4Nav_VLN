import torch
import torch.nn as nn # Imported for type hinting Optional[torch.Tensor] if nn.Module used later
import numpy as np
import heapq
from typing import Dict, List, Tuple, Optional, Set, Any

NodeType = int

class GraphNode:
    """
    Represents a node in the Semantic Topological Graph.
    It stores its position, a descriptive embedding, an associated LTM token,
    and visit statistics.
    """
    __slots__ = ('id', 'position', 'descriptor', 
                 'ltm_token', 
                 'creation_time', 'visits')

    def __init__(self, node_id: NodeType, position: np.ndarray, 
                 initial_descriptor: torch.Tensor, timestamp: int):
        self.id: NodeType = node_id
        self.position: np.ndarray = position # [x, y, z]
        self.descriptor: torch.Tensor = initial_descriptor 

        self.ltm_token: Optional[torch.Tensor] = None
        
        self.creation_time: int = timestamp
        self.visits: int = 1 # How many times this node has been "focused" or updated

    def update_descriptor(self, new_descriptor: torch.Tensor, 
                          new_position: Optional[np.ndarray] = None) -> None:
        """
        Updates the node's descriptor by averaging with the new one.
        Optionally updates position if provided (e.g., averaging position of multiple sightings).
        The LTM token update is managed externally via `set_ltm_token`.
        """
        # The paper mentions averaging edge weights. For node descriptors, averaging upon

        if self.visits > 0: # Avoid division by zero if somehow visits is 0
            self.descriptor = (self.descriptor * self.visits + new_descriptor) / (self.visits + 1)
            if new_position is not None:
                self.position = (self.position * self.visits + new_position) / (self.visits + 1)
        else: # First effective "visit" for descriptor update
             self.descriptor = new_descriptor
             if new_position is not None:
                 self.position = new_position
        self.visits += 1
        
    def set_ltm_token(self, token: torch.Tensor) -> None:
        """
        Sets or updates the Long-Term Memory (LTM) state token for this graph node.
        The provided token is expected to be the 2*d_emb dimensional output
        from the LTM's ReversibleTransformer.
        """
        if token.ndim == 0: # Should be a vector
             print(f"Warning: Attempting to set LTM token with scalar for GraphNode {self.id}. Token shape: {token.shape}")
        self.ltm_token = token

    def get_ltm_token(self) -> Optional[torch.Tensor]:
        """Returns the current LTM state token for this graph node."""
        return self.ltm_token

    def __repr__(self) -> str:
        ltm_token_shape = self.ltm_token.shape if self.ltm_token is not None else "None"
        descriptor_shape = self.descriptor.shape if self.descriptor is not None else "None"
        return (f"GraphNode(id={self.id}, pos={self.position.round(2).tolist()}, visits={self.visits}, "
                f"desc_shape={descriptor_shape}, ltm_token_shape={ltm_token_shape})")


class SemanticGraph:
    """
    Maintains a dynamic directed graph G = (V, E) where nodes V correspond to
    landmarks or intersections, and edges E encode traversability and cost.
    """
    def __init__(self, 
                 node_creation_similarity_threshold: float = 0.5, 
                 alpha_distance_weight: float = 1.0, 
                 beta_instruction_cost_weight: float = 0.5,
                 device: Optional[torch.device] = None): # Device for tensor operations
        """
        Initializes the Semantic Topological Graph.

        Args:
            node_creation_similarity_threshold (float): Delta (δ). If min_u ||v_t - phi(u)||_2 > δ,
                                                        a new node is created. This is a distance threshold.
            alpha_distance_weight (float): Alpha (α) for edge weighting (distance component).
            beta_instruction_cost_weight (float): Beta (β) for edge weighting (instruction cost component).
            device (Optional[torch.device]): PyTorch device for descriptor tensors.
        """
        self.nodes: Dict[NodeType, GraphNode] = {}
        self.adj: Dict[NodeType, Dict[NodeType, float]] = {} # Adjacency list: source_id -> {target_id: weight}
        self.pred: Dict[NodeType, Set[NodeType]] = {}       # Predecessors: target_id -> {source_id}

        self.node_creation_similarity_threshold: float = node_creation_similarity_threshold
        self.alpha_distance_weight: float = alpha_distance_weight
        self.beta_instruction_cost_weight: float = beta_instruction_cost_weight
        
        self.device: torch.device = device or torch.device('cpu')
        self._next_node_id_counter: int = 0
        self._timestamp: int = 0 # Internal counter for node creation time, or steps

    def _generate_new_node_id(self) -> NodeType:
        """Generates a new unique ID for a graph node."""
        new_id = self._next_node_id_counter
        self._next_node_id_counter += 1
        return new_id

    def add_or_update_node(self, 
                           current_embedding: torch.Tensor, # v_t
                           current_position_np: np.ndarray  # p_t
                          ) -> GraphNode:
        """
        Decides whether to create a new semantic node or update an existing one
        based on the similarity of current_embedding to existing node descriptors.
        The node's descriptor phi(u) is its aggregated embedding.

        Args:
            current_embedding (torch.Tensor): The d_emb-dimensional embedding of the current observation (v_t).
            current_position_np (np.ndarray): The 3D [x,y,z] position of the current observation (p_t).

        Returns:
            GraphNode: The created or identified (and potentially updated) graph node.
        """
        self._timestamp += 1
        current_embedding_dev = current_embedding.to(self.device)

        min_distance = float('inf')
        closest_node: Optional[GraphNode] = None

        if not self.nodes: # First node in the graph
            new_node_id = self._generate_new_node_id()
            created_node = GraphNode(new_node_id, current_position_np, 
                                     current_embedding_dev.clone(), self._timestamp)
            self.nodes[new_node_id] = created_node
            self.adj[new_node_id] = {}
            self.pred[new_node_id] = set()
            return created_node

        # Find the closest existing node
        for node_id, node in self.nodes.items():
            # L2 norm for distance: ||v_t - phi(u)||_2
            distance = torch.linalg.norm(current_embedding_dev - node.descriptor.to(self.device)).item()
            if distance < min_distance:
                min_distance = distance
                closest_node = node
        
        # Decide based on threshold δ
        if closest_node is not None and min_distance <= self.node_creation_similarity_threshold:
            # Found a similar existing node; update its descriptor and position (by averaging)
            closest_node.update_descriptor(current_embedding_dev, new_position=current_position_np)
            return closest_node
        else:
            # No sufficiently similar node found, or graph was empty; create a new node
            new_node_id = self._generate_new_node_id()
            created_node = GraphNode(new_node_id, current_position_np, 
                                     current_embedding_dev.clone(), self._timestamp)
            self.nodes[new_node_id] = created_node
            self.adj[new_node_id] = {}
            self.pred[new_node_id] = set()
            return created_node

    def add_or_update_edge(self, 
                             source_node_id: NodeType, 
                             target_node_id: NodeType, 
                             instruction_cost: float = 0.0, # c_instr
                             force_update_weight: bool = False): # If true, recalculates weight even if edge exists
        """
        Adds or updates a directed edge from source_node to target_node.
        Edge weight w_{u_prev, u_curr} = α * ||p_prev - p_curr||_2 + β * c_instr.
        If the edge already exists, its weight is averaged (as per paper) unless force_update_weight is True.

        Args:
            source_node_id (NodeType): The ID of the starting node.
            target_node_id (NodeType): The ID of the ending node.
            instruction_cost (float): c_instr, cost associated with instruction complexity.
                                      This value is determined by higher-level modules (e.g., Planning or Agent).
            force_update_weight (bool): If True, the edge weight is set to the newly calculated
                                        value, ignoring averaging.
        """
        source_node = self.nodes.get(source_node_id)
        target_node = self.nodes.get(target_node_id)

        if not source_node or not target_node:
            # print(f"Warning: Source ({source_node_id}) or target ({target_node_id}) node not in graph for edge update.")
            return
        if source_node_id == target_node_id: # Avoid self-loops from this method
            return

        distance_euclidean = np.linalg.norm(source_node.position - target_node.position)
        new_weight = (self.alpha_distance_weight * distance_euclidean + 
                      self.beta_instruction_cost_weight * instruction_cost)

        if new_weight < 0: # Ensure non-negative weights for Dijkstra
            # print(f"Warning: Calculated negative edge weight ({new_weight:.2f}) for {source_node_id}->{target_node_id}. Clamping to 0.")
            new_weight = 0.0

        current_edges_from_source = self.adj.setdefault(source_node_id, {})
        
        if target_node_id in current_edges_from_source and not force_update_weight:
            # Edge exists, average weight as per paper ("averaged to smooth out noise")
            existing_weight = current_edges_from_source[target_node_id]
            # A simple running average. Could be refined if edge traversals are counted.
            current_edges_from_source[target_node_id] = (existing_weight + new_weight) / 2.0
        else:
            # New edge or forced update
            current_edges_from_source[target_node_id] = new_weight
        
        self.pred.setdefault(target_node_id, set()).add(source_node_id)

    def get_node(self, node_id: NodeType) -> Optional[GraphNode]:
        """Retrieves a graph node by its ID."""
        return self.nodes.get(node_id)

    def shortest_path(self, start_node_id: NodeType, goal_node_id: NodeType) -> Tuple[List[NodeType], float]:
        """
        Computes the shortest path from start_node_id to goal_node_id using Dijkstra's algorithm.

        Returns:
            Tuple[List[NodeType], float]: A list of node IDs forming the shortest path
                                          (including start and end), and the total cost.
                                          Returns empty list and float('inf') if no path exists.
        """
        if start_node_id not in self.nodes or goal_node_id not in self.nodes:
            return [], float('inf')
        if start_node_id == goal_node_id:
            return [start_node_id], 0.0

        pq: List[Tuple[float, NodeType]] = [(0.0, start_node_id)] # (cost, node_id)
        distances: Dict[NodeType, float] = {node: float('inf') for node in self.nodes}
        predecessors: Dict[NodeType, Optional[NodeType]] = {node: None for node in self.nodes}
        distances[start_node_id] = 0.0

        while pq:
            current_dist, current_node = heapq.heappop(pq)

            if current_dist > distances[current_node]:
                continue
            if current_node == goal_node_id: # Path found
                break 

            for neighbor_node, weight in self.adj.get(current_node, {}).items():
                distance_through_current = current_dist + weight
                if distance_through_current < distances[neighbor_node]:
                    distances[neighbor_node] = distance_through_current
                    predecessors[neighbor_node] = current_node
                    heapq.heappush(pq, (distance_through_current, neighbor_node))
        
        path: List[NodeType] = []
        if distances[goal_node_id] == float('inf'): # Goal not reachable
            return [], float('inf')

        # Reconstruct path
        crawl_node = goal_node_id
        while crawl_node is not None:
            path.append(crawl_node)
            crawl_node = predecessors[crawl_node]
        
        return path[::-1], distances[goal_node_id]

    def clear(self) -> None:
        """Clears all nodes and edges from the graph."""
        self.nodes.clear()
        self.adj.clear()
        self.pred.clear()
        self._next_node_id_counter = 0
        self._timestamp = 0

    def __len__(self) -> int:
        """Returns the number of nodes in the graph."""
        return len(self.nodes)

if __name__ == '__main__':
    # Example Usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = SemanticGraph(node_creation_similarity_threshold=0.7, device=device)
    
    emb_dim = 64 # Should match LTM's d_emb if these are v_t
    pos_a = np.array([0.0, 0.0, 0.0])
    emb_a = torch.rand(emb_dim, device=device)
    
    pos_b = np.array([1.0, 0.0, 0.0])
    emb_b = torch.rand(emb_dim, device=device)
    
    pos_c = np.array([1.0, 1.0, 0.0])
    emb_c = torch.rand(emb_dim, device=device)

    # Add nodes
    node_a_obj = graph.add_or_update_node(emb_a, pos_a)
    node_b_obj = graph.add_or_update_node(emb_b, pos_b)
    node_c_obj = graph.add_or_update_node(emb_c, pos_c)

    print(f"Graph after adding A, B, C: {len(graph)} nodes")
    for node_id, node_obj in graph.nodes.items():
        print(f"  {node_obj}")

    # Simulate LTM token update for node_a_obj
    # LTM token dim is 2 * d_emb (e.g., 2 * 128 if LTM internal d_emb is 128)
    # For testing, use dummy LTM token dim.
    dummy_ltm_token_dim = 128 
    mock_ltm_token_for_node_a = torch.randn(dummy_ltm_token_dim, device=device)
    node_a_obj.set_ltm_token(mock_ltm_token_for_node_a)
    print(f"  Set LTM token for Node A. Shape: {node_a_obj.get_ltm_token().shape}") # type: ignore
    assert node_a_obj.ltm_token is not None

    # Re-observe something similar to node A at a new position
    pos_a_prime = np.array([0.1, 0.1, 0.0])
    emb_a_prime_similar = emb_a + torch.rand(emb_dim, device=device) * 0.1 # Similar to A
    node_a_updated_obj = graph.add_or_update_node(emb_a_prime_similar, pos_a_prime)
    
    print(f"After observing similar to A: Node ID {node_a_updated_obj.id}, Visits {node_a_updated_obj.visits}")
    assert node_a_updated_obj.id == node_a_obj.id # Should update existing node A
    assert node_a_updated_obj.visits > 1
    # LTM token on node_a_obj would still be the old one, until explicitly set again
    # by the MappingModule after an LTM.write operation for node_a_obj.id.
    assert torch.equal(node_a_obj.get_ltm_token(), mock_ltm_token_for_node_a) # type: ignore

    # Add edges (caller, e.g. MappingModule, provides instruction_cost)
    graph.add_or_update_edge(node_a_obj.id, node_b_obj.id, instruction_cost=0.1)
    graph.add_or_update_edge(node_b_obj.id, node_c_obj.id, instruction_cost=0.5)
    graph.add_or_update_edge(node_c_obj.id, node_a_obj.id, instruction_cost=0.0) # Cycle back

    print("\nGraph Adjacency List (Source -> Target: Weight):")
    for src_id, targets in graph.adj.items():
        for target_id, weight in targets.items():
            print(f"  {src_id} -> {target_id}: {weight:.2f}")

    # Test shortest path
    path_a_to_c, cost_a_to_c = graph.shortest_path(node_a_obj.id, node_c_obj.id)
    print(f"\nShortest path from A ({node_a_obj.id}) to C ({node_c_obj.id}):")
    if path_a_to_c:
        print(f"  Path: {' -> '.join(map(str, path_a_to_c))}, Cost: {cost_a_to_c:.2f}")
        expected_path_ids = [node_a_obj.id, node_b_obj.id, node_c_obj.id]
        assert path_a_to_c == expected_path_ids
    else:
        print("  No path found.")

    graph.clear()
    print(f"\nGraph cleared. Number of nodes: {len(graph)}")
    assert len(graph) == 0
    
    print("\nSemanticGraph with modified GraphNode tests completed.")