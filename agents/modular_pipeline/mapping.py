import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

try:
    from mem4nav_core.spatial_representation.sparse_octree import SparseOctree, OctreeLeaf
    from mem4nav_core.spatial_representation.semantic_graph import SemanticGraph, GraphNode
    from mem4nav_core.memory_system.memory_retrieval import MemoryRetrieval, RetrievalResultItem
except ImportError:
    print("Warning: MappingModule using placeholder for core Mem4Nav components.")
    class OctreeLeaf:   
        def __init__(self, morton_code, **kwargs): self.morton_code = morton_code; self.ltm_token = None
        def set_ltm_token(self, token): self.ltm_token = token # Simplified
    class SparseOctree:   
        def __init__(self, *args, **kwargs): self.leaves = {}
        def _get_morton_code(self, pos_np: np.ndarray) -> int: return hash(pos_np.tobytes()) # Mock
        def insert_or_get_leaf(self, pos_np: np.ndarray, emb: torch.Tensor) -> OctreeLeaf:
            key = self._get_morton_code(pos_np)
            if key not in self.leaves: self.leaves[key] = OctreeLeaf(morton_code=key)
            return self.leaves[key]
        def get_leaf_by_code(self, code: int) -> Optional[OctreeLeaf]: return self.leaves.get(code)
        def clear(self): self.leaves.clear()

    class GraphNode:   
        def __init__(self, node_id, position, **kwargs): self.id = node_id; self.position = position; self.ltm_token = None
        def set_ltm_token(self, token): self.ltm_token = token # Simplified
    class SemanticGraph:   
        def __init__(self, *args, **kwargs): self.nodes = {}; self._next_node_id = 0
        def add_or_update_node(self, emb: torch.Tensor, pos_np: np.ndarray) -> GraphNode:
            # Simplified: always creates a new node for testing this module in isolation
            node_id = self._next_node_id; self._next_node_id +=1
            node = GraphNode(node_id, pos_np)
            self.nodes[node_id] = node
            return node
        def add_or_update_edge(self, src_node: GraphNode, tgt_node: GraphNode, cost:float=0.0): pass
        def get_node(self, node_id: int) -> Optional[GraphNode]: return self.nodes.get(node_id)
        def shortest_path(self, start_id: int, goal_id: int) -> Tuple[List[int], float]: return [], float('inf')
        def clear(self): self.nodes.clear(); self._next_node_id = 0

    class MemoryRetrieval(nn.Module):   
        def __init__(self, *args, **kwargs): super().__init__(); self.ltm = LongTermMemoryPlaceholder(); self.stm = ShortTermMemoryPlaceholder()
        def write_observation(self, unique_observation_key, object_id_for_stm, current_absolute_position, current_observation_embedding, current_semantic_node_position) -> torch.Tensor: return torch.rand(256) # Mock 2d_emb token
        def retrieve_memory(self, current_observation_embedding, current_absolute_position, current_semantic_node_position) -> Tuple[str, List[Any]]: return "LTM", []
        def clear_all_memory(self): self.ltm.clear(); self.stm.clear()
        def get_ltm_module(self): return self.ltm
    class LongTermMemoryPlaceholder:   
        def get_current_ltm_read_token_for_key(self, key, default_if_new=True): return torch.zeros(128) # Mock d_emb token
        def clear(self):pass
    class ShortTermMemoryPlaceholder:   
        def clear(self):pass
    RetrievalResultItem = Any


class MappingModule(nn.Module):
    """
    Manages map updates (Octree, Semantic Graph) and memory interactions (LTM, STM)
    for the Modular Pipeline Agent.
    """
    def __init__(self, 
                 config: Dict[str, Any], 
                 octree: SparseOctree, 
                 semantic_graph: SemanticGraph, 
                 memory_retriever: MemoryRetrieval,
                 device: torch.device):
        super().__init__()
        self.config = config.get('mapping', {}) # Specific config for this module
        self.octree = octree
        self.semantic_graph = semantic_graph
        self.memory_retriever = memory_retriever
        self.device = device

        self.current_semantic_node: Optional[GraphNode] = None
        self.previous_semantic_node: Optional[GraphNode] = None

        # Parameters for edge update, e.g., from config if instruction cost is determined here
        self.default_instruction_cost_for_edge = self.config.get('default_instruction_cost', 0.1)

    def _get_octree_key_from_pos(self, pos_tensor: torch.Tensor) -> int:
        """Helper to get Morton code from a position tensor."""
        # Assuming octree._get_morton_code is accessible or SparseOctree provides a public method
        return self.octree._get_morton_code(pos_tensor.cpu().numpy())

    def update_current_semantic_context(self, 
                                        current_absolute_pos_np: np.ndarray, 
                                        fused_embedding: torch.Tensor) -> GraphNode:
        """
        Updates or determines the current semantic graph node (u_c) based on position and embedding.
        Also updates graph edges if transitioning between semantic nodes.
        """
        # Determine/Update current semantic graph node u_c
        # This logic could be more sophisticated, e.g., checking if agent is still "at" previous node.
        # For now, we call add_or_update_node which handles similarity checks.
        new_current_semantic_node = self.semantic_graph.add_or_update_node(
            fused_embedding, current_absolute_pos_np
        )

        if self.current_semantic_node is None: # First semantic node encountered
            self.current_semantic_node = new_current_semantic_node
        elif self.current_semantic_node.id != new_current_semantic_node.id:
            # Agent has transitioned to a new semantic node
            self.previous_semantic_node = self.current_semantic_node
            self.current_semantic_node = new_current_semantic_node

            self.semantic_graph.add_or_update_edge(
                self.previous_semantic_node, 
                self.current_semantic_node,
                instruction_cost=self.default_instruction_cost_for_edge # Placeholder for instruction cost
            )
        else: 
            self.current_semantic_node = new_current_semantic_node 

        return self.current_semantic_node


    def update_maps_and_memory(self,
                               current_absolute_pos: torch.Tensor, # p_t (tensor)
                               fused_embedding: torch.Tensor,    # v_t (tensor, d_emb)
                               observation_details: Dict[str, Any] 
                              ):
        """
        Primary method to update all spatial and memory representations with a new observation.
        This is called at each agent step.
        """
        current_absolute_pos_np = current_absolute_pos.cpu().numpy()
        current_uc = self.update_current_semantic_context(current_absolute_pos_np, fused_embedding)
        p_u_c_tensor = torch.tensor(current_uc.position, dtype=torch.float32, device=self.device)


        octree_leaf_key = self._get_octree_key_from_pos(current_absolute_pos)
        octree_leaf_obj = self.octree.insert_or_get_leaf(current_absolute_pos_np, fused_embedding)
        
        # LTM write for the octree leaf
        new_ltm_token_for_octree_leaf = self.memory_retriever.write_observation(
            unique_observation_key=octree_leaf_key, # Integer key
            object_id_for_stm=observation_details.get('object_id_for_stm', 'unknown_object'),
            current_absolute_position=current_absolute_pos,
            current_observation_embedding=fused_embedding,
            current_semantic_node_position=p_u_c_tensor # Context for STM part of this write
        )

        if hasattr(octree_leaf_obj, 'set_ltm_token'):
             octree_leaf_obj.set_ltm_token(new_ltm_token_for_octree_leaf.detach().clone()) #type: ignore
        elif hasattr(octree_leaf_obj, 'ltm_token'):
             octree_leaf_obj.ltm_token = new_ltm_token_for_octree_leaf.detach().clone() #type: ignore
        else:
            print(f"Warning: OctreeLeaf for key {octree_leaf_key} does not have a method/attribute to store LTM token.")



        if self.config.get('enable_graph_node_ltm', True): # Make this configurable
            # The object_id_for_stm here is less direct for a graph node, use a generic one.
            # The current_semantic_node_position for STM write is still p_u_c.
            new_ltm_token_for_graph_node = self.memory_retriever.write_observation(
                unique_observation_key=current_uc.id, # Integer key (graph node ID)
                object_id_for_stm=f"semantic_node_{current_uc.id}",
                current_absolute_position=current_absolute_pos, # Observation occurs AT p_t
                current_observation_embedding=fused_embedding,   # Use same v_t
                current_semantic_node_position=p_u_c_tensor
            )

            if hasattr(current_uc, 'set_ltm_token'):
                current_uc.set_ltm_token(new_ltm_token_for_graph_node.detach().clone()) #type: ignore
            elif hasattr(current_uc, 'ltm_token'):
                current_uc.ltm_token = new_ltm_token_for_graph_node.detach().clone() #type: ignore
            else:
                print(f"Warning: GraphNode {current_uc.id} does not have method/attribute for LTM token.")
                
    def retrieve_memories_for_policy(self,
                                     current_absolute_pos: torch.Tensor, # p_t
                                     fused_embedding: torch.Tensor     # v_t
                                    ) -> Tuple[str, List[RetrievalResultItem]]:#type: ignore
        """Retrieves memories from STM/LTM for policy decision."""
        if self.current_semantic_node is None:

            print("Warning: `current_semantic_node` is None during memory retrieval. Using p_t as p_u_c.")
            p_u_c_tensor = current_absolute_pos
        else:
            p_u_c_tensor = torch.tensor(self.current_semantic_node.position, dtype=torch.float32, device=self.device)

        source, data = self.memory_retriever.retrieve_memory(
            current_observation_embedding=fused_embedding,
            current_absolute_position=current_absolute_pos,
            current_semantic_node_position=p_u_c_tensor
        )
        return source, data

    def get_semantic_graph(self) -> SemanticGraph:
        """Provides access to the semantic graph for the PlanningModule."""
        return self.semantic_graph
        
    def get_octree(self) -> SparseOctree:
        """Provides access to the sparse octree."""
        return self.octree

    def reset_state(self):
        """Resets mapping-specific state for a new episode."""
        self.current_semantic_node = None
        self.previous_semantic_node = None
        # Octree, SemanticGraph, MemoryRetriever are cleared by the agent usually.
        # If this module needs to clear them independently:
        # self.octree.clear()
        # self.semantic_graph.clear()
        # self.memory_retriever.clear_all_memory()
        print("MappingModule state reset.")


if __name__ == '__main__':
    print("--- Conceptual Test for MappingModule ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Mock configurations and components
    mock_mem_config = {
        'world_size': 50.0, 'octree_depth': 10, 'embedding_dim': 128, # d_emb
        'lt_depth': 2, 'lt_heads': 2, 'lt_max_elements': 100,
        'st_capacity': 20, 'eviction_lambda': 0.5
    }
    mock_sg_config = {'node_creation_similarity_threshold': 0.8}
    mock_mapping_config = {'mapping': {'default_instruction_cost': 0.1, 'enable_graph_node_ltm': True}}

    octree_inst = SparseOctree(
        world_size=mock_mem_config['world_size'],
        max_depth=mock_mem_config['octree_depth'],
        embedding_dim=mock_mem_config['embedding_dim'],
        device=device
    )
    sg_inst = SemanticGraph(
        node_creation_similarity_threshold=mock_sg_config['node_creation_similarity_threshold'],
        device=device
    )
    mem_ret_inst = MemoryRetrieval(
        ltm_embedding_dim=mock_mem_config['embedding_dim'],
        ltm_transformer_depth=mock_mem_config['lt_depth'],
        ltm_transformer_heads=mock_mem_config['lt_heads'],
        ltm_max_elements=mock_mem_config['lt_max_elements'],
        stm_capacity=mock_mem_config['st_capacity'],
        stm_eviction_lambda=mock_mem_config['eviction_lambda'],
        device=device
    )

    mapping_module = MappingModule(mock_mapping_config, octree_inst, sg_inst, mem_ret_inst, device)

    # Simulate a few steps
    for i in range(3):
        print(f"\n--- Mapping Step {i} ---")
        # Mock current agent state and observation
        p_t = torch.tensor([i * 1.0, i * 0.5, 0.0], dtype=torch.float32, device=device)
        v_t = torch.randn(mock_mem_config['embedding_dim'], device=device)
        obs_details = {'object_id_for_stm': f'object_{i}'}

        mapping_module.update_maps_and_memory(p_t, v_t, obs_details)

        if mapping_module.current_semantic_node:
            print(f"  Current semantic node: ID {mapping_module.current_semantic_node.id}, Pos {mapping_module.current_semantic_node.position.round(2)}")
            # Check if LTM token was set (mocked set_ltm_token would need to exist on GraphNode)
            # print(f"    LTM token (GraphNode): {mapping_module.current_semantic_node.ltm_token is not None}")
        
        octree_key_for_pt = mapping_module._get_octree_key_from_pos(p_t)
        leaf = octree_inst.get_leaf_by_code(octree_key_for_pt)
        if leaf:
            print(f"  Octree leaf {octree_key_for_pt} exists.")
            # Check if LTM token was set
            # print(f"    LTM token (OctreeLeaf): {leaf.ltm_token is not None}")


        # Simulate memory retrieval
        source, data = mapping_module.retrieve_memories_for_policy(p_t, v_t)
        print(f"  Retrieved from {source} memory: {len(data)} items.")

    mapping_module.reset_state()
    print("\nMappingModule conceptual test finished.")