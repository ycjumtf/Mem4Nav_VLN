# type: ignore  
import torch
import numpy as np
from typing import Any, Optional, Tuple, List, Dict, Union

# Assuming LongTermMemory and ShortTermMemory classes are accessible
# from .long_term_memory import LongTermMemory
# from .short_term_memory import ShortTermMemory, ShortTermMemoryEntry
# For standalone execution, ensure these are defined or imported correctly.
# We will use placeholder definitions if not run in the full project context for now.
try:
    from mem4nav_core.memory_system.long_term_memory import LongTermMemory
    from mem4nav_core.memory_system.short_term_memory import ShortTermMemory, ShortTermMemoryEntry
except ImportError:
    # Placeholder classes for standalone development/testing of this file's logic
    # In the full project, these will be the actual LTM and STM classes.
    class LongTermMemory(torch.nn.Module):   
        def __init__(self, *args, **kwargs): super().__init__(); print("Warning: Using Placeholder LTM")
        def write(self, key: Any, previous_read_token: torch.Tensor, observation_embedding: torch.Tensor) -> torch.Tensor: return torch.empty(0)
        def retrieve(self, obs_emb: torch.Tensor, pos: torch.Tensor, k: Optional[int]=None) -> List[Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]]: return []
        def get_current_ltm_read_token_for_key(self, key: Any, default_if_new: bool = True) -> torch.Tensor: return torch.empty(0)
        def clear(self): pass

    class ShortTermMemoryEntry:   
        def __init__(self, embedding: torch.Tensor, **kwargs): self.embedding = embedding; self.key = kwargs.get('key')
    
    class ShortTermMemory:   
        def __init__(self, *args, **kwargs): self.entries: List[ShortTermMemoryEntry] = []; print("Warning: Using Placeholder STM")
        def set_current_step(self, step: int): pass
        def insert(self, key: Any, obj_id: Any, rel_pos: np.ndarray, emb: torch.Tensor): self.entries.append(ShortTermMemoryEntry(key=key, embedding=emb))
        def retrieve(self, q_emb: torch.Tensor, q_rel_pos: np.ndarray, radius: float, top_k: int) -> List[ShortTermMemoryEntry]: return []
        def clear(self): self.entries.clear()
        def __len__(self): return len(self.entries)


RetrievalResultItem = Union[ShortTermMemoryEntry, Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]]

class MemoryRetrieval(nn.Module):
    """
    Orchestrates memory retrieval from Short-Term Memory (STM) and Long-Term Memory (LTM).
    Implements the multi-level retrieval strategy described in Mem4Nav. [cite: 87]
    Also provides a unified interface for writing observations to both memory systems.
    """
    def __init__(self,
                 # LTM Configuration (passed to LTM constructor)
                 ltm_embedding_dim: int,
                 ltm_transformer_depth: int,
                 ltm_transformer_heads: int,
                 ltm_max_elements: int = 10000,
                 ltm_hnsw_ef_search: int = 50, # Default HNSW ef for search
                 ltm_default_retrieval_k: int = 3, # m in paper for LTM top-m
                 # STM Configuration (passed to STM constructor)
                 stm_capacity: int = 128, # K in paper
                 stm_eviction_lambda: float = 0.5, # λ for STM eviction
                 # MemoryRetrieval specific parameters
                 stm_retrieval_spatial_radius: float = 3.0, # ε for STM spatial filter [cite: 85]
                 stm_retrieval_k: int = 3, # k for STM top-k (can differ from LTM's k)
                 stm_similarity_threshold: float = 0.7, # τ_stm_threshold (paper uses τ) [cite: 87, 172]
                 device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device('cpu')

        self.long_term_memory = LongTermMemory(
            embedding_dim=ltm_embedding_dim,
            transformer_depth=ltm_transformer_depth,
            transformer_heads=ltm_transformer_heads,
            max_elements=ltm_max_elements,
            hnsw_ef_search=ltm_hnsw_ef_search,
            default_retrieval_k=ltm_default_retrieval_k,
            device=self.device
            # Other LTM params (mlp_ratio, dropout, projector/decoder hidden_dims) can be added here from config
        )

        self.short_term_memory = ShortTermMemory(
            capacity=stm_capacity,
            eviction_lambda=stm_eviction_lambda,
            device=self.device
        )

        self.stm_retrieval_spatial_radius = stm_retrieval_spatial_radius
        self.stm_retrieval_k = stm_retrieval_k
        self.ltm_retrieval_k = ltm_default_retrieval_k # Use LTM's default k for its retrieval
        self.stm_similarity_threshold = stm_similarity_threshold

    def update_current_step(self, step: int):
        """Updates the current step/time for STM's recency calculations."""
        self.short_term_memory.set_current_step(step)

    def write_observation(self,
                          # Common identifiers
                          unique_observation_key: Any, # Integer key (e.g., Morton code) for LTM, also used for STM if applicable
                          object_id_for_stm: Any, # Semantic identifier for STM entry (e.g., "car")
                          # Observation data
                          current_absolute_position: torch.Tensor, # p_t (e.g., [x,y,z])
                          current_observation_embedding: torch.Tensor, # v_t (d_emb dimensional)
                          # Context for STM's relative position
                          current_semantic_node_position: torch.Tensor # p_u_c (position of current graph node)
                         ):
        """
        Writes a new observation to both STM and LTM.

        Args:
            unique_observation_key (Any): An integer key, typically Morton code of the octree leaf,
                                          used for LTM and can be used for STM's entry key.
            object_id_for_stm (Any): A semantic label for the observation, used by STM.
            current_absolute_position (torch.Tensor): Agent's current global position p_t.
            current_observation_embedding (torch.Tensor): Embedding v_t of the current observation.
            current_semantic_node_position (torch.Tensor): Global position p_u_c of the semantic graph
                                                           node to which STM is currently attached.
        """
        if not isinstance(unique_observation_key, int):
            # LTM typically uses integer keys for HNSW labels.
            # If not, LTM write needs to handle mapping or raise error.
            # For now, we assume it's compatible or LTM handles it.
            pass

        # --- Write to LTM ---
        # 1. Get the d_emb "previous_read_token_for_R_input" for this key from LTM
        ltm_input_read_token = self.long_term_memory.get_current_ltm_read_token_for_key(unique_observation_key)
        
        # 2. Perform the LTM write, which returns the new 2*d_emb token that should be stored
        #    by the spatial element (OctreeLeaf/GraphNode) associated with unique_observation_key.
        #    The calling code (e.g., MappingModule) will be responsible for updating that spatial element's token.
        newly_written_ltm_token = self.long_term_memory.write(
            unique_observation_key,
            ltm_input_read_token,
            current_observation_embedding
        )
        # The `newly_written_ltm_token` should be stored by the OctreeLeaf/GraphNode identified by `unique_observation_key`.

        # --- Write to STM ---
        # 1. Calculate relative position for STM: p_rel = p_t - p_u_c
        relative_position_for_stm = (current_absolute_position - current_semantic_node_position).cpu().numpy()
        
        # 2. Insert into STM
        self.short_term_memory.insert(
            key=unique_observation_key, # Can use the same key if it makes sense for STM's uniqueness
            object_id=object_id_for_stm,
            relative_position=relative_position_for_stm,
            embedding=current_observation_embedding
        )
        return newly_written_ltm_token # Return this so caller can update spatial element

    def retrieve_memory(self,
                        current_observation_embedding: torch.Tensor, # v_t
                        current_absolute_position: torch.Tensor,     # p_t
                        current_semantic_node_position: torch.Tensor # p_u_c for STM's relative query
                       ) -> Tuple[str, List[RetrievalResultItem]]:
        """
        Performs multi-level memory retrieval. [cite: 87]
        First attempts STM lookup. If successful (similarity >= threshold), returns STM results.
        Otherwise, falls back to LTM retrieval.

        Args:
            current_observation_embedding (torch.Tensor): v_t for querying.
            current_absolute_position (torch.Tensor): p_t for LTM query and STM relative calculation.
            current_semantic_node_position (torch.Tensor): p_u_c for STM relative query.

        Returns:
            Tuple[str, List[RetrievalResultItem]]:
                A tuple where the first element is a string indicating memory source ('STM' or 'LTM'),
                and the second element is a list of retrieved items.
                For STM: List of ShortTermMemoryEntry objects.
                For LTM: List of Tuples (key, recon_v, recon_p, recon_d).
        """
        # 1. Attempt STM Lookup [cite: 85]
        query_relative_position_for_stm = (current_absolute_position - current_semantic_node_position).cpu().numpy()
        
        stm_retrieved_entries: List[ShortTermMemoryEntry] = self.short_term_memory.retrieve(
            query_embedding=current_observation_embedding,
            query_relative_position=query_relative_position_for_stm,
            spatial_radius_epsilon=self.stm_retrieval_spatial_radius,
            top_k=self.stm_retrieval_k
        )

        if stm_retrieved_entries:
            # Check if highest similarity exceeds threshold
            stm_embeddings = torch.stack([entry.embedding for entry in stm_retrieved_entries])
            query_emb_2d = current_observation_embedding.unsqueeze(0) if current_observation_embedding.ndim == 1 else current_observation_embedding
            
            similarities = torch.nn.functional.cosine_similarity(query_emb_2d, stm_embeddings, dim=1)
            max_similarity = torch.max(similarities).item()

            if max_similarity >= self.stm_similarity_threshold: # [cite: 87]
                # Use STM results
                # The agent will aggregate these into m_STM if needed.
                return "STM", stm_retrieved_entries    

        # 2. Fallback to LTM Retrieval [cite: 87]
        # LTM retrieval needs v_t and p_t (absolute)
        ltm_retrieved_items: List[Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]] = self.long_term_memory.retrieve(
            current_observation_embedding=current_observation_embedding,
            current_position=current_absolute_position, # LTM's Proj uses absolute position
            k=self.ltm_retrieval_k
        )
        # The agent will aggregate these into m_LTM if needed.
        return "LTM", ltm_retrieved_items   

    def clear_all_memory(self):
        """Clears both STM and LTM."""
        self.short_term_memory.clear()
        self.long_term_memory.clear()

    def get_ltm_module(self) -> LongTermMemory:
        return self.long_term_memory
    
    def get_stm_module(self) -> ShortTermMemory:
        return self.short_term_memory

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configs (example)
    emb_dim = 64
    ltm_params = {
        'ltm_embedding_dim': emb_dim, 'ltm_transformer_depth': 2, 'ltm_transformer_heads': 2,
        'ltm_max_elements': 50, 'ltm_default_retrieval_k': 2
    }
    stm_params = {
        'stm_capacity': 10, 'stm_eviction_lambda': 0.5
    }
    retrieval_params = {
        'stm_retrieval_spatial_radius': 5.0, 'stm_retrieval_k': 2, 'stm_similarity_threshold': 0.5
    }

    memory_retriever = MemoryRetrieval(**ltm_params, **stm_params, **retrieval_params, device=device).to(device)

    # Simulate agent's context
    current_step = 0
    agent_abs_pos = torch.tensor([1.0, 2.0, 0.0], device=device)
    current_sem_node_abs_pos = torch.tensor([0.5, 0.5, 0.0], device=device) # p_u_c

    # --- Test Write ---
    memory_retriever.update_current_step(current_step)
    obs_key1 = 1001 # Morton code
    obj_id1 = "red_car"
    obs_emb1 = torch.randn(emb_dim, device=device)
    
    # Simulate an observation at agent_abs_pos
    print(f"Step {current_step}: Writing observation key {obs_key1}...")
    # This token is stored by the OctreeLeaf/GraphNode associated with obs_key1
    _ = memory_retriever.write_observation( 
        unique_observation_key=obs_key1,
        object_id_for_stm=obj_id1,
        current_absolute_position=agent_abs_pos,
        current_observation_embedding=obs_emb1,
        current_semantic_node_position=current_sem_node_abs_pos
    )
    print(f"  STM size: {len(memory_retriever.short_term_memory)}")
    print(f"  LTM HNSW count: {memory_retriever.long_term_memory.hnsw_index.get_current_count()}")


    # --- Simulate another observation ---
    current_step += 1
    memory_retriever.update_current_step(current_step)
    agent_abs_pos += torch.tensor([1.0, 0.0, 0.0], device=device) # Agent moved
    # Assume semantic node also changed or STM is relative to new one if agent moved far
    # current_sem_node_abs_pos = agent_abs_pos - torch.rand(3) 
    
    obs_key2 = 1002
    obj_id2 = "blue_bicycle"
    # Make obs_emb2 similar to obs_emb1 to test STM threshold
    obs_emb2 = obs_emb1 + torch.randn(emb_dim, device=device) * 0.01 
    
    print(f"\nStep {current_step}: Writing observation key {obs_key2}...")
    _ = memory_retriever.write_observation(
        unique_observation_key=obs_key2,
        object_id_for_stm=obj_id2,
        current_absolute_position=agent_abs_pos,
        current_observation_embedding=obs_emb2,
        current_semantic_node_position=current_sem_node_abs_pos # Assume same u_c for simplicity
    )
    print(f"  STM size: {len(memory_retriever.short_term_memory)}")
    print(f"  LTM HNSW count: {memory_retriever.long_term_memory.hnsw_index.get_current_count()}")

    # --- Test Retrieval ---
    print("\n--- Test Retrieval ---")
    memory_retriever.update_current_step(current_step) # Update current step for STM recency if relevant
    
    # Query with an embedding similar to obs_emb2 (which is similar to obs_emb1)
    query_v_t = obs_emb2 + torch.randn(emb_dim, device=device) * 0.001
    query_p_t = agent_abs_pos # Agent is at the location of the last observation
    
    print(f"Retrieving memory for query similar to last observation...")
    source, retrieved_data = memory_retriever.retrieve_memory(
        current_observation_embedding=query_v_t,
        current_absolute_position=query_p_t,
        current_semantic_node_position=current_sem_node_abs_pos
    )

    print(f"Retrieved from: {source}")
    if retrieved_data:
        print(f"Retrieved {len(retrieved_data)} items:")
        for item_idx, item_data in enumerate(retrieved_data):
            if source == "STM":
                entry: ShortTermMemoryEntry = item_data   
                print(f"  STM Item {item_idx}: Key={entry.key}, ObjID={entry.object_id if hasattr(entry, 'object_id') else 'N/A'}, Emb_Sim={torch.cosine_similarity(query_v_t.unsqueeze(0), entry.embedding.unsqueeze(0)).item():.3f}")
            elif source == "LTM":
                key, recon_v, recon_p, recon_d = item_data
                print(f"  LTM Item {item_idx}: Key={key}, Recon_V_Sim={torch.cosine_similarity(query_v_t.unsqueeze(0), recon_v.unsqueeze(0)).item():.3f}, Recon_P={recon_p.cpu().numpy().round(2)}")
    else:
        print("No items retrieved.")
    
    # Expect STM if similarity threshold (0.5) is met by obs_emb2/obs_emb1
    if len(memory_retriever.short_term_memory) > 0 : # Check if STM actually has items
      # Manually check similarity of query_v_t to STM items for assertion
      stm_entry_for_key2 = memory_retriever.short_term_memory.entry_map.get(obs_key2)
      if stm_entry_for_key2:
          sim_to_key2 = torch.cosine_similarity(query_v_t.unsqueeze(0), stm_entry_for_key2.embedding.unsqueeze(0)).item()
          if sim_to_key2 >= memory_retriever.stm_similarity_threshold:
              assert source == "STM", f"Expected STM retrieval, got {source}. Max STM sim was {sim_to_key2}"
          else:
              assert source == "LTM", f"Expected LTM retrieval due to low STM sim ({sim_to_key2}), got {source}"
      elif memory_retriever.long_term_memory.hnsw_index.get_current_count() > 0 :
          assert source == "LTM", "Expected LTM as STM entry for key2 not found or other STM entries low sim."


    # Test retrieval that should fallback to LTM (e.g., query far from STM items or very different embedding)
    print("\nRetrieving memory with a dissimilar query (expect LTM)...")
    query_v_t_dissimilar = torch.randn(emb_dim, device=device) # very different embedding
    query_p_t_far = agent_abs_pos + torch.tensor([100.0, 100.0, 0.0], device=device) # far away position

    source_ltm, retrieved_data_ltm = memory_retriever.retrieve_memory(
        current_observation_embedding=query_v_t_dissimilar,
        current_absolute_position=query_p_t_far, # LTM uses this for Proj
        current_semantic_node_position=current_sem_node_abs_pos # STM uses this - p_t for q_rel
    )
    print(f"Retrieved from: {source_ltm}")
    # This should be LTM unless STM happens to have something very generic that matches by chance
    if memory_retriever.long_term_memory.hnsw_index.get_current_count() > 0:
        assert source_ltm == "LTM"
    elif len(memory_retriever.short_term_memory) > 0: # If LTM is empty, might still get STM if some items are there
        pass # Could be STM if LTM is empty
    else: # Both empty
        assert not retrieved_data_ltm


    memory_retriever.clear_all_memory()
    print(f"\nMemories cleared. STM size: {len(memory_retriever.short_term_memory)}, LTM HNSW count: {memory_retriever.long_term_memory.hnsw_index.get_current_count()}")
    assert len(memory_retriever.short_term_memory) == 0
    assert memory_retriever.long_term_memory.hnsw_index.get_current_count() == 0

    print("\nMemoryRetrieval tests completed.")