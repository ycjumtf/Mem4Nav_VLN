# type: ignore
import torch
import numpy as np
from typing import List, Tuple, Any, Optional, Dict
import time # For precise timestamps if needed, though step index is often used

class ShortTermMemoryEntry:
    """Represents a single entry in the Short-Term Memory."""
    __slots__ = ('key', 'object_id', 'relative_position', 'embedding', 'timestamp', 'access_frequency', 'creation_step')

    def __init__(self, 
                 key: Any, 
                 object_id: Any, 
                 relative_position: np.ndarray, # p_rel (relative to current graph node)
                 embedding: torch.Tensor, # v (multimodal embedding)
                 creation_step: int, 
                 initial_access_frequency: int = 1):
        self.key: Any = key 
        self.object_id: Any = object_id
        self.relative_position: np.ndarray = relative_position # Should be (3,) for 3D
        self.embedding: torch.Tensor = embedding # d-dimensional
        self.timestamp: float = time.monotonic() 
        self.creation_step: int = creation_step 
        self.access_frequency: int = initial_access_frequency

    def update_on_reaccess(self, new_embedding: torch.Tensor, new_relative_position: np.ndarray, current_step: int):
        """Updates entry when it's accessed or re-observed."""
        self.embedding = new_embedding
        self.relative_position = new_relative_position
        self.timestamp = time.monotonic() # Update wall time recency
        self.creation_step = current_step # Update step recency
        self.access_frequency += 1

    def __repr__(self) -> str:
        return (f"STMEntry(key={self.key}, obj_id={self.object_id}, rel_pos={self.relative_position.tolist()}, "
                f"step={self.creation_step}, freq={self.access_frequency}, emb_shape={self.embedding.shape})")

class ShortTermMemory:
    """
    Fixed-capacity Short-Term Memory (STM) cache.
    Stores recent multimodal entries in relative coordinates for rapid local lookup.
    Uses a Frequency-and-Least-Frequently Used (FLFU) inspired eviction policy.
    """
    def __init__(self, 
                 capacity: int = 128, 
                 eviction_lambda: float = 0.5, # Balances frequency and recency for eviction
                 device: Optional[torch.device] = None):
        """
        Initializes the Short-Term Memory.

        Args:
            capacity (int): K, the maximum number of entries STM can hold[cite: 79].
            eviction_lambda (float): Lambda (λ) for the eviction score calculation[cite: 79].
                                     Balances frequency vs. recency.
            device (Optional[torch.device]): PyTorch device for embeddings.
        """
        if not (0 <= eviction_lambda <= 1):
            raise ValueError("eviction_lambda must be between 0 and 1.")

        self.capacity: int = capacity
        self.eviction_lambda: float = eviction_lambda # λ
        self.device: torch.device = device or torch.device('cpu')
        
        # Stores ShortTermMemoryEntry objects
        self.entries: List[ShortTermMemoryEntry] = []
        # For quick lookup of entry by its unique key
        self.entry_map: Dict[Any, ShortTermMemoryEntry] = {}
        
        self._current_step: int = 0 

    def set_current_step(self, step: int):
        """Updates the current time step, used for recency calculations."""
        self._current_step = step

    def insert(self, 
               key: Any, # A unique key for this specific observation instance
               object_id: Any, # 'o' from paper: "car", "traffic_light"
               relative_position: np.ndarray, # p_rel = p_t - p_u_c (position relative to current semantic graph node u_c)
               embedding: torch.Tensor, # v_t, multimodal embedding
               current_semantic_node_id: Optional[Any] = None # u_c, for context, not directly stored in entry per paper structure
               ):
        """
        Inserts a new observation or updates an existing one in the STM.

        Args:
            key (Any): A unique identifier for this memory entry (e.g., a unique ID for the observation event).
            object_id (Any): Identifier for the object or event type.
            relative_position (np.ndarray): Coordinate relative to current semantic node $p_{rel}$.
            embedding (torch.Tensor): Multimodal embedding $v \in \mathbb{R}^d$.
            current_semantic_node_id (Optional[Any]): The ID of the current semantic graph node $u_c$
                to which this STM is conceptually attached. While STM entries store $p_{rel}$, this
                context might be useful for managing multiple STMs if needed, though the paper implies
                one STM attached to the *current* node.
        """
        embedding = embedding.to(self.device)
        relative_position = np.asarray(relative_position, dtype=np.float32)

        if key in self.entry_map:
            # Update existing entry
            existing_entry = self.entry_map[key]
            existing_entry.update_on_reaccess(embedding, relative_position, self._current_step)
        else:
            # New entry
            if len(self.entries) >= self.capacity:
                self._evict_entry() # Evict one entry if capacity is reached

            new_entry = ShortTermMemoryEntry(
                key=key,
                object_id=object_id,
                relative_position=relative_position,
                embedding=embedding,
                creation_step=self._current_step
            )
            self.entries.append(new_entry)
            self.entry_map[key] = new_entry

    def _evict_entry(self):
        """
        Evicts an entry based on the FLFU-inspired policy:
        Score(e_i) = λ * freq(e_i) - (1-λ) * (t_now - τ_i) [cite: 79]
        The entry with the minimum score is evicted[cite: 80].
        t_now is self._current_step, τ_i is entry.creation_step
        """
        if not self.entries:
            return

        min_score = float('inf')
        eviction_candidate_idx = -1

        for i, entry in enumerate(self.entries):
            recency_penalty = self._current_step - entry.creation_step
            # Score(e_i) = λ * freq(e_i) - (1-λ) * (t_now - τ_i)
            # We want to evict the one with the *lowest* score.
            score = (self.eviction_lambda * entry.access_frequency - 
                     (1.0 - self.eviction_lambda) * recency_penalty)
            
            if score < min_score:
                min_score = score
                eviction_candidate_idx = i
        
        if eviction_candidate_idx != -1:
            evicted_entry = self.entries.pop(eviction_candidate_idx)
            del self.entry_map[evicted_entry.key]

    def retrieve(self, 
                 query_embedding: torch.Tensor, 
                 query_relative_position: np.ndarray, # q_rel
                 spatial_radius_epsilon: float, # ε for spatial filtering
                 top_k: int = 5
                 ) -> List[ShortTermMemoryEntry]: # Returns list of actual entry objects
        """
        Retrieves top-k relevant entries from STM.
        1. Filters entries within spatial_radius_epsilon of query_relative_position[cite: 81].
        2. Ranks filtered entries by cosine similarity with query_embedding[cite: 81].

        Args:
            query_embedding (torch.Tensor): v_t, current observation embedding for similarity.
            query_relative_position (np.ndarray): q_rel, relative query position for spatial filtering.
            spatial_radius_epsilon (float): ε, radius for spatial filtering.
            top_k (int): Number of top entries to return.

        Returns:
            List[ShortTermMemoryEntry]: List of top-k ShortTermMemoryEntry objects.
        """
        if not self.entries:
            return []

        query_embedding = query_embedding.to(self.device)
        query_relative_position = np.asarray(query_relative_position, dtype=np.float32)
        
        # 1. Spatial Filtering
        # C = {e_i : ||p_rel,i - q_rel|| <= ε} [cite: 81]
        spatially_relevant_entries: List[ShortTermMemoryEntry] = []
        for entry in self.entries:
            distance = np.linalg.norm(entry.relative_position - query_relative_position)
            if distance <= spatial_radius_epsilon:
                spatially_relevant_entries.append(entry)

        if not spatially_relevant_entries:
            return []

        # 2. Semantic Ranking (Cosine Similarity)
        # s_i = <v_t, v_i> / (||v_t|| ||v_i||) [cite: 81]
        entry_embeddings = torch.stack([entry.embedding for entry in spatially_relevant_entries])
        
        # Ensure query_embedding is 2D for cosine_similarity (batch_size=1, embedding_dim)
        if query_embedding.ndim == 1:
            query_embedding_2d = query_embedding.unsqueeze(0)
        else:
            query_embedding_2d = query_embedding
            
        similarities = torch.nn.functional.cosine_similarity(query_embedding_2d, entry_embeddings, dim=1)
        
        # Get top-k entries based on similarities
        # Sort in descending order of similarity
        num_to_retrieve = min(top_k, len(spatially_relevant_entries))
        if num_to_retrieve == 0:
            return []
            
        top_k_indices = torch.topk(similarities, k=num_to_retrieve).indices
        
        top_entries = [spatially_relevant_entries[i.item()] for i in top_k_indices]
        
        return top_entries

    def get_all_entries(self) -> List[ShortTermMemoryEntry]:
        return list(self.entries)

    def clear(self):
        """Clears all entries from STM."""
        self.entries.clear()
        self.entry_map.clear()
        self._current_step = 0 # Reset step counter

    def __len__(self) -> int:
        """Returns the current number of entries in STM."""
        return len(self.entries)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stm_capacity = 5
    emb_dim = 64
    stm = ShortTermMemory(capacity=stm_capacity, eviction_lambda=0.5, device=device)

    # Simulate some steps and insertions
    for i in range(stm_capacity + 3):
        stm.set_current_step(i) # Agent takes a step
        
        key = f"obs_{i}"
        obj_id = f"obj_type_{(i%3)}"
        # p_rel should be relative to the agent's current graph node.
        # For this test, let's assume it changes somewhat randomly.
        rel_pos = np.random.rand(3).astype(np.float32) * 5 
        embedding = torch.randn(emb_dim, device=device)
        
        print(f"\nStep {i}: Inserting entry with key '{key}'")
        stm.insert(key, obj_id, rel_pos, embedding)
        
        if i == 2: # Re-access an old entry to see frequency update
            print(f"Step {i}: Re-accessing 'obs_0'")
            # Assume obs_0's relative position and embedding might change slightly upon re-access
            updated_rel_pos_obs0 = stm.entry_map["obs_0"].relative_position + np.array([0.1,0.1,0.1], dtype=np.float32)
            updated_emb_obs0 = stm.entry_map["obs_0"].embedding + torch.randn(emb_dim, device=device) * 0.1
            stm.insert("obs_0", "obj_type_0", updated_rel_pos_obs0, updated_emb_obs0)


        print(f"STM ({len(stm)}/{stm.capacity}):")
        for entry_idx, entry_val in enumerate(stm.get_all_entries()):
            score = (stm.eviction_lambda * entry_val.access_frequency -
                     (1.0 - stm.eviction_lambda) * (stm._current_step - entry_val.creation_step))
            print(f"  {entry_idx}: {entry_val.key}, Freq: {entry_val.access_frequency}, Step: {entry_val.creation_step}, Score: {score:.2f}")
    
    # Test retrieval
    stm.set_current_step(stm_capacity + 3) # Update current time for retrieval context if needed
    query_emb = torch.randn(emb_dim, device=device)
    query_rel_pos = np.array([2.0, 2.0, 0.0], dtype=np.float32) 
    spatial_radius = 3.0
    top_k_retrieval = 3

    print(f"\nRetrieving items around {query_rel_pos.tolist()} with radius {spatial_radius}, top {top_k_retrieval}")
    retrieved_entries = stm.retrieve(query_emb, query_rel_pos, spatial_radius, top_k_retrieval)

    if retrieved_entries:
        print("Retrieved STM Entries:")
        for entry in retrieved_entries:
            dist_to_query = np.linalg.norm(entry.relative_position - query_rel_pos)
            sim_to_query = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), entry.embedding.unsqueeze(0)).item()
            print(f"  Key: {entry.key}, ObjID: {entry.object_id}, Dist: {dist_to_query:.2f}, Sim: {sim_to_query:.2f}, Freq: {entry.access_frequency}, Step: {entry.creation_step}")
    else:
        print("No entries retrieved.")

    stm.clear()
    print(f"\nSTM cleared. Size: {len(stm)}")
    assert len(stm) == 0
    
    print("\nShortTermMemory tests completed.")