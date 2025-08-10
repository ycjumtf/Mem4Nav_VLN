# type: ignore
import torch
import torch.nn as nn
import numpy as np
import hnswlib 
from typing import Any, Dict, Optional, Tuple, List

try:
    from .reversible_transformer import ReversibleTransformer
except ImportError: 
    from mem4nav_core.memory_system.reversible_transformer import ReversibleTransformer

class MLP(nn.Module):
    """A simple Multi-Layer Perceptron."""
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_dims: Optional[List[int]] = None, 
                 activation: nn.Module = nn.ReLU,   
                 dropout_p: float = 0.1,
                 output_activation: Optional[nn.Module] = None):   
        super().__init__()
        layers: List[nn.Module] = []
        current_dim = input_dim
        if hidden_dims:
            for h_dim in hidden_dims:
                layers.append(nn.Linear(current_dim, h_dim))
                layers.append(activation())
                if dropout_p > 0:
                    layers.append(nn.Dropout(p=dropout_p))
                current_dim = h_dim
        layers.append(nn.Linear(current_dim, output_dim))
        if output_activation:
            layers.append(output_activation())
            
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class LongTermMemory(nn.Module):
    """
    Long-Term Memory (LTM) system for Vision-and-Language Navigation.
    Stores spatially anchored observations using reversible memory tokens.
    """
    def __init__(
        self,
        embedding_dim: int, # Dimension of individual observation embeddings (v_t), d_emb
        transformer_depth: int,
        transformer_heads: int,
        transformer_mlp_ratio: float = 4.0,
        transformer_dropout: float = 0.1,
        max_elements: int = 10000,
        hnsw_ef_construction: int = 500,
        hnsw_m: int = 64,
        hnsw_ef_search: int = 200,
        default_retrieval_k: int = 3, # m in paper (top-m)
        # MLP configurations
        query_projection_config: Optional[Dict[str, Any]] = None, # For Proj([v_t; p_t])
        position_decoder_config: Optional[Dict[str, Any]] = None, # For pi_p
        descriptor_decoder_config: Optional[Dict[str, Any]] = None, # For pi_d
        cycle_consistency_decoder_config: Optional[Dict[str, Any]] = None, # For pi_v
        position_dim: int = 3,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.embedding_dim = embedding_dim # d_emb (dim of v_t, and input theta_r for R)
        self.ltm_token_dim = embedding_dim * 2 # Dim of tokens stored in HNSW (output of R)

        self.reversible_transformer = ReversibleTransformer(
            dim=self.ltm_token_dim, # ReversibleTransformer operates on concatenated input
            depth=transformer_depth,
            num_heads=transformer_heads,
            mlp_ratio=transformer_mlp_ratio,
            dropout=transformer_dropout
        ).to(self.device)

        self.default_retrieval_k = default_retrieval_k

        # Store HNSW parameters for re-initialization in clear()
        self._init_hnsw_max_elements = max_elements
        self._init_hnsw_ef_construction = hnsw_ef_construction
        self._init_hnsw_m = hnsw_m
        self._init_hnsw_ef_search = hnsw_ef_search # Store ef for search

        self.hnsw_index = hnswlib.Index(space='l2', dim=self.ltm_token_dim)
        self.hnsw_index.init_index(
            max_elements=self._init_hnsw_max_elements,
            ef_construction=self._init_hnsw_ef_construction,
            M=self._init_hnsw_m
        )
        self.hnsw_index.set_ef(self._init_hnsw_ef_search)

        self.stored_read_tokens_data = np.zeros((self._init_hnsw_max_elements, self.ltm_token_dim), dtype=np.float32)
        self.key_to_hnsw_label: Dict[Any, int] = {} # Maps user key to HNSW internal unique label
        self.hnsw_label_to_key: Dict[int, Any] = {} # Maps HNSW label back to user key
        self._next_hnsw_label: int = 0 # Counter for unique HNSW labels

        # --- MLP Projectors and Decoders ---
        # Query Projector: Proj([v_t (d_emb); p_t (pos_dim)]) -> query_for_hnsw (ltm_token_dim)
        query_proj_conf = query_projection_config or {}
        self.query_projector = MLP(
            input_dim=self.embedding_dim + position_dim,
            output_dim=self.ltm_token_dim, # Output should match HNSW token dimension
            hidden_dims=query_proj_conf.get('hidden_dims', [self.ltm_token_dim]), # Default: one hidden layer
            dropout_p=query_proj_conf.get('dropout', 0.1)
        ).to(self.device)

        # Position Decoder: pi_p(reconstructed_v_s_stored (d_emb)) -> p_hat (pos_dim)
        pos_dec_conf = position_decoder_config or {}
        self.position_decoder = MLP(
            input_dim=self.embedding_dim,
            output_dim=position_dim,
            hidden_dims=pos_dec_conf.get('hidden_dims', [self.embedding_dim // 2, position_dim * 4]),
            dropout_p=pos_dec_conf.get('dropout', 0.1)
        ).to(self.device)

        # Descriptor Decoder: pi_d(reconstructed_v_s_stored (d_emb)) -> d_hat (e.g., d_emb or other desc_dim)
        desc_dec_conf = descriptor_decoder_config or {}
        descriptor_output_dim = desc_dec_conf.get('output_dim', self.embedding_dim)
        self.descriptor_decoder = MLP(
            input_dim=self.embedding_dim,
            output_dim=descriptor_output_dim,
            hidden_dims=desc_dec_conf.get('hidden_dims', [self.embedding_dim]),
            dropout_p=desc_dec_conf.get('dropout', 0.1)
        ).to(self.device)
        
        # Cycle Consistency Decoder: pi_v(R^-1(R(...)) which is 2*d_emb) -> v_hat (d_emb)
        cycle_dec_conf = cycle_consistency_decoder_config or {}
        self.cycle_consistency_v_decoder = MLP(
            input_dim=self.ltm_token_dim, # Takes the full reconstructed pair
            output_dim=self.embedding_dim, # Projects back to original v_t's space
            hidden_dims=cycle_dec_conf.get('hidden_dims', [self.ltm_token_dim // 2]),
            dropout_p=cycle_dec_conf.get('dropout', 0.1)
        ).to(self.device)

    def write(self, key: Any, 
              previous_read_token_for_R_input: torch.Tensor, # d_emb dimensional (theta_r_prev for R)
              observation_embedding_v_t: torch.Tensor      # d_emb dimensional (v_t for R)
             ) -> torch.Tensor:
        """
        Encodes and stores/updates an observation associated with a key.
        The `previous_read_token_for_R_input` is the d_emb token derived from the
        previously stored 2*d_emb LTM state for this key (or zeros if new).
        
        Returns the new 2*d_emb LTM state token (theta_r_new) to be stored by the caller
        (e.g., in OctreeLeaf or GraphNode).
        """
        if not isinstance(key, int):
            # HNSW labels are integers. If keys are not ints, a mapping is needed
            # or LTM should internally manage this mapping to integer labels.
            # Current `_next_hnsw_label` strategy handles this if keys are just for our dicts.
             print(f"Warning: LTM key '{key}' is not an integer. HNSW labels are integers. Ensure key management is robust.")


        previous_read_token_for_R_input = previous_read_token_for_R_input.to(self.device).view(1, 1, self.embedding_dim)
        observation_embedding_v_t = observation_embedding_v_t.to(self.device).view(1, 1, self.embedding_dim)

        transformer_input = torch.cat([previous_read_token_for_R_input, observation_embedding_v_t], dim=-1)
        
        # new_ltm_state_token is theta_w_new, which becomes theta_r_new for storage
        new_ltm_state_token_2d_emb = self.reversible_transformer(transformer_input).squeeze(0).squeeze(0) # (ltm_token_dim)
        new_token_np = new_ltm_state_token_2d_emb.detach().cpu().numpy().astype(np.float32)

        current_hnsw_label = self.key_to_hnsw_label.get(key)

        if current_hnsw_label is not None: # Key exists, update its LTM state
            # Mark the old entry for deletion in HNSW
            try:
                self.hnsw_index.mark_deleted(current_hnsw_label)
            except RuntimeError as e: # May happen if label was never actually added or already marked
                print(f"Info: Could not mark HNSW label {current_hnsw_label} as deleted (key {key}): {e}")


        # Add the new/updated token with a new HNSW label to ensure HNSW sees the new vector
        if self._next_hnsw_label >= self._init_hnsw_max_elements:
            # This implies we've used up all unique labels up to max_elements.
            # If max_elements is truly the cap on HNSW entries, this is an issue.
            # If stored_read_tokens_data is a circular buffer, HNSW labels must still be unique in current index.
            # A more robust solution might involve a separate label pool or re-evaluating max_elements.
            # For now, error if we exceed the initial max_elements with unique labels.
            raise MemoryError(f"LTM HNSW ran out of unique labels. Max labels ({self._init_hnsw_max_elements}) reached.")
        
        new_assigned_label = self._next_hnsw_label
        self.hnsw_index.add_items(new_token_np.reshape(1, -1), np.array([new_assigned_label]))
        
        # Update our tracking
        # The stored_read_tokens_data array uses the new_assigned_label as its index.
        # This means self._init_hnsw_max_elements must be large enough for all *updates* if labels always increment.
        # Or, labels need to be recycled, which is complex with HNSWlib's Python API.
        # For now, assume _next_hnsw_label doesn't exceed array bounds due to _init_hnsw_max_elements check.
        self.stored_read_tokens_data[new_assigned_label] = new_token_np
        self.key_to_hnsw_label[key] = new_assigned_label
        self.hnsw_label_to_key[new_assigned_label] = key
        self._next_hnsw_label += 1
        
        return new_ltm_state_token_2d_emb # Return the new 2*d_emb token

    def _decode_retrieved_hnsw_labels(self, retrieved_hnsw_labels: np.ndarray
                                     ) -> List[Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Helper to decode HNSW labels into (key, v_s, p_hat, d_hat)."""
        results = []
        if retrieved_hnsw_labels.size == 0:
            return results

        for label_val in retrieved_hnsw_labels[0]: # knn_query returns labels as [[label1, label2, ...]]
            if label_val < 0 or label_val >= self._next_hnsw_label: # Invalid label
                continue 
            
            # HNSWlib can sometimes return marked-deleted items if not fully cleaned, skip them
            # However, our current logic assumes valid labels are returned.
            # A check: if label_val not in self.hnsw_label_to_key might be needed if HNSW returns old labels.

            retrieved_ltm_state_token_np = self.stored_read_tokens_data[label_val]
            retrieved_ltm_state_token_tensor = torch.from_numpy(retrieved_ltm_state_token_np).to(self.device).view(1, 1, self.ltm_token_dim)

            with torch.no_grad():
                reconstructed_input_pair = self.reversible_transformer.inverse(retrieved_ltm_state_token_tensor)
            
            # Second half is reconstructed v_s_stored (d_emb)
            reconstructed_v_s_stored = reconstructed_input_pair[:, :, self.embedding_dim:].squeeze(0).squeeze(0)

            with torch.no_grad():
                reconstructed_position_p_hat = self.position_decoder(reconstructed_v_s_stored)
                reconstructed_descriptor_d_hat = self.descriptor_decoder(reconstructed_v_s_stored)
            
            original_key = self.hnsw_label_to_key.get(label_val)
            if original_key is not None:
                results.append((
                    original_key,
                    reconstructed_v_s_stored.detach(), 
                    reconstructed_position_p_hat.detach(), 
                    reconstructed_descriptor_d_hat.detach()
                ))
        return results

    def retrieve(self, 
                 current_observation_embedding_v_t: torch.Tensor, # d_emb
                 current_absolute_position_p_t: torch.Tensor,     # pos_dim (e.g., 3D)
                 k: Optional[int] = None
                ) -> List[Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Retrieves k nearest LTM entries using the internal query projector Proj([v_t; p_t]).
        """
        if self.hnsw_index.get_current_count() == 0:
            return []
        k_to_use = k if k is not None else self.default_retrieval_k
        if k_to_use == 0: return []


        query_input_for_proj = torch.cat([
            current_observation_embedding_v_t.to(self.device).view(-1), 
            current_absolute_position_p_t.to(self.device).view(-1)
        ], dim=0)
        projected_query_2d_emb = self.query_projector(query_input_for_proj)
        
        return self.retrieve_with_direct_query(projected_query_2d_emb, k_to_use)

    def retrieve_with_direct_query(self, 
                                   query_vector_2d_emb: torch.Tensor, # ltm_token_dim (2*d_emb)
                                   k: Optional[int] = None
                                  ) -> List[Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Retrieves k nearest LTM entries using a pre-computed query vector.
        This is useful for agent backbones like FLAME that might compute their own LTM query.
        The query_vector_2d_emb should match the dimension of tokens stored in HNSW (ltm_token_dim).
        """
        if self.hnsw_index.get_current_count() == 0:
            return []
        k_to_use = k if k is not None else self.default_retrieval_k
        if k_to_use == 0: return []
        if query_vector_2d_emb.shape[0] != self.ltm_token_dim:
            raise ValueError(f"Direct query vector dim {query_vector_2d_emb.shape[0]} "
                             f"does not match LTM token dim {self.ltm_token_dim}")

        query_np = query_vector_2d_emb.detach().cpu().numpy().astype(np.float32)
        
        actual_k = min(k_to_use, self.hnsw_index.get_current_count())
        if actual_k == 0 : return []
            
        try:
            retrieved_hnsw_labels, _ = self.hnsw_index.knn_query(query_np, k=actual_k)
        except Exception as e:
            # This can happen if k > number of elements, even after min. HNSWlib can be finicky.
            print(f"Error during HNSW knn_query: {e}. Attempting with k=1 if possible.")
            if self.hnsw_index.get_current_count() > 0:
                try:
                    retrieved_hnsw_labels, _ = self.hnsw_index.knn_query(query_np, k=1)
                except Exception as e_k1:
                    print(f"HNSW knn_query failed even with k=1: {e_k1}. Returning empty.")
                    return []
            else:
                return []
        
        return self._decode_retrieved_hnsw_labels(retrieved_hnsw_labels)

    def get_current_ltm_read_token_for_key(self, key: Any, default_if_new: bool = True) -> torch.Tensor:
        """
        Retrieves the d_emb "logical" previous read token for a given key,
        to be used as input theta_r_prev to the ReversibleTransformer for the next LTM write.
        It derives this from the currently stored 2*d_emb LTM state token for the key.
        """
        hnsw_label = self.key_to_hnsw_label.get(key)
        
        if hnsw_label is not None and hnsw_label < self._next_hnsw_label: # Ensure label is valid
            current_stored_ltm_state_2d_emb_np = self.stored_read_tokens_data[hnsw_label]
            current_stored_ltm_state_2d_emb = torch.from_numpy(current_stored_ltm_state_2d_emb_np).to(self.device).view(1,1,self.ltm_token_dim)
            with torch.no_grad():
                # R^-1(current_2d_emb_token) -> (theta_r_prev_logical (d_emb), v_s_stored (d_emb))
                reconstructed_pair = self.reversible_transformer.inverse(current_stored_ltm_state_2d_emb)
                prev_theta_r_logical_input = reconstructed_pair[:, :, :self.embedding_dim].squeeze(0).squeeze(0) # First half
            return prev_theta_r_logical_input # This is d_emb
        elif default_if_new:
            return torch.zeros(self.embedding_dim, device=self.device) # d_emb zeros for new keys
        else:
            raise KeyError(f"Key '{key}' not found in LTM and default_if_new is False.")

    def get_cycle_consistency_reconstruction(self, 
                                             previous_read_token_for_R_input: torch.Tensor, # d_emb
                                             observation_embedding_v_t: torch.Tensor      # d_emb
                                            ) -> torch.Tensor:
        """ v_hat = pi_v ( R^-1 ( R(theta_r_prev || v_t) ) ) """
        prev_token = previous_read_token_for_R_input.to(self.device).view(1, 1, self.embedding_dim)
        obs_emb = observation_embedding_v_t.to(self.device).view(1, 1, self.embedding_dim)
        
        transformer_input = torch.cat([prev_token, obs_emb], dim=-1) # (1,1, 2*d_emb)
        
        encoded_token = self.reversible_transformer(transformer_input)    # (1,1, 2*d_emb)
        reconstructed_pair = self.reversible_transformer.inverse(encoded_token) # (1,1, 2*d_emb)
        
        # pi_v takes the full reconstructed pair (2*d_emb) and outputs v_hat (d_emb)
        v_hat = self.cycle_consistency_v_decoder(reconstructed_pair.squeeze(0).squeeze(0))
        return v_hat

    def clear(self):
        """Clears LTM state and re-initializes HNSW index."""
        print("Clearing LongTermMemory...")
        self.hnsw_index = hnswlib.Index(space='l2', dim=self.ltm_token_dim)
        self.hnsw_index.init_index(
            max_elements=self._init_hnsw_max_elements,
            ef_construction=self._init_hnsw_ef_construction,
            M=self._init_hnsw_m
        )
        self.hnsw_index.set_ef(self._init_hnsw_ef_search) # Reset ef for search
        
        self.stored_read_tokens_data.fill(0) # Reset stored tokens
        self.key_to_hnsw_label.clear()
        self.hnsw_label_to_key.clear()
        self._next_hnsw_label = 0

if __name__ == '__main__':
    print("--- Testing LongTermMemory with modifications ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    d_emb = 128 # Dimension of v_t, and input theta_r for R
    pos_dim = 3
    
    # Example MLP configs
    mlp_hidden_default = [d_emb // 2]
    ltm_config = {
        "embedding_dim": d_emb, "transformer_depth": 2, "transformer_heads": 2,
        "max_elements": 100, "default_retrieval_k": 2, "position_dim": pos_dim,
        "query_projection_config": {"hidden_dims": [d_emb*2], "dropout": 0.0}, # Output 2*d_emb
        "position_decoder_config": {"hidden_dims": mlp_hidden_default, "dropout": 0.0},
        "descriptor_decoder_config": {"hidden_dims": mlp_hidden_default, "output_dim": d_emb, "dropout": 0.0},
        "cycle_consistency_decoder_config": {"hidden_dims": [d_emb*2 // 2], "dropout": 0.0}, # Input 2*d_emb, output d_emb
        "device": device
    }
    ltm = LongTermMemory(**ltm_config) #type: ignore
    ltm.train() # Set to train for dropout if any, though MLPs here have dropout 0

    # Test Write
    key1 = 101 
    # For first write to key1, previous_read_token_for_R_input is zeros(d_emb)
    prev_read_token_input_for_R_key1 = ltm.get_current_ltm_read_token_for_key(key1)
    assert prev_read_token_input_for_R_key1.shape == (d_emb,)
    assert torch.all(prev_read_token_input_for_R_key1 == 0)
    
    obs_v1 = torch.randn(d_emb, device=device)
    # new_ltm_state_key1 is the 2*d_emb token to be stored by OctreeLeaf/GraphNode
    new_ltm_state_key1 = ltm.write(key1, prev_read_token_input_for_R_key1, obs_v1)
    print(f"Key {key1} written. LTM state token shape: {new_ltm_state_key1.shape}")
    assert new_ltm_state_key1.shape == (d_emb * 2,)

    # Test get_current_ltm_read_token_for_key for next write
    next_prev_read_token_input_for_R_key1 = ltm.get_current_ltm_read_token_for_key(key1)
    print(f"Next input theta_r for key {key1} (d_emb): {next_prev_read_token_input_for_R_key1.shape}")
    assert next_prev_read_token_input_for_R_key1.shape == (d_emb,)
    # This should be the first half of R_inv(new_ltm_state_key1)
    with torch.no_grad():
        expected_next_prev_input, _ = ltm.reversible_transformer.inverse(new_ltm_state_key1.view(1,1,-1)).chunk(2, dim=-1)
    assert torch.allclose(next_prev_read_token_input_for_R_key1, expected_next_prev_input.squeeze(), atol=1e-5)

    # Test Cycle Consistency path
    v_hat1 = ltm.get_cycle_consistency_reconstruction(prev_read_token_input_for_R_key1, obs_v1)
    print(f"Reconstructed v_hat1 for cycle consistency: {v_hat1.shape}")
    assert v_hat1.shape == (d_emb,)
    # Loss would be mse(v_hat1, obs_v1)

    # Test Retrieval
    ltm.eval()
    query_obs_emb = torch.randn(d_emb, device=device)
    query_pos = torch.randn(pos_dim, device=device)
    
    retrieved_items = ltm.retrieve(query_obs_emb, query_pos, k=1)
    print(f"\nRetrieved {len(retrieved_items)} items using internal projector:")
    if retrieved_items:
        k_ret, v_s, p_hat, d_hat = retrieved_items[0]
        print(f"  Key: {k_ret}, v_s: {v_s.shape}, p_hat: {p_hat.shape}, d_hat: {d_hat.shape}")
        assert v_s.shape == (d_emb,)
        assert p_hat.shape == (pos_dim,)
        assert d_hat.shape == (d_emb,) # As per descriptor_decoder_config output_dim

    # Test retrieve_with_direct_query (for FLAME-like scenario)
    direct_query_vec_2d_emb = torch.randn(d_emb * 2, device=device) # Matches LTM token dim
    retrieved_direct = ltm.retrieve_with_direct_query(direct_query_vec_2d_emb, k=1)
    print(f"\nRetrieved {len(retrieved_direct)} items using direct 2*d_emb query:")
    if retrieved_direct:
        k_ret_d, _, _, _ = retrieved_direct[0]
        print(f"  Key: {k_ret_d}")
    
    # Test clear method
    old_ef_search = ltm._init_hnsw_ef_search # Store before clear for assertion
    ltm.clear()
    print(f"\nLTM cleared. Current HNSW count: {ltm.hnsw_index.get_current_count()}")
    assert ltm.hnsw_index.get_current_count() == 0
    assert ltm._next_hnsw_label == 0
    assert len(ltm.key_to_hnsw_label) == 0
    assert ltm.hnsw_index.ef == old_ef_search # Check if ef for search was reset

    print("\nLongTermMemory tests with modifications completed.")