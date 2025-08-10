import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
# for type hinting if needed,
# though the loss function will take its output directly.
# from mem4nav_core.memory_system.long_term_memory import LongTermMemory



class NavigationLoss(nn.Module):
    """
    Calculates the navigation loss, typically for imitation learning.
    Assumes the model outputs logits for a discrete action space.
    """
    def __init__(self, ignore_index: int = -100):
        """
        Args:
            ignore_index (int): Specifies a target value that is ignored
                                and does not contribute to the input gradient.
                                Useful for padding in sequences.
        """
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self,
                predicted_action_logits: torch.Tensor,  # (BatchSize, NumActions) or (BatchSize, SeqLen, NumActions)
                ground_truth_action_ids: torch.Tensor   # (BatchSize,) or (BatchSize, SeqLen)
               ) -> torch.Tensor:
        """
        Computes the cross-entropy loss for action prediction.

        Args:
            predicted_action_logits (torch.Tensor): The logits output by the policy network.
                                                    Shape: (BatchSize, NumActions) for single step, or
                                                           (BatchSize, SeqLen, NumActions) for sequence.
            ground_truth_action_ids (torch.Tensor): The ground truth action IDs.
                                                     Shape: (BatchSize,) for single step, or
                                                            (Batch_Size, SeqLen) for sequence.

        Returns:
            torch.Tensor: The computed navigation loss (scalar).
        """
        if predicted_action_logits.ndim == 3 and ground_truth_action_ids.ndim == 2:
            # Reshape for sequence loss: (BatchSize * SeqLen, NumActions) and (BatchSize * SeqLen)
            return self.cross_entropy_loss(
                predicted_action_logits.view(-1, predicted_action_logits.size(-1)),
                ground_truth_action_ids.view(-1)
            )
        elif predicted_action_logits.ndim == 2 and ground_truth_action_ids.ndim == 1:
            # Single step loss
            return self.cross_entropy_loss(predicted_action_logits, ground_truth_action_ids)
        else:
            raise ValueError(
                f"Shape mismatch for logits ({predicted_action_logits.shape}) and labels ({ground_truth_action_ids.shape})"
            )


class CycleConsistencyLoss(nn.Module):
    """
    Calculates the cycle-consistency loss for the Reversible Transformer in LTM.
    L_cycle = E[||v - v_hat||^2_2], where v_hat = pi_v(R^-1(R(theta_r_prev || v_t))).
    """
    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. Default: 'mean'.
        """
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self,
                original_observation_embeddings: torch.Tensor,    # v (BatchSize, EmbeddingDim)
                reconstructed_observation_embeddings: torch.Tensor # v_hat (BatchSize, EmbeddingDim)
               ) -> torch.Tensor:
        """
        Computes the MSE loss between original and reconstructed observation embeddings.

        Args:
            original_observation_embeddings (torch.Tensor): The original observation embeddings (v).
            reconstructed_observation_embeddings (torch.Tensor): The reconstructed embeddings (v_hat)
                                                                 after passing through R, R^-1, and pi_v.
        Returns:
            torch.Tensor: The computed cycle-consistency loss.
        """
        if original_observation_embeddings.shape != reconstructed_observation_embeddings.shape:
            raise ValueError(
                f"Shape mismatch for original ({original_observation_embeddings.shape}) "
                f"and reconstructed ({reconstructed_observation_embeddings.shape}) embeddings."
            )
        return self.mse_loss(reconstructed_observation_embeddings, original_observation_embeddings)


class MaskedVisualReconstructionLoss(nn.Module):
    """
    Calculates the loss for masked visual reconstruction (e.g., for MAE-style pre-training).
    Computes loss (e.g., MSE) only on the masked patches/features.
    """
    def __init__(self, loss_type: str = 'mse', reduction: str = 'mean'):
        """
        Args:
            loss_type (str): Type of loss to apply ('mse' or 'l1'). Default: 'mse'.
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. Default: 'mean'.
        """
        super().__init__()
        if loss_type.lower() == 'mse':
            self.pixel_wise_loss = nn.MSELoss(reduction='none') # Calculate element-wise, then mask and reduce
        elif loss_type.lower() == 'l1':
            self.pixel_wise_loss = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported loss_type for MaskedVisualReconstructionLoss: {loss_type}")
        
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Unsupported reduction type: {reduction}")
        self.reduction = reduction

    def forward(self,
                predicted_features: torch.Tensor, # (B, NumTotalPatches, FeatureDim) - predictions for ALL patches
                target_features: torch.Tensor,    # (B, NumTotalPatches, FeatureDim) - ground truth for ALL patches
                mask: torch.Tensor                # (B, NumTotalPatches) - boolean, True for MASKED patches (where loss is computed)
               ) -> torch.Tensor:
        """
        Computes the reconstruction loss on the masked patches.

        Args:
            predicted_features (torch.Tensor): The features predicted by the decoder for all patches.
            target_features (torch.Tensor): The ground truth features for all patches.
            mask (torch.Tensor): A boolean mask indicating which patches were originally masked
                                 (and thus should be included in the loss calculation).
                                 Shape (BatchSize, NumTotalPatches), True for masked.
        Returns:
            torch.Tensor: The computed masked reconstruction loss.
        """
        if not (predicted_features.shape == target_features.shape):
            raise ValueError(f"Shape mismatch for predicted ({predicted_features.shape}) and target ({target_features.shape}) features.")
        if not (predicted_features.shape[:2] == mask.shape): # Batch and NumPatches must match
             raise ValueError(f"Shape mismatch for features ({predicted_features.shape[:2]}) and mask ({mask.shape}).")

        # Calculate loss for all patches (element-wise or feature-wise)
        loss_all_patches = self.pixel_wise_loss(predicted_features, target_features) # (B, NumTotalPatches, FeatureDim)
        
        # Average loss over the feature dimension if it's multi-dimensional (e.g. reconstructing patch features)
        if loss_all_patches.ndim > mask.ndim : # If loss is (B, N, D_feat) and mask is (B, N)
            loss_all_patches = loss_all_patches.mean(dim=-1) # (B, NumTotalPatches)

        # Apply mask: only consider loss for masked patches
        masked_loss = loss_all_patches * mask.float() # Element-wise product, only keeps loss for masked regions

        # Reduce the loss
        if self.reduction == 'mean':
            # Mean over only the masked patches. Denominator is number of True in mask.
            num_masked_elements = torch.sum(mask)
            if num_masked_elements == 0: # Avoid division by zero if no patches were masked (should not happen in MAE)
                return torch.tensor(0.0, device=predicted_features.device, requires_grad=predicted_features.requires_grad)
            return torch.sum(masked_loss) / num_masked_elements
        elif self.reduction == 'sum':
            return torch.sum(masked_loss)
        elif self.reduction == 'none':
            return masked_loss # Return loss per masked patch (or zero for unmasked)
        else: # Should not be reached due to init check
            return torch.tensor(0.0, device=predicted_features.device)


class CombinedAgentLoss(nn.Module):
    """
    Combines multiple losses, e.g., navigation loss and cycle-consistency loss for LTM.
    Can be extended to include visual reconstruction loss if needed for a specific training phase.
    """
    def __init__(self,
                 navigation_loss_weight: float = 1.0,
                 cycle_consistency_loss_weight: float = 0.1,
                 visual_reconstruction_loss_weight: float = 0.0, # Default to 0, enable if used
                 nav_loss_ignore_index: int = -100,
                 visual_recon_loss_type: str = 'mse'):
        super().__init__()
        self.navigation_loss_fn = NavigationLoss(ignore_index=nav_loss_ignore_index)
        self.cycle_consistency_loss_fn = CycleConsistencyLoss()
        self.visual_reconstruction_loss_fn = MaskedVisualReconstructionLoss(loss_type=visual_recon_loss_type)

        self.navigation_loss_weight = navigation_loss_weight
        self.cycle_consistency_loss_weight = cycle_consistency_loss_weight
        self.visual_reconstruction_loss_weight = visual_reconstruction_loss_weight

    def forward(self,
                # For Navigation Loss
                predicted_action_logits: Optional[torch.Tensor] = None,
                ground_truth_action_ids: Optional[torch.Tensor] = None,
                # For Cycle Consistency Loss
                original_observation_embeddings_for_cycle: Optional[torch.Tensor] = None,
                reconstructed_observation_embeddings_for_cycle: Optional[torch.Tensor] = None,
                # For Masked Visual Reconstruction Loss
                predicted_visual_features_for_recon: Optional[torch.Tensor] = None,
                target_visual_features_for_recon: Optional[torch.Tensor] = None,
                visual_recon_mask: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the combined loss.
        """
        loss_device = None
        if predicted_action_logits is not None: loss_device = predicted_action_logits.device
        elif original_observation_embeddings_for_cycle is not None: loss_device = original_observation_embeddings_for_cycle.device
        elif predicted_visual_features_for_recon is not None: loss_device = predicted_visual_features_for_recon.device
        else: loss_device = torch.device('cpu') # Fallback

        total_loss = torch.tensor(0.0, device=loss_device)
        loss_components: Dict[str, float] = {}

        # Navigation Loss
        if self.navigation_loss_weight > 0 and \
           predicted_action_logits is not None and \
           ground_truth_action_ids is not None:
            nav_loss = self.navigation_loss_fn(predicted_action_logits, ground_truth_action_ids)
            total_loss += self.navigation_loss_weight * nav_loss
            loss_components['navigation_loss'] = nav_loss.item()
        else:
            loss_components['navigation_loss'] = 0.0

        # Cycle Consistency Loss
        if self.cycle_consistency_loss_weight > 0 and \
           original_observation_embeddings_for_cycle is not None and \
           reconstructed_observation_embeddings_for_cycle is not None:
            cycle_loss = self.cycle_consistency_loss_fn(
                original_observation_embeddings_for_cycle,
                reconstructed_observation_embeddings_for_cycle
            )
            total_loss += self.cycle_consistency_loss_weight * cycle_loss
            loss_components['cycle_consistency_loss'] = cycle_loss.item()
        else:
            loss_components['cycle_consistency_loss'] = 0.0
        
        # Masked Visual Reconstruction Loss
        if self.visual_reconstruction_loss_weight > 0 and \
           predicted_visual_features_for_recon is not None and \
           target_visual_features_for_recon is not None and \
           visual_recon_mask is not None:
            vis_recon_loss = self.visual_reconstruction_loss_fn(
                predicted_visual_features_for_recon,
                target_visual_features_for_recon,
                visual_recon_mask
            )
            total_loss += self.visual_reconstruction_loss_weight * vis_recon_loss
            loss_components['visual_reconstruction_loss'] = vis_recon_loss.item()
        else:
            loss_components['visual_reconstruction_loss'] = 0.0
            
        loss_components['total_loss'] = total_loss.item()
        return total_loss, loss_components


if __name__ == '__main__':
    print("--- Testing Loss Functions (with MaskedVisualReconstructionLoss) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test NavigationLoss (same as before)
    # ...

    # Test CycleConsistencyLoss (same as before)
    # ...

    # Test MaskedVisualReconstructionLoss
    print("\nTesting MaskedVisualReconstructionLoss...")
    batch_size_mae, num_patches_mae, feature_dim_mae = 2, 196, 256 # Example: 14x14 patches
    mae_loss_fn = MaskedVisualReconstructionLoss(loss_type='mse', reduction='mean')

    predicted_feats = torch.randn(batch_size_mae, num_patches_mae, feature_dim_mae, device=device)
    target_feats = torch.randn(batch_size_mae, num_patches_mae, feature_dim_mae, device=device)
    
    # Create a mask where ~75% of patches are masked (True for masked)
    mask_mae = torch.rand(batch_size_mae, num_patches_mae, device=device) > 0.25 
    if mask_mae.sum() == 0: # Ensure some are masked for test
        mask_mae[0,0] = True

    loss_mae = mae_loss_fn(predicted_feats, target_feats, mask_mae)
    print(f"  Masked Visual Reconstruction Loss (MSE): {loss_mae.item()}")
    assert loss_mae.ndim == 0
    assert loss_mae.item() >= 0

    mae_loss_fn_l1 = MaskedVisualReconstructionLoss(loss_type='l1', reduction='sum')
    loss_mae_l1_sum = mae_loss_fn_l1(predicted_feats, target_feats, mask_mae)
    print(f"  Masked Visual Reconstruction Loss (L1, sum): {loss_mae_l1_sum.item()}")
    assert loss_mae_l1_sum.ndim == 0


    # Test CombinedAgentLoss with visual reconstruction
    print("\nTesting CombinedAgentLoss with Visual Reconstruction...")
    combined_loss_fn_full = CombinedAgentLoss(
        navigation_loss_weight=1.0,
        cycle_consistency_loss_weight=0.5,
        visual_reconstruction_loss_weight=0.2, # Enable visual recon loss
        visual_recon_loss_type='mse'
    ).to(device)

    # Mock data for combined loss
    num_actions = 4
    logits_single = torch.randn(batch_size_mae, num_actions, device=device)
    labels_single = torch.randint(0, num_actions, (batch_size_mae,), device=device)
    original_v = torch.randn(batch_size_mae, feature_dim_mae, device=device) # Assuming d_emb = feature_dim_mae for cycle loss
    reconstructed_v_hat = original_v + torch.randn_like(original_v) * 0.1

    total_loss_full, components_full = combined_loss_fn_full(
        predicted_action_logits=logits_single,
        ground_truth_action_ids=labels_single,
        original_observation_embeddings_for_cycle=original_v,
        reconstructed_observation_embeddings_for_cycle=reconstructed_v_hat,
        predicted_visual_features_for_recon=predicted_feats,
        target_visual_features_for_recon=target_feats,
        visual_recon_mask=mask_mae
    )
    print(f"  Combined Total Loss (all active): {total_loss_full.item()}")
    print(f"  Loss Components (all active): {components_full}")
    assert components_full['navigation_loss'] > 0
    assert components_full['cycle_consistency_loss'] > 0
    assert components_full['visual_reconstruction_loss'] > 0

    print("\ntraining_utils/losses.py tests completed with MAE loss.")