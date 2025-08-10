import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from PIL import Image 
try:
    from external_models.unidepth.models import UniDepthV2 
    from external_models.unidepth.utils.misc import load_pretrained as load_unidepth_config #  Use its own config (if available)
    ACTUAL_UNIDEPTH_IMPORTED = True
except ImportError as e:
    print(f"Warning: Failed to import actual UniDepth model: {e}. Using mock UniDepth.")    
    ACTUAL_UNIDEPTH_IMPORTED = False

    class UniDepthV2(nn.Module):   
        def __init__(self, config_path: Optional[str] = None, pretrained_path: Optional[str] = None, 
                     internal_feature_dim: int = 256, **kwargs):
            super().__init__()
            self.config_path = config_path
            self.pretrained_path = pretrained_path
            self.internal_feature_dim = internal_feature_dim
            self.depth_prediction_head = nn.Conv2d(3, 1, kernel_size=3, padding=1)
            self.mock_camera_encoder = nn.Linear(10, 64)
            self.mock_feature_extractor_f = nn.AdaptiveAvgPool2d((1,1))
            print(f"MockUniDepth: Initialized. Config: {config_path}, Pretrained: {pretrained_path}")

        def forward(self, rgb_image_batch: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
            dense_depth_map = torch.sigmoid(self.depth_prediction_head(rgb_image_batch)) * 10.0
            batch_size = rgb_image_batch.shape[0]
            mock_cam_input = torch.randn(batch_size, 10, device=rgb_image_batch.device)
            camera_embedding_c_t = self.mock_camera_encoder(mock_cam_input)
            features_f_t = torch.randn(batch_size, self.internal_feature_dim, device=rgb_image_batch.device)

            return {
                'depth': dense_depth_map,       # Predicted depth (B, 1, H, W)
                'cam_embed': camera_embedding_c_t, 
                'aux_feats': features_f_t     
            }


class VisualFrontend(nn.Module):
    """
    Visual frontend to extract RGB features (v_t^RGB).
    Consists of a ResNet-50 backbone followed by a Vision Transformer (ViT).
    Includes a masked autoencoder (MAE) style reconstruction head for Phase 1 pre-training.
    """
    def __init__(self,
                 vit_output_dim: int,
                 mae_reconstruction_dim: int = 768, 
                 resnet_arch: str = "resnet50",
                 resnet_pretrained: bool = True,
                 vit_patch_size: int = 16, # Standard ViT patch size
                 vit_embed_dim: int = 768, 
                 vit_depth: int = 6,       
                 vit_num_heads: int = 8,
                 vit_mlp_ratio: float = 4.0,
                 img_size: int = 224):
        super().__init__()
        self.vit_output_dim = vit_output_dim
        self.mae_reconstruction_dim = mae_reconstruction_dim
        self.img_size = img_size
        self.patch_size = vit_patch_size
        self.num_patches = (img_size //vit_patch_size) ** 2
        self.vit_embed_dim = vit_embed_dim

        
        if resnet_arch == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if resnet_pretrained else None)
        elif resnet_arch == "resnet34":
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if resnet_pretrained else None)
        else:
            raise ValueError(f"Unsupported ResNet architecture: {resnet_arch}")
        
        self.resnet_patch_extractor = nn.Sequential(*list(resnet.children())[:-3]) 
        self.resnet_feature_dim = 1024 if resnet_arch == "resnet50" else 256
        
 
        self.patch_projection = nn.Conv2d(self.resnet_feature_dim, vit_embed_dim, kernel_size=1)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, vit_embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, vit_embed_dim)) # +1 for CLS token
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vit_embed_dim,
            nhead=vit_num_heads,
            dim_feedforward=int(vit_embed_dim * vit_mlp_ratio),
            dropout=0.1, # Standard dropout
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=vit_depth)
        self.norm = nn.LayerNorm(vit_embed_dim)

        # MLP head for v_rgb output (from CLS token)
        self.rgb_feature_head = nn.Linear(vit_embed_dim, vit_output_dim)

        self.mae_decoder_embed = nn.Linear(vit_embed_dim, vit_embed_dim, bias=True) # To process visible patch embeddings
        self.mae_mask_token = nn.Parameter(torch.zeros(1, 1, vit_embed_dim))
        mae_decoder_layer = nn.TransformerEncoderLayer(
            d_model=vit_embed_dim, nhead=vit_num_heads,
            dim_feedforward=int(vit_embed_dim * mlp_ratio),
            dropout=0.1, batch_first=True
        )
        self.mae_decoder = nn.TransformerEncoder(mae_decoder_layer, num_layers=max(1, vit_depth // 2))
        self.mae_decoder_norm = nn.LayerNorm(vit_embed_dim)
        self.mae_decoder_pred = nn.Linear(vit_embed_dim, self.mae_reconstruction_dim, bias=True) # Predict ResNet patch features

        self.image_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self._initialize_weights()

    def _initialize_weights(self):

        nn.init.normal_(self.pos_embed, std=.02)
        nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.mae_mask_token, std=.02)
        # Initialize MAE decoder pred linear layer
        nn.init.xavier_uniform_(self.mae_decoder_pred.weight)
        nn.init.constant_(self.mae_decoder_pred.bias, 0)


    def _extract_patches_resnet(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts patch features using ResNet backbone and projects them."""
        x = self.resnet_patch_extractor(x)  # (B, C_resnet, H_img/16, W_img/16)
        x = self.patch_projection(x)      # (B, vit_embed_dim, H_img/16, W_img/16)
        # Flatten: (B, vit_embed_dim, NumPatches) -> (B, NumPatches, vit_embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x

    def forward_encoder(self, x: torch.Tensor, visible_indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes visible patches.
        Args:
            x (torch.Tensor): Input images (B, C, H, W).
            visible_indices (Optional[torch.Tensor]): (B, NumVisiblePatches) indices of visible patches for MAE.
                                                      If None, all patches are visible (standard ViT).
        Returns:
            Tuple:
                - x_visible_with_cls (torch.Tensor): Embeddings of visible patches + CLS token (B, NumVisible+1, D_vit).
                - full_sequence_indices (torch.Tensor): Indices for restoring full sequence (B, NumPatches).
                - cls_token_output (torch.Tensor): Output of CLS token after encoder (B, D_vit). Used for v_rgb.
        """
        patches = self._extract_patches_resnet(x) # (B, NumPatches, D_vit)
        batch_size, num_patches, _ = patches.shape

        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # (B, 1, D_vit)
        
        if visible_indices is not None: # MAE-style encoding
            # Gather only visible patches
            patches_visible = torch.gather(patches, dim=1, index=visible_indices.unsqueeze(-1).expand(-1, -1, self.vit_embed_dim))
            # Prepend CLS token to visible patches
            x_visible_with_cls = torch.cat((cls_tokens, patches_visible), dim=1)
            # Add positional embedding only to visible tokens + CLS
            pos_embed_visible = torch.cat([
                self.pos_embed[:, :1, :], 
                torch.gather(self.pos_embed[:, 1:, :].expand(batch_size, -1, -1), # Patch pos_embeds
                             dim=1, index=visible_indices.unsqueeze(-1).expand(-1, -1, self.vit_embed_dim))
            ], dim=1)
            x_visible_with_cls = x_visible_with_cls + pos_embed_visible
            
            full_sequence_indices = torch.arange(num_patches, device=x.device).unsqueeze(0).expand(batch_size, -1) # Placeholder

        else: 
            x_visible_with_cls = torch.cat((cls_tokens, patches), dim=1)
            x_visible_with_cls = x_visible_with_cls + self.pos_embed # Add pos_embed to all
            full_sequence_indices = torch.arange(num_patches, device=x.device).unsqueeze(0).expand(batch_size, -1) # All indices

        encoded_sequence = self.transformer_encoder(x_visible_with_cls) # (B, NumVisible+1, D_vit)
        encoded_sequence = self.norm(encoded_sequence)
        
        cls_token_output = encoded_sequence[:, 0] # Output of CLS token (B, D_vit)
        
        return encoded_sequence, full_sequence_indices, cls_token_output


    def forward_mae_decoder(self, x_visible_encoded: torch.Tensor, 
                            full_sequence_indices: torch.Tensor, # (B, NumPatches) - not used in this simplified decoder
                            visible_indices: torch.Tensor # (B, NumVisiblePatches)
                           ) -> torch.Tensor:
        """
        MAE decoder: takes encoded visible patches, adds mask tokens, and reconstructs.
        Simplified version. A full MAE has more elaborate index shuffling.
        """
        # x_visible_encoded includes CLS token at [:,0,:]. We use patch embeddings [:,1:,:]
        encoded_visible_patches = x_visible_encoded[:, 1:, :] # (B, NumVisible, D_vit)
        batch_size, num_visible, embed_dim = encoded_visible_patches.shape
        
        num_masked_patches = self.num_patches - num_visible
        mask_tokens = self.mae_mask_token.expand(batch_size, num_masked_patches, -1) # (B, NumMasked, D_vit)


        decoder_input_full = torch.cat([encoded_visible_patches, mask_tokens], dim=1) # (B, NumPatches, D_vit)
        

        decoder_input_full = decoder_input_full + self.pos_embed[:, 1:, :].expand(batch_size, -1, -1)

        decoded_patches = self.mae_decoder(decoder_input_full)
        decoded_patches = self.mae_decoder_norm(decoded_patches)
        
        reconstructed_patches = self.mae_decoder_pred(decoded_patches) # (B, NumPatches, mae_reconstruction_dim)

        return reconstructed_patches


    def forward(self, rgb_image_batch: torch.Tensor) -> torch.Tensor:
        """ Standard forward pass for v_t^RGB feature extraction. """
        _, _, cls_token_output = self.forward_encoder(rgb_image_batch, visible_indices=None)
        v_rgb = self.rgb_feature_head(cls_token_output)
        return v_rgb

    def forward_reconstruction(self, rgb_image_batch: torch.Tensor, 
                               mask_ratio: float = 0.75
                               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for MAE-style masked reconstruction (Phase 1 pre-training).
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - reconstructed_patches (B, NumPatches, mae_reconstruction_dim): Output of MAE decoder.
                - target_patches (B, NumPatches, mae_reconstruction_dim): Original patch features to reconstruct.
                - reconstruction_mask (B, NumPatches): Boolean mask, True for patches that were masked and should be predicted.
        """
        # 1. Get original patch features (these will be the reconstruction target)
        original_patches_for_target = self._extract_patches_resnet(rgb_image_batch) # (B, NumPatches, D_vit)
 
        if self.mae_reconstruction_dim != self.vit_embed_dim:
            print(f"Warning/TODO: MAE reconstruction dim ({self.mae_reconstruction_dim}) "
                  f"differs from ViT embed dim ({self.vit_embed_dim}). "
                  "Ensure target patches for MAE are correctly defined.")
        target_patches = original_patches_for_target # (B, NumPatches, D_vit)

        # 2. Generate mask: select visible and masked patches
        batch_size, num_total_patches, _ = original_patches_for_target.shape
        num_visible_patches = int(num_total_patches * (1 - mask_ratio))
        
        # Randomly shuffle indices and pick visible ones
        noise = torch.rand(batch_size, num_total_patches, device=rgb_image_batch.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        visible_indices = ids_shuffle[:, :num_visible_patches] # (B, NumVisible)
        masked_indices = ids_shuffle[:, num_visible_patches:]   # (B, NumMasked)

        # 3. Encode only visible patches
        encoded_sequence_visible, _, _ = self.forward_encoder(rgb_image_batch, visible_indices=visible_indices)
        
        # 4. Decode using MAE decoder (predicts features for all patches, including masked ones)
        reconstructed_all_patches = self.forward_mae_decoder(encoded_sequence_visible, ids_shuffle, visible_indices)

        # 5. Prepare mask for loss calculation (True for patches that were masked)
        reconstruction_mask = torch.zeros(batch_size, num_total_patches, dtype=torch.bool, device=rgb_image_batch.device)
        reconstruction_mask.scatter_(dim=1, index=masked_indices, value=True)
        
        return reconstructed_all_patches, target_patches, reconstruction_mask


class DepthFeatureNetwork(nn.Module):
    """
    Computes v_t^Depth = MLP(CA(F_t, C_t))
    F_t: Auxiliary features from UniDepth's depth module.
    C_t: Camera embedding from UniDepth.
    """
    def __init__(self, f_t_dim: int, c_t_dim: int, output_dim: int, 
                 num_attn_heads: int = 4, mlp_hidden_mult: int = 2):
        super().__init__()
        self.f_t_dim = f_t_dim
        self.c_t_dim = c_t_dim
        self.output_dim = output_dim

        combined_dim = f_t_dim + c_t_dim
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, combined_dim * mlp_hidden_mult),
            nn.ReLU(),
            nn.Linear(combined_dim * mlp_hidden_mult, output_dim)
        )
        # If using MHA:
        # self.output_mlp = nn.Linear(f_t_dim, output_dim) 


    def forward(self, features_f_t: torch.Tensor, camera_embedding_c_t: torch.Tensor) -> torch.Tensor:

        combined_features = torch.cat([features_f_t, camera_embedding_c_t], dim=1)
        v_depth = self.mlp(combined_features)
        return v_depth
        



class MultimodalFeatureProcessor(nn.Module):
    """
    Processes RGB panoramas to extract fused multimodal features for Mem4Nav.
    Combines a visual frontend for RGB features and UniDepth for depth features.
    """
    def __init__(self,
                 visual_frontend_output_dim: int,
                 unidepth_config_path: str, # Path to UniDepth's own config JSON
                 unidepth_model_weights_path: str, # Path to UniDepth's .pth weights
                 unidepth_output_depth_feats_dim: int, # This is v_t^Depth's dimension
                 device: Optional[torch.device] = None,
                 vf_resnet_arch: str = "resnet50", # For VisualFrontend
                 vf_vit_embed_dim: int = 768, # For VisualFrontend
                 vf_vit_depth: int = 6,       # For VisualFrontend
                 vf_vit_heads: int = 8        # For VisualFrontend
                 ):
        super().__init__()
        self.device = device or torch.device('cpu')
        
        self.visual_frontend = VisualFrontend(
            vit_output_dim=visual_frontend_output_dim,
            resnet_arch=vf_resnet_arch,
            vit_embed_dim=vf_vit_embed_dim,
            vit_depth=vf_vit_depth,
            vit_num_heads=vf_vit_heads,
            # mae_reconstruction_dim will be vit_embed_dim for simplicity now
            mae_reconstruction_dim=vf_vit_embed_dim 
        ).to(self.device)
        
        if ACTUAL_UNIDEPTH_IMPORTED:
            # Load UniDepth config
            unidepth_cfg_dict = load_unidepth_config(unidepth_config_path) # UniDepth's own config loader

            self.unidepth_model = UniDepthV2(config=unidepth_cfg_dict).to(self.device) # Pass the dict
            
            print(f"Loading UniDepth weights from: {unidepth_model_weights_path}")
            try:
                checkpoint = torch.load(unidepth_model_weights_path, map_location='cpu')
                # UniDepth often stores weights under 'model' key, sometimes 'state_dict'
                weights_to_load = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
                self.unidepth_model.load_state_dict(weights_to_load)
                print("UniDepth weights loaded successfully.")
            except Exception as e:
                print(f"Error loading UniDepth weights from {unidepth_model_weights_path}: {e}. Model might be random.")
            self.unidepth_model.eval() # Set to eval mode
        else: # Use mock
            self.unidepth_model = UniDepthV2(   
                config_path=unidepth_config_path, # Mock just prints this
                pretrained_path=unidepth_model_weights_path,
                internal_feature_dim=unidepth_output_depth_feats_dim # For consistency with mock
            ).to(self.device)

        # Network to compute v_t^Depth = MLP(CA(F_t, C_t))
        mock_f_t_dim = self.config.get('unidepth_ft_dim_from_model', 256) 
        mock_c_t_dim = self.config.get('unidepth_ct_dim_from_model', 64)  
        self.depth_feature_network = DepthFeatureNetwork(
            f_t_dim=mock_f_t_dim, 
            c_t_dim=mock_c_t_dim, 
            output_dim=unidepth_output_depth_feats_dim
        ).to(self.device)

        self.fused_embedding_dim = visual_frontend_output_dim + unidepth_output_depth_feats_dim

    def process_panorama(self, rgb_panorama_pil_or_tensor: Any, 
                         unidepth_input_extras: Optional[Dict] = None) -> \
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, Any]]]:
        """
        Processes a single RGB panorama image.
        Args:
            rgb_panorama_pil_or_tensor: Input RGB panorama (e.g., PIL Image or pre-transformed Tensor (C,H,W)).
            unidepth_input_extras (Optional[Dict]): Extra inputs for UniDepth if needed (e.g., intrinsics as tensor).
        Returns:
            Tuple (batch size is 1 for all tensors):
                - fused_multimodal_embedding (torch.Tensor): v_t = [v_rgb; v_depth] (1, D_fused)
                - dense_depth_map (torch.Tensor): (1, 1, H_depth, W_depth) from UniDepth
                - point_cloud (Optional[torch.Tensor]): (N, 3) unprojected from depth (placeholder for now)
                - additional_outputs (Dict): {'cam_embed_c_t': Tensor, 'v_rgb': Tensor, 'v_depth': Tensor}
        """
        if not isinstance(rgb_panorama_pil_or_tensor, torch.Tensor):
            vf_input_tensor = self.visual_frontend.image_transform(rgb_panorama_pil_or_tensor).unsqueeze(0).to(self.device)
        else:
            vf_input_tensor = rgb_panorama_pil_or_tensor.unsqueeze(0).to(self.device) if rgb_panorama_pil_or_tensor.ndim == 3 else rgb_panorama_pil_or_tensor.to(self.device)
        
        # Make sure it's B,C,H,W where B=1
        if vf_input_tensor.ndim != 4 or vf_input_tensor.shape[1] != 3:
            raise ValueError(f"Input tensor for visual_frontend must be (B,3,H,W) or (3,H,W), got {vf_input_tensor.shape}")
        if vf_input_tensor.shape[0] != 1:
             print(f"Warning: MFP expects batch size 1 for panorama processing, got {vf_input_tensor.shape[0]}. Processing first item.")
             vf_input_tensor = vf_input_tensor[0:1]


        with torch.no_grad(): # Assuming inference for feature extraction
            v_rgb = self.visual_frontend(vf_input_tensor) # (1, D_rgb)

            # Prepare input for UniDepth (expects B,3,H,W, often normalized differently or specific size)
            unidepth_input_dict = {'image': vf_input_tensor}
            if unidepth_input_extras:
                unidepth_input_dict.update(unidepth_input_extras)
            
            # Actual UniDepth forward pass
            # UniDepthV2 returns a dict: {'depth': tensor, 'confidence': tensor, 'cam_embed': tensor, 'aux_feats': tensor, ...}
            unidepth_output = self.unidepth_model(unidepth_input_dict['image'], **unidepth_input_dict) # Pass image and any extras
            
            dense_depth_map = unidepth_output['depth']      # (B, 1, H, W)
            camera_embedding_c_t = unidepth_output.get('cam_embed') # (B, D_cam)
            features_f_t = unidepth_output.get('aux_feats')       # (B, D_f)

            if camera_embedding_c_t is None or features_f_t is None:
                print("Warning: UniDepth output missing 'cam_embed' or 'aux_feats'. Using zeros for v_depth.")
                # Use placeholder dimensions if real ones are not available yet
                # These should match DepthFeatureNetwork input dims
                d_depth_out_dim = self.depth_feature_network.output_dim 
                v_depth = torch.zeros(v_rgb.shape[0], d_depth_out_dim, device=self.device)
                if camera_embedding_c_t is None: camera_embedding_c_t = torch.zeros(v_rgb.shape[0], self.depth_feature_network.c_t_dim, device=self.device)
            else:
                # Ensure dimensions match what DepthFeatureNetwork expects
                if features_f_t.shape[1] != self.depth_feature_network.f_t_dim or \
                   camera_embedding_c_t.shape[1] != self.depth_feature_network.c_t_dim:

                    print(f"CRITICAL Dimension Mismatch for DepthFeatureNetwork: "
                          f"F_t expected {self.depth_feature_network.f_t_dim}, got {features_f_t.shape[1]}. "
                          f"C_t expected {self.depth_feature_network.c_t_dim}, got {camera_embedding_c_t.shape[1]}. "
                          f"Using zeros for v_depth.")
                    v_depth = torch.zeros(v_rgb.shape[0], self.depth_feature_network.output_dim, device=self.device)
                else:
                    v_depth = self.depth_feature_network(features_f_t, camera_embedding_c_t) # (B, D_depth_out)
            
            fused_multimodal_embedding = torch.cat([v_rgb, v_depth], dim=1)
        
        additional_outputs = {
            'cam_embed_c_t': camera_embedding_c_t.detach() if camera_embedding_c_t is not None else None,
            'v_rgb': v_rgb.detach(),
            'v_depth': v_depth.detach()
        }
        # Point cloud generation still separate
        return fused_multimodal_embedding, dense_depth_map.detach(), None, additional_outputs


def unproject_depth_to_pointcloud(depth_map: torch.Tensor, 
                                  camera_intrinsics: Dict[str, float], # fx, fy, cx, cy
                                  depth_scale: float = 1.0,
                                  max_depth_val: float = 100.0, # To filter out very far points
                                  min_depth_val: float = 0.1   # To filter out too close/invalid points
                                 ) -> torch.Tensor:
    """
    Unprojects a depth map to a 3D point cloud.
    Assumes depth_map is (H, W) or (1, H, W) from a single image.
    """
    if depth_map.ndim == 3 and depth_map.shape[0] == 1: # (1, H, W)
        depth_map = depth_map.squeeze(0) # (H, W)
    elif depth_map.ndim != 2:
        raise ValueError(f"Depth map must be (H,W) or (1,H,W), got {depth_map.shape}")
    
    H, W = depth_map.shape
    device = depth_map.device

    fx = camera_intrinsics['fx']
    fy = camera_intrinsics['fy']
    cx = camera_intrinsics['cx']
    cy = camera_intrinsics['cy']

    u_coords = torch.arange(0, W, device=device, dtype=torch.float32)
    v_coords = torch.arange(0, H, device=device, dtype=torch.float32)
    v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing='ij') # HxW, HxW

    Z = depth_map * depth_scale # (H, W)

    valid_mask = (Z > min_depth_val) & (Z < max_depth_val) & torch.isfinite(Z)
    
    X = (u_grid[valid_mask] - cx) * Z[valid_mask] / fx
    Y = (v_grid[valid_mask] - cy) * Z[valid_mask] / fy
    Z_valid = Z[valid_mask]
    
    point_cloud = torch.stack([X, Y, Z_valid], dim=1) # (N_valid, 3)
    
    return point_cloud


if __name__ == '__main__':
    # This test requires actual UniDepth model and weights to run properly.
    # For now, it will use the mock if UniDepth is not correctly set up in external_models.
    print("--- Testing Feature Utils (with actual UniDepth if available) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a dummy config for MFP
    # CRITICAL: User must provide valid paths for unidepth_config_path and unidepth_model_weights_path
    # and ensure the unidepth_output_depth_feats_dim, mock_f_t_dim, mock_c_t_dim are consistent with their UniDepth model.
    # Example assumes UniDepth output aux_feats is 256D and cam_embed is 64D, and v_depth output is 128D.
    mock_mfp_config = {
        'visual_frontend_output_dim': 256,
        'unidepth_config_path': "path/to/your/unidepth_v2_vitl14.json", # Placeholder - e.g., from uploaded files
        'unidepth_model_weights_path': "path/to/your/unidepth_v2_vitl14_nodepth2mask.pth", # Placeholder
        'unidepth_output_depth_feats_dim': 128, # Dimension of v_depth
        'vf_resnet_arch': "resnet34", # Lighter ResNet for faster testing
        'vf_vit_depth': 3, 'vf_vit_heads': 4, 'vf_vit_embed_dim': 384, # Smaller ViT for testing
        # These should match the input dimensions of DepthFeatureNetwork
        'unidepth_ft_dim_from_model': 256, # Example dimension for F_t
        'unidepth_ct_dim_from_model': 64   # Example dimension for C_t
    }
    
    # --- Test VisualFrontend MAE part ---
    print("\nTesting VisualFrontend MAE...")
    vf = VisualFrontend(
        vit_output_dim=mock_mfp_config['visual_frontend_output_dim'],
        resnet_arch=mock_mfp_config['vf_resnet_arch'],
        vit_embed_dim=mock_mfp_config['vf_vit_embed_dim'],
        vit_depth=mock_mfp_config['vf_vit_depth'],
        vit_num_heads=mock_mfp_config['vf_vit_heads'],
        mae_reconstruction_dim=mock_mfp_config['vf_vit_embed_dim'] # Reconstruct to ViT embed dim
    ).to(device)
    vf.eval()
    dummy_img_batch = torch.randn(2, 3, 224, 224, device=device) # Batch of 2 images
    reconstructed, target, mask = vf.forward_reconstruction(dummy_img_batch, mask_ratio=0.75)
    print(f"  MAE reconstructed shape: {reconstructed.shape}") # (B, NumPatches, mae_reconstruction_dim)
    print(f"  MAE target shape: {target.shape}")         # (B, NumPatches, D_vit_embed)
    print(f"  MAE mask shape: {mask.shape}, True count: {mask.sum()}") # (B, NumPatches)
    assert reconstructed.shape[0] == 2 and reconstructed.shape[2] == mock_mfp_config['vf_vit_embed_dim']
    assert target.shape == reconstructed.shape
    assert mask.shape[0] == 2 and mask.shape[1] == vf.num_patches


    # --- Test MultimodalFeatureProcessor ---
    print("\nTesting MultimodalFeatureProcessor...")
    # Use try-except as this might fail if paths are not valid or UniDepth code has issues
    try:
        # Try to use one of the uploaded unidepth configs if paths are placeholders
        # This is a HACK for testing, user should provide correct paths in actual config.
        if mock_mfp_config['unidepth_config_path'] == "path/to/your/unidepth_v2_vitl14.json":
            # Check if any uploaded config exists
            if os.path.exists("configs/config_v2_vitl14.json"): # Check if it's in a root 'configs' dir
                 mock_mfp_config['unidepth_config_path'] = "configs/config_v2_vitl14.json"
            elif os.path.exists("external_models/unidepth/configs/config_v2_vitl14.json"):
                 mock_mfp_config['unidepth_config_path'] = "external_models/unidepth/configs/config_v2_vitl14.json"
            else: # Try to use a contentFetchId if available (won't work directly like this in script)
                print("Fallback: Provide actual UniDepth config path for full test.")


        processor = MultimodalFeatureProcessor(**mock_mfp_config, device=device)
        processor.eval()
        
        dummy_pil_image = Image.new('RGB', (320, 240), color='red')
        
        fused_emb, depth_map, _, add_outputs = processor.process_panorama(dummy_pil_image)
        print(f"  Processor fused_embedding shape: {fused_emb.shape}")
        print(f"  Processor depth_map shape: {depth_map.shape}")
        print(f"  Processor v_rgb shape: {add_outputs['v_rgb'].shape}")
        print(f"  Processor v_depth shape: {add_outputs['v_depth'].shape}")

        expected_fused_dim = mock_mfp_config['visual_frontend_output_dim'] + mock_mfp_config['unidepth_output_depth_feats_dim']
        assert fused_emb.shape == (1, expected_fused_dim)
        assert add_outputs['v_rgb'].shape == (1, mock_mfp_config['visual_frontend_output_dim'])
        assert add_outputs['v_depth'].shape == (1, mock_mfp_config['unidepth_output_depth_feats_dim'])
        assert depth_map.ndim == 4 and depth_map.shape[0]==1 and depth_map.shape[1]==1 # B,1,H,W

        # Test unprojection (using depth map from processor)
        test_depth_map_mfp = depth_map.squeeze(0) # (1,H,W) -> (H,W)
        mock_intrinsics_mfp = {'fx': 250.0, 'fy': 250.0, 'cx': depth_map.shape[3]/2 - 0.5, 'cy': depth_map.shape[2]/2 - 0.5}
        
        point_cloud_mfp = unproject_depth_to_pointcloud(test_depth_map_mfp, mock_intrinsics_mfp)
        print(f"  Generated point cloud shape from MFP depth: {point_cloud_mfp.shape}")
        if point_cloud_mfp.numel() > 0:
            assert point_cloud_mfp.ndim == 2 and point_cloud_mfp.shape[1] == 3

    except Exception as e:
        print(f"ERROR during MultimodalFeatureProcessor test (likely due to UniDepth path/config): {e}")
        import traceback
        traceback.print_exc()

    print("\nmem4nav_core/perception_processing/feature_utils.py tests/updates completed.")