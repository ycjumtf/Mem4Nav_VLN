import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image # For type hinting and potential pre-processing


try:
    from mem4nav_core.perception_processing.feature_utils import MultimodalFeatureProcessor, unproject_depth_to_pointcloud
except ImportError:
    print("Warning: PerceptionModule using placeholder for MultimodalFeatureProcessor due to import error.")
    class MultimodalFeatureProcessor(nn.Module): # type: ignore
        def __init__(self, visual_frontend_output_dim: int, unidepth_model_path: Optional[str],
                     unidepth_internal_feature_dim: int, device: Optional[torch.device]):
            super().__init__()
            self.device = device or torch.device('cpu')
            self.fused_embedding_dim = visual_frontend_output_dim + unidepth_internal_feature_dim
            print(f"Placeholder MFP initialized with fused_dim: {self.fused_embedding_dim}")
        def process_panorama(self, rgb_panorama_pil_or_tensor: Any) -> \
                Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
            # Mock output
            batch_size = 1
            if isinstance(rgb_panorama_pil_or_tensor, torch.Tensor) and rgb_panorama_pil_or_tensor.ndim == 4:
                batch_size = rgb_panorama_pil_or_tensor.shape[0]
            
            fused_emb = torch.randn(batch_size, self.fused_embedding_dim, device=self.device)
            depth_map = torch.rand(batch_size, 1, 512, 512, device=self.device) # Mock depth map H, W for UniDepth
            cam_embed = torch.randn(batch_size, 64, device=self.device) # Mock camera embedding
            return fused_emb, depth_map, cam_embed, None # No point cloud from mock

    def unproject_depth_to_pointcloud(depth_map: torch.Tensor, 
                                      camera_intrinsics: Dict[str, float],
                                      depth_scale: float = 1.0) -> torch.Tensor:
        print("Warning: Using placeholder for unproject_depth_to_pointcloud.")
        if depth_map.ndim == 4: # B, C, H, W
            depth_map = depth_map.squeeze(1) # B, H, W
        
        # Return a dummy point cloud of shape (Batch, Num_points, 3)
        # or just one (Num_points, 3) if batching is handled outside
        num_dummy_points = 100 
        if depth_map.ndim == 3: # Batched depth maps
            return torch.randn(depth_map.shape[0], num_dummy_points, 3, device=depth_map.device)
        else: # Single depth map
            return torch.randn(num_dummy_points, 3, device=depth_map.device)


class PerceptionModule(nn.Module):
    """
    Handles visual input processing for the Modular Pipeline Agent.
    It uses the MultimodalFeatureProcessor to extract fused embeddings (v_t)
    and other relevant information like depth maps or point clouds from raw observations.
    """
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__()
        self.config = config.get('perception', {}) # Specific config for this module
        self.device = device

        processor_config = self.config.get('multimodal_feature_processor', {})
        self.feature_processor = MultimodalFeatureProcessor(
            visual_frontend_output_dim=processor_config.get('vf_output_dim', 256),
            unidepth_model_path=processor_config.get('unidepth_model_path'), # Critical path
            unidepth_internal_feature_dim=processor_config.get('unidepth_internal_feature_dim', 128),
            device=self.device
        )
        
        # Camera intrinsics: can be fixed, or part of observation, or from config
        # These are needed for unprojecting depth to point cloud.
        self.camera_intrinsics = processor_config.get('camera_intrinsics')
        if self.camera_intrinsics is None:
            print("Warning: PerceptionModule camera_intrinsics not provided in config. Using default mock values.")
            self.camera_intrinsics = {'fx': 256.0, 'fy': 256.0, 'cx': 255.5, 'cy': 255.5} # Example for 512x512 output
        
        self.generate_point_cloud = self.config.get('generate_point_cloud', False) # Whether to output point cloud

    def process_observation(self, raw_observation: Dict[str, Any]) -> \
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, Any]]]:
        """
        Processes a raw observation from the environment.

        Args:
            raw_observation (Dict[str, Any]): A dictionary containing sensor data.
                Expected keys:
                - 'rgb' (PIL.Image.Image or torch.Tensor): The RGB image/panorama.
                - 'current_pose' (np.ndarray or torch.Tensor): Agent's current global pose
                  (e.g., [x, y, z] or [x, y, yaw] or full pose matrix).
                - Optional: 'camera_intrinsics_override' (Dict): To override default intrinsics.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, Any]]]:
                - fused_multimodal_embedding (torch.Tensor): v_t, shape (D_fused,).
                - current_absolute_position (torch.Tensor): p_t, shape (3,) representing [x,y,z].
                - point_cloud (Optional[torch.Tensor]): Generated 3D point cloud (N, 3), if enabled.
                - additional_perception_info (Optional[Dict[str, Any]]): For other info like object IDs,
                  camera embeddings, etc. For now, includes camera embedding from UniDepth.
        """
        rgb_input = raw_observation['rgb']
        pose_input = raw_observation['current_pose']

        if isinstance(pose_input, np.ndarray):
            # Assuming pose_input gives at least [x, y, z]
            # For p_t, we typically need the 3D position.
            current_absolute_position = torch.tensor(pose_input[:3], dtype=torch.float32, device=self.device)
        elif isinstance(pose_input, torch.Tensor):
            current_absolute_position = pose_input[:3].clone().detach().to(self.device, dtype=torch.float32)
        else:
            raise TypeError("Unsupported type for 'current_pose' in observation.")

        # Process panorama/image through the feature processor
        # The feature_processor.process_panorama expects a single image or a batch of one.
        # It returns (fused_emb (1,D), depth_map (1,1,H,W), cam_embed (1,D_cam), point_cloud (None initially))
        fused_embedding_batch, depth_map_batch, camera_embedding_batch, _ = \
            self.feature_processor.process_panorama(rgb_input)

        # Squeeze batch dimension as this module processes one observation at a time for the agent
        fused_multimodal_embedding = fused_embedding_batch.squeeze(0) # (D_fused,)
        depth_map = depth_map_batch.squeeze(0) # (1, H_depth, W_depth)
        camera_embedding_c_t = camera_embedding_batch.squeeze(0) if camera_embedding_batch is not None else None #(D_cam,)


        point_cloud: Optional[torch.Tensor] = None
        if self.generate_point_cloud:
            intrinsics_to_use = raw_observation.get('camera_intrinsics_override', self.camera_intrinsics)
            if intrinsics_to_use:
                point_cloud = unproject_depth_to_pointcloud(depth_map, intrinsics_to_use)
            else:
                print("Warning: Cannot generate point cloud as camera intrinsics are missing.")
        
        additional_info = {
            'camera_embedding_ct': camera_embedding_c_t
            # Future: add detected object IDs, semantic segmentation, etc.
            # 'object_id': 'some_detected_object' # Example for STM
        }
        
        return fused_multimodal_embedding, current_absolute_position, point_cloud, additional_info

if __name__ == '__main__':
    print("--- Testing PerceptionModule ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Example Config for PerceptionModule and its underlying MultimodalFeatureProcessor
    mock_perception_config = {
        'perception': { # Config for PerceptionModule itself
            'generate_point_cloud': True,
            'multimodal_feature_processor': { # Config for MultimodalFeatureProcessor
                'vf_output_dim': 256,
                'unidepth_model_path': None,  # No real model for mock
                'unidepth_internal_feature_dim': 128,
                'camera_intrinsics': {'fx': 250.0, 'fy': 250.0, 'cx': 159.5, 'cy': 119.5} # Example for 320x240
            }
        }
    }

    perception_module = PerceptionModule(mock_perception_config, device)
    perception_module.eval() # Set to evaluation mode

    # Create a dummy raw observation
    # In a real VLN setup, this would come from the environment simulator
    try:
        dummy_rgb_pil = Image.new('RGB', (320, 240), color='blue')
    except ImportError: # If Pillow is not installed for some reason in a minimal test
        print("Pillow not installed, cannot create dummy PIL image for full test.")
        dummy_rgb_pil = torch.rand(3, 240, 320) # Use tensor instead

    mock_raw_observation = {
        'rgb': dummy_rgb_pil,
        'current_pose': np.array([10.0, 5.0, 1.5]) # x, y, z
        # Optionally: 'camera_intrinsics_override': {...}
    }

    fused_emb, abs_pos, pc, add_info = perception_module.process_observation(mock_raw_observation)

    print(f"PerceptionModule Output:")
    print(f"  Fused Embedding (v_t) shape: {fused_emb.shape}")
    expected_fused_dim = mock_perception_config['perception']['multimodal_feature_processor']['vf_output_dim'] + \
                         mock_perception_config['perception']['multimodal_feature_processor']['unidepth_internal_feature_dim']
    assert fused_emb.shape == (expected_fused_dim,)
    
    print(f"  Current Absolute Position (p_t): {abs_pos.cpu().numpy()}")
    assert abs_pos.shape == (3,)
    
    if pc is not None:
        print(f"  Point Cloud shape: {pc.shape}")
        if pc.numel() > 0: # if not empty
             assert pc.ndim == 2 and pc.shape[1] == 3
    else:
        print("  Point Cloud: Not generated or empty.")
        
    if add_info:
        print(f"  Additional Info: { {k: v.shape if isinstance(v, torch.Tensor) else v for k,v in add_info.items()} }")
        assert 'camera_embedding_ct' in add_info

    print("\nPerceptionModule tests completed.")