import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

# Assuming other modules will be importable from these paths
from mem4nav_core.perception_processing.feature_utils import MultimodalFeatureProcessor
from mem4nav_core.spatial_representation.sparse_octree import SparseOctree, OctreeLeaf
from mem4nav_core.spatial_representation.semantic_graph import SemanticGraph, GraphNode
from mem4nav_core.memory_system.memory_retrieval import MemoryRetrieval, RetrievalResultItem

from .perception import PerceptionModule 
from .planning import PlanningModule
from .control import ControlModule
from ..base_vln_agent import BaseVLNAgent 

class BaseVLNAgent(nn.Module):
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
    def reset(self, instruction_data: Optional[Dict[str, Any]] = None) -> None: # type: ignore
        raise NotImplementedError
    def step(self, observation: Dict[str, Any], instruction_data: Optional[Dict[str, Any]] = None) -> Any: # type: ignore
        raise NotImplementedError


# Define the sub-modules (simplified interfaces for now, will be detailed in their own files)
# We'll import them properly once they are created. For now, this helps structure ModularAgent.

class PerceptionModule(nn.Module):
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__()
        self.device = device
        self.processor_config = config.get('multimodal_feature_processor', {})
        self.feature_processor = MultimodalFeatureProcessor(
            visual_frontend_output_dim=self.processor_config.get('vf_output_dim', 256),
            unidepth_model_path=self.processor_config.get('unidepth_model_path'),
            unidepth_internal_feature_dim=self.processor_config.get('unidepth_internal_feature_dim', 128),
            device=device
        )
        # Example: Intrinsics might be fixed or come with observation
        self.camera_intrinsics = self.processor_config.get('camera_intrinsics', {'fx':256,'fy':256,'cx':255.5,'cy':255.5})

    def process_observation(self, raw_observation: Dict[str, Any]) -> \
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Processes raw observation to extract features and state.
        raw_observation: {'rgb': PILImage or Tensor, 'current_pose': np.array([x,y,yaw])}
        Returns:
            - fused_embedding (v_t)
            - current_absolute_pos (p_t as tensor)
            - point_cloud (optional)
        """
        rgb_image = raw_observation['rgb'] # Assuming PIL Image or pre-transformable tensor
        current_pose_numpy = raw_observation['current_pose'] # [x, y, z] or [x,y,yaw]
        
        # For simplicity, p_t is just the position part. In reality, full pose might be used.
        current_absolute_pos_tensor = torch.tensor(current_pose_numpy[:3], dtype=torch.float32, device=self.device)

        fused_embedding, depth_map, _, _ = self.feature_processor.process_panorama(rgb_image)
        # point_cloud = unproject_depth_to_pointcloud(depth_map, self.camera_intrinsics) # If needed here
        
        return fused_embedding.squeeze(0), current_absolute_pos_tensor, None # point_cloud placeholder


class MappingModule(nn.Module):
    def __init__(self, config: Dict[str, Any], memory_retriever: MemoryRetrieval, 
                 octree: SparseOctree, semantic_graph: SemanticGraph, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        self.memory_retriever = memory_retriever
        self.octree = octree
        self.semantic_graph = semantic_graph
        self.current_semantic_node: Optional[GraphNode] = None # u_c

    def update_maps_and_memory(self,
                               current_absolute_pos: torch.Tensor, # p_t
                               fused_embedding: torch.Tensor, # v_t
                               observation_info: Dict[str, Any]): # e.g. {'object_id': 'car', 'octree_key': 12345}
        """Updates octree, semantic graph, and LTM/STM."""
        octree_key = observation_info.get('octree_key') # Morton code from current_absolute_pos
        obj_id_for_stm = observation_info.get('object_id', 'generic_observation')

        # 1. Ensure Octree Leaf exists (or is created)
        # The actual leaf object might be needed if it stores its own LTM token directly.
        # For now, assume LTM manages tokens internally by key.
        # OctreeLeaf creation/access:
        _ = self.octree.insert_or_get_leaf(current_absolute_pos.cpu().numpy(), fused_embedding)
        
        # 2. Determine current semantic graph node (u_c)
        # This might involve calling semantic_graph.add_or_update_node
        # For simplicity, assume self.current_semantic_node is managed externally or based on planning
        if self.current_semantic_node is None:
            # Fallback or initialize semantic node. In a real agent, this is tied to navigation state.
            # This node's position is p_u_c
            # For this example, let's try to create/update based on current pos/embedding
            # This logic should ideally be more robust in a full planning cycle.
            self.current_semantic_node = self.semantic_graph.add_or_update_node(
                fused_embedding, current_absolute_pos.cpu().numpy()
            )
            print(f"Mapping: Current semantic node set/updated to {self.current_semantic_node.id}")

        # 3. Write to LTM and STM via MemoryRetriever
        # This requires p_u_c for STM
        p_u_c_tensor = torch.tensor(self.current_semantic_node.position, dtype=torch.float32, device=self.device)

        newly_written_ltm_token = self.memory_retriever.write_observation(
            unique_observation_key=octree_key, # Use octree Morton code as the unique integer key
            object_id_for_stm=obj_id_for_stm,
            current_absolute_position=current_absolute_pos,
            current_observation_embedding=fused_embedding,
            current_semantic_node_position=p_u_c_tensor
        )

        # 4. Store the newly_written_ltm_token in the corresponding spatial element
        # This is crucial: the OctreeLeaf or GraphNode needs to store this token.
        octree_leaf_obj = self.octree.get_leaf_by_code(octree_key)
        if octree_leaf_obj:
            # Assuming OctreeLeaf has a field like `current_ltm_token` (2d_emb)
            # For now, we'll print. The actual storage mechanism needs refinement in OctreeLeaf.
            # octree_leaf_obj.ltm_current_token = newly_written_ltm_token # This is the modification discussed
            # print(f"Debug: Octree leaf {octree_key} would store LTM token of shape {newly_written_ltm_token.shape}")
            pass 
        
        # If graph nodes also store LTM tokens directly tied to them (not just via octree leaves):
        graph_node_obj = self.semantic_graph.get_node(self.current_semantic_node.id) #type: ignore
        if graph_node_obj:
            # graph_node_obj.ltm_current_token = newly_written_ltm_token # If graph node also has independent LTM
            pass

    def retrieve_memories(self,
                          current_absolute_pos: torch.Tensor, # p_t
                          fused_embedding: torch.Tensor, # v_t
                         ) -> Tuple[str, List[RetrievalResultItem]]:
        if self.current_semantic_node is None:
            # Handle case where semantic context isn't set; maybe default to LTM or error
            print("Warning: current_semantic_node not set in MappingModule for retrieval. Defaulting p_u_c to p_t.")
            p_u_c_tensor = current_absolute_pos
        else:
            p_u_c_tensor = torch.tensor(self.current_semantic_node.position, dtype=torch.float32, device=self.device)

        return self.memory_retriever.retrieve_memory(
            current_observation_embedding=fused_embedding,
            current_absolute_position=current_absolute_pos,
            current_semantic_node_position=p_u_c_tensor
        )

    def update_current_semantic_node(self, node: GraphNode):
        self.current_semantic_node = node


class PlanningModule(nn.Module): # Placeholder
    def __init__(self, config: Dict[str, Any], semantic_graph: SemanticGraph, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        self.semantic_graph = semantic_graph
        self.current_plan: List[int] = [] # List of semantic node IDs
        self.plan_index: int = 0

    def generate_plan(self, start_node_id: int, goal_node_id: int, instruction_info: Optional[Dict] = None) -> bool:
        # In a full system, this might use an LLM to parse instructions into goal_node_id or subgoals
        # For now, assumes goal_node_id is given.
        print(f"Planning: Generating plan from {start_node_id} to {goal_node_id}")
        path_nodes, cost = self.semantic_graph.shortest_path(start_node_id, goal_node_id)
        if path_nodes:
            self.current_plan = path_nodes
            self.plan_index = 0 # Start at the first node of the plan
            print(f"Planning: Path found: {self.current_plan} with cost {cost:.2f}")
            return True
        print(f"Planning: No path found from {start_node_id} to {goal_node_id}")
        self.current_plan = []
        return False

    def get_next_waypoint_info(self, current_pose_abs: torch.Tensor) -> Optional[Dict[str, Any]]:
        if not self.current_plan or self.plan_index >= len(self.current_plan):
            return None # Plan exhausted or no plan

        # Get current target semantic node from plan
        target_node_id = self.current_plan[self.plan_index]
        target_node_obj = self.semantic_graph.get_node(target_node_id)

        if not target_node_obj:
            self.plan_index +=1 # Skip if node somehow disappeared
            return self.get_next_waypoint_info(current_pose_abs)

        target_pos_np = target_node_obj.position
        
        # Check if current waypoint (semantic node) is reached
        # This threshold should be configurable
        dist_to_target_node = np.linalg.norm(current_pose_abs.cpu().numpy() - target_pos_np)
        waypoint_reached_threshold = self.config.get('waypoint_reached_threshold', 1.5) # meters

        if dist_to_target_node < waypoint_reached_threshold:
            print(f"Planning: Reached waypoint/semantic node {target_node_id}. Advancing plan.")
            self.plan_index += 1
            if self.plan_index >= len(self.current_plan):
                print("Planning: Plan completed.")
                return None # Plan finished
            # Get next node in plan
            next_target_node_id = self.current_plan[self.plan_index]
            next_target_node_obj = self.semantic_graph.get_node(next_target_node_id)
            if not next_target_node_obj: return None # Should not happen if plan is valid
            target_pos_np = next_target_node_obj.position

        return {'target_position': torch.tensor(target_pos_np, dtype=torch.float32, device=self.device),
                'target_node_id': self.current_plan[self.plan_index]}
                
    def is_goal_reached(self, current_pose_abs: torch.Tensor, goal_node_id: int) -> bool:
        if not self.current_plan or self.plan_index < len(self.current_plan) -1:
            return False # Plan not finished or current target is not the final goal
        
        if self.current_plan[self.plan_index] != goal_node_id:
             return False # Current target is not the actual goal node

        target_node_obj = self.semantic_graph.get_node(goal_node_id)
        if not target_node_obj: return False
        
        dist_to_goal_node = np.linalg.norm(current_pose_abs.cpu().numpy() - target_node_obj.position)
        goal_reached_threshold = self.config.get('goal_reached_threshold', 1.5)
        return dist_to_goal_node < goal_reached_threshold


class ControlModule(nn.Module): # Placeholder
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        # Parameters for PID or other low-level controller
        self.pos_tolerance = config.get('pos_tolerance', 0.2) # meters
    
    def compute_action_to_waypoint(self, current_pose_abs: torch.Tensor, 
                                   waypoint_abs_pos: torch.Tensor) -> str: # Returns discrete action
        # Simplified: determine a discrete action ('forward', 'turn_left', 'turn_right', 'stop')
        # This is a very basic controller. Real controller would be more complex.
        direction_vector = waypoint_abs_pos - current_pose_abs[:len(waypoint_abs_pos)] # Ensure compatible dims
        distance = torch.linalg.norm(direction_vector)

        if distance < self.pos_tolerance:
            return "stop" 
        
        # Assume current_pose includes yaw for orientation, or we use bearings
        # For now, just move forward if not at goal. A real system needs orientation.
        # This is a placeholder for actual low-level motion command generation.
        # If current_pose_abs has yaw: current_yaw = current_pose_abs[2]
        # desired_yaw = torch.atan2(direction_vector[1], direction_vector[0])
        # angle_diff = (desired_yaw - current_yaw + np.pi) % (2*np.pi) - np.pi
        # if abs(angle_diff) > some_turn_threshold: return "turn_left" or "turn_right"
        return "forward"


class PolicyNetwork(nn.Module): # Placeholder for the "lightweight policy network"
    def __init__(self, config: Dict[str, Any], input_dim: int, num_actions: int, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        # Example: MLP policy
        # input_dim = dim(v_t) + dim(aggregated_memory) + dim(waypoint_representation)
        self.network = nn.Sequential(
            nn.Linear(input_dim, config.get('policy_hidden_dim', 128)),
            nn.ReLU(),
            nn.Linear(config.get('policy_hidden_dim', 128), num_actions)
        ).to(device)
        self.num_actions = num_actions

    def forward(self, policy_input: torch.Tensor) -> torch.Tensor: # Returns action logits
        return self.network(policy_input)

    def get_action(self, policy_input: torch.Tensor, explore: bool = False) -> int: # Returns discrete action index
        logits = self.forward(policy_input)
        if explore:
            # Add exploration (e.g., sample from softmax)
            action = torch.distributions.Categorical(logits=logits).sample().item()
        else:
            action = torch.argmax(logits, dim=-1).item()
        return action


class ModularAgent(BaseVLNAgent):
    """
    Hierarchical Modular Pipeline Agent for Vision-and-Language Navigation,
    augmented with Mem4Nav.
    """
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)

        self.agent_config = config.get('modular_pipeline_agent', {})
        mem_config = config.get('mem4nav', {})
        octree_config = self.agent_config.get('octree', {})
        sg_config = self.agent_config.get('semantic_graph', {})
        
        # --- Initialize Mem4Nav Core Components ---
        self.octree = SparseOctree(
            world_size=mem_config.get('world_size', 100.0),
            max_depth=mem_config.get('octree_depth', 16),
            embedding_dim=mem_config.get('embedding_dim', 256), # This is d_emb for LTM
            device=device
        )
        self.semantic_graph = SemanticGraph(
            node_creation_similarity_threshold=sg_config.get('node_creation_similarity_threshold', 0.5),
            alpha_distance_weight=sg_config.get('alpha_distance_weight', 1.0),
            beta_instruction_cost_weight=sg_config.get('beta_instruction_cost_weight', 0.5),
            device=device
        )
        self.memory_retriever = MemoryRetrieval(
            ltm_embedding_dim=mem_config.get('embedding_dim', 256), # This is v_t's dim
            ltm_transformer_depth=mem_config.get('lt_depth', 6),
            ltm_transformer_heads=mem_config.get('lt_heads', 8),
            ltm_max_elements=mem_config.get('lt_max_elements', 10000),
            stm_capacity=mem_config.get('st_capacity', 128),
            # ... other memretriever params from config ...
            device=device
        )

        # --- Initialize Agent Modules ---
        self.perception_module = PerceptionModule(self.agent_config.get('perception', {}), device)
        self.mapping_module = MappingModule(self.agent_config.get('mapping', {}), 
                                            self.memory_retriever, self.octree, self.semantic_graph, device)
        self.planning_module = PlanningModule(self.agent_config.get('planning', {}), self.semantic_graph, device)
        self.control_module = ControlModule(self.agent_config.get('control', {}), device)

        # --- Policy Network ---
        # Determine input_dim based on what's fed to policy
        # Example: fused_emb_dim + aggregated_mem_dim + waypoint_info_dim
        # This needs to be calculated based on feature_utils output and memory aggregation.
        fused_emb_dim = self.perception_module.feature_processor.fused_embedding_dim
        # For now, assume memory is also aggregated to fused_emb_dim for simplicity
        # Waypoint info (e.g., relative direction vector to waypoint) could be small, e.g., 3D
        policy_input_dim = fused_emb_dim + fused_emb_dim + 3 # Placeholder
        self.action_space = ["forward", "turn_left", "turn_right", "stop"] # Example
        self.policy_network = PolicyNetwork(self.agent_config.get('policy_network', {}),
                                            policy_input_dim, len(self.action_space), device)
        
        self.current_instruction_info: Optional[Dict] = None
        self.goal_node_id: Optional[int] = None
        self.current_step_count: int = 0


    def reset(self, instruction_data: Optional[Dict[str, Any]] = None) -> None:
        """Resets agent state for a new episode."""
        print("ModularAgent: Resetting for new episode.")
        self.mapping_module.octree.clear()
        self.mapping_module.semantic_graph.clear()
        self.mapping_module.memory_retriever.clear_all_memory()
        self.mapping_module.current_semantic_node = None # Reset current u_c context
        
        self.planning_module.current_plan = []
        self.planning_module.plan_index = 0
        
        self.current_step_count = 0
        self.memory_retriever.update_current_step(self.current_step_count)

        self.current_instruction_info = instruction_data
        if instruction_data:
            # Extract goal information, potentially involves LLM or pre-processing
            # For now, assume 'goal_node_id' is directly available or can be derived.
            self.goal_node_id = instruction_data.get('goal_node_id') # Example
            # Or, if goal is by position:
            # goal_pos_np = instruction_data.get('goal_position_xyz_numpy')
            # if goal_pos_np is not None:
            #    goal_node_obj = self.mapping_module.semantic_graph.add_or_update_node(
            #        torch.zeros(self.perception_module.feature_processor.fused_embedding_dim, device=self.device), # dummy embedding for goal node
            #        goal_pos_np
            #    )
            #    self.goal_node_id = goal_node_obj.id

            print(f"ModularAgent: Instruction loaded. Goal node ID: {self.goal_node_id}")
        else:
            self.goal_node_id = None
            print("ModularAgent: No instruction data provided for reset.")

    def _get_octree_key_from_pos(self, pos_tensor: torch.Tensor) -> int:
        """Helper to get Morton code from a position tensor."""
        return self.octree._get_morton_code(pos_tensor.cpu().numpy()) # Accessing protected member for brevity

    def _aggregate_memory_for_policy(self, mem_source: str, mem_data: List[RetrievalResultItem], 
                                     target_dim: int) -> torch.Tensor:
        """Aggregates retrieved memory into a single tensor for the policy."""
        if not mem_data:
            return torch.zeros(target_dim, device=self.device)

        # Simple aggregation: average of embeddings
        # More complex aggregation (e.g., attention) could be used.
        all_embeddings: List[torch.Tensor] = []
        for item in mem_data:
            if mem_source == "STM":
                entry: ShortTermMemoryEntry = item #type: ignore
                all_embeddings.append(entry.embedding)
            elif mem_source == "LTM":
                _, recon_v, _, _ = item # key, recon_v, recon_p, recon_d
                all_embeddings.append(recon_v)
        
        if not all_embeddings:
            return torch.zeros(target_dim, device=self.device)
            
        stacked_embeddings = torch.stack(all_embeddings).to(self.device)
        aggregated_embedding = torch.mean(stacked_embeddings, dim=0)
        
        # Project to target_dim if necessary (e.g., if LTM/STM embeddings differ from policy needs)
        # For now, assume embeddings are compatible or use a learned projection layer.
        if aggregated_embedding.shape[0] != target_dim:
            # This would require a projection layer. For now, we expect match or truncation/padding.
            # print(f"Warning: Aggregated memory dim {aggregated_embedding.shape[0]} != target {target_dim}")
            # Fallback: take first part or zero pad (crude)
            if aggregated_embedding.shape[0] > target_dim:
                return aggregated_embedding[:target_dim]
            else:
                padded = torch.zeros(target_dim, device=self.device)
                padded[:aggregated_embedding.shape[0]] = aggregated_embedding
                return padded
        return aggregated_embedding


    def step(self, observation: Dict[str, Any]) -> str: # Returns a discrete action string
        """
        Performs one step of navigation.
        observation: {'rgb': PILImage, 'current_pose': np.array([x,y,z]) or [x,y,yaw]}
        """
        self.current_step_count += 1
        self.memory_retriever.update_current_step(self.current_step_count)

        # 1. Perception: Process raw observation
        # v_t is the fused multimodal embedding
        v_t, p_t, _ = self.perception_module.process_observation(observation)
        # p_t is current absolute position tensor

        # 2. Mapping: Update maps and memories
        # This needs a unique key for the current location, e.g., Morton code from p_t.
        # And an object_id if available from perception for STM.
        current_octree_key = self._get_octree_key_from_pos(p_t)
        obs_info_for_mapping = {
            'octree_key': current_octree_key,
            'object_id': observation.get('detected_object_id', f'obs_at_step_{self.current_step_count}')
        }
        self.mapping_module.update_maps_and_memory(p_t, v_t, obs_info_for_mapping)

        # 3. Planning: Determine current plan and next waypoint
        # This is a simplified planning logic.
        # If no plan or current semantic node, try to make one to the goal.
        if not self.planning_module.current_plan and self.goal_node_id is not None:
            current_sem_node = self.mapping_module.current_semantic_node
            if current_sem_node:
                self.planning_module.generate_plan(current_sem_node.id, self.goal_node_id, self.current_instruction_info) #type: ignore
            else: # Cannot plan without a start semantic node
                print("Warning: No current semantic node to start planning from.")


        waypoint_info = self.planning_module.get_next_waypoint_info(p_t) # p_t is current absolute pose
        
        # Check if goal is reached (based on final node in plan)
        if self.goal_node_id is not None and self.planning_module.is_goal_reached(p_t, self.goal_node_id):
            print(f"ModularAgent: Goal node {self.goal_node_id} reached!")
            return "stop"

        if waypoint_info is None and self.goal_node_id is not None: # Plan finished, but not at final goal node yet (e.g. threshold diff) or no plan
             # This could mean we need to re-plan or issue stop if truly stuck.
             # For now, if no waypoint, and not at goal, maybe stop.
             # Or, if goal_node_id is set, but plan could not be made to it, it's a problem.
            print("ModularAgent: No current waypoint and goal not confirmed reached. Stopping.")
            return "stop"
        elif waypoint_info is None and self.goal_node_id is None: # No goal, no plan
            print("ModularAgent: No goal and no plan. Stopping.")
            return "stop"


        # 4. Memory Retrieval
        mem_source, mem_data = self.mapping_module.retrieve_memories(p_t, v_t)
        
        # Aggregate memory for policy (example: average retrieved embeddings)
        # Assuming policy network input_dim was set up for this.
        # Let's say aggregated memory should match v_t's dimension for simplicity here.
        aggregated_memory_embedding = self._aggregate_memory_for_policy(mem_source, mem_data, v_t.shape[0])


        # 5. Policy Decision
        # Prepare input for the policy network
        # Example: [current_fused_embedding (v_t); aggregated_memory_embedding; waypoint_direction_vector]
        if waypoint_info:
            waypoint_pos_tensor = waypoint_info['target_position'].to(self.device)
            # Relative direction to waypoint (simplified representation)
            waypoint_direction = waypoint_pos_tensor - p_t[:len(waypoint_pos_tensor)] # Use compatible part of p_t
            # Normalize, or ensure fixed dim. For now, use as is (if 3D)
            waypoint_repr_for_policy = waypoint_direction[:3] # Make sure it's 3D
            if waypoint_repr_for_policy.shape[0] < 3: # Pad if needed
                padded_waypoint_repr = torch.zeros(3, device=self.device)
                padded_waypoint_repr[:waypoint_repr_for_policy.shape[0]] = waypoint_repr_for_policy
                waypoint_repr_for_policy = padded_waypoint_repr
        else: # No waypoint (e.g. plan finished or failed)
            waypoint_repr_for_policy = torch.zeros(3, device=self.device) # Zero vector if no waypoint


        policy_input_tensor = torch.cat([
            v_t, 
            aggregated_memory_embedding,
            waypoint_repr_for_policy 
        ], dim=0).unsqueeze(0) # Add batch dimension

        # Check if policy_input_tensor matches self.policy_network input_dim
        expected_policy_input_dim = self.policy_network.network[0].in_features
        if policy_input_tensor.shape[1] != expected_policy_input_dim:
            # This indicates a mismatch in setup, very important to get right.
            # For now, we'll just print a warning.
             print(f"CRITICAL WARNING: Policy input dim mismatch! Expected {expected_policy_input_dim}, Got {policy_input_tensor.shape[1]}")
             # Fallback: try to make it work by truncating or padding (very bad idea for real model)
             if policy_input_tensor.shape[1] > expected_policy_input_dim:
                 policy_input_tensor = policy_input_tensor[:, :expected_policy_input_dim]
             else:
                 padding = torch.zeros((policy_input_tensor.shape[0], expected_policy_input_dim - policy_input_tensor.shape[1]), device=self.device)
                 policy_input_tensor = torch.cat([policy_input_tensor, padding], dim=1)


        action_idx = self.policy_network.get_action(policy_input_tensor)
        action_str = self.action_space[action_idx]

        # 6. (Optional) Control: Refine action or convert to low-level commands
        # If action_str is high-level (e.g. "go_to_waypoint"), control module would be used.
        # If action_str is already "forward", "turn_left", etc., it's directly usable by some envs.
        # For now, we assume policy network outputs one of self.action_space strings.
        # If the policy network produced, say, a target pose, then control_module would generate motor commands.
        # Here, if action_str is e.g. "forward", and we need continuous control, the control_module would take over.
        # This example assumes discrete actions are output by policy.

        print(f"Step {self.current_step_count}: v_t shape {v_t.shape}, p_t {p_t.cpu().numpy().round(1)}, Mem: {mem_source} ({len(mem_data)} items), Waypoint: {waypoint_info.get('target_node_id') if waypoint_info else 'None'}, Action: {action_str}")
        
        if action_str == "stop" and waypoint_info is not None: # if policy decides to stop before waypoint reached
            print("Policy decided to stop before reaching current waypoint.")

        return action_str


if __name__ == '__main__':
    # This is a conceptual test. Running it requires proper configs and mock environment.
    print("--- Conceptual Test for ModularAgent ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Example Config (replace paths and details with actual ones)
    mock_config = {
        'mem4nav': {
            'world_size': 50.0, 'octree_depth': 10, 'embedding_dim': 128+64, # vf_out + unidepth_internal
            'lt_depth': 2, 'lt_heads': 2, 'lt_max_elements': 100,
            'st_capacity': 20,
            # ... other mem4nav params
        },
        'modular_pipeline_agent': {
            'octree': {},
            'semantic_graph': {'node_creation_similarity_threshold': 0.8},
            'perception': {
                'multimodal_feature_processor': {
                    'vf_output_dim': 128,
                    'unidepth_model_path': None, # Path to actual UniDepth model
                    'unidepth_internal_feature_dim': 64,
                    'camera_intrinsics': {'fx':128,'fy':128,'cx':127.5,'cy':127.5}
                }
            },
            'mapping': {},
            'planning': {'waypoint_reached_threshold': 0.5, 'goal_reached_threshold': 0.5},
            'control': {},
            'policy_network': {'policy_hidden_dim': 64}
        }
    }

    agent = ModularAgent(mock_config, device)

    # Mock instruction
    # In a real scenario, goal_node_id might be determined after some initial exploration
    # or from parsing the instruction string.
    # Let's assume after first step, a start node is made, and we set a goal.
    mock_instruction = {'text': "go to the red door", 'goal_node_id': 5} # Assume node 5 is the goal
    agent.reset(mock_instruction)

    # Simulate a few steps
    from PIL import Image
    for step_num in range(5):
        print(f"\n--- Agent Step {step_num} ---")
        # Mock observation (replace with actual environment observation)
        mock_obs = {
            'rgb': Image.new('RGB', (256, 256), color=(np.random.randint(0,255),0,0)),
            'current_pose': np.array([step_num * 0.5, 0.0, 0.0]), # Agent moves along x-axis
            'detected_object_id': f'obj_at_step_{step_num}'
        }
        
        # Manually set/update current semantic node for mapping context in this test
        # In a real agent, this would be based on navigation logic / graph traversal
        if agent.mapping_module.current_semantic_node is None or step_num % 2 == 0 : # Update u_c periodically
            # This is a hack for testing; real u_c update is more involved
            # based on where the agent thinks it is on the semantic graph.
            temp_emb = torch.randn(mock_config['mem4nav']['embedding_dim'], device=device)
            temp_pos_np = mock_obs['current_pose']
            node_u_c = agent.semantic_graph.add_or_update_node(temp_emb, temp_pos_np)
            agent.mapping_module.update_current_semantic_node(node_u_c)


        action = agent.step(mock_obs)
        print(f"Agent action: {action}")
        if action == "stop":
            print("Agent decided to stop.")
            break
    
    print("\nModularAgent conceptual test finished.")