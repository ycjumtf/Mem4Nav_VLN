import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

# Assuming SemanticGraph and GraphNode are accessible
try:
    from mem4nav_core.spatial_representation.semantic_graph import SemanticGraph, GraphNode
except ImportError:
    print("Warning: PlanningModule using placeholder for SemanticGraph components.")
    # Placeholders for standalone development
    class GraphNode: # type: ignore
        def __init__(self, node_id: int, position: np.ndarray): self.id = node_id; self.position = position
    class SemanticGraph: # type: ignore
        def __init__(self, *args, **kwargs): self.nodes: Dict[int, GraphNode] = {}
        def get_node(self, node_id: int) -> Optional[GraphNode]: return self.nodes.get(node_id)
        def shortest_path(self, start_id: int, goal_id: int) -> Tuple[List[int], float]:
            # Mock path for testing
            if start_id in self.nodes and goal_id in self.nodes:
                if start_id == goal_id: return ([start_id], 0.0)
                # Simple mock: direct path if both exist, or a linear path if intermediate nodes are implied
                mock_path = [start_id]
                if goal_id != start_id + 1 and goal_id != start_id: # try to add intermediate mock node
                    if start_id + 1 in self.nodes: mock_path.append(start_id + 1)
                mock_path.append(goal_id)

                mock_cost = 0.0
                for i in range(len(mock_path) -1):
                    pos1 = self.nodes[mock_path[i]].position
                    pos2 = self.nodes[mock_path[i+1]].position
                    mock_cost += np.linalg.norm(pos1 - pos2) if pos1 is not None and pos2 is not None else 1.0
                return mock_path, mock_cost
            return [], float('inf')


class PlanningModule(nn.Module):
    """
    Handles high-level planning for the Modular Pipeline Agent.
    It uses the SemanticGraph to find a sequence of semantic waypoints (graph nodes)
    from a starting node to a goal node.
    """
    def __init__(self, 
                 config: Dict[str, Any], 
                 semantic_graph: SemanticGraph, 
                 device: torch.device):
        super().__init__()
        self.module_config = config.get('planning', {}) # Specific config for this module
        self.semantic_graph = semantic_graph
        self.device = device

        # Current high-level plan: a list of semantic graph node IDs
        self.current_plan_node_ids: List[int] = []
        self.current_plan_index: int = 0 # Index of the next waypoint in self.current_plan_node_ids

        # Thresholds from config
        self.waypoint_reached_threshold: float = self.module_config.get('waypoint_reached_threshold', 1.5) # meters
        self.goal_reached_threshold: float = self.module_config.get('goal_reached_threshold', 1.0) # meters

        self.global_goal_node_id: Optional[int] = None

    def set_global_goal(self, goal_node_id: Optional[int], instruction_info: Optional[Dict] = None):
        """
        Sets the overall navigation goal for the planner.
        instruction_info is a placeholder for future LLM-based instruction parsing.
        """
        self.global_goal_node_id = goal_node_id
        self.current_plan_node_ids = [] # Clear previous plan when a new global goal is set
        self.current_plan_index = 0
        if goal_node_id is not None:
            print(f"PlanningModule: Global goal set to semantic node ID: {goal_node_id}")
        else:
            print("PlanningModule: Global goal cleared.")

    def generate_plan_to_goal(self, start_node_id: int) -> bool:
        """
        Generates a high-level plan (sequence of semantic graph nodes) from the
        start_node_id to the currently set global_goal_node_id.

        Args:
            start_node_id (int): The ID of the semantic graph node to start planning from.

        Returns:
            bool: True if a plan was successfully generated, False otherwise.
        """
        if self.global_goal_node_id is None:
            print("PlanningModule: Cannot generate plan, global goal not set.")
            self.current_plan_node_ids = []
            self.current_plan_index = 0
            return False
        
        if start_node_id == self.global_goal_node_id:
            print(f"PlanningModule: Start node {start_node_id} is already the global goal.")
            self.current_plan_node_ids = [start_node_id] # Plan is just the goal itself
            self.current_plan_index = 0 # Point to the goal
            return True

        print(f"PlanningModule: Generating plan from semantic node {start_node_id} to {self.global_goal_node_id}...")
        path_nodes, cost = self.semantic_graph.shortest_path(start_node_id, self.global_goal_node_id)

        if path_nodes:
            self.current_plan_node_ids = path_nodes
            self.current_plan_index = 0 # Start at the beginning of the plan
            print(f"PlanningModule: Plan found: {self.current_plan_node_ids} with cost {cost:.2f}")
            # The first node in path_nodes should be start_node_id.
            # The agent is at/near start_node_id, so the first *target* waypoint is typically path_nodes[1].
            # We will advance the index if current waypoint is reached.
            return True
        else:
            print(f"PlanningModule: No path found from {start_node_id} to {self.global_goal_node_id}.")
            self.current_plan_node_ids = []
            self.current_plan_index = 0
            return False

    def get_current_waypoint_info(self, current_agent_abs_pos_np: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Gets information about the next semantic waypoint in the current plan.
        Advances the plan if the current waypoint is deemed reached.

        Args:
            current_agent_abs_pos_np (np.ndarray): Agent's current absolute [x,y,z] position.

        Returns:
            Optional[Dict[str, Any]]: Info about the next waypoint, e.g.,
                {'target_position_tensor': torch.Tensor, 'target_node_id': int, 'is_final_goal': bool},
                or None if no plan, plan completed, or error.
        """
        if not self.current_plan_node_ids or self.current_plan_index >= len(self.current_plan_node_ids):
            # print("PlanningModule: No active plan or plan completed.")
            return None

        current_target_node_id = self.current_plan_node_ids[self.current_plan_index]
        current_target_node_obj = self.semantic_graph.get_node(current_target_node_id)

        if not current_target_node_obj:
            print(f"PlanningModule: Error - Node ID {current_target_node_id} from plan not found in graph. Advancing plan.")
            self.current_plan_index += 1
            return self.get_current_waypoint_info(current_agent_abs_pos_np) # Try next one

        target_pos_np = current_target_node_obj.position
        
        # Check if current waypoint (semantic node) is reached
        dist_to_target_node = np.linalg.norm(current_agent_abs_pos_np - target_pos_np)

        is_final_node_in_plan = (self.current_plan_index == len(self.current_plan_node_ids) - 1)
        threshold_to_use = self.goal_reached_threshold if is_final_node_in_plan and current_target_node_id == self.global_goal_node_id else self.waypoint_reached_threshold

        if dist_to_target_node < threshold_to_use:
            print(f"PlanningModule: Reached waypoint/semantic node {current_target_node_id} (Dist: {dist_to_target_node:.2f} < Thr: {threshold_to_use:.2f}). Advancing plan.")
            self.current_plan_index += 1
            if self.current_plan_index >= len(self.current_plan_node_ids):
                print("PlanningModule: Current plan fully traversed.")
                return None # Plan finished
            
            # Get next node in plan
            next_target_node_id = self.current_plan_node_ids[self.current_plan_index]
            next_target_node_obj = self.semantic_graph.get_node(next_target_node_id)
            if not next_target_node_obj:
                print(f"PlanningModule: Error - Next node ID {next_target_node_id} from plan not found. Plan aborted.")
                self.current_plan_node_ids = [] # Abort plan
                return None
            target_pos_np = next_target_node_obj.position
            current_target_node_id = next_target_node_id # Update current target

        is_final_waypoint_the_global_goal = (current_target_node_id == self.global_goal_node_id)

        return {
            'target_position_tensor': torch.tensor(target_pos_np, dtype=torch.float32, device=self.device),
            'target_node_id': current_target_node_id,
            'is_final_goal_of_plan': is_final_waypoint_the_global_goal and \
                                     (self.current_plan_index == len(self.current_plan_node_ids) - 1)
        }

    def is_overall_goal_reached(self, current_agent_abs_pos_np: np.ndarray) -> bool:
        """
        Checks if the agent has reached the vicinity of the global_goal_node_id.
        This is a stricter check specifically for the final destination.
        """
        if self.global_goal_node_id is None:
            return False # No goal set

        # Check if current plan's last node is the global goal and if we are targeting it
        if not self.current_plan_node_ids or self.current_plan_node_ids[-1] != self.global_goal_node_id:
             # Current plan doesn't lead to global goal, or no plan exists
             # We might need to replan if agent is close to goal but plan is wrong/finished early
             pass # Continue to check direct distance to goal node

        goal_node_obj = self.semantic_graph.get_node(self.global_goal_node_id)
        if not goal_node_obj:
            # print(f"PlanningModule: Global goal node ID {self.global_goal_node_id} not found in graph for goal check.")
            return False
        
        dist_to_goal_node = np.linalg.norm(current_agent_abs_pos_np - goal_node_obj.position)
        
        if dist_to_goal_node < self.goal_reached_threshold:
            # Additionally, ensure that if there was a plan, we are at its end
            # and that end was indeed the goal.
            plan_completed_at_goal = (
                self.current_plan_index >= len(self.current_plan_node_ids) -1 and \
                self.current_plan_node_ids and \
                self.current_plan_node_ids[len(self.current_plan_node_ids)-1] == self.global_goal_node_id
            )
            # If no plan, but close to goal, consider it reached.
            # Or if plan completed and was for this goal.
            if not self.current_plan_node_ids or plan_completed_at_goal:
                 return True
        return False

    def needs_replan(self, current_start_node_id: Optional[int]) -> bool:
        """Determines if a replan is necessary."""
        if self.global_goal_node_id is None:
            return False # No goal to plan to
        if not self.current_plan_node_ids: # No plan exists
            return True 
        if self.current_plan_index >= len(self.current_plan_node_ids): # Plan finished
            return False # Don't replan if finished, goal check should trigger stop
        if current_start_node_id is not None and self.current_plan_node_ids[0] != current_start_node_id:
            # If current plan's start doesn't match where agent currently thinks it starts from
            # This might happen if agent gets "lost" or deviates significantly.
            # print("PlanningModule: Agent's current semantic node differs from plan start. Consider replan.")
            # return True # This can be aggressive, disable for now.
            pass
        return False # Default: don't replan unless necessary

    def reset_plan(self):
        """Clears the current plan state."""
        self.current_plan_node_ids = []
        self.current_plan_index = 0
        # self.global_goal_node_id = None # Keep global goal unless explicitly changed by set_global_goal
        print("PlanningModule: Plan reset.")


if __name__ == '__main__':
    print("--- Conceptual Test for PlanningModule ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Mock SemanticGraph
    mock_sg = SemanticGraph(device=device)
    # Add some nodes to the mock graph for testing
    node0_pos = np.array([0.0, 0.0, 0.0])
    node1_pos = np.array([5.0, 0.0, 0.0])
    node2_pos = np.array([5.0, 5.0, 0.0])
    node3_pos = np.array([0.0, 5.0, 0.0]) # A non-goal node
    
    mock_sg.nodes[0] = GraphNode(0, node0_pos)
    mock_sg.nodes[1] = GraphNode(1, node1_pos)
    mock_sg.nodes[2] = GraphNode(2, node2_pos)
    mock_sg.nodes[3] = GraphNode(3, node3_pos)

    # Mock planner config
    mock_planner_config = {'planning': {'waypoint_reached_threshold': 0.5, 'goal_reached_threshold': 0.5}}
    planner = PlanningModule(mock_planner_config, mock_sg, device)

    # Test 1: Set goal and generate plan
    print("\nTest 1: Set goal and generate plan")
    planner.set_global_goal(goal_node_id=2) # Goal is node 2
    plan_success = planner.generate_plan(start_node_id=0)
    assert plan_success
    assert planner.current_plan_node_ids == [0, 1, 2] # Based on mock SemanticGraph shortest_path

    # Test 2: Get waypoints
    print("\nTest 2: Get waypoints")
    # Agent at start_node_id=0 (pos [0,0,0])
    agent_pos_np = np.array([0.0, 0.0, 0.0]) 
    wp_info = planner.get_current_waypoint_info(agent_pos_np) # Should target node 0 initially
    print(f"Waypoint Info: {wp_info}")
    assert wp_info['target_node_id'] == 0 
    assert not wp_info['is_final_goal_of_plan']

    # Simulate agent moving to waypoint 0 (it's already there)
    agent_pos_np = node0_pos 
    wp_info = planner.get_current_waypoint_info(agent_pos_np) # Reaches node 0, plan advances
    print(f"Waypoint Info (after reaching node 0): {wp_info}")
    assert wp_info['target_node_id'] == 1 # Next target is node 1
    assert torch.allclose(wp_info['target_position_tensor'].cpu(), torch.tensor(node1_pos, dtype=torch.float32))

    # Simulate agent moving to waypoint 1
    agent_pos_np = node1_pos + np.array([0.1, -0.1, 0.0]) # Near node 1
    wp_info = planner.get_current_waypoint_info(agent_pos_np) # Reaches node 1, plan advances
    print(f"Waypoint Info (after reaching node 1): {wp_info}")
    assert wp_info['target_node_id'] == 2 # Next target is node 2 (the goal)
    assert wp_info['is_final_goal_of_plan']

    # Simulate agent moving to final goal (node 2)
    agent_pos_np = node2_pos - np.array([0.05, 0.05, 0.0]) # Near node 2
    is_goal = planner.is_overall_goal_reached(agent_pos_np) # Should not be True yet, as get_current_waypoint_info hasn't been called to advance past it
    assert not is_goal 
    
    wp_info = planner.get_current_waypoint_info(agent_pos_np) # Reaches node 2
    print(f"Waypoint Info (after reaching node 2 - the goal): {wp_info}") # Should be None as plan is complete
    assert wp_info is None 

    is_goal_final = planner.is_overall_goal_reached(agent_pos_np) # Now check
    print(f"Is overall goal reached (after plan completion): {is_goal_final}")
    assert is_goal_final

    # Test 3: No path
    print("\nTest 3: No path")
    planner.set_global_goal(goal_node_id=99) # Non-existent goal
    plan_success_no_path = planner.generate_plan(start_node_id=0)
    assert not plan_success_no_path
    assert not planner.current_plan_node_ids

    planner.reset_plan()
    print("PlanningModule conceptual test finished.")