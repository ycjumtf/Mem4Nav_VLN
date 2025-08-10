from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Optional, List, Tuple, Union


# from mem4nav_core.memory_system.memory_retrieval import MemoryRetrieval, RetrievalResultItem
# from mem4nav_core.perception_processing.feature_utils import MultimodalFeatureProcessor

class BaseVLNAgent(nn.Module, ABC):
    """
    Abstract Base Class for Vision-and-Language Navigation (VLN) agents.
    Defines the common interface for all agent architectures to be used
    within the Mem4Nav experimental framework.
    """
    def __init__(self,
                 agent_config: Dict[str, Any],
                 memory_retriever: Optional[Any] = None, # Type: MemoryRetrieval
                 feature_processor: Optional[Any] = None, # Type: MultimodalFeatureProcessor
                 device: Optional[torch.device] = None):
        """
        Initializes the base VLN agent.

        Args:
            agent_config (Dict[str, Any]): Agent-specific configuration parameters.
            memory_retriever (Optional[MemoryRetrieval]): An instance of the MemoryRetrieval system.
                                                           If None, the agent might operate without Mem4Nav
                                                           or initialize its own.
            feature_processor (Optional[MultimodalFeatureProcessor]): An instance for processing sensor
                                                                     data into features.
            device (Optional[torch.device]): The PyTorch device to run the agent on.
        """
        super().__init__()
        self.agent_config = agent_config
        self.memory_retriever = memory_retriever
        self.feature_processor = feature_processor
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Common state variables for an episode
        self.current_episode_id: Optional[str] = None
        self.instruction_tokens: Optional[List[Any]] = None # Processed instruction
        self.history: List[Dict[str, Any]] = [] # History of states, observations, actions, etc.
        self.current_step_in_episode: int = 0

        # To be defined by subclasses, e.g., output dim of the policy head
        self.action_space_size: Optional[int] = None


    @abstractmethod
    def reset(self, episode_id: str, instruction: Any, initial_observation: Dict[str, Any]):
        """
        Resets the agent's state for a new episode.

        Args:
            episode_id (str): Unique identifier for the new episode.
            instruction (Any): The navigation instruction for the episode (can be text, tokens, etc.).
            initial_observation (Dict[str, Any]): The initial observation from the environment
                                                 (e.g., RGB image, current pose).
        """
        self.current_episode_id = episode_id
        self.instruction_tokens = self._process_instruction(instruction)
        self.history = []
        self.current_step_in_episode = 0
        if self.memory_retriever:
            self.memory_retriever.clear_all_memory()
            self.memory_retriever.update_current_step(self.current_step_in_episode)
        print(f"Agent reset for episode: {episode_id}")

    @abstractmethod
    def _process_instruction(self, raw_instruction: Any) -> Any:
        """
        Processes the raw navigation instruction into a format usable by the agent.
        This might involve tokenization, embedding, etc.

        Args:
            raw_instruction (Any): The raw instruction from the dataset.

        Returns:
            Any: The processed instruction.
        """
        pass

    @abstractmethod
    def _process_observation(self, observation: Dict[str, Any]) -> \
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict[str, Any]]]:
        """
        Processes the raw observation from the environment to extract relevant features.
        This is where `feature_utils.MultimodalFeatureProcessor` would be used.

        Args:
            observation (Dict[str, Any]): Raw observation from the environment
                                         (e.g., {'rgb': PIL_Image, 'depth': np.array, 'pose': np.array}).

        Returns:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict[str, Any]]]:
                - current_observation_embedding (v_t): The fused multimodal embedding for memory systems.
                - current_absolute_position (p_t): Agent's current global position.
                - current_semantic_node_position (p_u_c): Position of current semantic graph node.
                - additional_processed_obs (Dict): Other processed information for the policy.
        """
        pass


    def _update_memory_and_maps(self,
                                unique_key: Any,
                                obs_embedding_v_t: torch.Tensor,
                                abs_position_p_t: torch.Tensor,
                                sem_node_pos_p_u_c: torch.Tensor,
                                stm_object_id: Optional[Any] = "observation"):
        """
        Helper method to write to Mem4Nav systems (STM and LTM) via MemoryRetriever.
        This method would also be responsible for updating the spatial representations (Octree, Graph)
        with the new observation and getting/updating their LTM tokens.
        This part needs careful design: Spatial representations hold LTM tokens. MemoryRetriever.write
        computes the *new* LTM token. This new token needs to be stored back into the spatial element.

        Args:
            unique_key: Key for the observation (e.g., Morton code from p_t).
            obs_embedding_v_t: Fused multimodal embedding.
            abs_position_p_t: Current absolute position.
            sem_node_pos_p_u_c: Current semantic node position.
            stm_object_id: Object ID for STM.
        """
        if self.memory_retriever:
            # This `write_observation` updates LTM internal state and STM.
            # It also returns the new LTM token that the spatial element (octree leaf/graph node)
            # associated with `unique_key` should now store.
            new_ltm_token_for_spatial_element = self.memory_retriever.write_observation(
                unique_observation_key=unique_key,
                object_id_for_stm=stm_object_id,
                current_absolute_position=abs_position_p_t,
                current_observation_embedding=obs_embedding_v_t,
                current_semantic_node_position=sem_node_pos_p_u_c
            )
            # The agent (or its mapping component) needs to ensure that the
            # OctreeLeaf or GraphNode corresponding to `unique_key` gets its
            # `ltm_current_token` updated to `new_ltm_token_for_spatial_element`.
            # This logic might live in a specific `mapping_module` for the agent.
            # For base agent, we just note that memory_retriever was called.
            # print(f"BaseAgent: Memory updated for key {unique_key}. New LTM token generated.")
            pass # Actual update of OctreeLeaf/GraphNode LTM token handled by specific agent's mapping logic.

    def _retrieve_from_memory(self,
                             obs_embedding_v_t: torch.Tensor,
                             abs_position_p_t: torch.Tensor,
                             sem_node_pos_p_u_c: torch.Tensor
                             ) -> Tuple[Optional[str], List[Any]]: # Type: List[RetrievalResultItem]
        """
        Helper method to retrieve from Mem4Nav.
        """
        if self.memory_retriever:
            source, retrieved_items = self.memory_retriever.retrieve_memory(
                current_observation_embedding=obs_embedding_v_t,
                current_absolute_position=abs_position_p_t,
                current_semantic_node_position=sem_node_pos_p_u_c
            )
            return source, retrieved_items
        return None, []

    @abstractmethod
    def _get_policy_inputs(self,
                           processed_observation: Dict[str, Any],
                           retrieved_memory_source: Optional[str],
                           retrieved_memory_items: List[Any], # Type: List[RetrievalResultItem]
                           instruction_tokens: Any
                           ) -> Dict[str, Any]:
        """
        Prepares the final set of inputs for the agent's policy network.
        This includes current processed observation, retrieved memories (STM/LTM),
        and processed instruction. This is where aggregation of memory items
        into a final m_t might happen, or where they are prepared for attention.

        Args:
            processed_observation (Dict[str, Any]): Output from _process_observation.
            retrieved_memory_source (Optional[str]): 'STM' or 'LTM'.
            retrieved_memory_items (List[RetrievalResultItem]): Data from memory.
            instruction_tokens (Any): Processed instruction.

        Returns:
            Dict[str, Any]: Inputs ready for the policy model's forward pass.
        """
        pass

    @abstractmethod
    def _policy_step(self, policy_inputs: Dict[str, Any]) -> Any:
        """
        Performs a forward pass through the agent's policy model to decide an action.

        Args:
            policy_inputs (Dict[str, Any]): Inputs prepared by _get_policy_inputs.

        Returns:
            Any: The action chosen by the policy (e.g., action index, waypoint).
        """
        pass

    def step(self, observation: Dict[str, Any]) -> Any:
        """
        The main method called at each time step of an episode.
        It processes the observation, interacts with memory, and decides an action.

        Args:
            observation (Dict[str, Any]): Raw observation from the environment.

        Returns:
            Any: The action to take in the environment.
        """
        self.current_step_in_episode += 1
        if self.memory_retriever:
            self.memory_retriever.update_current_step(self.current_step_in_episode)

        # 1. Process current observation (get v_t, p_t, p_u_c, etc.)
        obs_emb_v_t, abs_pos_p_t, sem_node_pos_p_u_c, additional_proc_obs = \
            self._process_observation(observation)

        # Ensure essential components are available if memory is used
        if self.memory_retriever and \
           (obs_emb_v_t is None or abs_pos_p_t is None or sem_node_pos_p_u_c is None):
            raise ValueError("Observation processing did not yield necessary embeddings/positions for memory operations.")

        # 2. (Optional for some agents, critical for others) Update memory systems and spatial maps
        # This step's placement (before or after retrieval) can vary.
        # For Mem4Nav, historical observations are written. So, this observation is now history.
        # The `unique_key` would come from `abs_pos_p_t` (e.g. Morton code).
        # This needs to be implemented carefully in the concrete agent.
        # For now, we assume that if an agent uses Mem4Nav, it will call _update_memory_and_maps.
        # A concrete agent's _process_observation might return the unique_key.
        # Example: unique_key_for_mem = self._get_key_from_position(abs_pos_p_t)
        # if unique_key_for_mem and obs_emb_v_t and abs_pos_p_t and sem_node_pos_p_u_c:
        #    self._update_memory_and_maps(unique_key_for_mem, obs_emb_v_t, abs_pos_p_t, sem_node_pos_p_u_c)


        # 3. Retrieve relevant memories
        retrieved_source, retrieved_items = None, []
        if self.memory_retriever and obs_emb_v_t is not None and abs_pos_p_t is not None and sem_node_pos_p_u_c is not None:
            retrieved_source, retrieved_items = self._retrieve_from_memory(
                obs_emb_v_t, abs_pos_p_t, sem_node_pos_p_u_c
            )

        # 4. Prepare inputs for the policy
        policy_inputs = self._get_policy_inputs(
            additional_proc_obs or {}, # Pass other processed observation data
            retrieved_source,
            retrieved_items,
            self.instruction_tokens
        )

        # 5. Decide action using the policy
        action = self._policy_step(policy_inputs)

        # 6. Record history (optional, for analysis or some RL approaches)
        self.history.append({
            'step': self.current_step_in_episode,
            'raw_observation': observation, # Could be large, use with caution
            'processed_observation_extras': additional_proc_obs,
            'retrieved_memory_source': retrieved_source,
            # 'retrieved_memory_items': retrieved_items, # Could be large
            'action': action
        })

        return action

    def get_history(self) -> List[Dict[str, Any]]:
        """Returns the history of the current episode."""
        return self.history

    # --- Methods for training (to be called by Trainer) ---
    def forward(self, batch: Dict[str, Any]) -> Any:
        """
        Defines the forward pass for training.
        This will vary greatly depending on the agent architecture and training regime
        (e.g., imitation learning, reinforcement learning).
        Subclasses must implement this if they are to be trained.
        For imitation learning, this might take a batch of (observation, instruction)
        and produce action logits.
        """
        raise NotImplementedError("Forward pass for training must be implemented by subclass.")

    def compute_loss(self, model_outputs: Any, batch_ground_truth: Dict[str, Any]) -> torch.Tensor:
        """
        Computes the loss for training.
        Subclasses must implement this.
        """
        raise NotImplementedError("Loss computation must be implemented by subclass.")


if __name__ == '__main__':
    # This is an abstract class and cannot be instantiated directly.
    # Example of how a concrete agent might inherit:

    class DummyAgent(BaseVLNAgent):
        def __init__(self, agent_config, memory_retriever=None, feature_processor=None, device=None):
            super().__init__(agent_config, memory_retriever, feature_processor, device)
            print("DummyAgent initialized.")
            # Example: define a simple policy network
            self.policy_net = nn.Linear(10, 4).to(self.device) # Input 10, output 4 actions
            self.action_space_size = 4

        def _process_instruction(self, raw_instruction: Any) -> Any:
            print(f"DummyAgent processing instruction: {raw_instruction}")
            return {"text": str(raw_instruction), "tokens": [0,1,2]} # Dummy processed instruction

        def _process_observation(self, observation: Dict[str, Any]) -> \
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict[str, Any]]]:
            print(f"DummyAgent processing observation: {observation.keys()}")
            # Dummy processed observation
            dummy_emb_dim = self.agent_config.get("embedding_dim", 64)
            obs_emb_v_t = torch.randn(dummy_emb_dim, device=self.device)
            abs_pos_p_t = torch.tensor(observation.get('pose', [0,0,0]), dtype=torch.float32, device=self.device)
            sem_node_pos_p_u_c = abs_pos_p_t - torch.tensor([0.1, 0.1, 0.0], dtype=torch.float32, device=self.device) # Dummy
            
            # Simulate getting a unique key (e.g., Morton code)
            # In a real agent, this would come from spatial processing.
            unique_key_for_mem = int(abs_pos_p_t[0].item() * 100) # Dummy key
            
            # Update memory (typically called here or by a mapping module)
            if self.memory_retriever and obs_emb_v_t is not None: # Check if memory_retriever exists
                 # This is where the call to self._update_memory_and_maps would conceptually happen
                 # For DummyAgent, we can call it directly if we assume it also handles mapping.
                 # Let's assume DummyAgent needs to make this call explicitly.
                 # print(f"DummyAgent: Preparing to update memory for key {unique_key_for_mem}")
                 # self._update_memory_and_maps(unique_key_for_mem, obs_emb_v_t, abs_pos_p_t, sem_node_pos_p_u_c)
                 pass # Actual call to _update_memory_and_maps would be more involved with mapping components.


            return obs_emb_v_t, abs_pos_p_t, sem_node_pos_p_u_c, {"image_features": torch.randn(10, device=self.device)}

        def _get_policy_inputs(self, processed_observation_extras: Dict[str, Any],
                               retrieved_memory_source: Optional[str],
                               retrieved_memory_items: List[Any], #Type: List[RetrievalResultItem]
                               instruction_tokens: Any) -> Dict[str, Any]:
            print(f"DummyAgent getting policy inputs. Memory source: {retrieved_memory_source}, Items: {len(retrieved_memory_items)}")
            # Aggregate or use memory items here. For dummy, just use image_features.
            policy_input_features = processed_observation_extras.get("image_features", torch.randn(10, device=self.device))
            return {"features": policy_input_features}

        def _policy_step(self, policy_inputs: Dict[str, Any]) -> Any:
            features = policy_inputs["features"]
            action_logits = self.policy_net(features)
            action = torch.argmax(action_logits).item()
            print(f"DummyAgent policy step. Input features shape: {features.shape}, Action: {action}")
            return action

        def reset(self, episode_id: str, instruction: Any, initial_observation: Dict[str, Any]):
            super().reset(episode_id, instruction, initial_observation)
            # Process initial observation to update memory, etc.
            obs_emb_v_t, abs_pos_p_t, sem_node_pos_p_u_c, _ = self._process_observation(initial_observation)
            # Example: update memory with initial observation
            # unique_key_for_mem = int(abs_pos_p_t[0].item() * 100) # Dummy key
            # if self.memory_retriever and obs_emb_v_t is not None:
            #    self._update_memory_and_maps(unique_key_for_mem, obs_emb_v_t, abs_pos_p_t, sem_node_pos_p_u_c)
            print(f"DummyAgent: Initial observation processed for episode {episode_id}.")



    print("BaseVLNAgent definition complete.")