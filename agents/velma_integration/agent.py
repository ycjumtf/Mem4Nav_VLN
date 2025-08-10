# type: ignore
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import os
import json
import time
import random
import shutil

try:
    from mem4nav_core.perception_processing.feature_utils import MultimodalFeatureProcessor 
    from mem4nav_core.spatial_representation.sparse_octree import SparseOctree
    from mem4nav_core.spatial_representation.semantic_graph import SemanticGraph, GraphNode
    from mem4nav_core.memory_system.memory_retrieval import MemoryRetrieval, RetrievalResultItem, ShortTermMemoryEntry
except ImportError:

    print("Warning: VelmaMem4NavAgent using placeholders for Mem4Nav core components.")
    class MultimodalFeatureProcessor(nn.Module):   
        def __init__(self, *args, **kwargs): super().__init__(); self.fused_embedding_dim=384
        def process_panorama(self, *args, **kwargs): return torch.randn(1,384), torch.randn(1,1,224,224), torch.randn(1,64), None
    class SparseOctree:   
        def __init__(self, *args, **kwargs): pass
        def _get_morton_code(self, pos_np: np.ndarray) -> int: return hash(pos_np.tobytes())
    class GraphNode:   
        def __init__(self,node_id,pos,**kwargs): self.id=node_id; self.position=pos
    class SemanticGraph:   
        def __init__(self, *args, **kwargs): self.nodes={}; self._nid_counter=0
        def add_or_update_node(self, emb, pos): nid=self._nid_counter; self.nodes[nid]=GraphNode(nid,pos); self._nid_counter+=1; return self.nodes[nid]
    class ShortTermMemoryEntry:   
        def __init__(self, object_id, relative_position, **kwargs): self.object_id=object_id; self.relative_position=relative_position
    RetrievalResultItem = Any   
    class MemoryRetrieval(nn.Module):   
        def __init__(self, *args, **kwargs): super().__init__()
        def write_observation(self, *args, **kwargs): return torch.randn(256) 
        def retrieve_memory(self, *args, **kwargs) -> Tuple[str, List[Any]]: return "LTM", []
        def update_current_step(self, step): pass
        def get_ltm_module(self): return LongTermMemoryPlaceholder()   
    class LongTermMemoryPlaceholder:   
        def get_current_ltm_read_token_for_key(self, key, default_if_new=True): return torch.zeros(128)



try:
    from ..modular_pipeline.perception import PerceptionModule 
    from ..modular_pipeline.mapping import MappingModule       
except ImportError:
    print("Warning: VelmaMem4NavAgent using placeholder for Perception/Mapping modules.")
    class PerceptionModule(nn.Module):   
        def __init__(self, config: Dict[str, Any], device: torch.device):
            super().__init__()
            self.device = device
            processor_config = config.get('multimodal_feature_processor', {})
            self.feature_processor = MultimodalFeatureProcessor(
                visual_frontend_output_dim=processor_config.get('vf_output_dim', 256),
                unidepth_model_path=processor_config.get('unidepth_model_path'),
                unidepth_internal_feature_dim=processor_config.get('unidepth_internal_feature_dim', 128),
                device=device
            )
        def process_observation(self, raw_observation: Dict[str, Any]) -> \
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, Any]]]:
            fused_embedding, _, camera_embedding, _ = self.feature_processor.process_panorama(raw_observation['rgb'])
            current_pos = torch.tensor(raw_observation['current_pose'][:3], dtype=torch.float32, device=self.device)
            return fused_embedding.squeeze(0), current_pos, None, {'camera_embedding_ct': camera_embedding.squeeze(0)}

    class MappingModule(nn.Module):   
        def __init__(self, config: Dict[str, Any], memory_retriever: MemoryRetrieval, octree: SparseOctree, semantic_graph: SemanticGraph, device: torch.device):
            super().__init__(); self.memory_retriever = memory_retriever; self.octree = octree; self.semantic_graph=semantic_graph; self.current_semantic_node = None; self.device = device
        def update_maps_and_memory(self, p_t, v_t, obs_info):
            # Simplified: create/update current semantic node based on p_t, v_t
            self.current_semantic_node = self.semantic_graph.add_or_update_node(v_t, p_t.cpu().numpy())
            p_u_c = torch.tensor(self.current_semantic_node.position, dtype=torch.float32, device=self.device)
            octree_key = self.octree._get_morton_code(p_t.cpu().numpy())
            self.memory_retriever.write_observation(octree_key, obs_info.get('object_id','obj'), p_t, v_t, p_u_c)
        def retrieve_memories_for_policy(self, p_t, v_t):
            p_u_c = torch.tensor(self.current_semantic_node.position, dtype=torch.float32, device=self.device) if self.current_semantic_node else p_t
            return self.memory_retriever.retrieve_memory(v_t, p_t, p_u_c)
        def reset_state(self): self.current_semantic_node = None



try:
    from external_models.VELMA_main.vln.agent import Agent as VelmaBaseAgent 
    from external_models.VELMA_main.vln.env import ClipEnv 
    from external_models.VELMA_main.llm.query_llm import OpenAI_LLM 
    from transformers import AutoTokenizer, AutoModelForCausalLM 
except ImportError:
    print("Warning: VELMA specific imports not found. Using placeholders for VelmaBaseAgent, ClipEnv, OpenAI_LLM.")
    def get_nav_from_actions(actions, instance, env):
        return type('Navigation', (), {
            'actions': actions,
            'instance': instance,
            'env': env,
            'validate_action': lambda x: x if x in ['forward', 'left', 'right', 'turn_around', 'stop'] else 'stop',
            'step': lambda x: None
        })()
    class VelmaBaseAgent:   
        def __init__(self, query_func, env, instance, prompt_template_prefix): self.query_func = query_func; self.env=env; self.instance=instance; self.init_prompt=prompt_template_prefix; self.landmarks=instance.get('landmarks',[])
        def run(self,max_steps): actions=['stop']; nav_lines=['obs1','act_desc']; q_count=1; return type('Nav',(),{'actions':actions})(),nav_lines,[],q_count
        def extract_next_action(self, output, prompt_text): return 'forward' 
    class ClipEnv:   
        def __init__(self, *args, **kwargs): self.current_pano_id = "mock_pano_0"; self.current_heading_deg=0.0; self.current_pose_xyz=np.array([0.,0.,0.])
        def get_current_observation_for_mem4nav(self): return {'rgb': torch.rand(3,224,224), 'current_pose': self.current_pose_xyz} 
        def step(self, action_str): self.current_pano_id=f"mock_pano_{random.randint(1,100)}"; self.current_heading_deg=(self.current_heading_deg+30)%360; self.current_pose_xyz += np.random.rand(3) # mock
    class OpenAI_LLM:   
         def __init__(self, *args, **kwargs): pass
         def query_api(self, prompt): return prompt + " forward" 

try:
    from ..base_vln_agent import BaseVLNAgent
except ImportError:
    class BaseVLNAgent(nn.Module):   
        def __init__(self, config, device): super().__init__(); self.config=config; self.device=device
        def reset(self, instruction_data=None): pass
        def step(self, observation, instruction_data=None): return "stop"


class VelmaMem4NavAgent(BaseVLNAgent):
    """
    VELMA agent augmented with Mem4Nav.
    It uses Mem4Nav to generate structured memory summaries that are injected
    into the LLM's prompt.
    """
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        
        self.agent_config = config.get('velma_agent', {})
        mem_config = config.get('mem4nav', {})
        octree_config = self.agent_config.get('octree', {}) 
        sg_config = self.agent_config.get('semantic_graph', {})

        self.octree = SparseOctree(
            world_size=mem_config.get('world_size', 100.0),
            max_depth=mem_config.get('octree_depth', 16),
            embedding_dim=mem_config.get('embedding_dim', 256),
            device=device
        )
        self.semantic_graph = SemanticGraph(
            node_creation_similarity_threshold=sg_config.get('node_creation_similarity_threshold', 0.5),
            device=device
        )
        self.memory_retriever = MemoryRetrieval(
            ltm_embedding_dim=mem_config.get('embedding_dim', 256),
            ltm_transformer_depth=mem_config.get('lt_depth', 6),
            ltm_transformer_heads=mem_config.get('lt_heads', 8),

            device=device
        )
        self.perception_module = PerceptionModule(self.agent_config.get('perception', {}), device)
        self.mapping_module = MappingModule(self.agent_config.get('mapping', {}),
                                            self.memory_retriever, self.octree, self.semantic_graph, device)

        self.llm_model_name = self.agent_config.get('llm_model', 'openai/text-davinci-003')
        self.hf_auth_token = self.agent_config.get('hf_auth_token')
        self.llm = None
        self.tokenizer = None
        self._setup_llm()

  
        self.velma_env = ClipEnv(
            graph_dir=self.agent_config.get('graph_dir', 'path/to/velma/graph_data'),

            panoCLIP=None 
        )
        

        self.prompt_template_prefix = self.agent_config.get(
            'prompt_template_prefix',
            'Navigate to the described target location!\nAction Space: forward, left, right, turn_around, stop\nNavigation Instructions: "{instruction_text}"\n'
        )
        self.max_llm_prompt_tokens = self.agent_config.get('max_llm_prompt_tokens', 2000) 
        self.current_episode_instance: Optional[Dict] = None
        self.velma_internal_agent: Optional[VelmaBaseAgent] = None # To reuse VELMA's stepping logic

    def _setup_llm(self):
        if self.llm_model_name.startswith("openai/"):
            self.llm = OpenAI_LLM(model_name=self.llm_model_name.split("/", 1)[1], 
                                  api_key=os.environ.get("OPENAI_API_KEY")) # Or from config
            self.tokenizer = None 
        else: 
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name, token=self.hf_auth_token)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token # Common practice
            
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name, 
                torch_dtype=torch.float16, # Or bfloat16 for newer GPUs
                device_map="auto", 
                token=self.hf_auth_token
            ).eval() # Set to eval mode for inference
        print(f"VELMA LLM ({self.llm_model_name}) setup complete.")

    def _verbalize_memory(self, mem_source: str, mem_data: List[RetrievalResultItem]) -> str:
        """Converts retrieved Mem4Nav data into verbalized text for LLM prompt."""
        if not mem_data:
            return "Past memory: None."

        verbalized_memories = []
        # Paper Appendix A.2.2: "at (xj,yj) saw oj status sj"
        for item in mem_data[:3]: # Limit to top 3 for prompt brevity as in paper's LTM example
            if mem_source == "STM":
                entry: ShortTermMemoryEntry = item   

                pos_str = f"relative_pos ({entry.relative_position[0]:.1f},{entry.relative_position[1]:.1f})"
                verbalized_memories.append(f"at {pos_str} recently saw {entry.object_id}")
            elif mem_source == "LTM":
                key, recon_v, recon_p, recon_d = item

                pos_str = f"approx_abs_pos ({recon_p[0]:.1f},{recon_p[1]:.1f})"
   
                obj_desc_from_ltm = f"something (key {key})"
                verbalized_memories.append(f"at {pos_str} previously saw {obj_desc_from_ltm}")
        
        if not verbalized_memories:
            return "Past memory: None."
        return "Past memory: " + " | ".join(verbalized_memories) + "."

    def _build_augmented_query_func_for_velma(self):
        """
        Creates the query_func for VELMA's internal agent, augmented with Mem4Nav.
        This function will be called by VELMA's agent at each step *before* querying the LLM.
        """
        def augmented_query_func(prompt_text_from_velma: str, llm_hints: Optional[Dict] = None) -> Tuple[str, int, Dict]:
 
      
            raw_obs_for_mem4nav = self.velma_env.get_current_observation_for_mem4nav()
            # raw_obs_for_mem4nav = {'rgb': ..., 'current_pose': np.array([x,y,z])}

            v_t, p_t, _, perception_extras = self.perception_module.process_observation(raw_obs_for_mem4nav)
            
   
            # It internally updates/gets u_c.
            obs_details_for_mapping = {
                'object_id_for_stm': perception_extras.get('detected_object_id', 'scene_view'),
                # octree_key will be derived inside mapping_module from p_t
            }
            self.mapping_module.update_maps_and_memory(p_t, v_t, obs_details_for_mapping)

      
            mem_source, mem_data = self.mapping_module.retrieve_memories_for_policy(p_t, v_t)

            verbalized_memory_text = self._verbalize_memory(mem_source, mem_data)


            final_prompt_for_llm = prompt_text_from_velma.replace("{{memory_text}}", verbalized_memory_text)

            if self.tokenizer: 
                tokenized_prompt = self.tokenizer.encode(final_prompt_for_llm)
                if len(tokenized_prompt) > self.max_llm_prompt_tokens:
  
                    print(f"Warning: Prompt length {len(tokenized_prompt)} exceeds max {self.max_llm_prompt_tokens}")

            
            api_calls = 0
            if self.llm_model_name.startswith("openai/"):
                llm_response_text = self.llm.query_api(final_prompt_for_llm)   
                api_calls = 1
            else: 
                inputs = self.tokenizer(final_prompt_for_llm, return_tensors="pt", padding=True, truncation=True, max_length=self.max_llm_prompt_tokens).to(self.device) #type: ignore
                with torch.no_grad():
                    # Generation config can be tuned (do_sample, temperature, max_new_tokens etc.)
                    # VELMA typically generates just the next action.
                    outputs = self.llm.generate(**inputs, max_new_tokens=10, pad_token_id=self.tokenizer.eos_token_id)   
                llm_response_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)   
            

            full_output_sequence = final_prompt_for_llm + " " + llm_response_text.strip()

            return full_output_sequence, api_calls, llm_hints if llm_hints is not None else {} # Pass through hints

        return augmented_query_func

    def reset(self, instruction_data: Optional[Dict[str, Any]] = None) -> None:
        """Resets the agent for a new episode."""
        super().reset(instruction_data)
        self.mapping_module.reset_state()
        self.memory_retriever.clear_all_memory()
        self.current_episode_instance = instruction_data # This is the instance dict from VELMA's dataset
        

        velma_prompt_template_prefix = self.prompt_template_prefix.format(
            instruction_text=self.current_episode_instance['navigation_text']   
        ) + "\nMemory Cues: {{memory_text}}\nAction Sequence:\n" # Ensure placeholder for Mem4Nav

        self.velma_internal_agent = VelmaBaseAgent(
            query_func=self._build_augmented_query_func_for_velma(),
            env=self.velma_env,
            instance=self.current_episode_instance,   
            prompt_template_prefix=velma_prompt_template_prefix
        )
        # Reset VELMA's environment to the start of the episode
        self.velma_env.init_state(self.current_episode_instance['route_panoids'][0], self.current_episode_instance['start_heading'])   

        self.memory_retriever.update_current_step(0)
        print(f"VelmaMem4NavAgent: Reset for episode {self.current_episode_instance['idx'] if self.current_episode_instance else 'N/A'}") #type: ignore

    def step(self, observation: Dict[str, Any], instruction_data: Optional[Dict[str, Any]] = None) -> str:   
        """
        Performs one step of navigation using VELMA's logic augmented by Mem4Nav.
        The 'observation' here is VELMA's observation, which might be different from
        what our PerceptionModule takes. We need to ensure PerceptionModule gets raw RGB.
        """
        if not self.velma_internal_agent or not self.current_episode_instance:
            raise RuntimeError("Agent not reset. Call reset() before step().")


        if not hasattr(self.velma_internal_agent, 'nav'): # First step after init
             self.velma_internal_agent.nav = get_nav_from_actions(['init'], self.current_episode_instance, self.velma_env)   


        try:
            from external_models.VELMA_main.vln.prompt_builder import get_navigation_lines
        except ImportError:
            def get_navigation_lines(*args, **kwargs): return ["Mock observation line.", f"{len(args[0].actions if args[0] else [])}."], [False, False] #type: ignore

        step_idx_for_nav_lines = len(self.velma_internal_agent.nav.actions)   
        current_nav_lines, _ = get_navigation_lines(
            self.velma_internal_agent.nav,   
            self.velma_env,
            self.velma_internal_agent.landmarks,   
            self.current_episode_instance.get('traffic_flow'),   
            step_id=step_idx_for_nav_lines 
        )
        
        if current_nav_lines and current_nav_lines[-1].strip().endswith('.'):
            if current_nav_lines[-1].strip()[:-1].isdigit():
                 current_nav_lines.pop()


        prompt_for_llm = self.velma_internal_agent.init_prompt + "\n".join(current_nav_lines)   
        if not prompt_for_llm.endswith("\n"):
            prompt_for_llm += "\n"
        if len(self.velma_internal_agent.nav.actions) < 55 :   
             prompt_for_llm += f"{len(self.velma_internal_agent.nav.actions)}."   


        full_llm_output_sequence, _, _ = self.velma_internal_agent.query_func(prompt_for_llm, None) #type: ignore
        
        # Extract action from LLM's full output sequence
        action_str = self.velma_internal_agent.extract_next_action(full_llm_output_sequence, prompt_for_llm)   
        action_str = self.velma_internal_agent.nav.validate_action(action_str)   

        if action_str != "stop":
            self.velma_internal_agent.nav.step(action_str)   
            self.velma_env.step(action_str) # Update VELMA's environment state

        self.memory_retriever.update_current_step(self.current_step_count + 1) # Assuming current_step_count is managed by BaseVLNAgent or ModularAgent
        
        return action_str


if __name__ == '__main__':
    print("--- Conceptual Test for VelmaMem4NavAgent ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Mock configuration
    mock_agent_config = {
        'mem4nav': { # For Mem4Nav components
            'world_size': 50.0, 'octree_depth': 10, 'embedding_dim': 256, # d_emb (vf_out + unidepth_internal)
            'lt_depth': 2, 'lt_heads': 2, 'lt_max_elements': 50,
            'st_capacity': 10,
        },
        'velma_agent': { # For VELMA specific parts
            'llm_model': 'openai/text-davinci-003', # Mock will use simple echo
            'graph_dir': 'path/to/dummy/graph_data', # Needs to exist or ClipEnv mock needs to be better
            'perception': { # For our PerceptionModule used by VELMA agent
                'multimodal_feature_processor': {
                    'vf_output_dim': 128,
                    'unidepth_model_path': None,
                    'unidepth_internal_feature_dim': 128,
                }
            },
            'mapping': {} # Config for our MappingModule
        }
    }
    if not os.path.exists('path/to/dummy/graph_data'): os.makedirs('path/to/dummy/graph_data')


    velma_mem4nav_agent = VelmaMem4NavAgent(mock_agent_config, device)

    mock_episode_instance = {
        'idx': 'test_episode_1',
        'navigation_text': "Go to the red car then turn left at the big building.",
        'route_panoids': ['mock_pano_0', 'mock_pano_1', 'mock_pano_2'], # Sequence of panoids
        'start_heading': 0.0,
        'landmarks': ['red car', 'big building'] # From VELMA's landmark extractor
    }

    velma_mem4nav_agent.reset(mock_episode_instance)

    for i in range(3):
        print(f"\n--- VelmaMem4NavAgent Step {i+1} ---")

        action = velma_mem4nav_agent.step(observation={}) 
        print(f"  Agent Action: {action}")
        if action == "stop":
            print("  Episode stopped by agent.")
            break
    
    print("\nVelmaMem4NavAgent conceptual test finished.")
    if os.path.exists('path/to/dummy/graph_data'): shutil.rmtree('path/to/dummy')