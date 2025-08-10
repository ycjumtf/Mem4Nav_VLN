# type: ignore
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset 
import os
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Iterable

try:
    from .losses import CombinedAgentLoss, MaskedVisualReconstructionLoss 
    from .optimizers import create_optimizer, create_scheduler
    from ..agents.base_vln_agent import BaseVLNAgent 
    from mem4nav_core.memory_system.long_term_memory import LongTermMemory 
    from mem4nav_core.perception_processing.feature_utils import VisualFrontend 
    from ..evaluation_utils.evaluator import Evaluator
except ImportError:
    print("Warning: Trainer using placeholders for some imported modules due to import error.")

    class BaseVLNDataset(Dataset):   
        def __init__(self, num_samples=10, phase_name="unknown"): self.num_samples = num_samples; self.phase_name=phase_name
        def __len__(self): return self.num_samples
        def __getitem__(self, idx):
            if self.phase_name == "phase1_visual": # For visual reconstruction
                return {'panoramas': torch.randn(3, 224, 224), 'masks': torch.rand(1, (224//16)**2) > 0.75}
            elif self.phase_name == "phase2_ltm_synthetic": # For LTM cycle consistency
                return {'prev_ltm_token_input': torch.randn(128), 'observation_embedding': torch.randn(128)}
            elif self.phase_name == "phase3_vln": # For E2E VLN
                 return {'dummy_episode_data': torch.randn(10), 'gt_actions': torch.randint(0,4,(1,))}
            return {}

    class CombinedAgentLoss(nn.Module):   
        def __init__(self, **kwargs): super().__init__(); self.cycle_consistency_loss_fn=lambda x,y: torch.tensor(0.1); self.navigation_loss_fn=lambda x,y: torch.tensor(0.3); self.visual_reconstruction_loss_fn=lambda x,y,z: torch.tensor(0.2)
        def forward(self, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
            mock_loss = torch.tensor(0.5, requires_grad=True)
            return mock_loss, {'total_loss': 0.5, 'navigation_loss':0.3, 'cycle_consistency_loss':0.1, 'visual_reconstruction_loss': 0.2}
    class MaskedVisualReconstructionLoss(nn.Module):   
        def __init__(self, **kwargs): super().__init__()
        def forward(self, *args) -> torch.Tensor: return torch.tensor(0.2, requires_grad=True)
    def create_optimizer(*args, **kwargs): return optim.Adam([nn.Parameter(torch.randn(1))], lr=1e-3)
    def create_scheduler(*args, **kwargs): return None
    class VisualFrontend(nn.Module):   
        def __init__(self, *args, **kwargs): super().__init__()
        def forward_reconstruction(self, panoramas, mask_ratio): return torch.randn(panoramas.shape[0], 196, 256), torch.randn(panoramas.shape[0], 196, 256), torch.ones(panoramas.shape[0], 196, dtype=torch.bool)
    class LongTermMemory(nn.Module):   
        def __init__(self): super().__init__()
        def get_cycle_consistency_reconstruction(self, prev_token, obs_emb): return obs_emb + 0.01
    class BaseVLNAgent(nn.Module):   
        def __init__(self, *args, **kwargs): super().__init__(); self.device = torch.device('cpu'); self.visual_frontend: Optional[VisualFrontend] = None; self.ltm_module: Optional[LongTermMemory] = None
        def get_visual_frontend_params(self): return self.parameters()
        def get_visual_reconstruction_head_params(self): return self.parameters() # Should be distinct
        def get_ltm_params(self): return self.parameters()
        def get_policy_params(self): return self.parameters()
        def get_all_trainable_params_for_e2e(self): return self.parameters()
        def get_ltm_module(self) -> Optional[LongTermMemory]: return self.ltm_module
        def get_visual_frontend(self) -> Optional[VisualFrontend]: return self.visual_frontend
        def process_batch_for_training(self, batch_data, device): return torch.randn(2,4), torch.randint(0,4,(2,)), torch.randn(2,128), torch.randn(2,128) # logits, gt_actions, orig_v, recon_v
    class Evaluator:   
        def __init__(self, agent, dataloader, env_graph, eval_config, device): pass
        def evaluate(self): print("Mock Evaluator: Evaluating..."); return {'SPL': 0.05, "TC": 5.0}


class Trainer:
    """
    Orchestrates the three-phase training process for Mem4Nav-augmented agents.
    """
    def __init__(self,
                 agent: BaseVLNAgent,
                 train_dataloaders: Dict[str, DataLoader],
                 val_dataloader_vln: Optional[DataLoader],
                 evaluator: Optional[Evaluator], # Pass the evaluator instance
                 config: Dict[str, Any],
                 device: torch.device):
        self.agent = agent
        self.train_dataloaders = train_dataloaders
        self.val_dataloader_vln = val_dataloader_vln
        self.evaluator = evaluator # Store the evaluator
        self.config = config
        self.training_config = config.get('training', {})
        self.device = device

        self.current_phase_idx: int = 0 # 0 for Phase 1, 1 for Phase 2, 2 for Phase 3
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.best_val_metric: float = -float('inf') if self.training_config.get('higher_is_better_val_metric', True) else float('inf')

        self.checkpoint_dir = self.training_config.get('checkpoint_dir_template', "{output_dir_root}/{experiment_name}/checkpoints").format(
            output_dir_root=config.get('experiment',{}).get('output_dir_root','./outputs'),
            experiment_name=config.get('experiment',{}).get('name','default_exp')
        )
        ensure_directory_exists(self.checkpoint_dir)

        self.combined_loss_fn = CombinedAgentLoss(
            navigation_loss_weight=self.training_config.get('nav_loss_weight', 1.0),
            cycle_consistency_loss_weight=self.training_config.get('cycle_loss_weight', 0.1),
            visual_reconstruction_loss_weight=self.training_config.get('visual_recon_loss_weight', 1.0), # Weight for phase 1
            nav_loss_ignore_index=self.training_config.get('nav_loss_ignore_index', -100),
            visual_recon_loss_type=self.training_config.get('visual_recon_loss_type', 'mse')
        ).to(self.device)


    def _setup_optimizer_and_scheduler(self, phase_config: Dict[str, Any], parameters_to_train: Iterable[nn.Parameter]):
        opt_config = phase_config.get('optimizer', self.training_config.get('optimizer', {})) # Fallback to global
        optimizer = create_optimizer(parameters_to_train, opt_config)

        scheduler_config = phase_config.get('scheduler', self.training_config.get('scheduler', {}))
        
        num_epochs_phase = phase_config.get('epochs', 1)
        dataloader_key_map = ["phase1_visual", "phase2_ltm_synthetic", "phase3_vln"]
        dataloader_key = dataloader_key_map[self.current_phase_idx]

        num_training_steps_phase, num_warmup_steps_phase = None, None
        if dataloader_key in self.train_dataloaders and self.train_dataloaders[dataloader_key] is not None:
            num_training_steps_phase = len(self.train_dataloaders[dataloader_key]) * num_epochs_phase
            grad_accum = phase_config.get('gradient_accumulation_steps', 1)
            num_training_steps_phase //= grad_accum

            warmup_ratio_cfg = scheduler_config.get('num_warmup_steps_ratio')
            warmup_abs_cfg = scheduler_config.get('num_warmup_steps')
            if warmup_abs_cfg is not None:
                num_warmup_steps_phase = warmup_abs_cfg
            elif warmup_ratio_cfg is not None and num_training_steps_phase is not None:
                 num_warmup_steps_phase = int(num_training_steps_phase * warmup_ratio_cfg)
        
        scheduler = create_scheduler(optimizer, scheduler_config,
                                     num_training_steps=num_training_steps_phase,
                                     num_warmup_steps=num_warmup_steps_phase)
        return optimizer, scheduler


    def _freeze_parameters(self, model: nn.Module, params_to_train_names: Optional[List[str]] = None):
        """Freezes all parameters and then unfreezes specified ones."""
        for param in model.parameters():
            param.requires_grad = False
        if params_to_train_names:
            for name, param in model.named_parameters():
                if any(train_name_part in name for train_name_part in params_to_train_names):
                    param.requires_grad = True
        # Verify
        # for name, param in model.named_parameters():
        #     if param.requires_grad: print(f"  Training param: {name}")

    def _get_param_name_keywords_for_phase(self, phase_idx: int) -> List[str]:
        """
        Returns keywords to identify parameters to be trained in a given phase.
        The agent needs to be structured such that its components can be identified by names.
        Example: 'visual_frontend.encoder', 'ltm_module.reversible_transformer', 'policy_net'
        """
        phase_configs = self.training_config.get('phases', {})
        if phase_idx == 0: # Phase 1: Visual Frontend + its reconstruction head
            return phase_configs.get('phase1_visual', {}).get('trainable_param_keywords', ['visual_frontend'])
        elif phase_idx == 1: # Phase 2: LTM Reversible Transformer + pi_v decoder
            return phase_configs.get('phase2_ltm', {}).get('trainable_param_keywords', ['ltm_module.reversible_transformer', 'ltm_module.cycle_consistency_v_decoder'])
        elif phase_idx == 2: # Phase 3: End-to-End Navigation (e.g., policy, all of LTM, parts of VF)
            return phase_configs.get('phase3_e2e_nav', {}).get('trainable_param_keywords', ['policy_network', 'ltm_module', 'visual_frontend.rgb_feature_head']) # Example
        else:
            raise ValueError(f"Invalid phase index: {phase_idx}")

    def train_phase_1_visual(self, phase_config: Dict[str, Any]):
        print(f"\n--- Starting Training Phase 1: Visual Frontend ({phase_config.get('epochs',0)} epochs) ---")
        self.current_phase_idx = 0
        
        param_keywords = self._get_param_name_keywords_for_phase(0)
        self._freeze_parameters(self.agent, params_to_train_names=param_keywords)
        
        trainable_params = [p for p in self.agent.parameters() if p.requires_grad]
        if not trainable_params:
            print("Phase 1: No trainable parameters found based on keywords. Skipping.")
            return

        optimizer, scheduler = self._setup_optimizer_and_scheduler(phase_config, trainable_params)
        dataloader = self.train_dataloaders.get("phase1_visual")
        if not dataloader:
            print("Warning: Dataloader for 'phase1_visual' not found. Skipping Phase 1.")
            return

        visual_frontend_module = getattr(self.agent, 'visual_frontend', None) or \
                                (getattr(self.agent, 'perception_module', None) and \
                                 getattr(self.agent.perception_module, 'feature_processor', None) and \
                                 getattr(self.agent.perception_module.feature_processor, 'visual_frontend', None)) # type: ignore

        if not visual_frontend_module or not hasattr(visual_frontend_module, 'forward_reconstruction'):
            print("Error: Agent's visual_frontend or its 'forward_reconstruction' method not found. Skipping Phase 1.")
            return

        num_epochs = phase_config.get('epochs', 10)
        grad_accum_steps = phase_config.get('gradient_accumulation_steps', 1)

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.agent.train()
            epoch_loss = 0.0
            num_batches_processed = 0
            optimizer.zero_grad()

            for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Phase 1 Epoch {epoch+1}/{num_epochs}")):
                panoramas = batch_data['panoramas'].to(self.device) # (B, C, H, W)
                # Masks for MAE can be generated here or come from dataloader
                mask_ratio = phase_config.get('mae_mask_ratio', 0.75)
                
                # Call the visual frontend's reconstruction method
                # It should return: (reconstructed_patches, target_patches, boolean_loss_mask)
                reconstructed_feats, target_feats, loss_calc_mask = \
                    visual_frontend_module.forward_reconstruction(panoramas, mask_ratio=mask_ratio)

                loss, loss_comps = self.combined_loss_fn(
                    predicted_visual_features_for_recon=reconstructed_feats,
                    target_visual_features_for_recon=target_feats,
                    visual_recon_mask=loss_calc_mask
                )
                vis_recon_loss = loss_comps.get('visual_reconstruction_loss', 0.0)

                if grad_accum_steps > 1:
                    loss = loss / grad_accum_steps
                
                loss.backward()

                if (batch_idx + 1) % grad_accum_steps == 0:
                    if phase_config.get('grad_clip_norm', None):
                        nn.utils.clip_grad_norm_(trainable_params, phase_config['grad_clip_norm'])
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += vis_recon_loss # Log the unscaled recon loss component
                num_batches_processed += 1
                self.global_step +=1
                
                if batch_idx % self.training_config.get('log_interval', 100) == 0 and num_batches_processed > 0 :
                    print(f"  P1 Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, Visual Recon Loss: {vis_recon_loss:.4f}")


            avg_epoch_loss = epoch_loss / num_batches_processed if num_batches_processed > 0 else 0
            print(f"Phase 1 - Epoch {epoch+1}/{num_epochs} - Avg Visual Recon Loss: {avg_epoch_loss:.4f}")
            if scheduler: scheduler.step()
            self._save_checkpoint(f"phase1_epoch{epoch+1}")


    def train_phase_2_ltm(self, phase_config: Dict[str, Any]):
        print(f"\n--- Starting Training Phase 2: LTM Cycle-Consistency ({phase_config.get('epochs',0)} epochs) ---")
        self.current_phase_idx = 1

        param_keywords = self._get_param_name_keywords_for_phase(1)
        self._freeze_parameters(self.agent, params_to_train_names=param_keywords)
        
        trainable_params = [p for p in self.agent.parameters() if p.requires_grad]
        if not trainable_params:
            print("Phase 2: No trainable LTM parameters found. Skipping.")
            return
            
        optimizer, scheduler = self._setup_optimizer_and_scheduler(phase_config, trainable_params)
        dataloader = self.train_dataloaders.get("phase2_ltm_synthetic")
        if not dataloader:
            print("Warning: Dataloader for 'phase2_ltm_synthetic' not found. Skipping Phase 2.")
            return

        ltm_module = self.agent.get_ltm_module()
        if not ltm_module:
            print("Error: Agent does not provide access to LTM module (agent.get_ltm_module()). Skipping Phase 2.")
            return

        num_epochs = phase_config.get('epochs', 5)
        grad_accum_steps = phase_config.get('gradient_accumulation_steps', 1)

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.agent.train() # Ensure LTM (and its pi_v decoder) are in train mode
            epoch_loss = 0.0
            num_batches_processed = 0
            optimizer.zero_grad()

            for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Phase 2 Epoch {epoch+1}/{num_epochs}")):
                prev_ltm_token_input = batch_data['prev_ltm_token_input'].to(self.device) # d_emb
                obs_embedding_v_t = batch_data['observation_embedding'].to(self.device)    # d_emb

                reconstructed_v_hat = ltm_module.get_cycle_consistency_reconstruction(
                    prev_ltm_token_input, obs_embedding_v_t
                )
                
                loss, loss_comps = self.combined_loss_fn(
                    original_observation_embeddings_for_cycle=obs_embedding_v_t,
                    reconstructed_observation_embeddings_for_cycle=reconstructed_v_hat
                )
                cycle_loss = loss_comps.get('cycle_consistency_loss', 0.0)
                
                if grad_accum_steps > 1:
                    loss = loss / grad_accum_steps
                loss.backward()

                if (batch_idx + 1) % grad_accum_steps == 0:
                    if phase_config.get('grad_clip_norm', None):
                        nn.utils.clip_grad_norm_(trainable_params, phase_config['grad_clip_norm'])
                    optimizer.step()
                    optimizer.zero_grad()
                
                if scheduler and phase_config.get('scheduler_step_per_batch', False):
                     scheduler.step()

                epoch_loss += cycle_loss
                num_batches_processed += 1
                self.global_step +=1

                if batch_idx % self.training_config.get('log_interval', 100) == 0 and num_batches_processed > 0:
                     print(f"  P2 Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, Cycle Loss: {cycle_loss:.4f}")

            avg_epoch_loss = epoch_loss / num_batches_processed if num_batches_processed > 0 else 0
            print(f"Phase 2 - Epoch {epoch+1}/{num_epochs} - Avg Cycle Loss: {avg_epoch_loss:.4f}")
            if scheduler and not phase_config.get('scheduler_step_per_batch', False):
                scheduler.step()
            self._save_checkpoint(f"phase2_epoch{epoch+1}")


    def train_phase_3_e2e_nav(self, phase_config: Dict[str, Any]):
        print(f"\n--- Starting Training Phase 3: End-to-End Navigation ({phase_config.get('epochs',0)} epochs) ---")
        self.current_phase_idx = 2

        param_keywords = self._get_param_name_keywords_for_phase(2)
        self._freeze_parameters(self.agent, params_to_train_names=param_keywords)
        
        if phase_config.get('freeze_visual_backbone_in_phase3', False):
            if hasattr(self.agent, 'get_visual_backbone_params_names'): # Agent must define this
                self._freeze_parameters(self.agent, params_to_train_names=[kw for kw in param_keywords if kw not in self.agent.get_visual_backbone_params_names()])   
            else:
                print("Warning: 'freeze_visual_backbone_in_phase3' is True, but agent has no 'get_visual_backbone_params_names()' method.")

        trainable_params = [p for p in self.agent.parameters() if p.requires_grad]
        if not trainable_params:
            print("Phase 3: No trainable parameters found. Skipping.")
            return

        optimizer, scheduler = self._setup_optimizer_and_scheduler(phase_config, trainable_params)
        dataloader = self.train_dataloaders.get("phase3_vln")
        if not dataloader:
            print("Warning: Dataloader for 'phase3_vln' not found. Skipping Phase 3.")
            return
        
        num_epochs = phase_config.get('epochs', 30)
        grad_accum_steps = phase_config.get('gradient_accumulation_steps', 1)
        use_cycle_loss_jointly = self.training_config.get('use_cycle_loss_in_phase3', True) and \
                                 self.combined_loss_fn.cycle_consistency_loss_weight > 0


        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.agent.train()
            epoch_total_loss, epoch_nav_loss, epoch_cycle_loss = 0.0, 0.0, 0.0
            num_batches_processed = 0
            optimizer.zero_grad()

            for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Phase 3 Epoch {epoch+1}/{num_epochs}")):
                # Agent's process_batch_for_training method should handle the VLN episode batch,
                # perform teacher forcing or policy rollouts, and return:
                # - predicted_action_logits
                # - ground_truth_action_ids
                # - Optionally: original_v_for_cycle, reconstructed_v_for_cycle (if LTM was used and cycle loss is active)
                if not hasattr(self.agent, 'process_batch_for_training'):
                    print("ERROR: Agent must implement 'process_batch_for_training(batch_data, device)' for Phase 3.")
                    return

                outputs = self.agent.process_batch_for_training(batch_data, self.device)

                pred_logits = outputs.get('action_logits')
                gt_actions = outputs.get('gt_action_ids')
                orig_v_cycle = outputs.get('original_v_cycle') if use_cycle_loss_jointly else None
                recon_v_cycle = outputs.get('reconstructed_v_cycle') if use_cycle_loss_jointly else None

                if pred_logits is None or gt_actions is None:
                    print("ERROR: Agent's process_batch_for_training did not return 'action_logits' or 'gt_action_ids'.")
                    continue

                loss, loss_comps = self.combined_loss_fn(
                    predicted_action_logits=pred_logits,
                    ground_truth_action_ids=gt_actions,
                    original_observation_embeddings_for_cycle=orig_v_cycle,
                    reconstructed_observation_embeddings_for_cycle=recon_v_cycle
                )
                
                if grad_accum_steps > 1:
                    loss = loss / grad_accum_steps
                loss.backward()

                if (batch_idx + 1) % grad_accum_steps == 0:
                    if phase_config.get('grad_clip_norm', None):
                        nn.utils.clip_grad_norm_(trainable_params, phase_config['grad_clip_norm'])
                    optimizer.step()
                    optimizer.zero_grad()

                if scheduler and phase_config.get('scheduler_step_per_batch', True):
                     scheduler.step()

                epoch_total_loss += loss_comps.get('total_loss', loss.item() * grad_accum_steps)
                epoch_nav_loss += loss_comps.get('navigation_loss', 0.0)
                epoch_cycle_loss += loss_comps.get('cycle_consistency_loss', 0.0)
                num_batches_processed += 1
                self.global_step +=1

                if batch_idx % self.training_config.get('log_interval', 100) == 0 and num_batches_processed > 0:
                    print(f"  P3 Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, "
                          f"Total Loss: {loss_comps.get('total_loss',0):.4f} "
                          f"(Nav: {loss_comps.get('navigation_loss',0):.4f}, Cycle: {loss_comps.get('cycle_consistency_loss',0):.4f})")

            avg_epoch_loss = epoch_total_loss / num_batches_processed if num_batches_processed > 0 else 0
            avg_nav_loss = epoch_nav_loss / num_batches_processed if num_batches_processed > 0 else 0
            avg_cycle_loss = epoch_cycle_loss / num_batches_processed if num_batches_processed > 0 else 0
            print(f"Phase 3 - Epoch {epoch+1}/{num_epochs} - Avg Total Loss: {avg_epoch_loss:.4f} "
                  f"(Nav: {avg_nav_loss:.4f}, Cycle: {avg_cycle_loss:.4f})")

            if scheduler and not phase_config.get('scheduler_step_per_batch', True):
                scheduler.step()

            if self.val_dataloader_vln and self.evaluator and \
               (epoch + 1) % self.training_config.get('eval_interval_epochs', 1) == 0:
                val_metrics = self.evaluator.evaluate() # Evaluator runs on self.val_dataloader_vln
                print(f"Validation after Epoch {epoch+1}: {val_metrics}")
                primary_metric_name = self.training_config.get('primary_val_metric', 'SPL')
                primary_metric_val = val_metrics.get(primary_metric_name, -float('inf'))
                
                is_better = (primary_metric_val > self.best_val_metric) if \
                            self.training_config.get('higher_is_better_val_metric', True) else \
                            (primary_metric_val < self.best_val_metric)
                
                if is_better:
                    self.best_val_metric = primary_metric_val
                    print(f"New best validation {primary_metric_name}: {self.best_val_metric:.4f}. Saving best model.")
                    self._save_checkpoint(f"phase3_best_val_epoch{epoch+1}")
            
            self._save_checkpoint(f"phase3_epoch{epoch+1}")


    def train(self):
        print(f"=== Starting Mem4Nav Agent Training (Output Dir: {self.checkpoint_dir}) ===")
        phase_configs = self.training_config.get('phases', {})
        
        if phase_configs.get('phase1_visual', {}).get('enabled', False):
            self.train_phase_1_visual(phase_configs['phase1_visual'])
        else:
            print("Skipping Phase 1: Visual Frontend Fine-tuning (disabled in config).")

        if phase_configs.get('phase2_ltm', {}).get('enabled', False):
            self.train_phase_2_ltm(phase_configs['phase2_ltm'])
        else:
            print("Skipping Phase 2: LTM Pre-training (disabled in config).")

        if phase_configs.get('phase3_e2e_nav', {}).get('enabled', True):
            self.train_phase_3_e2e_nav(phase_configs['phase3_e2e_nav'])
        else:
            print("Skipping Phase 3: End-to-End Navigation Fine-tuning (disabled in config).")
        
        print("=== Mem4Nav Agent Training Finished ===")

    def _save_checkpoint(self, checkpoint_name_suffix: str):
        if not self.checkpoint_dir: return
        
        # TODO: Get optimizer and scheduler states if they need to be saved for resumption
        state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'phase_idx': self.current_phase_idx,
            'agent_state_dict': self.agent.state_dict(),
            'best_val_metric': self.best_val_metric,
            'config_training_at_save': self.training_config # Save training part of config
            # 'optimizer_state_dict': optimizer.state_dict(), # If optimizer is instance var
            # 'scheduler_state_dict': scheduler.state_dict(), # If scheduler is instance var
        }
        filename = os.path.join(self.checkpoint_dir, f"ckpt_{checkpoint_name_suffix}.pth")
        torch.save(state, filename)
        print(f"Saved checkpoint: {filename}")

    def load_checkpoint(self, checkpoint_path: str, 
                        # load_optimizer: bool = False, optimizer: Optional[optim.Optimizer]=None,
                        # load_scheduler: bool = False, scheduler: Optional[Any]=None
                        ): # Optimizer/scheduler loading needs them to be passed or part of self
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint file not found at {checkpoint_path}")
            return
        
        print(f"Loading checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.device)
        
        self.agent.load_state_dict(state['agent_state_dict'])
        self.current_epoch = state.get('epoch', 0)
        self.global_step = state.get('global_step', 0)
        self.current_phase_idx = state.get('phase_idx', 0)
        self.best_val_metric = state.get('best_val_metric', -float('inf') if self.training_config.get('higher_is_better_val_metric', True) else float('inf'))
        
        # TODO: Handle optimizer and scheduler state loading if they are passed or member variables
        # if load_optimizer and optimizer and 'optimizer_state_dict' in state:
        #     optimizer.load_state_dict(state['optimizer_state_dict'])
        # if load_scheduler and scheduler and 'scheduler_state_dict' in state:
        #     scheduler.load_state_dict(state['scheduler_state_dict'])
        print(f"Loaded checkpoint. Resuming at phase index {self.current_phase_idx}, epoch {self.current_epoch}, global_step {self.global_step}.")