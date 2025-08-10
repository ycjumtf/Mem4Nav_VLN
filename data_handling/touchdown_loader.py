import torch
import json
import os
from typing import Dict, Any, Optional, List

try:
    from .base_loader import BaseVLNDataset
    # from .utils import tokenize_instructions # Tokenization is handled by base class
except ImportError:
    print("Warning: TouchdownDatasetLoader using placeholder for BaseVLNDataset due to import error.")
    class BaseVLNDataset(torch.utils.data.Dataset):   
        def __init__(self, config, split, tokenizer=None, max_instruction_length=128, name="Base"):
            self.config = config.get(name.lower(), config)
            self.dataset_root_path = self.config.get('dataset_path', './datasets')
            self.specific_dataset_path = self.config.get('specific_path')
            self.split = split
            self.tokenizer = tokenizer
            self.max_instruction_length = max_instruction_length
            self.data: List[Dict[str, Any]] = []
            self._load_data()
        def _load_data(self): raise NotImplementedError
        def __len__(self): return len(self.data)
        def __getitem__(self, index): return self._preprocess_instance(self.data[index])
        def _preprocess_instance(self, instance_data):
            if self.tokenizer and 'instruction_text' in instance_data:
                mock_tokens = torch.randint(0, 100, (self.max_instruction_length,))
                mock_attn_mask = torch.ones(self.max_instruction_length, dtype=torch.long)
                instance_data['instruction_tokens_ids'] = mock_tokens
                instance_data['instruction_attention_mask'] = mock_attn_mask
            return instance_data
    try:
        from transformers import PreTrainedTokenizerBase
    except ImportError:
        PreTrainedTokenizerBase = Any   


class TouchdownDatasetLoader(BaseVLNDataset):
    """
    Dataset loader for the Touchdown Vision-and-Language Navigation dataset.
    """
    def __init__(self,
                 config: Dict[str, Any],
                 split: str,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,   
                 max_instruction_length: int = 128,
                 name: str = "TouchdownDatasetLoader"):
        """
        Initializes the Touchdown dataset loader.

        Args:
            config (Dict[str, Any]): Configuration dictionary. Expected to contain:
                - 'specific_path': Path to the root of the Touchdown dataset.
                - 'landmarks_file_path' (Optional): Path to a JSON file containing
                                                     pre-extracted landmarks for episodes.
            split (str): The dataset split to load (e.g., "train", "dev", "test_seen", "test_unseen").
                         Touchdown often has 'dev', 'train', 'test'. The paper mentions testing on Touchdown Dev/Test.
            tokenizer (Optional[PreTrainedTokenizerBase]): Tokenizer for instructions.
            max_instruction_length (int): Max length for tokenized instructions.
            name (str): Name for this loader instance.
        """
        self.landmarks_data: Optional[Dict[str, Any]] = None
        self.landmarks_file_path = config.get('touchdown_loader', {}).get('landmarks_file_path')

        super().__init__(config, split, tokenizer, max_instruction_length, name)
        # Note: self.specific_dataset_path should point to the root of the Touchdown dataset folder
        # e.g., /path/to/datasets/touchdown/

    def _load_data(self) -> None:
        """
        Loads Touchdown data from JSON line files.
        Populates `self.data` with a list of episode dictionaries.
        """
        if not self.specific_dataset_path:
            raise ValueError(f"Configuration for '{self.name}' must specify 'specific_path' to the Touchdown dataset.")


        data_file_path = os.path.join(self.specific_dataset_path, "data", f"{self.split}.json")
 
        if not os.path.exists(data_file_path):
            data_file_path_alt = os.path.join(self.specific_dataset_path, f"{self.split}.json")
            if not os.path.exists(data_file_path_alt):
                raise FileNotFoundError(f"Touchdown data file not found at '{data_file_path}' or '{data_file_path_alt}' for split '{self.split}'.")
            data_file_path = data_file_path_alt
            
        print(f"Loading Touchdown data from: {data_file_path}")

        with open(data_file_path, 'r') as f:
            for line in f:
                try:
                    instance = json.loads(line.strip())
                    self.data.append(instance)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line due to JSON decode error: {e} in file {data_file_path}")
        
        if not self.data:
            print(f"Warning: No data loaded for Touchdown split '{self.split}' from '{data_file_path}'.")
            return

        if self.landmarks_file_path:
            lm_path = os.path.join(self.dataset_root_path, self.landmarks_file_path) 
            if not os.path.exists(lm_path):
                # Try relative to specific_dataset_path
                lm_path_alt = os.path.join(self.specific_dataset_path, self.landmarks_file_path)
                if not os.path.exists(lm_path_alt):
                    print(f"Warning: Landmarks file '{self.landmarks_file_path}' not found at '{lm_path}' or '{lm_path_alt}'. Proceeding without pre-extracted landmarks.")
                else:
                    lm_path = lm_path_alt
            
            if os.path.exists(lm_path):
                print(f"Loading landmarks from: {lm_path}")
                try:
                    with open(lm_path, 'r') as f_lm:
                      
                        loaded_landmarks_data = json.load(f_lm)
                        if 'instances' in loaded_landmarks_data:
                            self.landmarks_data = loaded_landmarks_data['instances']
                        else:
                            self.landmarks_data = loaded_landmarks_data # Assume direct map if 'instances' not present
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse landmarks JSON file '{lm_path}': {e}")
                except Exception as e:
                    print(f"Warning: Error loading landmarks file '{lm_path}': {e}")


        print(f"TouchdownDatasetLoader: Loaded {len(self.data)} instances for split '{self.split}'.")


    def _preprocess_instance(self, instance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocesses a single Touchdown data instance.
        """
        processed_instance = super()._preprocess_instance(instance_data)

        if 'route_id' not in processed_instance:
            if 'id' in processed_instance:
                processed_instance['route_id'] = processed_instance['id']
            elif 'episode_id' in processed_instance:
                 processed_instance['route_id'] = processed_instance['episode_id']
            else:
            
                pass # Allow it to pass, agent should handle missing keys if critical

        # Rename 'navigation_text' to 'instruction_text' if that's what base expects
        if 'navigation_text' in processed_instance and 'instruction_text' not in processed_instance:
            processed_instance['instruction_text'] = processed_instance['navigation_text']
        
        # Ensure 'instruction_text' is present for tokenization by base class
        if 'instruction_text' not in processed_instance:
             processed_instance['instruction_text'] = "" # Default to empty string if missing

        # Target panoid (usually the last in the route)
        if 'route_panoids' in processed_instance and processed_instance['route_panoids']:
            processed_instance['target_panoid'] = processed_instance['route_panoids'][-1]
        else:
            processed_instance['route_panoids'] = [] # Ensure it exists
            processed_instance['target_panoid'] = None
            # print(f"Warning: 'route_panoids' missing or empty for instance_id {processed_instance.get('route_id')}")

        # Start heading (ensure it's a float)
        if 'start_heading' in processed_instance:
            try:
                processed_instance['start_heading'] = float(processed_instance['start_heading'])
            except ValueError:
                print(f"Warning: Could not convert start_heading '{processed_instance['start_heading']}' to float for instance {processed_instance.get('route_id')}. Setting to 0.0.")
                processed_instance['start_heading'] = 0.0
        else:
            processed_instance['start_heading'] = 0.0 # Default if missing

        route_id_str = str(processed_instance.get('route_id', ''))
        if self.landmarks_data and route_id_str in self.landmarks_data:
            # Example: VELMA landmarks structure might be {'unfiltered': [...]}
            # We might need a utility to filter these as per paper or VELMA code
            # For now, store what's available.
            landmark_info = self.landmarks_data[route_id_str]
            if isinstance(landmark_info, dict) and 'unfiltered' in landmark_info:
                 processed_instance['landmarks_unfiltered'] = landmark_info['unfiltered']
                 # A more refined 'landmarks' field could be added after filtering
                 # processed_instance['landmarks'] = filter_landmarks_utility(landmark_info['unfiltered'])
            else: # If structure is different
                processed_instance['landmarks_raw'] = landmark_info
        else:
            processed_instance['landmarks_unfiltered'] = []




        return processed_instance


if __name__ == '__main__':
    print("--- Testing TouchdownDatasetLoader ---")

    # Create a dummy config
    # Make sure to create dummy data files as per this structure for the test to run
    dummy_dataset_root = "./tmp_datasets"
    dummy_touchdown_specific_path = os.path.join(dummy_dataset_root, "touchdown_vln")
    dummy_data_dir = os.path.join(dummy_touchdown_specific_path, "data")
    os.makedirs(dummy_data_dir, exist_ok=True)

    mock_config_td = {
        'dataset_path': dummy_dataset_root, # Root for all datasets
        'TouchdownDatasetLoader': { # Config specific to this loader
            'specific_path': dummy_touchdown_specific_path, # Path to touchdown dataset
            'num_mock_samples': 3, # Not used by actual loader, but for mock base
            'landmarks_file_path': 'landmarks/touchdown_landmarks_mock.json' # Relative to dataset_root_path
        }
    }

    # Create dummy dev.json for Touchdown
    dummy_dev_data = [
        {"route_id": "td_dev_0", "navigation_text": "Walk forward past the cafe and stop at the corner.", "route_panoids": ["pano1", "pano2", "pano3"], "start_heading": 90.0},
        {"route_id": "td_dev_1", "navigation_text": "Turn left and go towards the big tree.", "route_panoids": ["panoA", "panoB"], "start_heading": "0"},
        {"route_id": "td_dev_2", "navigation_text": "Find the entrance.", "route_panoids": ["panoX"], "start_heading": 180}
    ]
    with open(os.path.join(dummy_data_dir, "dev.json"), "w") as f:
        for item in dummy_dev_data:
            f.write(json.dumps(item) + "\n")

    # Create dummy landmarks file
    dummy_landmarks_dir = os.path.join(dummy_dataset_root, "landmarks")
    os.makedirs(dummy_landmarks_dir, exist_ok=True)
    dummy_landmarks_data = {
        "instances": {
            "td_dev_0": {"unfiltered": ["cafe", "corner"]},
            "td_dev_1": {"unfiltered": ["big tree"]}
        }
    }
    with open(os.path.join(dummy_landmarks_dir, "touchdown_landmarks_mock.json"), "w") as f:
        json.dump(dummy_landmarks_data, f)


    # Mock Tokenizer
    class MockTokenizerForTD:
        def __init__(self, max_len=25): self.max_length = max_len; self.pad_token_id=0
        def __call__(self, text_list, padding, truncation, max_length, return_tensors, return_attention_mask):
            input_ids_batch = []
            attention_mask_batch = []
            for text in text_list if isinstance(text_list, list) else [text_list]:
                tokens = [hash(t) % 1000 + 1 for t in text.split()[:max_length]] # Mock token IDs
                padded_tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))
                attn_mask = [1] * len(tokens) + [0] * (max_length - len(tokens))
                input_ids_batch.append(padded_tokens)
                attention_mask_batch.append(attn_mask)
            return {
                'input_ids': torch.tensor(input_ids_batch, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask_batch, dtype=torch.long)
            }
    
    mock_tokenizer_td = MockTokenizerForTD(max_len=30)

    try:
        td_dataset = TouchdownDatasetLoader(
            config=mock_config_td, 
            split="dev", 
            tokenizer=mock_tokenizer_td, 
            max_instruction_length=30
        )
        print(f"Touchdown Dataset size: {len(td_dataset)}")
        assert len(td_dataset) == 3

        if len(td_dataset) > 0:
            sample_item_0 = td_dataset[0]
            print(f"\nSample item [0]: ID {sample_item_0['route_id']}")
            print(f"  Instruction: {sample_item_0['instruction_text']}")
            print(f"  Tokens shape: {sample_item_0['instruction_tokens_ids'].shape}")
            print(f"  Target Pano: {sample_item_0['target_panoid']}")
            print(f"  Landmarks: {sample_item_0.get('landmarks_unfiltered')}")
            assert sample_item_0['instruction_tokens_ids'].shape == (30,)
            assert sample_item_0['target_panoid'] == "pano3"
            assert "cafe" in sample_item_0.get('landmarks_unfiltered', [])

            sample_item_1 = td_dataset[1]
            print(f"\nSample item [1]: ID {sample_item_1['route_id']}")
            print(f"  Start Heading: {sample_item_1['start_heading']} (type: {type(sample_item_1['start_heading'])})")
            assert isinstance(sample_item_1['start_heading'], float)
            assert "big tree" in sample_item_1.get('landmarks_unfiltered', [])
            
            sample_item_2 = td_dataset[2] # No landmarks for this one in mock
            assert not sample_item_2.get('landmarks_unfiltered')


    except Exception as e:
        print(f"Error during TouchdownDatasetLoader test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up dummy files and directories
        import shutil
        if os.path.exists(dummy_dataset_root):
            shutil.rmtree(dummy_dataset_root)

    print("\nTouchdownDatasetLoader tests completed.")