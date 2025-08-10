import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Optional, List, Callable
# from transformers import PreTrainedTokenizerBase # For type hinting tokenizer

try:
    from transformers import PreTrainedTokenizerBase
except ImportError:
    print("Warning: PreTrainedTokenizerBase not found. Using Any for tokenizer type hint in BaseVLNDatasetLoader.")
    PreTrainedTokenizerBase = Any # type: ignore


class BaseVLNDataset(Dataset):
    """
    Abstract Base Class for Vision-and-Language Navigation (VLN) datasets.
    All specific VLN dataset loaders (e.g., Touchdown, Map2Seq) should inherit from this class.
    """
    def __init__(self,
                 config: Dict[str, Any],
                 split: str,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 max_instruction_length: int = 128, # Default, can be configured
                 name: str = "BaseVLNDataset"):
        """
        Initializes the base VLN dataset.

        Args:
            config (Dict[str, Any]): Configuration dictionary, typically containing paths
                                     to data files, feature directories, etc.
            split (str): The dataset split to load (e.g., "train", "dev", "test").
            tokenizer (Optional[PreTrainedTokenizerBase]): Tokenizer for processing
                                                           natural language instructions.
            max_instruction_length (int): Maximum length for tokenized instructions.
            name (str): A name for the dataset loader instance.
        """
        super().__init__()
        self.config = config.get(name.lower(), config) # Get dataset-specific config or general config
        self.dataset_root_path = self.config.get('dataset_path', './datasets') # Root for all VLN datasets
        self.specific_dataset_path = self.config.get('specific_path') # Path to the specific dataset (e.g., .../touchdown)

        self.split = split
        self.tokenizer = tokenizer
        self.max_instruction_length = max_instruction_length
        self.name = name

        self.data: List[Dict[str, Any]] = [] # This will store the loaded and preprocessed data instances

        # Load the actual data - to be implemented by subclasses
        self._load_data()

    def _load_data(self) -> None:
        """
        Abstract method to load data from files.
        Subclasses must implement this to populate `self.data`.
        Each item in `self.data` should be a dictionary representing a single
        navigation episode or instruction-trajectory pair.
        """
        raise NotImplementedError("Subclasses must implement _load_data().")

    def __len__(self) -> int:
        """Returns the total number of instances in the loaded split."""
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Retrieves a single processed data instance by index.

        Args:
            index (int): The index of the data instance.

        Returns:
            Dict[str, Any]: A dictionary containing the processed data for
                            one episode/instance. Structure depends on the
                            specific dataset but should include tokenized instructions,
                            goal information, image/feature paths, etc.
        """
        if not (0 <= index < len(self.data)):
            raise IndexError(f"Index {index} out of bounds for dataset of size {len(self.data)}.")
        
        raw_instance = self.data[index]
        return self._preprocess_instance(raw_instance)

    def _preprocess_instance(self, instance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract or common method for preprocessing a single raw data instance.
        This can include tokenizing instructions, formatting paths, etc.
        Subclasses can override or extend this.

        Args:
            instance_data (Dict[str, Any]): A raw data instance (one item from self.data).

        Returns:
            Dict[str, Any]: The processed data instance.
        """
        # Example: Tokenize instruction text if tokenizer is provided
        if self.tokenizer and 'instruction_text' in instance_data:
            # This tokenization utility would ideally be in data_handling/utils.py
            # For now, a simplified version:
            tokenized_instruction = self.tokenizer(
                instance_data['instruction_text'],
                padding='max_length',
                truncation=True,
                max_length=self.max_instruction_length,
                return_tensors="pt", # Return PyTorch tensors
                return_attention_mask=True
            )
            # Squeeze if batch dim is 1 (tokenizer often adds it)
            instance_data['instruction_tokens_ids'] = tokenized_instruction['input_ids'].squeeze(0)
            instance_data['instruction_attention_mask'] = tokenized_instruction['attention_mask'].squeeze(0)
        
        # Other common preprocessing can be added here.
        return instance_data

    def get_num_instances(self) -> int:
        """Helper method to get dataset size."""
        return self.__len__()


if __name__ == '__main__':
    print("--- Testing BaseVLNDataset (Conceptual) ---")

    # This base class is abstract and cannot be instantiated directly
    # without implementing _load_data and _preprocess_instance.
    # We can test its structure by creating a minimal concrete subclass.

    class MockVLNDataset(BaseVLNDataset):
        def __init__(self, config, split, tokenizer=None, max_instr_len=50):
            super().__init__(config, split, tokenizer, max_instr_len, name="MockVLNDataset")

        def _load_data(self):
            # Load mock data
            num_samples = self.config.get('num_mock_samples', 10)
            for i in range(num_samples):
                self.data.append({
                    'id': f'{self.split}_{i}',
                    'instruction_text': f"This is mock instruction {i} for the {self.split} split.",
                    'goal_info': {'type': 'dummy_goal', 'details': f'goal_details_{i}'},
                    # ... other dataset-specific fields
                })
            print(f"MockVLNDataset: Loaded {len(self.data)} mock instances for split '{self.split}'.")

        def _preprocess_instance(self, instance_data: Dict[str, Any]) -> Dict[str, Any]:
            # Call base preprocessing (which handles tokenization if tokenizer exists)
            processed_instance = super()._preprocess_instance(instance_data)
            
            # Add any mock-specific preprocessing
            processed_instance['is_preprocessed_mock'] = True
            if 'instruction_tokens_ids' in processed_instance:
                print(f"  Instance {instance_data['id']} instruction tokenized, shape: {processed_instance['instruction_tokens_ids'].shape}")
            return processed_instance

    # Example usage:
    mock_config_data = {
        'MockVLNDataset': { # Corresponds to dataset name
            'dataset_path': '/tmp/datasets', # Root for all datasets
            'specific_path': '/tmp/datasets/mock_vln', # Path to this specific mock dataset
            'num_mock_samples': 5
        }
    }

    # Mock Tokenizer (e.g., from Hugging Face)
    class MockTokenizer:
        def __init__(self, max_len=50): self.max_length = max_len; self.pad_token_id=0
        def __call__(self, text, padding, truncation, max_length, return_tensors, return_attention_mask):
            print(f"MockTokenizer called for: '{text[:30]}...'")
            # Simulate tokenization
            tokens = [random.randint(1, 1000) for _ in range(min(len(text.split()), max_length))]
            padded_tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))
            attention_mask = [1] * len(tokens) + [0] * (max_length - len(tokens))
            return {
                'input_ids': torch.tensor([padded_tokens], dtype=torch.long),
                'attention_mask': torch.tensor([attention_mask], dtype=torch.long)
            }

    mock_tokenizer_inst = MockTokenizer(max_len=20)
    
    try:
        dataset = MockVLNDataset(mock_config_data, "train", tokenizer=mock_tokenizer_inst, max_instr_len=20)
        print(f"Dataset size: {len(dataset)}")
        if len(dataset) > 0:
            sample_item = dataset[0]
            print(f"Sample item [0]: {sample_item['id']}")
            assert 'instruction_tokens_ids' in sample_item
            assert 'is_preprocessed_mock' in sample_item
            assert sample_item['instruction_tokens_ids'].shape == (20,)

    except Exception as e:
        print(f"Error during BaseVLNDataset test: {e}")
        import traceback
        traceback.print_exc()

    print("\nBaseVLNDataset conceptual test finished.")