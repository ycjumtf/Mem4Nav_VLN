import torch
import re
import string
from typing import List, Dict, Any, Optional
# from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

try:
    from transformers import PreTrainedTokenizerBase
except ImportError:
    print("Warning: PreTrainedTokenizerBase not found. Using Any for tokenizer type hint in data_handling/utils.py.")
    PreTrainedTokenizerBase = Any #type: ignore


def clean_text(text: str) -> str:
    """
    Basic text cleaning: lowercase, remove punctuation, strip whitespace.
    """
    text = text.lower()

    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = text.strip()
    text = re.sub(r'\s+', ' ', text) # Normalize whitespace
    return text


def tokenize_instructions(instructions: List[str],
                          tokenizer: PreTrainedTokenizerBase, # type: ignore
                          max_length: int,
                          clean_first: bool = False,
                          padding_strategy: str = "max_length", # "longest", "max_length", "do_not_pad"
                          truncation_strategy: bool = True
                         ) -> Dict[str, torch.Tensor]:
    """
    Tokenizes a batch of instruction strings.

    Args:
        instructions (List[str]): A list of raw instruction strings.
        tokenizer (PreTrainedTokenizerBase): A Hugging Face tokenizer.
        max_length (int): Maximum sequence length for padding/truncation.
        clean_first (bool): Whether to apply basic cleaning before tokenization.
        padding_strategy (str): Padding strategy for the tokenizer.
        truncation_strategy (bool): Truncation strategy for the tokenizer.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing:
            - 'input_ids': Tensor of token IDs (BatchSize, SeqLen).
            - 'attention_mask': Tensor of attention masks (BatchSize, SeqLen).
            - (Potentially other tokenizer outputs like 'token_type_ids').
    """
    if not tokenizer:
        raise ValueError("Tokenizer must be provided for tokenizing instructions.")

    if clean_first:
        processed_instructions = [clean_text(instr) for instr in instructions]
    else:
        processed_instructions = instructions

    tokenized_output = tokenizer(
        processed_instructions,
        padding=padding_strategy,
        truncation=truncation_strategy,
        max_length=max_length,
        return_tensors="pt",  # Return PyTorch tensors
        return_attention_mask=True
    )
    return tokenized_output


def build_vocab_from_sequences(sequences: List[List[str]],
                               min_freq: int = 1,
                               add_special_tokens: bool = True,
                               pad_token: str = "<PAD>",
                               unk_token: str = "<UNK>",
                               bos_token: str = "<BOS>",
                               eos_token: str = "<EOS>") -> Dict[str, int]:
    """
    Builds a vocabulary (word to index mapping) from a list of tokenized sequences.
    Useful for action spaces or other discrete symbolic sequences if not using a
    pre-trained tokenizer's vocabulary.

    Args:
        sequences (List[List[str]]): A list of sequences, where each sequence is a list of tokens.
        min_freq (int): Minimum frequency for a token to be included in the vocabulary.
        add_special_tokens (bool): Whether to add PAD, UNK, BOS, EOS tokens.
        pad_token, unk_token, bos_token, eos_token (str): Special token strings.

    Returns:
        Dict[str, int]: A dictionary mapping tokens (str) to integer indices.
    """
    token_counts: Dict[str, int] = {}
    for seq in sequences:
        for token in seq:
            token_counts[token] = token_counts.get(token, 0) + 1

    vocab = {}
    if add_special_tokens:
        if pad_token: vocab[pad_token] = len(vocab)
        if unk_token: vocab[unk_token] = len(vocab)
        if bos_token: vocab[bos_token] = len(vocab)
        if eos_token: vocab[eos_token] = len(vocab)

    for token, count in token_counts.items():
        if count >= min_freq:
            if token not in vocab: # Avoid re-adding special tokens if they were in sequences
                vocab[token] = len(vocab)
    
    return vocab


def instruction_to_tensor(instruction_tokens: List[int],
                          vocab_or_tokenizer: Any, # Can be a vocab dict or a tokenizer
                          max_length: Optional[int] = None) -> torch.Tensor:
    """
    Converts a list of instruction tokens (already integer IDs if from tokenizer,
    or strings if to be mapped by vocab) into a padded tensor.
    This is a simplified example; `tokenize_instructions` is more robust for text.
    """
    if isinstance(vocab_or_tokenizer, dict): # It's a vocab
        pad_idx = vocab_or_tokenizer.get("<PAD>", 0) # Assume PAD is 0 if not found
        # This path assumes instruction_tokens are strings to be mapped by vocab
        # This function is more for action sequences or similar, not raw text.
        # For raw text, use tokenize_instructions.
        # For now, let's assume instruction_tokens are already integer IDs.
        ids = instruction_tokens
    elif hasattr(vocab_or_tokenizer, 'pad_token_id'): # It's a Hugging Face tokenizer
        pad_idx = vocab_or_tokenizer.pad_token_id
        ids = instruction_tokens # Assume already tokenized to IDs
    else:
        raise TypeError("vocab_or_tokenizer must be a vocab dict or a Hugging Face tokenizer.")

    if max_length:
        if len(ids) < max_length:
            ids = ids + [pad_idx] * (max_length - len(ids))
        elif len(ids) > max_length:
            ids = ids[:max_length]
            
    return torch.tensor(ids, dtype=torch.long)


if __name__ == '__main__':
    print("--- Testing data_handling/utils.py ---")

    text1 = "  Go Left, then Right!! At the **Store**...  "
    cleaned1 = clean_text(text1)
    print(f"Original: '{text1}' -> Cleaned: '{cleaned1}'")
    assert cleaned1 == "go left then right at the store"

    class MockHFTokenizer:
        def __init__(self, model_max_length=30):
            self.model_max_length = model_max_length
            self.pad_token_id = 0
            self.vocab = {"<PAD>":0, "this":1, "is":2, "a":3, "test":4, "instruction":5, ".":6}
            self.unk_token_id = 7 # Example

        def __call__(self, texts: List[str], padding, truncation, max_length, return_tensors, return_attention_mask):
            batch_input_ids = []
            batch_attn_mask = []
            for text in texts:
                tokens = text.lower().split()
                ids = [self.vocab.get(t, self.unk_token_id) for t in tokens]
                
                if truncation and len(ids) > max_length:
                    ids = ids[:max_length]
                
                attn_mask = [1] * len(ids)
                
                if padding == "max_length" or (padding == "longest" and len(ids) < max_length): # simplified longest
                    pad_len = max_length - len(ids)
                    ids.extend([self.pad_token_id] * pad_len)
                    attn_mask.extend([0] * pad_len)
                
                batch_input_ids.append(ids)
                batch_attn_mask.append(attn_mask)
            
            if return_tensors == "pt":
                return {
                    "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(batch_attn_mask, dtype=torch.long)
                }
            return {}

    mock_hf_tokenizer = MockHFTokenizer(model_max_length=10)
    instructions_batch = [
        "This is a test instruction.",
        "Another short one."
    ]
    tokenized_batch = tokenize_instructions(instructions_batch, mock_hf_tokenizer, max_length=10)
    print("\nTokenized Batch:")
    print(f"  Input IDs: {tokenized_batch['input_ids']}")
    print(f"  Attention Mask: {tokenized_batch['attention_mask']}")
    assert tokenized_batch['input_ids'].shape == (2, 10)
    assert tokenized_batch['attention_mask'].shape == (2, 10)


    # Test build_vocab_from_sequences
    action_sequences = [
        ["forward", "turn_left", "forward", "stop"],
        ["turn_right", "forward", "stop"]
    ]
    action_vocab = build_vocab_from_sequences(action_sequences, min_freq=1)
    print(f"\nBuilt Action Vocab: {action_vocab}")
    assert "<PAD>" in action_vocab and "forward" in action_vocab and "stop" in action_vocab
    assert len(action_vocab) == 4 + 4 # 4 special + 4 actions

    example_token_ids = [10, 20, 30, 0, 0] # Already includes padding
    tensor_ids = instruction_to_tensor(example_token_ids, mock_hf_tokenizer) # Max length not specified, uses original
    print(f"\nInstruction to Tensor (no max_length): {tensor_ids}")
    assert torch.equal(tensor_ids, torch.tensor([10,20,30,0,0], dtype=torch.long))

    tensor_ids_padded = instruction_to_tensor([10,20], mock_hf_tokenizer, max_length=5)
    print(f"Instruction to Tensor (with max_length): {tensor_ids_padded}")
    assert torch.equal(tensor_ids_padded, torch.tensor([10,20,0,0,0], dtype=torch.long))

    print("\ndata_handling/utils.py tests completed.")