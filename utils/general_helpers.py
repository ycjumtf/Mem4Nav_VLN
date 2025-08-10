import torch
import numpy as np
import random
import os
import time
from typing import List, Union, Any, Dict, Sequence

def set_random_seed(seed: int, deterministic_cudnn: bool = False) -> None:
    """
    Sets the random seed for Python's `random`, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
        deterministic_cudnn (bool): If True, sets torch.backends.cudnn.deterministic = True
                                    and torch.backends.cudnn.benchmark = False. This can
                                    make CUDA operations deterministic but might impact performance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # For all GPUs
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            # Default PyTorch behavior, often True for performance if input sizes don't vary much
            torch.backends.cudnn.benchmark = True 
    print(f"Random seed set to: {seed}")


def calculate_euclidean_path_length(path_xyz: Union[np.ndarray, List[np.ndarray], List[List[float]]]) -> float:
    """
    Calculates the total Euclidean length of a path defined by a sequence of XYZ coordinates.

    Args:
        path_xyz (Union[np.ndarray, List[np.ndarray]]): A sequence of 3D points.
            Can be a 2D NumPy array (NumPoints, 3) or a list of NumPy arrays/lists.

    Returns:
        float: The total length of the path. Returns 0.0 if the path has fewer than 2 points.
    """
    if not isinstance(path_xyz, np.ndarray):
        path_points = np.array(path_xyz, dtype=np.float32)
    else:
        path_points = path_xyz.astype(np.float32)

    if path_points.ndim != 2 or path_points.shape[1] != 3:
        if path_points.size == 0 or (path_points.ndim == 1 and path_points.shape[0] == 3): # Single point
             return 0.0
        raise ValueError("path_xyz must be a 2D array of shape (NumPoints, 3) or convertible to it.")

    if len(path_points) < 2:
        return 0.0

    total_length = 0.0
    for i in range(len(path_points) - 1):
        p1 = path_points[i]
        p2 = path_points[i+1]
        distance = np.linalg.norm(p2 - p1)
        total_length += distance
    
    return float(total_length)


def ensure_directory_exists(dir_path: str) -> None:
    """
    Ensures that a directory exists. If it doesn't, it creates it.

    Args:
        dir_path (str): The path to the directory.
    """
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
            # print(f"Created directory: {dir_path}") # Optional: log or print
        except Exception as e:
            # Use logging in a real app
            print(f"Error creating directory {dir_path}: {e}")
            raise
    elif not os.path.isdir(dir_path):
        raise NotADirectoryError(f"Path '{dir_path}' exists but is not a directory.")


def format_time_duration(seconds: float) -> str:
    """
    Formats a duration in seconds into a human-readable string (e.g., "1h 23m 45.6s").

    Args:
        seconds (float): The duration in seconds.

    Returns:
        str: A human-readable string representing the duration.
    """
    if seconds < 0:
        return "Invalid duration (negative)"

    days = int(seconds // (24 * 3600))
    seconds %= (24 * 3600)
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    
    if seconds > 0 or not parts: # Show seconds if it's the only unit or non-zero
        parts.append(f"{seconds:.2f}s" if isinstance(seconds, float) and not seconds.is_integer() else f"{int(seconds)}s")
        
    return " ".join(parts) if parts else "0s"


def move_to_device(data: Any, device: torch.device) -> Any:
    """
    Recursively moves tensors within a nested data structure (dicts, lists, tuples)
    to the specified PyTorch device. Non-tensor elements are returned as is.

    Args:
        data (Any): The data to move. Can be a tensor, list, tuple, or dict.
        device (torch.device): The target PyTorch device.

    Returns:
        Any: The data structure with all tensors moved to the specified device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(x, device) for x in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(x, device) for x in data)
    else:
        return data # Return as is if not a tensor or common collection

def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Counts the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("--- Testing General Helper Utilities ---")

    # 1. Test set_random_seed
    print("\n1. Testing set_random_seed...")
    set_random_seed(42)
    py_rand1 = random.randint(0, 1000)
    np_rand1 = np.random.rand(2)
    torch_rand1 = torch.randn(2)
    
    set_random_seed(42) # Reset seed
    py_rand2 = random.randint(0, 1000)
    np_rand2 = np.random.rand(2)
    torch_rand2 = torch.randn(2)
    
    assert py_rand1 == py_rand2
    assert np.array_equal(np_rand1, np_rand2)
    assert torch.equal(torch_rand1, torch_rand2)
    print(f"  Seed setting test passed (py: {py_rand1}, np: {np_rand1}, torch: {torch_rand1})")

    # 2. Test calculate_euclidean_path_length
    print("\n2. Testing calculate_euclidean_path_length...")
    path1 = np.array([[0,0,0], [3,0,0], [3,4,0]]) # Length 3 + 4 = 7
    length1 = calculate_euclidean_path_length(path1)
    print(f"  Length of path {path1.tolist()}: {length1:.2f}")
    assert np.isclose(length1, 7.0)

    path2 = [[0,0,0], [1,1,1]] # Length sqrt(3)
    length2 = calculate_euclidean_path_length(path2)
    print(f"  Length of path {path2}: {length2:.2f}")
    assert np.isclose(length2, np.sqrt(3))

    path3 = [[1,2,3]] # Single point
    length3 = calculate_euclidean_path_length(path3)
    print(f"  Length of path {path3}: {length3:.2f}")
    assert np.isclose(length3, 0.0)
    
    path4: List[np.ndarray] = [] # Empty path
    length4 = calculate_euclidean_path_length(path4)
    print(f"  Length of empty path: {length4:.2f}")
    assert np.isclose(length4, 0.0)

    # 3. Test ensure_directory_exists
    print("\n3. Testing ensure_directory_exists...")
    test_dir = "./tmp_test_helpers_dir/subdir"
    ensure_directory_exists(test_dir)
    assert os.path.isdir(test_dir)
    print(f"  Directory '{test_dir}' ensured/created.")
    # Test with existing dir
    ensure_directory_exists(test_dir) 
    print(f"  Ensuring existing directory '{test_dir}' again (should do nothing).")
    # Clean up
    import shutil
    if os.path.exists("./tmp_test_helpers_dir"):
        shutil.rmtree("./tmp_test_helpers_dir")

    # 4. Test format_time_duration
    print("\n4. Testing format_time_duration...")
    assert format_time_duration(5.25) == "5.25s"
    assert format_time_duration(65) == "1m 5s"
    assert format_time_duration(3661.5) == "1h 1m 1.50s"
    assert format_time_duration(86400 + 7200 + 180 + 10) == "1d 2h 3m 10s"
    assert format_time_duration(0) == "0s"
    print(f"  format_time_duration(3661.5) -> '{format_time_duration(3661.5)}'")
    print(f"  format_time_duration(90000) -> '{format_time_duration(90000)}'")

    # 5. Test move_to_device
    print("\n5. Testing move_to_device...")
    if torch.cuda.is_available():
        target_device = torch.device("cuda")
        print(f"  CUDA available, testing with device: {target_device}")
    else:
        target_device = torch.device("cpu")
        print(f"  CUDA not available, testing with device: {target_device}")

    data_struct: Dict[str, Any] = {
        "a": torch.randn(2,2),
        "b": [torch.randn(3), "string", {"c": torch.randn(1)}],
        "d": 123
    }
    moved_data = move_to_device(data_struct, target_device)
    assert moved_data["a"].device == target_device
    assert moved_data["b"][0].device == target_device
    assert moved_data["b"][2]["c"].device == target_device
    assert moved_data["b"][1] == "string"
    assert moved_data["d"] == 123
    print(f"  move_to_device test passed.")

    # 6. Test count_trainable_parameters
    print("\n6. Testing count_trainable_parameters...")
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10,5)
            self.fc2 = nn.Linear(5,1)
            self.non_trainable = nn.Parameter(torch.randn(3), requires_grad=False)
    
    net = SimpleNet()
    # fc1: 10*5 (weights) + 5 (bias) = 55
    # fc2: 5*1 (weights) + 1 (bias) = 6
    # Total = 61
    num_params = count_trainable_parameters(net)
    print(f"  Trainable parameters in SimpleNet: {num_params}")
    assert num_params == (10*5 + 5) + (5*1 + 1)

    print("\nutils/general_helpers.py tests completed.")