import yaml
import os
from typing import Dict, Any, Optional, List, Union
import logging

logger = logging.getLogger(__name__)

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        Dict[str, Any]: A dictionary representing the loaded configuration.

    Raises:
        FileNotFoundError: If the config_path does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    if not os.path.exists(config_path):
        error_msg = f"Configuration file not found at: {config_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        if config_data is None: # Handles empty YAML file case
            logger.warning(f"Configuration file at {config_path} is empty or contains only null values.")
            return {}
        logger.info(f"Successfully loaded configuration from: {config_path}")
        return config_data
    except yaml.YAMLError as e:
        error_msg = f"Error parsing YAML configuration file at {config_path}: {e}"
        logger.error(error_msg)
        raise
    except Exception as e:
        error_msg = f"An unexpected error occurred while loading config {config_path}: {e}"
        logger.error(error_msg)
        raise


def merge_configs(base_config: Dict[str, Any],
                  override_config: Dict[str, Any],
                  allow_new_keys: bool = True) -> Dict[str, Any]:
    """
    Recursively merges two configuration dictionaries.
    Values from `override_config` will take precedence over `base_config`.

    Args:
        base_config (Dict[str, Any]): The base configuration dictionary.
        override_config (Dict[str, Any]): The dictionary with overriding values.
        allow_new_keys (bool): If True, keys present in override_config but not
                               in base_config will be added. If False, an error
                               might be raised or new keys ignored (current impl. adds them).

    Returns:
        Dict[str, Any]: The merged configuration dictionary.
    """
    merged = base_config.copy() # Start with a copy of the base

    for key, override_value in override_config.items():
        if key in merged:
            base_value = merged[key]
            if isinstance(base_value, dict) and isinstance(override_value, dict):
                merged[key] = merge_configs(base_value, override_value, allow_new_keys)
            elif isinstance(base_value, list) and isinstance(override_value, list):
                # For lists, one common strategy is to replace, another is to extend.
                # Here, we replace for simplicity. Can be configured if needed.
                merged[key] = override_value
            else:
                # Override value directly
                merged[key] = override_value
        elif allow_new_keys:
            merged[key] = override_value
        else:
            logger.warning(f"Key '{key}' in override_config not found in base_config and allow_new_keys is False. Ignoring.")
            # Or raise ValueError(f"Key '{key}' in override_config not found in base_config.")

    return merged


def get_config_value(config: Dict[str, Any],
                     key_path: Union[str, List[str]],
                     default_value: Optional[Any] = None,
                     required: bool = False) -> Any:
    """
    Safely retrieves a value from a nested configuration dictionary using a dot-separated path.

    Args:
        config (Dict[str, Any]): The configuration dictionary.
        key_path (Union[str, List[str]]): A dot-separated string (e.g., "optimizer.params.lr")
                                         or a list of keys representing the path to the value.
        default_value (Optional[Any]): The value to return if the key is not found.
        required (bool): If True and the key is not found, raises a KeyError.
                         If False and not found, returns default_value.

    Returns:
        Any: The retrieved configuration value, or the default_value.

    Raises:
        KeyError: If the key is required and not found.
    """
    if isinstance(key_path, str):
        keys = key_path.split('.')
    elif isinstance(key_path, list):
        keys = key_path
    else:
        raise TypeError("key_path must be a dot-separated string or a list of keys.")

    current_level = config
    for key in keys:
        if isinstance(current_level, dict) and key in current_level:
            current_level = current_level[key]
        else:
            if required:
                full_path = ".".join(keys)
                error_msg = f"Required configuration key '{full_path}' not found."
                logger.error(error_msg)
                raise KeyError(error_msg)
            return default_value
    return current_level


def save_config_to_yaml(config_data: Dict[str, Any], output_path: str) -> None:
    """
    Saves a configuration dictionary to a YAML file.

    Args:
        config_data (Dict[str, Any]): The configuration dictionary to save.
        output_path (str): The path where the YAML file will be saved.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(config_data, f, indent=2, sort_keys=False)
        logger.info(f"Configuration successfully saved to: {output_path}")
    except Exception as e:
        error_msg = f"Error saving configuration to {output_path}: {e}"
        logger.error(error_msg)
        raise


if __name__ == '__main__':
    print("--- Testing Config Parser Utilities ---")

    # Create dummy config files for testing
    os.makedirs("./tmp_test_configs", exist_ok=True)
    base_config_content = {
        "model": {"name": "BaseAgent", "type": "modular"},
        "dataset": {"name": "touchdown", "path": "/data/touchdown", "batch_size": 32},
        "training": {"epochs": 10, "lr": 1e-4, "optimizer": {"name": "adam", "beta1": 0.9}},
        "logging": {"level": "INFO"}
    }
    override_config_content = {
        "dataset": {"batch_size": 64, "augmentation": True}, # Override batch_size, add augmentation
        "training": {"lr": 5e-5, "scheduler": {"name": "steplr"}}, # Override lr, add scheduler
        "new_section": {"param_x": "value_x"} # Add new section
    }

    with open("./tmp_test_configs/base.yaml", "w") as f:
        yaml.dump(base_config_content, f)
    with open("./tmp_test_configs/override.yaml", "w") as f:
        yaml.dump(override_config_content, f)
    with open("./tmp_test_configs/empty.yaml", "w") as f:
        f.write("") # Empty file
    with open("./tmp_test_configs/invalid.yaml", "w") as f:
        f.write("key: value\nkey_no_val:") # Invalid YAML

    # 1. Test load_yaml_config
    print("\n1. Testing load_yaml_config...")
    loaded_base = load_yaml_config("./tmp_test_configs/base.yaml")
    assert loaded_base["model"]["name"] == "BaseAgent"
    print(f"  Loaded base config: {loaded_base}")

    loaded_empty = load_yaml_config("./tmp_test_configs/empty.yaml")
    assert loaded_empty == {}
    print(f"  Loaded empty config: {loaded_empty}")
    
    try:
        load_yaml_config("./tmp_test_configs/non_existent.yaml")
    except FileNotFoundError:
        print("  Correctly caught FileNotFoundError.")
    try:
        load_yaml_config("./tmp_test_configs/invalid.yaml")
    except yaml.YAMLError:
        print("  Correctly caught YAMLError.")

    # 2. Test merge_configs
    print("\n2. Testing merge_configs...")
    merged = merge_configs(loaded_base, override_config_content)
    print(f"  Merged config: {merged}")
    assert merged["dataset"]["batch_size"] == 64 # Overridden
    assert merged["dataset"]["augmentation"] is True # Added
    assert merged["training"]["lr"] == 5e-5 # Overridden
    assert "scheduler" in merged["training"] # Added nested
    assert merged["model"]["name"] == "BaseAgent" # From base
    assert "new_section" in merged # New section added

    # 3. Test get_config_value
    print("\n3. Testing get_config_value...")
    lr_val = get_config_value(merged, "training.lr")
    assert lr_val == 5e-5
    print(f"  Got 'training.lr': {lr_val}")

    model_name = get_config_value(merged, ["model", "name"])
    assert model_name == "BaseAgent"
    print(f"  Got ['model', 'name']: {model_name}")

    default_val_test = get_config_value(merged, "model.non_existent_key", default_value="default")
    assert default_val_test == "default"
    print(f"  Got 'model.non_existent_key' with default: {default_val_test}")

    try:
        get_config_value(merged, "dataset.path.deep", required=True)
    except KeyError:
        print("  Correctly caught KeyError for required missing key.")
    
    scheduler_name = get_config_value(merged, "training.scheduler.name", required=True)
    assert scheduler_name == "steplr"
    print(f"  Got 'training.scheduler.name': {scheduler_name}")


    # 4. Test save_config_to_yaml
    print("\n4. Testing save_config_to_yaml...")
    saved_config_path = "./tmp_test_configs/merged_saved.yaml"
    save_config_to_yaml(merged, saved_config_path)
    assert os.path.exists(saved_config_path)
    loaded_saved_config = load_yaml_config(saved_config_path)
    assert loaded_saved_config == merged # Check if content is the same
    print(f"  Saved and re-loaded config matches original merged config.")

    # Clean up dummy files
    import shutil
    shutil.rmtree("./tmp_test_configs")
    print("\nutils/config_parser.py tests completed and cleanup done.")