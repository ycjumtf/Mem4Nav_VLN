import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau, LambdaLR
from typing import Dict, Any, Iterable, List, Optional, Union
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

def get_parameter_groups(model: torch.nn.Module,
                         weight_decay: float,
                         no_decay_keywords: Optional[List[str]] = None,
                         custom_param_group_configs: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Sets up parameter groups for the optimizer, typically to apply weight decay
    selectively (e.g., not to biases and LayerNorm weights).

    Args:
        model (torch.nn.Module): The model whose parameters are to be optimized.
        weight_decay (float): The base weight decay value.
        no_decay_keywords (Optional[List[str]]): A list of keywords (e.g., "bias", ".norm.weight")
                                                that identify parameters to exclude from weight decay.
                                                Defaults to common ones if None.
        custom_param_group_configs (Optional[List[Dict[str, Any]]]):
            A list of custom parameter group configurations. Each dict can specify
            'params' (a list of parameter names or a filter function) and other
            optimizer-specific options like 'lr', 'weight_decay'.
            If None, uses default grouping (decay vs no_decay).

    Returns:
        List[Dict[str, Any]]: A list of parameter groups suitable for an optimizer.
    """
    if no_decay_keywords is None:
        no_decay_keywords = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight", "ln_1.weight", "ln_2.weight", "ln_final.weight"]

    if custom_param_group_configs:
        # TODO: Implement logic to assign parameters based on custom_param_group_configs.
        # This would involve matching parameter names to the 'params' spec in each config.
        # For now, we'll stick to the simpler decay/no_decay split.
        print("Warning: custom_param_group_configs not yet fully implemented in get_parameter_groups. Using default decay/no_decay.")

    optimizer_grouped_parameters = []
    decay_parameters_names = set()
    all_param_names = {name for name, _ in model.named_parameters() if _.requires_grad}

    # Identify parameters for weight decay
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not any(no_decay_keyword in name for no_decay_keyword in no_decay_keywords):
            decay_parameters_names.add(name)

    decay_params = [param for name, param in model.named_parameters()
                    if name in decay_parameters_names and param.requires_grad]
    no_decay_params = [param for name, param in model.named_parameters()
                       if name not in decay_parameters_names and param.requires_grad]

    optimizer_grouped_parameters.append({
        "params": decay_params,
        "weight_decay": weight_decay,
    })
    optimizer_grouped_parameters.append({
        "params": no_decay_params,
        "weight_decay": 0.0,
    })
    
    # Sanity check: ensure all trainable params are in some group
    grouped_params_count = sum(len(group["params"]) for group in optimizer_grouped_parameters)
    trainable_params_count = sum(1 for p in model.parameters() if p.requires_grad)
    if grouped_params_count != trainable_params_count:
        print(f"Warning: Mismatch in parameter counts for optimizer groups. "
              f"Grouped: {grouped_params_count}, Trainable: {trainable_params_count}")
        # This might happen if some parameters were missed. For robust handling,
        # one might add a final group for any remaining parameters.

    return optimizer_grouped_parameters


def create_optimizer(model_or_params: Union[torch.nn.Module, Iterable[torch.nn.Parameter], List[Dict[str, Any]]],
                     optimizer_config: Dict[str, Any]) -> optim.Optimizer:
    """
    Creates an optimizer based on the provided configuration.

    Args:
        model_or_params (Union[torch.nn.Module, Iterable[torch.nn.Parameter], List[Dict[str, Any]]]):
            Either the model (its parameters will be used), an iterable of parameters,
            or a list of parameter groups.
        optimizer_config (Dict[str, Any]): Configuration for the optimizer.
            Expected keys:
            - 'name' (str): Name of the optimizer (e.g., "adamw", "sgd", "adam").
            - 'lr' (float): Learning rate.
            - 'weight_decay' (float, optional): Weight decay. Default 0.
            - 'beta1', 'beta2' (float, optional): Betas for Adam/AdamW. Default 0.9, 0.999.
            - 'eps' (float, optional): Epsilon for Adam/AdamW. Default 1e-8.
            - 'momentum' (float, optional): Momentum for SGD. Default 0.
            - 'no_decay_keywords' (List[str], optional): Keywords for params to exclude from wd.
                                                         Only used if model_or_params is nn.Module.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """
    opt_name = optimizer_config.get("name", "adamw").lower()
    lr = optimizer_config.get("lr", 1e-4)
    weight_decay = optimizer_config.get("weight_decay", 0.0)

    params_to_optimize: Union[Iterable[torch.nn.Parameter], List[Dict[str, Any]]]
    if isinstance(model_or_params, torch.nn.Module):
        no_decay_kw = optimizer_config.get("no_decay_keywords")
        params_to_optimize = get_parameter_groups(model_or_params, weight_decay, no_decay_keywords=no_decay_kw)
    elif isinstance(model_or_params, list) and all(isinstance(p_group, dict) for p_group in model_or_params):
        params_to_optimize = model_or_params # Already parameter groups
    elif isinstance(model_or_params, Iterable):
        params_to_optimize = model_or_params # Iterable of parameters
    else:
        raise TypeError("model_or_params must be nn.Module, iterable of Parameters, or list of param_groups.")


    if opt_name == "adamw":
        beta1 = optimizer_config.get("beta1", 0.9)
        beta2 = optimizer_config.get("beta2", 0.999)
        eps = optimizer_config.get("eps", 1e-8)
        optimizer = optim.AdamW(params_to_optimize, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
        # Note: if get_parameter_groups applied weight_decay, the optimizer's top-level weight_decay
        # might be redundant or interact. AdamW handles weight decay correctly per group if groups are passed.
        # If params_to_optimize is not already grouped, AdamW's weight_decay applies to all.
        # If params_to_optimize *is* from get_parameter_groups, then AdamW's top-level weight_decay
        # argument here might be ignored or conflict. It's safer if get_parameter_groups
        # fully specifies weight_decay for each group and the main AdamW weight_decay is 0 or unused.
        # However, typical HuggingFace Trainer sets global wd and AdamW handles it.
        # The get_parameter_groups already sets group-specific wd.
    elif opt_name == "adam":
        beta1 = optimizer_config.get("beta1", 0.9)
        beta2 = optimizer_config.get("beta2", 0.999)
        eps = optimizer_config.get("eps", 1e-8)
        optimizer = optim.Adam(params_to_optimize, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
    elif opt_name == "sgd":
        momentum = optimizer_config.get("momentum", 0.0)
        optimizer = optim.SGD(params_to_optimize, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer name: {opt_name}. "
                         "Supported: 'adamw', 'adam', 'sgd'.")

    return optimizer


def create_scheduler(optimizer: optim.Optimizer,
                     scheduler_config: Dict[str, Any],
                     num_training_steps: Optional[int] = None, # Needed for some schedulers like linear/cosine with warmup
                     num_warmup_steps: Optional[int] = None
                    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]: # type: ignore
    """
    Creates a learning rate scheduler based on the provided configuration.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to create the scheduler.
        scheduler_config (Dict[str, Any]): Configuration for the scheduler.
            Expected keys:
            - 'name' (str): Name of the scheduler (e.g., "steplr", "cosine_annealing", "reduce_on_plateau", "linear_warmup").
            - 'step_size' (int, optional): For StepLR.
            - 'gamma' (float, optional): For StepLR.
            - 'T_max' (int, optional): For CosineAnnealingLR (often num_epochs or num_training_steps).
            - 'eta_min' (float, optional): For CosineAnnealingLR.
            - 'factor', 'patience', 'threshold' (optional): For ReduceLROnPlateau.
            - 'num_warmup_steps_ratio' (float, optional): For linear/cosine warmup (ratio of total training steps). If 'num_warmup_steps' is given, it's used directly.
        num_training_steps (Optional[int]): Total number of training steps. Required for some schedulers.
        num_warmup_steps (Optional[int]): Number of warmup steps. Calculated if ratio is given.

    Returns:
        Optional[torch.optim.lr_scheduler._LRScheduler]: The initialized scheduler, or None if no scheduler specified.
    """
    scheduler_name = scheduler_config.get("name")
    if not scheduler_name:
        return None

    scheduler_name = scheduler_name.lower()

    if scheduler_name == "steplr":
        step_size = scheduler_config.get("step_size", 30)
        gamma = scheduler_config.get("gamma", 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == "cosine_annealing":
        if num_training_steps is None and "T_max" not in scheduler_config:
            raise ValueError("T_max (or num_training_steps for derivation) is required for CosineAnnealingLR.")
        T_max = scheduler_config.get("T_max", num_training_steps)
        eta_min = scheduler_config.get("eta_min", 0.0)
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min) # type: ignore
    elif scheduler_name == "reduce_on_plateau":
        factor = scheduler_config.get("factor", 0.1)
        patience = scheduler_config.get("patience", 10)
        threshold = scheduler_config.get("threshold", 1e-4)
        return ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, threshold=threshold)
    elif scheduler_name in ["linear_warmup", "cosine_warmup"]: # Using transformers schedulers
        try:
            from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
        except ImportError:
            raise ImportError("Hugging Face Transformers library is required for 'linear_warmup' or 'cosine_warmup' schedulers.")

        if num_training_steps is None:
            raise ValueError("num_training_steps is required for warmup schedulers.")
        
        actual_num_warmup_steps = num_warmup_steps
        if actual_num_warmup_steps is None:
            warmup_ratio = scheduler_config.get("num_warmup_steps_ratio", 0.1)
            actual_num_warmup_steps = int(num_training_steps * warmup_ratio)
        
        if scheduler_name == "linear_warmup":
            return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=actual_num_warmup_steps, num_training_steps=num_training_steps) # type: ignore
        else: # cosine_warmup
            return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=actual_num_warmup_steps, num_training_steps=num_training_steps) # type: ignore
    
    elif scheduler_name == "lambda_lr": # Example for a custom lambda scheduler
        # Expects 'lr_lambda_func_str' in config which is a string representation of a lambda function
        # e.g., "lambda epoch: 0.95 ** epoch"
        # This is advanced and potentially unsafe if eval-ing strings, use with caution.
        lr_lambda_str = scheduler_config.get("lr_lambda_func_str")
        if lr_lambda_str:
            try:
                # WARNING: eval is a security risk if lr_lambda_str is not trusted.
                lr_lambda_func = eval(lr_lambda_str)
                return LambdaLR(optimizer, lr_lambda=lr_lambda_func)
            except Exception as e:
                raise ValueError(f"Could not evaluate lr_lambda_func_str: {lr_lambda_str}. Error: {e}")
        else:
            raise ValueError("'lr_lambda_func_str' required for LambdaLR scheduler.")
    else:
        print(f"Warning: Unsupported scheduler name: {scheduler_name}. No scheduler will be used.")
        return None


if __name__ == '__main__':
    print("--- Testing Optimizer and Scheduler Creation ---")
    
    # Dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_extractor = nn.Linear(10, 20) # Example "backbone" part
            self.norm = nn.LayerNorm(20)
            self.classifier_bias = nn.Parameter(torch.randn(5))
            self.head = nn.Linear(20, 5)
    
    model = DummyModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Test Optimizer Creation
    print("\nTesting Optimizer Creation...")
    opt_config_adamw = {
        "name": "adamw",
        "lr": 1e-3,
        "weight_decay": 0.01,
        "no_decay_keywords": ["bias", "norm.weight"] # Test selective weight decay
    }
    optimizer_adamw = create_optimizer(model, opt_config_adamw)
    print(f"  AdamW Optimizer created: {optimizer_adamw}")
    assert isinstance(optimizer_adamw, optim.AdamW)
    # Check parameter groups for weight decay
    assert len(optimizer_adamw.param_groups) == 2
    # Params like 'classifier_bias' and 'norm.weight' should have wd=0.0
    # Params like 'feature_extractor.weight', 'head.weight' should have wd=0.01
    for group in optimizer_adamw.param_groups:
        for param in group['params']:
            param_name = [name for name, p_model in model.named_parameters() if p_model is param][0]
            if any(ndkw in param_name for ndkw in opt_config_adamw["no_decay_keywords"]):
                assert group['weight_decay'] == 0.0, f"Param {param_name} should have 0 wd."
            else:
                assert group['weight_decay'] == 0.01, f"Param {param_name} should have non-zero wd."


    opt_config_sgd = {"name": "sgd", "lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4}
    # For SGD, pass all params as one group if not using custom grouping for SGD specific wd
    optimizer_sgd = create_optimizer(model.parameters(), opt_config_sgd)
    print(f"  SGD Optimizer created: {optimizer_sgd}")
    assert isinstance(optimizer_sgd, optim.SGD)

    # Test Scheduler Creation
    print("\nTesting Scheduler Creation...")
    total_steps = 1000
    warmup_steps_count = 100

    sched_config_steplr = {"name": "StepLR", "step_size": 300, "gamma": 0.1}
    scheduler_steplr = create_scheduler(optimizer_adamw, sched_config_steplr)
    print(f"  StepLR Scheduler created: {scheduler_steplr}")
    assert isinstance(scheduler_steplr, StepLR)

    sched_config_cosine = {"name": "cosine_annealing", "T_max": total_steps, "eta_min": 1e-6}
    scheduler_cosine = create_scheduler(optimizer_adamw, sched_config_cosine)
    print(f"  CosineAnnealingLR Scheduler created: {scheduler_cosine}")
    assert isinstance(scheduler_cosine, CosineAnnealingLR)

    # Test with warmup (requires transformers)
    try:
        sched_config_linear_warmup = {"name": "linear_warmup", "num_warmup_steps_ratio": 0.1}
        scheduler_linear_warmup = create_scheduler(optimizer_adamw, sched_config_linear_warmup, num_training_steps=total_steps)
        print(f"  Linear Warmup Scheduler created: {scheduler_linear_warmup}")
        assert scheduler_linear_warmup is not None # Type check would need transformers imported

        sched_config_cosine_warmup_abs = {"name": "cosine_warmup"} # num_warmup_steps passed directly
        scheduler_cosine_warmup_abs = create_scheduler(optimizer_adamw, sched_config_cosine_warmup_abs,
                                                       num_training_steps=total_steps, num_warmup_steps=warmup_steps_count)
        print(f"  Cosine Warmup Scheduler (abs steps) created: {scheduler_cosine_warmup_abs}")
        assert scheduler_cosine_warmup_abs is not None

    except ImportError:
        print("  Skipping warmup scheduler tests as 'transformers' library is not installed.")
    except ValueError as ve:
        print(f"  Skipping warmup scheduler tests due to ValueError: {ve}")


    # Simulate a few steps with optimizer and scheduler
    print("\nSimulating training steps with AdamW and StepLR...")
    initial_lr = optimizer_adamw.param_groups[0]['lr']
    for epoch in range(2): # Simulate epochs
        for step in range(10): # Simulate steps within epoch
            # loss.backward()
            optimizer_adamw.step()
            optimizer_adamw.zero_grad()
        # Schedulers like StepLR are often called per epoch
        if scheduler_steplr:
             scheduler_steplr.step()
        current_lr = optimizer_adamw.param_groups[0]['lr']
        print(f"  Epoch {epoch+1}, LR: {current_lr}")
    assert current_lr < initial_lr # LR should have changed after some epochs with StepLR

    print("\ntraining_utils/optimizers.py tests completed.")