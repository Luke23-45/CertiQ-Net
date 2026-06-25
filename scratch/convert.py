
import yaml
from pathlib import Path

def convert_cli_to_hydra(file_path):
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    
    if "model" not in data or "init_args" not in data["model"]:
        return False
        
    init_args = data["model"]["init_args"]
    
    hydra_data = {}
    
    # 1. Model Backbone
    if "model" in init_args and "class_path" in init_args["model"]:
        backbone = init_args["model"]
        hydra_model = {"_target_": backbone["class_path"]}
        if "init_args" in backbone and backbone["init_args"]:
            hydra_model.update(backbone["init_args"])
        hydra_data["model"] = hydra_model
    
    # 2. Loss config
    if "loss_fn" in init_args and "init_args" in init_args["loss_fn"] and init_args["loss_fn"]["init_args"]:
        hydra_data["loss"] = init_args["loss_fn"]["init_args"]
        
    # 3. Trainer / Misc params
    trainer_params = {}
    for k, v in init_args.items():
        if k not in ["model", "loss_fn", "input_normalization", "lam"]:
            if not k.startswith("dual_") and k not in ["target_kl_cert", "initial_policy_kl_weight"]:
                trainer_params[k] = v
                
    if trainer_params:
        hydra_data["trainer"] = trainer_params
        
    # 4. Lagrangian params
    lagrangian_params = {}
    for k, v in init_args.items():
        if k.startswith("dual_") or k in ["target_kl_cert", "initial_policy_kl_weight"]:
            lagrangian_params[k] = v
            
    if lagrangian_params:
        hydra_data["lagrangian"] = lagrangian_params
        
    # 5. Environment coupling
    if "lam" in init_args:
        hydra_data["lam"] = init_args["lam"]
        
    out_yaml = "# @package _global_\n" + yaml.dump(hydra_data, sort_keys=False)
    
    with open(file_path, "w") as f:
        f.write(out_yaml)
    return True

configs_dir = Path("configs/cli/model")
for f in configs_dir.glob("*.yaml"):
    convert_cli_to_hydra(f)
    print(f"Converted {f}")

