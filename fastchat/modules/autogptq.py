from pathlib import Path

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

def load_autogptq_quantized(model_path, gptq_config: BaseQuantizeConfig, device, use_triton=False, trust_remote_code=False):
    model_path = Path(model_path)
    
    # find checkpoint
    ckpt_params = find_autogptq_ckpt(model_path, gptq_config, device)
    
    # load model
    base_params = {
        'use_triton': use_triton,
        'trust_remote_code': trust_remote_code,
        'device': "cuda:0" if device == 'cuda' else 'cpu',
    }
    params = {**base_params, **ckpt_params}
    print(params)
    model = AutoGPTQForCausalLM.from_quantized(model_path, **params)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    return model, tokenizer
    

def find_autogptq_ckpt(model_path, gptq_config: BaseQuantizeConfig, device):
    pt_path = None
    for ext in ['.safetensors', '.pt', '.bin']:
        found = list(model_path.glob(f"*{ext}"))
        if len(found) > 0:
            if len(found) > 1:
                print(f'More than one {ext} model has been found. The last one will be selected. It could be wrong.')
            pt_path = found[-1]
            break
        
    if pt_path is None:
        print("The model could not be loaded because its checkpoint file in .bin/.pt/.safetensors format could not be located.")
        return
    
    use_safetensors = pt_path.suffix == '.safetensors'
    use_config_file = (model_path / "quantize_config.json").exists() # use config if exists

    ckpt_params = {
        'model_basename': pt_path.stem,
        'use_safetensors': use_safetensors,
        'quantize_config': None if use_config_file else gptq_config,
    }

    return ckpt_params