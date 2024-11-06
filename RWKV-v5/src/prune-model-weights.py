'''
load a model, guess its version (x52, x58, x59 ...), 
check if it has cls or mlp,
based on the model version, drop unnecessary weights, and save the model back (optional)
'''

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch
from torch.nn import functional as F
import sys


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python prune-model-weights.py <model_name> [--writeback]")
        sys.exit(1)

    is_written = "--writeback" in sys.argv

    model_name = sys.argv[1]
    print(f"Model name: {model_name}")
    if not model_name.endswith('.pth'):
        print("Error: Model name must end with .pth")
        sys.exit(1)

    model_out_name = model_name.replace('.pth', '-pruned.pth')
    print(f"Output model name: {model_out_name}")

    w = torch.load(model_name, map_location='cpu', weights_only=True)
    # breakpoint()
    # print(w.keys())
    keys = list(w.keys())

    # same as rwkv/model.py RWKV::__init__ line 366
    version = 4
    has_mlp = False
    has_cls = False 

    keywords_to_print = []
    keywords_to_print += ['blocks.0']

    for x in keys:
        for keywords in keywords_to_print:
            if keywords in x:
                print(x)

        if 'ln_x' in x:
            version = max(5, version)
        if 'gate.weight' in x:
            version = max(5.1, version)
        if int(version) == 5 and 'att.time_decay' in x:
            if len(w[x].shape) > 1:
                if w[x].shape[1] > 1:
                    version = max(5.2, version)
        if 'key1.weight' in x: # xzl
            version = max(5.8, version)
        if 'key_diag' in x:  # xzl
            version = max(5.9, version)
        if 'ffn.key1.weight' in x:
            version = max(5.94, version)
        if 'ffn.key_diag' in x:
            version = max(5.95, version)
        if 'time_maa' in x:
            print("SISIXIXIX")
            version = max(6, version)
        if 'head_l1.weight' in x:             
            has_cls = True
        if 'mlp.fc1.weight' in x: 
            has_mlp = True

    print(f'Model detected: v{version:.2f} {"cls:YES" if has_cls else "cls:no"} {"mlp:YES" if has_mlp else "mlp:no"}')

    # check for consistency (also document) 
    if has_cls: 
        assert("head_l1.weight" in keys) 
        # head_l2.weight is constructed at load time
        # we cannot delete 'head.weight', which is used to construct the head l2 projection, at 
        # model load time. will delete after that 

    if has_mlp:
        pass 

    # these are keywords
    keys_to_delete = []
    # legacy designs
    keys_to_delete.append('head_l1fc1')
    keys_to_delete.append('head_l1fc2')

    if version == 5.8 or version == 5.9:
        keys_to_delete.append('.att.key.weight')
        keys_to_delete.append('.att.value.weight')
        keys_to_delete.append('.att.receptance.weight')
        keys_to_delete.append('.fnn.receptance.weight')

    for x in keys: 
        for keywords in keys_to_delete:
            if keywords in x:
                print(f"will delete: {x}, shape: {w[x].shape}, size: {w[x].nelement() * w[x].element_size() / 1024:.2f} KB")
                del w[x]

    # keys = list(w.keys())
    # print("after deleteion", keys)

    if is_written:
        torch.save(w, model_out_name)
        print(f"Model saved to {model_out_name}")

