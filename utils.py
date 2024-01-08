import contextlib
from itertools import chain

import torch


def slicepart(l, block_size, slice_inside_block=slice(None, None)):
    return list(
        chain(
            *(
                l[i : i + block_size][slice_inside_block]
                for i in range(0, len(l), block_size)
            )
        )
    )


def normalize(x):
    x = x - x.min()
    x = x / x.max()
    return x


patch_module_list = [
    torch.nn.Linear,
    torch.nn.Conv2d,
    torch.nn.MultiheadAttention,
    torch.nn.GroupNorm,
    torch.nn.LayerNorm,
]


def manual_cast_forward(dtype):
    def forward_wrapper(self, *args, **kwargs):
        if isinstance(self, torch.nn.MultiheadAttention):
            target_dtype = torch.float32  
        else:
            target_dtype = dtype
        org_dtype = next(self.parameters()).dtype
        self.to(target_dtype)
        args = [
            arg.to(target_dtype) 
            if isinstance(arg, torch.Tensor) 
            else arg 
            for arg in args
        ]
        kwargs = {
            k: v.to(target_dtype) 
            if isinstance(v, torch.Tensor) 
            else v 
            for k, v in kwargs.items()
        }
        result = self.org_forward(*args, **kwargs)
        self.to(org_dtype)
        
        if isinstance(result, tuple):
            result = tuple(
                i.to(dtype) 
                if isinstance(i, torch.Tensor) 
                else i 
                for i in result
            )
        elif isinstance(result, torch.Tensor):
            result = result.to(dtype)
        return result
    return forward_wrapper


@contextlib.contextmanager
def manual_cast(dtype):
    patched = {}
    for module_type in patch_module_list:
        if hasattr(module_type, "org_forward"):
            continue
        org_forward = module_type.forward
        module_type.forward = manual_cast_forward(dtype)
        module_type.org_forward = org_forward
        patched[module_type] = True
    try:
        yield None
    finally:
        for module_type in patched:
            module_type.forward = module_type.org_forward
            delattr(module_type, "org_forward")