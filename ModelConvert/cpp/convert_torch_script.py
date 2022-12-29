import torch
import sys

sys.path.insert(0, './ModelConvert')

model = torch.load(r"D:\Vortex\SVN\遗留物\1.10-left_efficient_0919.pt")

example = torch.rand(1, 3, 224, 224).to('cuda:0')

traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("left_torch_script.pt")
