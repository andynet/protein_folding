

# %%
import torch

# %%
# fp = "/faststorage/project/deeply_thinking_potato/data/our_input/temp/1r9dA01.pssm"
# with open(fp) as f:
#     lines = f.readlines()

# %%
# fp = "/faststorage/project/deeply_thinking_potato/data/our_input/temp/1r9dA01.pssm"
# with open(fp) as f:
#     f.readline()
#     f.readline()
#     f.readline()
#     while True:
#         line = f.readline()
#         print(line)
#         if line == '\n':
#             break
#         if line == '\EOF':
#             pass

# %%
# f = open(fp)
# for line in f:
#    print(line)

# %%
# tmp = torch.load("/faststorage/project/deeply_thinking_potato/data/our_input/pssm/2lgqA00_pssm.pt")
tmp = torch.load("/faststorage/project/deeply_thinking_potato/data/our_input/pssm/1r9dA01_pssm.pt")