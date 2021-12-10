# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 12/10/21 6:28 PM
@desc:
"""

import torch
from alphafold2_pytorch import Alphafold2

model = Alphafold2(
    dim=256,
    depth=2,
    heads=8,
    dim_head=64,
    predict_coords=True,
    structure_module_type='se3',
    # use SE3 Transformer - if set to False, will use E(n)-Transformer, Victor and Max Welling's new paper
    structure_module_dim=4,  # se3 transformer dimension
    structure_module_depth=1,  # depth
    structure_module_heads=1,  # heads
    structure_module_dim_head=16,  # dimension of heads
    structure_module_refinement_iters=2,  # number of equivariant coordinate refinement iterations
    structure_num_global_nodes=1  # number of global nodes for the structure module, only works with SE3 transformer
).cuda()

seq = torch.randint(0, 21, (2, 64)).cuda()
msa = torch.randint(0, 21, (2, 5, 60)).cuda()
mask = torch.ones_like(seq).bool().cuda()
msa_mask = torch.ones_like(msa).bool().cuda()

coords = model(
    seq,
    msa,
    mask=mask,
    msa_mask=msa_mask
)  # (2, 64 * 3, 3)  <-- 3 atoms per residue
