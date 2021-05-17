import numpy as np
from torch.utils.data import DataLoader
from data.dataset_in_torch import ProteinDataset, collate_pool
import torch
from network.proteinGCN import ProteinGCN
from arguments import buildParser


parser = buildParser()
args = parser.parse_args()

dataset = ProteinDataset(pkl_dir=args.pkl_dir,
                         atom_init_filename=args.atom_init)
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_pool, shuffle=True, num_workers=10, pin_memory=False)


def getInputs(inputs, target):
    """Move inputs and targets to cuda"""

    input_var = [inputs[0].cuda(), inputs[1].cuda(), inputs[2].cuda(), inputs[3].cuda(), inputs[4].cuda()]
    target_var = target.cuda()
    return input_var, target_var


parser = buildParser()
args = parser.parse_args()
for (protein_atom_fea, nbr_fea, nbr_fea_idx, atom_amino_idx, atom_mask), (target, protein_id) in loader:

    structures, _ = dataset[0]
    h_b = structures[1].shape[-1]  # 2nd dim of structures
    # build model
    kwargs = {
        'pkl_dir': args.pkl_dir,  # Root directory for data
        'atom_init': args.atom_init,  # Atom Init filename
        'h_a': args.h_a,  # Dim of the hidden atom embedding learnt
        'h_g': args.h_g,  # Dim of the hidden graph embedding after pooling
        'n_conv': args.n_conv,  # Number of GCN layers

        'random_seed': args.seed,  # Seed to fix the simulation
        'lr': args.lr,  # Learning rate for optimizer
        'h_b': h_b  # Dim of the bond embedding initialization
    }

    print("Let's use", torch.cuda.device_count(), "GPUs and Data Parallel Model.")
    model = ProteinGCN(**kwargs)
    model = torch.nn.DataParallel(model)
    model.cuda()

    model.train()

    inputs, target = getInputs([protein_atom_fea, nbr_fea, nbr_fea_idx, atom_amino_idx, atom_mask], target)
    out = model(inputs)
    out = model.module.mask_remove(out)

    print(out)

