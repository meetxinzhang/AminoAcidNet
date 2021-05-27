import numpy as np
from arguments import build_parser
from data.build_pickle import make_pickle
from data.loader_from_pkl import get_loader

parser = build_parser()
args = parser.parse_args()

# make_h5_dataset(json_dir=args.json_dir, h5_dir=args.h5_dir, h5_name=args.h5_name)
# loader = construct_dataloader_from_h5(filename=args.h5_dir+args.h5_name, batch_size=args.batch_size)
make_pickle(json_dir=args.json_dir, pkl_dir=args.pkl_dir)
loader = get_loader(pkl_dir=args.pkl_dir)

for (atom_fea, atom_3d, neighbor_map, res_idx), (affinity, pdb_id) in loader:

    print('main(): ', np.shape(atom_fea))
    pass




















# dataset = ProteinDataset(pkl_dir=args.pkl_dir,
#                          atom_init_filename=args.atom_init)
# loader = DataLoader(dataset, batch_size=3, collate_fn=collate_pool, shuffle=True, num_workers=10, pin_memory=False)
#
#
# def getInputs(inputs, target):
#     """Move inputs and targets to cuda"""
#
#     input_var = [inputs[0].cuda(), inputs[1].cuda(), inputs[2].cuda(), inputs[3].cuda(), inputs[4].cuda()]
#     target_var = target.cuda()
#     return input_var, target_var
#
#
# parser = buildParser()
# args = parser.parse_args()
# for (atom_fea, nbr_fea, nbr_fea_idx, atom_amino_idx, atom_mask), (affinity, protein_id) in loader:
#
#     structures, _ = dataset[0]
#     h_b = structures[1].shape[-1]  # 2nd dim of structures
#     # build model
#     kwargs = {
#         'pkl_dir': args.pkl_dir,  # Root directory for data
#         'atom_init': args.atom_init,  # Atom Init filename
#         'h_a': args.h_a,  # Dim of the hidden atom embedding learnt
#         'h_g': args.h_g,  # Dim of the hidden graph embedding after pooling
#         'n_conv': args.n_conv,  # Number of GCN layers
#
#         'random_seed': args.seed,  # Seed to fix the simulation
#         'lr': args.lr,  # Learning rate for optimizer
#         'h_b': h_b  # Dim of the bond embedding initialization
#     }
#
#     # print("Let's use", torch.cuda.device_count(), "GPUs and Data Parallel Model.")
#     model = ProteinGCN(**kwargs)
#     # model = torch.nn.DataParallel(model)
#     # model.cuda()
#
#     model.train()
#
#     # inputs, target = getInputs([atom_fea, nbr_fea, nbr_fea_idx, atom_amino_idx, atom_mask], affinity)
#     inputs = [atom_fea, nbr_fea, nbr_fea_idx, atom_amino_idx, atom_mask]
#     out = model(inputs)
#     out = model.module.mask_remove(out)
#
#     print(out)

