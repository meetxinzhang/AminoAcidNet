from data.load_from_pkl import get_loader

from arguments import build_parser
parser = build_parser()
args = parser.parse_args()


# make_pickle(json_dir='/media/zhangxin/Raid0/dataset/PP/single_complex/bind_sites/json_dir/2/',
#             pkl_dir='/media/zhangxin/Raid0/dataset/PP/single_complex/bind_sites/pkl/2/')
loader = get_loader(pkl_dir='/media/zhangxin/Raid0/dataset/PP/single_complex/bind_sites/pkl_PyG/2/')

# models

for [pos, atom_fea, edge_idx, edge_attr, res_idx], affinity in loader:
    # [n_atom, 3], [n_atom, 5],  [n_atom, n_nei], [n_atom, m_nei, 2], [n_atom]
    print('main(): ', affinity)
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

