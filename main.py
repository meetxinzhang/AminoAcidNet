import torch.nn.init

from data_engineer.load_from_pkl import get_loader

from arguments import build_parser
parser = build_parser()
args = parser.parse_args()


# make_pickle(json_dir='/media/zhangxin/Raid0/dataset/PP/single_complex/bind_sites/json_dir/2/',
#             pkl_dir='/media/zhangxin/Raid0/dataset/PP/single_complex/bind_sites/pkl/2/')
loader = get_loader(pkl_dir='/media/zhangxin/Raid0/dataset/PP/single_complex/bind_sites/pkl/2/',
                    affinities_path='/media/zhangxin/Raid0/dataset/PP/index/INDEX_general_PP.2019')

# models
from network.mol_conv import AbstractConv
from network.mol_pooling import AbstractPooling
# conv1 = AtomConv(kernel_num=32, k_size=10)
# pool1 = AtomPooling(kernel_size=4, stride=4)

conv1 = AbstractConv(kernel_num=32, k_size=10, in_channels=1, node_fea_dim=5, edge_fea_dim=2)
pool1 = AbstractPooling(kernel_size=4, stride=4)

conv2 = AbstractConv(kernel_num=16, k_size=10, in_channels=32, node_fea_dim=1)
pool2 = AbstractPooling(kernel_size=4, stride=4)

conv3 = AbstractConv(kernel_num=4, k_size=5, in_channels=16, node_fea_dim=1)
pool3 = AbstractPooling(kernel_size=4, stride=4)

for [pos, atom_fea, edge_idx, edge_attr, res_idx, atom_mask], affinity in loader:
    # [bs, n_atom, 3], [bs, n_atom, 5],  [bs, n_atom, n_nei], [bs, n_atom, m_nei, 2], [bs, n_atom], [bs]
    # add channel dimension
    pos = pos.unsqueeze(1)
    atom_fea = atom_fea.unsqueeze(1)
    edge_idx = edge_idx.unsqueeze(1)
    edge_attr = edge_attr.unsqueeze(1)

    h1, pos1 = conv1(pos, atom_fea, atom_mask, edge_idx, edge_attr)
    p_pos, p_fea, p_ridx, p_mask = pool1(pos1, h1, res_idx, atom_mask)

    h2, pos2 = conv2(p_pos, p_fea, p_mask)
    p_pos2, p_fea2, p_ridx2, p_mask2 = pool2(pos2, h2, p_ridx, p_mask)

    h3, pos3 = conv3(p_pos2, p_fea2, p_mask2)
    p_pos3, p_fea3, p_ridx3, p_mask3 = pool3(pos3, h3, p_ridx2, p_mask2)

    print(p_pos)
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
#         'pkl_dir': args.pkl_dir,  # Root directory for data_engineer
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

