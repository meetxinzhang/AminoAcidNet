
from torch.utils.data import DataLoader
from data.dataset_in_torch import ProteinDataset, collate_pool

dataset = ProteinDataset(pkl_dir='/media/zhangxin/Raid0/dataset/PP/pkl/',
                         atom_init_filename='/media/zhangxin/Raid0/dataset/PP/pkl/protein_atom_init.json')
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_pool, shuffle=True, num_workers=10, pin_memory=False)

for (protein_atom_fea, nbr_fea, nbr_fea_idx, atom_amino_idx, atom_mask), (target, protein_id) in loader:
    print(protein_id)

