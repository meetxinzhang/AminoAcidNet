# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 7/26/22 10:26 AM
@desc:
ALL parameters are >0, and ==0 indicates None or unknown.
"""
import torch


class Atom:
    def __init__(self, name, id=0, n_bond=4):
        """bonds: a list involves Bond id"""
        self.name = name
        self.id = id
        self.n_bond = n_bond

    def to_tensor(self):
        return torch.Tensor([self.name, self.id, self.n_bond])


class Bond:
    def __init__(self, name, id=0, distance=2, electrons=0, t1=0, t2=0):
        self.name = name
        self.id = id
        self.distance = distance
        self.electrons = electrons  # single/double-bond
        self.t1 = t1  # terminal 1
        self.t2 = t2  # terminal 2

    def to_tensor(self):
        return torch.Tensor[self.name, self.id, self.distance, self.electrons, self.t1, self.t2]


class Formula:
    def __init__(self, name, id=0, atoms=None, bonds=None):
        self.name = name
        self.id = id
        if self.chemical_check(atoms, bonds):
            self.atoms = atoms
            self.bonds = bonds

    def chemical_check(self, atoms, bonds):
        """check in-degree of all atoms
        @atoms: tensor [name, id, n_bond], assert atoms are sorted by id.
        @bonds: ditto  [name, id, distance, electrons, t1, t2]
        algorithm ideas: bonds->slice->flatten->sort->count == atoms->slice->sort ?
        """
        terminal_12 = torch.index_select(bonds, dim=1, index=torch.Tensor([4, 5]))  # t1, t2
        terminals = terminal_12.flatten()  # .view(.numel()) # .reshape(-1)
        sorted_terminals = torch.sort(terminals, dim=0)
        _, count1 = sorted_terminals.unique(return_count=True)

        bond_degree = atoms[:, -1]
        assert bond_degree.equal(count1), 'chemical checking failed'

        # _bonds = sum(bonds, [])  # multiple dimension to one dimension, sort
        # for atom in atoms:
        #     assert atom.n_bond == len(atom.bonds)
        #     assert atom.n_bond ==
        pass



