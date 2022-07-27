# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 7/26/22 10:26 AM
@desc:
ALL parameters are >0, and ==0 indicates None or unknown.
"""


class Atom:
    def __init__(self, name, id=0, n_bond=4, bonds=None):
        """bonds: a list involves Bond id"""
        self.name = name
        self.id = id
        self.n_bond = n_bond
        self.bonds = bonds


class Bond:
    def __init__(self, name, id=0, distance=2, electrons=0, fromm=0, to=0):
        self.name = name
        self.id = id
        self.distance = distance
        self.electrons = electrons
        self.fromm = fromm
        self.to = to


class Formula:
    def __init__(self, name, id=0, atoms=None, bonds=None):
        self.name = name
        self.id = id
        if self.chemical_check(atoms, bonds):
            self.atoms = atoms
            self.bonds = bonds

    def chemical_check(self, atoms, bonds):
        """check in-degree of all atoms
        if atoms.n_bonds == frequency in bonds ?

        """
        _bonds = sum(bonds, []).sort()  # multiple dimension to one dimension, sort
        num_the_atom_in_bonds =
        for atom in atoms:
            assert atom.n_bond == 
        return False
