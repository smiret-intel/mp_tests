from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ase import Atoms

def load_atoms(k, v):
    s = Structure.from_dict(v["structure"])
    # Preprocessed structures so we don't need this anymore
    # sg = SpacegroupAnalyzer(s)
    # s = sg.get_conventional_standard_structure()
    a = s.to_ase_atoms()
    a.info["mp-id"] = k
    a.info["elastic-constants-ieee"] = v["elastic_tensor"]["ieee_format"]
    a.info["bulk-modulus-reuss"] = v["bulk_modulus"]["reuss"]
    return a


def get_isolated_energy_per_atom(calc, symbol):
    """
    Construct a non-periodic cell containing a single atom and compute its energy.
    """
    single_atom = Atoms(
        symbol,
        positions=[(0.1, 0.1, 0.1)],
        cell=(20, 20, 20),
        pbc=(False, False, False),
    )
    single_atom.calc = calc
    energy_per_atom = single_atom.get_potential_energy()
    if hasattr(calc, "__del__"):
        calc.__del__()
    del single_atom
    return energy_per_atom


mp_species = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
]
