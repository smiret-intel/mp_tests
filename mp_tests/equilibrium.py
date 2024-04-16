from kim_test_utils.test_driver.core import KIMTestDriver

import numpy as np
import os
from tqdm import tqdm
from ase import Atoms
from ase.spacegroup.symmetrize import FixSymmetry
from ase.constraints import UnitCellFilter
from ase.optimize import BFGS, FIRE, BFGSLineSearch

from mp_tests.mp_test_driver import MPTestDriver
from mp_tests.utils import get_isolated_energy_per_atom
from mp_tests.utils import load_atoms, mp_species

class EquilibriumCrystalStructure(MPTestDriver):
    def _calculate(self, maxstep=0.05, ftol=1e-8, it=10000):
        """
        Performs calculation of equilibrium crystal structure and writes output to TinyDB file

        Example:

        test = EquilibriumCrystalStructure(
                "SW_StillingerWeber_1985_Si__MO_405512056662_006",
                db_name = "mp.json"
            )

        atoms = ase.build.bulk('Si', 'diamond', a=5.43)

        test(atoms=atoms, maxstep=0.05, ftol=1e-4, it=10000)


        Parameters:
            maxstep : float
                Used to set the maximum distance an atom can move per iteration
            ftol : float
                Convergence criterion of the forces on atoms
            it : int
                Maximum number of iterations for the minimization
        """
        gt_lengths = self.atoms.cell.lengths().tolist()
        gt_angles = self.atoms.cell.angles().tolist()
        
        symmetry = FixSymmetry(self.atoms)
        self.atoms.set_constraint(symmetry)
        atoms_wrapped = UnitCellFilter(self.atoms)
        # Optimize
        opt = BFGSLineSearch(atoms_wrapped, maxstep=maxstep)  # logfile=None)
        try:
            converged = opt.run(fmax=ftol, steps=it)
            iteration_limits_reached = not converged
            minimization_stalled = False
        except:
            minimization_stalled = True
            iteration_limits_reached = False
            return
        if minimization_stalled or iteration_limits_reached:
            self.success = False
        else:
            self.success = True
        forces = self.atoms.get_forces()
        stress = self.atoms.get_stress()
        # Compute the average energy per atom after subtracting out the energies of the
        # isolated atoms
        energy_isolated = sum(
            [
                get_isolated_energy_per_atom(self._calc, sym)
                for sym in self.atoms.get_chemical_symbols()
            ]
        )
        energy_per_atom = (
            self.atoms.get_potential_energy() - energy_isolated
        ) / self.atoms.get_global_number_of_atoms()
        # print("Minimization " +
        #      ("converged" if not minimization_stalled else "stalled") +
        #      " after " +
        #      (("hitting the maximum of "+str(ITER)) if iteration_limits_reached else str(opt.nsteps)) +
        #      " steps.")
        # print("Maximum force component: " +
        #      str(np.max(np.abs(forces)))+" eV/Angstrom")
        # print("Maximum stress component: " +
        #      str(np.max(np.abs(stress)))+" eV/Angstrom^3")
        # print("==== Minimized structure obtained from ASE ====")
        # print("symbols = ", self.atoms.get_chemical_symbols())
        # print("basis = ", self.atoms.get_scaled_positions())
        # print("cellpar = ", self.atoms.get_cell())
        # print("forces = ", forces)
        # print("stress = ", stress)
        # print("energy per atom = ", energy_per_atom)
        # print("===============================================")

        lengths = self.atoms.cell.lengths().tolist()
        angles = self.atoms.cell.angles().tolist()

        self.insert_mp_outputs(
            self.atoms.info["mp-id"], "cell-lengths", gt_lengths, lengths
        )
        self.insert_mp_outputs(
            self.atoms.info["mp-id"], "cell-angles", gt_angles, angles
        )

    def mp_tests(self):
        """Loads all structures with computed elastic constants from Materials Project and computes
        elastic constants for it if the model supports the species present
        """
        import pickle

        mp_dict = pickle.load(open("%s/%s/mp_elasticity_conventional_4-9-24.pkl" %(os.path.dirname(__file__), "data"), "rb"))
        for k, v in tqdm(mp_dict.items()):
            atoms = load_atoms(k, v)
            self(atoms)
