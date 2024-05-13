from kim_test_utils.test_driver.core import KIMTestDriver

from tqdm import tqdm

import os

from ase.units import GPa

from mp_tests.mp_test_driver import MPTestDriver

from mp_tests.elastic import (
    ElasticConstants,
    calc_bulk,
    find_nearest_isotropy,
    get_unique_components_and_reconstruct_matrix,
)

from pymatgen.core.tensors import Tensor
from pymatgen.io.ase import AseAtomsAdaptor

import numpy as np

from mp_tests.utils import load_atoms, mp_species

from mp_tests.equilibrium import EquilibriumCrystalStructure

# TODO: Logging
class Elasticity(MPTestDriver):
    def _calculate(self, method="energy-condensed", bulk_calc=True, ignore_relax= False, **kwargs):
        """
        Performs calculation of elastic constants as implemented in elastic.py and writes output to TinyDB file. 
        Will minimize crystal structure first using EquilibriumCrystalStructure test

        Example:

        test = Elasticity(
                "SW_StillingerWeber_1985_Si__MO_405512056662_006",
                db_name = "mp.json"
            )

        atoms = ase.build.bulk('Si', 'diamond', a=5.43)

        test(atoms=atoms)


        Parameters:
            method : string
                Used to set the maximum distance an atom can move per iteration
            bulk_calc : bool
                Whether or not to compute bulk modulus from full elastic tensor
            ignore_relax: bool
                If True will proceed with elastic test even if structure wasn't successfully relaxed
            ** kwargs : 
                Other arguements which are passed to EquilibriumCrystalStructure
        """
        eq_test = EquilibriumCrystalStructure(
            self._calc, supported_species=self.supported_species, db_name=self.db_name
        )
        eq_test(self.atoms, **kwargs)
        if not ignore_relax:
            if eq_test.success is False:
                return
        self.atoms = eq_test.atoms
        del self.atoms.constraints

        print('\nE L A S T I C  C O N S T A N T  C A L C U L A T I O N S\n')

        moduli = ElasticConstants(self.atoms, condensed_minimization_method="bfgs")
        elastic_constants, error_estimate, success = moduli.results(
            optimize=False, method=method
        )
        if success:
            if bulk_calc:
                try:
                    bulk = calc_bulk(elastic_constants)
                    bulk /= GPa
                    self.insert_mp_outputs(
                        self.atoms.info["mp-id"],
                        "bulk-modulus-reuss",
                        self.atoms.info["bulk-modulus-reuss"],
                        bulk,
                    )
                except Exception as e:
                    self.insert_mp_outputs(
                        self.atoms.info["mp-id"],
                        "bulk-modulus-reuss",
                        self.atoms.info["bulk-modulus-reuss"],
                        {"error": str(e)},
                    )

            # Apply unit conversion
            elastic_constants /= GPa
            error_estimate /= GPa
            units = "GPa"

            # Compute nearest isotropic constants and distance
            #try:
            #    d_iso, bulk_iso, shear_iso = find_nearest_isotropy(elastic_constants)
            #    got_iso = True
            #except:
            #    got_iso = False  # Failure can occur if elastic constants are
                # not positive definite
            # Echo output
            #print("\nR E S U L T S\n")
            # print('Elastic constants [{}]:'.format(units))
            # print(np.array_str(elastic_constants, precision=5, max_line_width=100, suppress_small=True))
            # print()
            # print('Error estimate [{}]:'.format(units))
            # print(np.array_str(error_estimate, precision=5, max_line_width=100, suppress_small=True))
            # print()
            # print('Bulk modulus [{}] = {:.5f}'.format(units,bulk))
            # print()
            # if got_iso:
            #    print('Nearest matrix of isotropic elastic constants:')
            #    print('Distance to isotropic state [-]  = {:.5f}'.format(d_iso))
            #    print('Isotropic bulk modulus      [{}] = {:.5f}'.format(units,bulk_iso))
            #    print('Isotropic shear modulus     [{}] = {:.5f}'.format(units,shear_iso))
            # else:
            #    print('WARNING: Nearest isotropic state not computed.')

            # Not sure if necessary but convert to IEEE format to match MP
            try:
                t = Tensor.from_voigt(elastic_constants)
                elastic_constants = t.convert_to_ieee(
                    AseAtomsAdaptor.get_structure(self.atoms)
                ).voigt

                self.insert_mp_outputs(
                    self.atoms.info["mp-id"],
                    "elastic-constants-ieee",
                    self.atoms.info["elastic-constants-ieee"],
                    elastic_constants.tolist(),
                )
            except Exception as e:
                    self.insert_mp_outputs(
                    self.atoms.info["mp-id"],
                    "elastic-constants-ieee",
                    self.atoms.info["elastic-constants-ieee"],
                    {"error": str(e)},
                )
                
        else:
                self.insert_mp_outputs(
                self.atoms.info["mp-id"],
                "elastic-constants-ieee",
                self.atoms.info["elastic-constants-ieee"],
                {"error" : str(elastic_constants)}
            )

if __name__ == "__main__":
    '''
    from matgl.ext.ase import PESCalculator
    import matgl

    model = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    calc = PESCalculator(model)
    test = Elasticity(
        calc,
        supported_species=[
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
        ],
    )'''
    #test = Elasticity("SW_StillingerWeber_1985_Si__MO_405512056662_006", db_name='si.json')
    from mace.calculators import mace_mp
    model = mace_mp(default_dtype="float64")
    test = Elasticity(model,supported_species=mp_species, db_name='mp.json' )
    test.mp_tests(job_n=0, n_calcs=10,it=10,ignore_relax=True, method= "stress-condensed-fast")
