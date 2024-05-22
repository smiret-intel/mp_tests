from mp_tests import Elasticity, EquilibriumCrystalStructure
from mp_tests.utils import mp_species

import os

os.environ["KUSP_SERVER_CONFIG"] = (
    "/store/code/ai4science/kusp/example/kusp_config.yaml"
)

test = Elasticity(
    "KUSP__MO_000000000000_000",
    supported_species=mp_species,
    db_name="mp_from_kusp.json",
)
# job_n: job number
# n_calcs: number of tests per job
test.mp_tests(it=10000, n_calcs=5, ignore_relax=True, method="stress-condensed-fast")

# test = Elasticity("KUSP__MO_000000000000_000", supported_species=mp_species)
# test.mp_tests(it=100, ignore_relax=True)#, method="stress-condensed-fast")


## Use this to run through relaxation simulations.
# test = EquilibriumCrystalStructure("KUSP__MO_000000000000_000", supported_species=mp_species)
# test.mp_tests(it=100)


# from mp_tests import Elasticity
# from mp_tests.utils import mp_species

# test = Elasticity(calc, supported_species=mp_species, db_name="mp_from_calculator.json")
# test.mp_tests(it=10, n_calcs=5, ignore_relax=True, method="stress-condensed-fast")
