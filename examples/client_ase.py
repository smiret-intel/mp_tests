import sys

sys.path.append("/store/code/open-catalyst/public-repo/matsciml")
import os

os.environ["KUSP_SERVER_CONFIG"] = (
    "/store/code/ai4science/mp_tests/examples/kusp_config.yaml"
)
import numpy as np
from ase import Atoms, io
from ase.calculators.kim import KIM

# Initialize KIM Model
model = KIM("KUSP__MO_000000000000_000")

config = io.read("./examples/Si.xyz")

# Set it as calculator
config.set_calculator(model)

# Compute energy/forces
energy = config.get_potential_energy()
forces = config.get_forces()

print(f"Forces: {forces}")
print(f"Energy: {energy}")
