from tinydb import TinyDB
import datetime
from kim_test_utils.test_driver.core import KIMTestDriver
#from jobflow import job
import os
from mp_tests.utils import load_atoms
from tqdm import tqdm
class MPTestDriver(KIMTestDriver):
    """
    Base class for tests performed over Materials Project data. 

    Parameters:
        model : string or ASE Calculator
            Name of KIM Model or ASE Calculator object to use to compute energy, forces, etc.
        
        supported_species : list[string]
            List of chemical species supported by the model. 
            Only needed if an ASE Calculator is passed in.

        db_name : string
            Name of TinyDB file to write outputs to


    """
    def __init__(self, model, supported_species=None, db_name="mp.json"):
        super().__init__(model)
        self.db_name = db_name
        self.supported_species = []
        if hasattr(self._calc,"species_map"):
            for k,v in self._calc.species_map.items():
                self.supported_species.append(k)
        else:
            if len(supported_species)>0:
                self.supported_species = supported_species
            else:
                raise Exception("'supported_species' must be given if passing a calculator instead of a KIM model")
    #@job jobflow job
    def __call__(self, atoms, **kwargs):
        self._setup(atoms, **kwargs)
        check = self.check_supported_species()
        if check:
            self.atoms.calc = self._calc
            results = self._calculate(**kwargs)
            return results
    def check_supported_species(self):
        for a in set(self.atoms.get_chemical_symbols()):
            if a not in self.supported_species:
                return False
        return True

    
    def mp_tests(self, job_n=0, n_calcs=10733,**kwargs):
        """Loads all structures with computed elastic constants from Materials Project and computes
        elastic constants for it if the model supports the species present
        """
        import pickle

        mp_dict = pickle.load(open("%s/%s/mp_elasticity_conventional_4-9-24.pkl" %(os.path.dirname(__file__), "data"), "rb"))
        for k, v in tqdm(list(mp_dict.items())[job_n*n_calcs:(job_n+1)*n_calcs]):
            atoms = load_atoms(k, v)
            self(atoms, **kwargs)

    def insert_mp_outputs(self, mp_id, property_name, gt, comp):
        db = TinyDB(self.db_name)
        db.insert({'mp-id':mp_id, property_name:{'computed':comp, 'ground_truth': gt}, 'timestamp': str(datetime.datetime.now())})
        
