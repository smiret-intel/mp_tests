from matsciml.preprocessing.atoms_to_graphs import AtomsToGraphs
import numpy as np
from matsciml.models.base import ForceRegressionTask
from ase.calculators.calculator import Calculator
from torch_geometric.data import Batch, Data
from matsciml.preprocessing.atoms_to_graphs import AtomsToGraphs
from pymatgen.io.ase import AseAtomsAdaptor
from matsciml.datasets.utils import point_cloud_featurization, concatenate_keys
from matsciml.datasets.base import PointCloudDataset
from math import pi

class MatSciMLCalculator(Calculator):


    implemented_properties = ["energy", "free_energy"]

    def __init__(self, model, transforms, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.model = model
        self.transforms = transforms

        self.energy = None
        self.forces = None

        if isinstance(self.model, ForceRegressionTask):
            self.implemented_properties.append('forces')
            self.implemented_properties.append('stress')
        
        self._parameters_changed = False

    def _parse_structure(self, data, return_dict):
        structure = data.get("structure", None)
        if structure is None:
            raise ValueError(
                "Structure not found in data - workflow needs a structure to use!",
            )
        coords = torch.from_numpy(structure.cart_coords).float()
        system_size = len(coords)
        return_dict["pos"] = coords
        chosen_nodes = PointCloudDataset.choose_dst_nodes(system_size, True)
        src_nodes, dst_nodes = chosen_nodes["src_nodes"], chosen_nodes["dst_nodes"]
        atom_numbers = torch.LongTensor(structure.atomic_numbers)
        # uses one-hot encoding featurization
        pc_features = point_cloud_featurization(
            atom_numbers[src_nodes],
            atom_numbers[dst_nodes],
            100,
        )
        # keep atomic numbers for graph featurization
        return_dict["atomic_numbers"] = atom_numbers
        return_dict["pc_features"] = pc_features
        return_dict["sizes"] = system_size
        return_dict.update(**chosen_nodes)
        return_dict["distance_matrix"] = torch.from_numpy(
            structure.distance_matrix,
        ).float()
        # grab lattice properties
        space_group = structure.get_space_group_info()[-1]
        return_dict["natoms"] = len(atom_numbers)
        lattice_params = torch.FloatTensor(
            structure.lattice.abc
            + tuple(a * (pi / 180.0) for a in structure.lattice.angles),
        )
        lattice_features = {
            "space_group": space_group,
            "lattice_params": lattice_params,
        }
        return_dict["lattice_features"] = lattice_features

    def _convert_atoms_to_graph(self, atoms):
        print (atoms.arrays)
        structure = AseAtomsAdaptor.get_structure(atoms)
        data = {"structure": structure}
        return_dict = {}
        self._parse_structure(data, return_dict)
        for i,transform in enumerate(self.transforms):
            return_dict = transform(return_dict)

        return concatenate_keys(
            [return_dict],
            pad_keys=["pc_features", "force", "stress"],
            unpacked_keys=["sizes", "src_nodes", "dst_nodes"],
        )

    def calculate(self,
        atoms=None,
        properties=["energy", "forces"],
        system_changes=["positions", "numbers", "cell", "pbc"],
    ):

        """
        Inherited method from the ase Calculator class that is called by
        get_property()

        Parameters
        ----------
        atoms : Atoms
            Atoms object whose properties are desired

        properties : list of str
            List of what needs to be calculated.  Can be any combination
            of 'energy' and 'forces'.

        system_changes : list of str
            List of what has changed since last calculation.  Can be any
            combination of these six: 'positions', 'numbers', 'cell',
            and 'pbc'.
        """
        pos = atoms.get_positions()
        if np.any(np.isnan(pos)):
            return
        else:
            super().calculate(atoms, properties, system_changes)

            graph_data = self._convert_atoms_to_graph(atoms)

            model_output = self.model(graph_data)
            if "force" in model_output:
                self.results = {"energy": model_output["energy"].detach().numpy(),
                        "forces": model_output["force"].detach().numpy()}
                self.results['stress'] = self._compute_virial_stress(self.results["forces"], atoms.get_positions(), atoms.get_volume())
    @staticmethod
    def _compute_virial_stress(forces, coords, volume):
        """Compute the virial stress in Voigt notation.

        Parameters
        ----------
        forces : 2D array
            Partial forces on all atoms (padding included)

        coords : 2D array
            Coordinates of all atoms (padding included)

        volume : float
            Volume of cell

        Returns
        -------
        stress : 1D array
            stress in Voigt order (xx, yy, zz, yz, xz, xy)
        """
        stress = np.zeros(6)
        stress[0] = -np.dot(forces[:, 0], coords[:, 0]) / volume
        stress[1] = -np.dot(forces[:, 1], coords[:, 1]) / volume
        stress[2] = -np.dot(forces[:, 2], coords[:, 2]) / volume
        stress[3] = -np.dot(forces[:, 1], coords[:, 2]) / volume
        stress[4] = -np.dot(forces[:, 0], coords[:, 2]) / volume
        stress[5] = -np.dot(forces[:, 0], coords[:, 1]) / volume

        return stress

if __name__ == "__main__":

    #class m:
    #    regress_forces=True

    from e3nn.o3 import Irreps
    from mace.modules.blocks import RealAgnosticInteractionBlock
    from torch import nn
    import torch
    from torch.nn import LayerNorm, SiLU

    from matsciml.datasets.utils import element_types
    from matsciml.models import (
        FAENet,
        GalaPotential,
        M3GNet,
        MEGNet,
        PLEGNNBackbone,
        TensorNet,
    )
    from matsciml.models.pyg.mace import MACEWrapper
    from matsciml.datasets.transforms import (
    DistancesTransform,
    FrameAveraging,
    GraphVariablesTransform,
    MGLDataTransform,
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
    COMShift,
    )

    available_models = {
    "egnn": {
        "encoder_class": PLEGNNBackbone,
        "encoder_kwargs": {
            "embed_in_dim": 1,
            "embed_hidden_dim": 32,
            "embed_out_dim": 128,
            "embed_depth": 5,
            "embed_feat_dims": [128, 128, 128],
            "embed_message_dims": [128, 128, 128],
            "embed_position_dims": [64, 64],
            "embed_edge_attributes_dim": 0,
            "embed_activation": "relu",
            "embed_residual": True,
            "embed_normalize": True,
            "embed_tanh": True,
            "embed_activate_last": False,
            "embed_k_linears": 1,
            "embed_use_attention": False,
            "embed_attention_norm": "sigmoid",
            "readout": "sum",
            "node_projection_depth": 3,
            "node_projection_hidden_dim": 128,
            "node_projection_activation": "relu",
            "prediction_out_dim": 1,
            "prediction_depth": 3,
            "prediction_hidden_dim": 128,
            "prediction_activation": "relu",
        },
        "output_kwargs": {
            "norm": LayerNorm(128),
            "hidden_dim": 128,
            "activation": SiLU,
            "lazy": False,
            "input_dim": 128,
        },
    },
    "faenet": {
        "encoder_class": FAENet,
        "encoder_kwargs": {
            "average_frame_embeddings": False,
            "pred_as_dict": False,
            "hidden_channels": 128,
            "out_dim": 128,
            "tag_hidden_channels": 0,
        },
        "output_kwargs": {"lazy": False, "input_dim": 128, "hidden_dim": 128},
    },
    "gala": {
        "encoder_class": GalaPotential,
        "encoder_kwargs": {
            "D_in": 100,
            "depth": 2,
            "hidden_dim": 64,
            "merge_fun": "concat",
            "join_fun": "concat",
            "invariant_mode": "full",
            "covariant_mode": "full",
            "include_normalized_products": True,
            "invar_value_normalization": "momentum",
            "eqvar_value_normalization": "momentum_layer",
            "value_normalization": "layer",
            "score_normalization": "layer",
            "block_normalization": "layer",
            "equivariant_attention": False,
            "tied_attention": True,
            "encoder_only": True,
        },
        "output_kwargs": {"lazy": False, "input_dim": 64, "hidden_dim": 64},
    },
    "m3gnet": {
        "encoder_class": M3GNet,
        "encoder_kwargs": {
            "element_types": element_types(),
            "return_all_layer_output": True,
        },
        "output_kwargs": {"lazy": False, "input_dim": 64, "hidden_dim": 64},
    },
    "megnet": {
        "encoder_class": MEGNet,
        "encoder_kwargs": {
            "edge_feat_dim": 2,
            "node_feat_dim": 128,
            "graph_feat_dim": 9,
            "num_blocks": 4,
            "hiddens": [256, 256, 128],
            "conv_hiddens": [128, 128, 128],
            "s2s_num_layers": 5,
            "s2s_num_iters": 4,
            "output_hiddens": [64, 64],
            "is_classification": False,
            "encoder_only": True,
        },
        "output_kwargs": {"lazy": False, "input_dim": 640, "hidden_dim": 640},
    },
    "tensornet": {
        "encoder_class": TensorNet,
        "encoder_kwargs": {
            "element_types": element_types(),
            "num_rbf": 8,
            "max_n": 2,
            "max_l": 2,
        },
        # element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
        # units: int = 64,
        # ntypes_state: int | None = None,
        # dim_state_embedding: int = 0,
        # dim_state_feats: int | None = None,
        # include_state: bool = False,
        # nblocks: int = 2,
        # num_rbf: int = 32,
        # max_n: int = 3,
        # max_l: int = 3,
        # rbf_type: Literal["Gaussian", "SphericalBessel"] = "Gaussian",
        # use_smooth: bool = False,
        # activation_type: Literal["swish", "tanh", "sigmoid", "softplus2", "softexp"] = "swish",
        # cutoff: float = 5.0,
        # equivariance_invariance_group: str = "O(3)",
        # dtype: torch.dtype = matgl.float_th,
        # width: float = 0.5,
        # readout_type: Literal["set2set", "weighted_atom", "reduce_atom"] = "weighted_atom",
        # task_type: Literal["classification", "regression"] = "regression",
        # niters_set2set: int = 3,
        # nlayers_set2set: int = 3,
        # field: Literal["node_feat", "edge_feat"] = "node_feat",
        # is_intensive: bool = True,
        # ntargets: int = 1,
        "output_kwargs": {"lazy": False, "input_dim": 64, "hidden_dim": 64},
    },
    "mace": {
        "encoder_class": MACEWrapper,
        "encoder_kwargs": {
            "r_max": 6.0,
            "num_bessel": 3,
            "num_polynomial_cutoff": 3,
            "max_ell": 2,
            "interaction_cls": RealAgnosticInteractionBlock,
            "interaction_cls_first": RealAgnosticInteractionBlock,
            "num_interactions": 2,
            "atom_embedding_dim": 64,
            "MLP_irreps": Irreps("256x0e"),
            "avg_num_neighbors": 10.0,
            "correlation": 1,
            "radial_type": "bessel",
            "gate": nn.Identity(),
        },
        "output_kwargs": {"lazy": False, "input_dim": 128, "hidden_dim": 128},
    },
    "generic": {
        "output_kwargs": {
            "norm": LayerNorm(128),
            "hidden_dim": 128,
            "activation": "SiLU",
            "lazy": False,
            "input_dim": 128,
        },
        "lr": 0.0001,
    },
    }
    model = ForceRegressionTask(**available_models['faenet'])
    model.load_state_dict(
        torch.load('/Users/efuemmel/Downloads/faenet_mat_traj_mar25_24.ckpt', map_location=torch.device("cpu")), strict=False
    )
    transforms = [
        PeriodicPropertiesTransform(cutoff_radius=6.5, adaptive_cutoff=True),
        PointCloudToGraphTransform(
            "pyg",
            cutoff_dist=20.0,
            node_keys=["pos", "atomic_numbers"],
        ),
        FrameAveraging(frame_averaging="3D", fa_method="stochastic")
    ]
    calc = MatSciMLCalculator(model, transforms)
    from mp_tests.utils import mp_species
    from mp_tests import Elasticity
    test = Elasticity(calc,supported_species=mp_species, db_name='mp.json' )
    test.mp_tests(it=10,ignore_relax=True, method= "stress-condensed-fast")
