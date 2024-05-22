from __future__ import annotations

from e3nn.o3 import Irreps
from mace.modules.blocks import RealAgnosticInteractionBlock, RealAgnosticResidualInteractionBlock
from torch import nn
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
import torch
from mace.modules import MACE, ScaleShiftMACE


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
            "hidden_dim": 128,
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
            "base_model": ScaleShiftMACE,
            "num_atom_embedding": 89,
            "r_max": 6.0,
            "num_bessel": 10,
            "num_polynomial_cutoff": 5.0,
            "max_ell": 3,
            "interaction_cls": RealAgnosticResidualInteractionBlock,
            "interaction_cls_first": RealAgnosticResidualInteractionBlock,
            "num_interactions": 2,
            "atom_embedding_dim": 128,
            "MLP_irreps": Irreps("16x0e"),
            "avg_num_neighbors": 10.0,
            "correlation": 3,
            "radial_type": "bessel",
            "gate": nn.Identity(),
            ###
            # fmt: off
            "atomic_energies": torch.Tensor([-3.6672, -1.3321, -3.4821, -4.7367, -7.7249, -8.4056, -7.3601, -7.2846, -4.8965, 0.0000, -2.7594, -2.8140, -4.8469, -7.6948, -6.9633, -4.6726, -2.8117, -0.0626, -2.6176, -5.3905, -7.8858, -10.2684, -8.6651, -9.2331, -8.3050, -7.0490, -5.5774, -5.1727, -3.2521, -1.2902, -3.5271, -4.7085, -3.9765, -3.8862, -2.5185, 6.7669, -2.5635, -4.9380, -10.1498, -11.8469, -12.1389, -8.7917, -8.7869, -7.7809, -6.8500, -4.8910, -2.0634, -0.6396, -2.7887, -3.8186, -3.5871, -2.8804, -1.6356, 9.8467, -2.7653, -4.9910, -8.9337, -8.7356, -8.0190, -8.2515, -7.5917, -8.1697, -13.5927, -18.5175, -7.6474, -8.1230, -7.6078, -6.8503, -7.8269, -3.5848, -7.4554, -12.7963, -14.1081, -9.3549, -11.3875, -9.6219, -7.3244, -5.3047, -2.3801, 0.2495, -2.3240, -3.7300, -3.4388, -5.0629, -11.0246, -12.2656, -13.8556, -14.9331, -15.2828])
            # fmt: on
        },
        "output_kwargs": {"lazy": False, "input_dim": 256, "hidden_dim": 256},
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
