#!/usr/bin/env python3
"""
Compute elasticity matrix for an arbitrary crystal structure defined by a
periodic cell and basis atom positions. Note that this can also be applied to
representative volume elements (RVEs) of non-crystlline materials with the RVE
taken to be periodic.

The code also computes the distance from the obtained elasticity tensor to
the nearest isotropic tensor using. This provides a measure of the anisotropy
of the crystal. Note that this calculation can fail if the elasticity tensor
is not positive definite.

The following methods for computing the elastic constants are supported:

(I) energy-condensed : Compute elastic constants from the hessian of the
condensed strain energy density, W_eff(eps) = min_d W(eps,d), where d are
the displacements of the internal atoms (aside from preventing rigid-body
translation) and eps is the strain.

(II) stress-condensed : Compute elastic constants from the jacobian of the
condensed stress, sig_eff(eps) = sig(eps,dmin), where dmin = arg min_d
W(eps,d). A faster, less accurate version of this approach
(stress-condensed-fast), is also provided.

(III) energy-full : Compute elastic constants from the hessian of the full
strain energy density, W(eps,d). This involves an algebraic manipulation
to account for the effect of atom relaxation; see eqn (27), in Tadmor et
al, Phys. Rev. B, 59:235-245, 1999.

All three methods give similar results with differences beyond the first or
second digit due to the numerical differentiation. The `energy-condensed` and
`energy-full` approaches have comparable accuracy, but full Hessian is *much*
slower. The `stress-condensed` approach is significantly faster than
`energy-condensed`, but is less accurate. The default apporach is
`energy-condensed`. The other methods are retained in the code for testing
purposes or, for very large systems, the `stress-condensed` method is an
option.

=== Theory ===

All three approaches involves numerical differentiation of energy or stress
expressions in terms of strains represented in Voigt notation. This is requires
the introduction of factors to account for the Voigt form as explained below.

The code assumes a Voigt notation with the following ordering:
1 = 11,  2 = 22,  3 = 33,  4 = 23,  5 = 13,  6 = 12  ---------------(1)

In this notation, the linear elastic stress-strain relation for infinitesimal
defomation is give by:

sig_m = C_mn eps_n -------------------------------------------------(2)

where

sig_1 = sig_11,  sig_2 = sig_22,  sig_3 = sig_33
sig_4 = sig_23,  sig_5 = sig_13,  sig_6 = sig_12

eps_1 = eps_11,  eps_2 = eps_22,  eps_3 = eps_33
eps_4 = gam_23,  eps_5 = gam_13,  eps_6 = gam_12 -------------------(3)

and gam_ij = 2 eps_ij.

In the above sig_ij and eps_ij are the components of the stress and strain
tensors in a Cartesian basis. The same basis is used to define the orientation
of the periodic cell and basis atom positions.

=== Energy-based calculation of elastic constants ===

The strain energy density of a linear elastic meterial is

W = 1/2 c_ijkl eps_ij eps_kl ---------------------------------------(4)

where c_ijkl are the components of the 4th-order elasticity tensor.
In Voigt notation strain energy density has the form:

W = 1/2 C_mn eps_m eps_n -------------------------------------------(5)

where C_mn are the components of elasticity matrix. The symmetries in the
summation of Eqn (4) (which includes 81 terms) are captured by the factors
of two in the 4, 5, 6 components of eps_* in Eqn (3).

Eqn (4) implies that the elasticity tensor components are given by

c_ijkl = d^2 W / d eps_ij d eps_kl ---------------------------------(6)

Substituting into Eqn (6), the definition of W in Eqn (5), a chain rule
must be applied:

c_ijkl = (d^2 W / d eps_m d eps_n )(d eps_m / d eps_ij)(d eps_n / d eps_kl)

                                                                    (7)

Transforming to Voigt notation and accounting for the factors of two
in the shear terms in Eqn (3), we have

C_mn = d^2 W / d eps_m d eps_n               for m,n <=3

C_mn = (1/2) d^2 W / d eps_m d eps_n         for m<=3, n>3  or
                                                 m>3, n<=3

C_mn = (1/4) d^2 W / d eps_m d eps_n         for m,n > 3

=== Stress-based calculation of elastic constants ===

An alternative aproach to computing the elastic constants is to use the
stress-strain relation in Eqn (2), which in full tensorial notation is

sig_ij = c_ijkl eps_kl ---------------------------------------------(8)

This relation implies that

c_ijkl = d sig_ij / d eps_kl ---------------------------------------(9)

Substituting in Eqn (2) for the stress and applying the chain rule,

c_ijkl = (d sig_m / d eps_n)(d eps_n / d eps_kl) ------------------(10)

where m is the Voigt index for ij. Transforming to Voigt notation and
accounting for the factors of two in the shear terms in Eqn (3), we have

C_mn = d sig_m / d eps_n                     for m,n <=3

C_mn = (1/2) d sig_m / d eps_n               for n > 3

Revisions:

2019/04/13 Ellad Tadmor added ability to do diamond)
2022/02/01 Chloe Zeller (lammps compatibility - specific usage case)
2022/02/22 Ellad Tadmor generalized to arbitrary crystal structures

"""

import numpy as np
from numpy.linalg import inv
from numpy.linalg import eig
from numpy.linalg import matrix_rank
from ase.optimize.precon import PreconLBFGS
from ase.units import GPa

from ase.constraints import ExpCellFilter, UnitCellFilter
import numdifftools as ndt
from numdifftools.step_generators import MaxStepGenerator
from scipy.optimize import minimize
import math

# Encoding of symmetry restrictions on elasticity tensors.
# The keys are the Voigt indices of non-independent components.
# The values are a pair of lists representing the linear combination
# of the unique compoonents that is used to determine the non-unique component
# specified in the key. The first list is the coefficients, the second
# is the indices. If a non-independent component is zero, this is indicated
# by a value of None. Any components not listed as a key are assumed to
# be independent. Only the upper triangle (i<j) is listed

CUBIC_EQN = {
    (1, 3): ([1], [(1, 2)]),
    (1, 4): None,
    (1, 5): None,
    (1, 6): None,
    (2, 2): ([1], [(1, 1)]),
    (2, 3): ([1], [(1, 2)]),
    (2, 4): None,
    (2, 5): None,
    (2, 6): None,
    (3, 3): ([1], [(1, 1)]),
    (3, 4): None,
    (3, 5): None,
    (3, 6): None,
    (4, 5): None,
    (4, 6): None,
    (5, 5): ([1], [(4, 4)]),
    (5, 6): None,
    (6, 6): ([1], [(4, 4)]),
}

HEXAGONAL_EQN = {
    (1, 4): None,
    (1, 5): None,
    (1, 6): None,
    (2, 2): ([1], [(1, 1)]),
    (2, 3): ([1], [(1, 3)]),
    (2, 4): None,
    (2, 5): None,
    (2, 6): None,
    (3, 4): None,
    (3, 5): None,
    (3, 6): None,
    (4, 5): None,
    (4, 6): None,
    (5, 5): ([1], [(4, 4)]),
    (5, 6): None,
    (6, 6): ([0.5, -0.5], [(1, 1), (1, 2)]),
}

TRIGONAL_EQN = {
    (1, 4): None,
    (1, 6): None,
    (2, 2): ([1], [(1, 1)]),
    (2, 3): ([1], [(1, 3)]),
    (2, 4): None,
    (2, 5): ([-1], [(1, 5)]),
    (2, 6): None,
    (3, 4): None,
    (3, 5): None,
    (3, 6): None,
    (4, 5): None,
    (4, 6): ([-1], [(1, 5)]),
    (5, 5): ([1], [(4, 4)]),
    (5, 6): None,
    (6, 6): ([0.5, -0.5], [(1, 1), (1, 2)]),
}

TETRAGONAL_EQN = {
    (1, 4): None,
    (1, 5): None,
    (1, 6): None,
    (2, 2): ([1], [(1, 1)]),
    (2, 3): ([1], [(1, 3)]),
    (2, 4): None,
    (2, 5): None,
    (2, 6): None,
    (3, 4): None,
    (3, 5): None,
    (3, 6): None,
    (4, 5): None,
    (4, 6): None,
    (5, 5): ([1], [(4, 4)]),
    (5, 6): None,
}

ORTHORHOMBIC_EQN = {
    (1, 4): None,
    (1, 5): None,
    (1, 6): None,
    (2, 4): None,
    (2, 5): None,
    (2, 6): None,
    (3, 4): None,
    (3, 5): None,
    (3, 6): None,
    (4, 5): None,
    (4, 6): None,
    (5, 6): None,
}

MONOCLINIC_EQN = {
    (1, 4): None,
    (1, 6): None,
    (2, 4): None,
    (2, 6): None,
    (3, 4): None,
    (3, 6): None,
    (4, 5): None,
    (5, 6): None,
}

TRICLINIC_EQN = {}

# error check typing in the above dicts
for eqn in (
    CUBIC_EQN,
    HEXAGONAL_EQN,
    TRIGONAL_EQN,
    TETRAGONAL_EQN,
    ORTHORHOMBIC_EQN,
    MONOCLINIC_EQN,
    TRICLINIC_EQN,
):
    assert sorted(list(set(eqn.keys()))) == sorted(list(eqn.keys()))  # only unique keys
    independent_components = []
    for dependent_component in eqn:
        if eqn[dependent_component] is not None:
            independent_components += eqn[dependent_component][1]
    # check that all components appearing in RHS of relations
    for independent_component in independent_components:
        assert not (independent_component in eqn)


def get_unique_components_and_reconstruct_matrix(elastic_constants, space_group_number):
    """
    From an elasticity matrix in Voigt order and a space group number, extract the
    elastic constants that should be unique (cij where first i is as low as possible, then j)
    Reconstruct the elasticity matrix based on the algebraic symmetry rules and check
    how much the original matrix violated the symmetry rules for both crystallography
    and material frame indifference

    Returns:
        * Names of unique elastic constants
        * List of unique elastic constants
        * Reconstructed matrix
    """
    assert 0 < space_group_number < 231

    if space_group_number < 3:
        eqn = TRICLINIC_EQN
    elif space_group_number < 16:
        eqn = MONOCLINIC_EQN
    elif space_group_number < 75:
        eqn = ORTHORHOMBIC_EQN
    elif space_group_number < 143:
        eqn = TETRAGONAL_EQN
    elif space_group_number < 168:
        eqn = TRIGONAL_EQN
    elif space_group_number < 195:
        eqn = HEXAGONAL_EQN
    else:
        eqn = CUBIC_EQN

    elastic_constants_names = []
    elastic_constants_values = []
    reconstructed_matrix = np.zeros((6, 6))
    # first, figure out which constants are unique and extract them
    for i in range(1, 7):
        for j in range(i, 7):
            if (i, j) not in eqn:
                elastic_constants_names.append("c" + str(i) + str(j))
                elastic_constants_values.append(elastic_constants[i - 1, j - 1])
                reconstructed_matrix[i - 1, j - 1] = elastic_constants[i - 1, j - 1]

    for dep_comp in eqn:
        if eqn[dep_comp] is None:
            continue
        else:
            dep_comp_zero_based = (dep_comp[0] - 1, dep_comp[1] - 1)
            for coeff, indep_comp in zip(*eqn[dep_comp]):
                indep_comp_zero_based = (indep_comp[0] - 1, indep_comp[1] - 1)
                reconstructed_matrix[dep_comp_zero_based] += (
                    coeff * reconstructed_matrix[indep_comp_zero_based]
                )

    reconstructed_matrix = (
        reconstructed_matrix
        + reconstructed_matrix.T
        - np.diag(reconstructed_matrix.diagonal())
    )

    # print ("\n########################################################################\n"
    #       "MAXIMUM DEVIATION FROM CRYSTAL SYMMETRY AND MATERIAL FRAME INDIFFERENCE: %f\n"%np.max(reconstructed_matrix-elastic_constants)+
    #       "########################################################################\n")

    return elastic_constants_names, elastic_constants_values, reconstructed_matrix


def minimize_PreconLBFGS(
    supercell,
    fmax=0.05,
    smax=0.05,
    steps=100000000,
    variable_cell_flag=True,
    logfile="-",
):
    """
    Use PreconLBFGS to Minimize cell energy with respect to cell shape and
    internal atom positions.

    PreconLBGS convergence behavior is as follows:
    - The solver returns True if it is able to converge within the optimizer
      iteration limits (which can be changed by the `steps` argument passed
      to `run`), otherwise it returns False.
    - The solver raises an exception in situations where the line search cannot
      improve the solution, typically due to an incompatibility between the
      potential's values for energy, forces, and/or stress.

    This routine attempts to minimizes the energy until the force and stress
    reduce below specified tolerances given a provided limit on the number of
    allowed steps. The code returns when convergence is achieved or no
    further progress can be made, either due to reaching the iteration step
    limit, or a stalled minimization due to line search failures.

    Parameters:
        supercell : ASE Atoms object
            Atomic configuration to be minimized.
        fmax : float
            Force convergence tolerance (the magnitude of the force on each
            atom must be less than this for convergence)
        smax : float
            Stress convergence tolerance (each component of the stress tensor
            must be less than this for converfence)
        steps : int
            Maximum number of iterations for the minimization
        variable_cell_flag : bool
            True to allow relaxation with respect to cell shape

    Returns:
        iteration_limits_reached : bool
            True if iteration limit was reached before the forces and stresses
            achieved the convergence target
        minimization_stalled : bool
            True if an exception was raised indicating that the minimization
            stalled due to a line search failure
    """
    opt = PreconLBFGS(supercell, variable_cell=variable_cell_flag, logfile=logfile)
    # @  supercell_wrapped = ExpCellFilter(supercell)
    # @  opt = PreconLBFGS(supercell_wrapped)
    try:
        converged = opt.run(fmax=fmax, smax=smax, steps=steps)
        iteration_limits_reached = not converged
        minimization_stalled = False
    except:
        minimization_stalled = True
        iteration_limits_reached = False
    return iteration_limits_reached, minimization_stalled

class ElasticConstants(object):
    """
    Compute the elastic constants of an arbitrary crystal
    through numerical differentiation
    """

    def __init__(self, supercell, condensed_minimization_method="bfgs"):
        """
        Class containing data and routines for elastic constant calculations

        Parameters:
            calc : ASE Atoms object
                ASE atoms object with calculator attached. This object
                contains the structure for which the elastic constant will be
                computed.
            condensed_minimization_method : string
                Used to define the minimization method to be used when relaxing
                out internal atom positions for a set strain. Two methods are
                supported:
                'bfgs'  : Use the native ASE Precon_LBFGS solver (default)
                'scipy' : Use the scipy default minimize method
        """
        self.supercell = supercell
        self.condensed_minimization_method = condensed_minimization_method
        self.natoms = supercell.get_global_number_of_atoms()
        # Store the original reference cell structure and volume, and atom
        # positions. These will overwritten if an initial cell relaxation is
        # requested.
        self.o_cell = self.supercell.get_cell()
        self.o_volume = self.supercell.get_volume()
        self.refpositions = self.supercell.get_positions()

    def voigt_to_matrix(self, voigt_vec):
        """
        Convert a voigt notation vector to a matrix

        Parameters:
            voigt_vec : float
                A numpy array containing the six strain components in
                Voigt ordering (xx, yy, zz, yz, xz, xy)

        Returns:
            matrix : float
               A 3x3 numpy array containg the strain tensor components
        """
        matrix = np.zeros((3, 3))
        matrix[0, 0] = voigt_vec[0]
        matrix[1, 1] = voigt_vec[1]
        matrix[2, 2] = voigt_vec[2]
        matrix[tuple([[1, 2], [2, 1]])] = voigt_vec[3]
        matrix[tuple([[0, 2], [2, 0]])] = voigt_vec[4]
        matrix[tuple([[0, 1], [1, 0]])] = voigt_vec[5]
        return matrix

    def get_energy_from_positions(self, pos):
        """
        Compute the supercell energy, given a set of positions for the
        internal atoms.

        Parameters:
            pos : float
                A numpy array containing the positions of all atoms in a
                flat concatenated form

        Returns:
            energy : float
               Potential energy of the supercell
        """
        self.supercell.set_positions(np.reshape(pos, (self.natoms, 3)))
        energy = self.supercell.get_potential_energy()
        return energy

    def get_gradient_from_positions(self, pos):
        """
        Compute the gradient of the supercell energy, given a set of positions
        for the internal atoms.

        Parameters:
            pos : float
                A numpy array containing the positions of all atoms in a
                flat concatenated form

        Returns:
            gradient : float
               The gradient of the potential energy of the force (the negative
               of the forces) in a flat concatenated form
        """
        self.supercell.set_positions(np.reshape(pos, (self.natoms, 3)))
        forces = self.supercell.get_forces()
        return -forces.flatten()

    def get_energy_from_strain_and_atom_displacements(self, strain_and_disps_vec):
        """
        Compute reference strain energy density for a given applied strain
        and internal atom positions for all but one atom constrained to
        prevent rigid-body translation.

        Parameters:
            strain_and_disps_vec : float
                A numpy array of length 6+(natoms-1)*3 containing the
                strain components in Voigt order and the free internal
                atom degrees of freedom

        Returns:
            energy density : float
                Potential energy of the supercell divided by its reference
                (unstrained) volume
        """
        if self.natoms < 2:
            return self.get_energy_from_strain(strain_and_disps_vec)

        # Set atom positions of the last N-1 atoms to reference positions
        # plus displacements, keeping first atom fixed
        defpositions = self.refpositions.copy()
        for i in range(self.natoms - 1):
            disp = strain_and_disps_vec[6 + i * 3 : 9 + i * 3]
            defpositions[i + 1] += disp
        self.supercell.set_positions(defpositions)

        # Apply strain to cell scaling the atom positions
        self.supercell.set_cell(self.o_cell, scale_atoms=False)
        strain_vec = strain_and_disps_vec[0:6]
        strain_mat = self.voigt_to_matrix(strain_vec)
        old_cell = self.o_cell
        new_cell = old_cell + np.dot(old_cell, strain_mat)
        self.supercell.set_cell(new_cell, scale_atoms=True)

        # Compute energy
        energy = self.supercell.get_potential_energy()

        return energy / self.o_volume

    def get_energy_from_strain(self, strain_vec):
        """
        Compute reference strain energy density for a given applied strain.

        Parameters:
            strain_vec : float
                A numpy array of length 6 containing the strain components in
                Voigt order

        Returns:
            energy density : float
                Potential energy of the supercell divided by its reference
                (unstrained) volume
        """
        self.supercell.set_cell(self.o_cell, scale_atoms=False)
        self.supercell.set_positions(self.refpositions)
        strain_mat = self.voigt_to_matrix(strain_vec)
        old_cell = self.o_cell
        new_cell = old_cell + np.dot(old_cell, strain_mat)
        self.supercell.set_cell(new_cell, scale_atoms=True)

        if self.natoms > 1:
            if self.condensed_minimization_method == "scipy":
                pos0 = self.supercell.get_positions().flatten()
                res = minimize(
                    self.get_energy_from_positions,
                    pos0,
                    jac=self.get_gradient_from_positions,
                )
                energy = res["fun"]
            elif self.condensed_minimization_method == "bfgs":
                fmax = 1e-5
                smax = 1e-5
                steps = 100000000
                iteration_limits_reached = True
                while iteration_limits_reached:
                    (
                        iteration_limits_reached,
                        minimization_stalled,
                    ) = minimize_PreconLBFGS(
                        self.supercell,
                        fmax=fmax,
                        smax=smax,
                        steps=steps,
                        variable_cell_flag=False,
                        logfile='-',
                    )
                energy = self.supercell.get_potential_energy()
            else:
                raise RuntimeError(
                    "Unknown condensed minimization method =",
                    self.condensed_minimization_method,
                )
        else:
            energy = self.supercell.get_potential_energy()

        return energy / self.o_volume

    def get_stress_from_strain(self, strain_vec):
        """
        Compute stress for a given applied strain.

        Parameters:
            strain_vec : float
                A numpy array of length 6 containing the strain components in
                Voigt order

        Returns:
            stress : float
                Cauchy stress of the supercell
        """
        energy = self.get_energy_from_strain(strain_vec)
        stress = self.supercell.get_stress()
        return stress

    def results(self, optimize=False, method="energy-condensed"):
        """
        Compute the elastic constants of supercell, relaxed if requested,
        using numerical differentiation

        Parameters:
            optimize : boolean
                If set to True, first minimize the energy of the supercell
                subject to strain and internal atom positions. Otherwise,
                compute the elastic constants for the as provided structure.
            method : string
                Select method for computing the elastic constants. The following
                methods are supported:
                'energy-condensed' : Compute elastic constants from the Hessian
                    of the condensed strain energy density (i.e. the enregy for
                    a given strain is relaxed with respect to internal atom
                    positions)
                'stress-condensed' : Compute elastic constants from the Jacobian
                    of the condensed stress (i.e. the stress for a given strain
                    where the energy is relaxed with respect to internal atom
                    positions)
                'stress-condensed-fast' : Same as 'stress-condensed' but
                    the numerical differentiation is performed with a single
                    step size (instead of a more accurate Richardson
                    extraploation). Useful for large systems where the cost of
                    the more accurate method is prohibitive.
                'energy-full' : Compute elastic constants from the full Hessian
                    relative to both strains and internal atom degrees of
                    freedom. This is followed by an algebraic manipulation to
                    account for the effect of atom relaxation on elastic
                    constants.
                In general, 'energy-condensed' is the preferred method.
                The 'stress-condensed' method is much faster, but generally less
                accurate. The 'energy-full' method has accuracy comparable
                to 'energy-condensed' but tends to be much slower due to the
                larger Hessian matrix that has to be computed.

        Returns:
            elastic_constants : float
                A 6x6 numpy array containing the elastic constants matrix
                in Voigt ordering. Units are the default returned by the
                calculator.
            estimated_error : float
                A 6x6 numpy array containing the standard error in the elastic
                constants returned by numdifftools. Same units as
                elastic_constants.
        """
        if optimize:
            fmax = 1e-5
            smax = 1e-5
            steps = 100000000
            iteration_limits_reached = True
            while iteration_limits_reached:
                iteration_limits_reached, minimization_stalled = minimize_PreconLBFGS(
                    self.supercell,
                    fmax=fmax,
                    smax=smax,
                    steps=steps,
                    variable_cell_flag=True,
                )
            # if minimization_stalled:
            # print("WARNING -- Minimization stalled before reaching requested tolerance: fmax={}, smax={}".format(fmax,smax))
            self.o_cell = self.supercell.get_cell()
            self.o_volume = self.supercell.get_volume()
            self.refpositions = self.supercell.get_positions()

        # Define different step generators for numerical differentiation to try
        if method == "stress-condensed-fast":
            sg = [1e-4]
        else:
            sg = [
                MaxStepGenerator(
                    base_step=1e-4,
                    num_steps=14,
                    use_exact_steps=True,
                    step_ratio=1.6,
                    offset=0,
                ),
                MaxStepGenerator(
                    base_step=1e-3,
                    num_steps=14,
                    use_exact_steps=True,
                    step_ratio=1.6,
                    offset=0,
                ),
                MaxStepGenerator(
                    base_step=1e-2,
                    num_steps=14,
                    use_exact_steps=True,
                    step_ratio=1.6,
                    offset=0,
                ),
                1e-4,
            ]

        success = True
        if method == "energy-full":
            # Compute stiffness matrix using full Hessian that includes
            # both strain components and internal atom positions
            func = self.get_energy_from_strain_and_atom_displacements
            for i in range(len(sg)):
                hess = ndt.Hessian(func, step=sg[i], full_output=True)
                try:
                    fullhessian, info = hess(
                        np.zeros(6 + (self.natoms - 1) * 3, dtype=float)
                    )
                    fullhessian_error_estimate = info.error_estimate
                    # Separate full Hessian into blocks
                    # (eps-eps, eps-disp, disp-disp)
                    hess_ee = fullhessian[0:6, 0:6]
                    hess_ed = fullhessian[0:6, 6:]
                    hess_dd = fullhessian[6:, 6:]
                    # Invert disp-disp block
                    hess_dd_inv = inv(hess_dd)
                    # Compute hessian accounting for basis atom relaxation. based
                    # on Eqn (27) in Tadmor et al, Phys. Rev. B, 59:235-245, 1999.
                    hessian = hess_ee - np.dot(
                        np.dot(hess_ed, hess_dd_inv), np.transpose(hess_ed)
                    )
                    # TODO: Figure out how to estimate error of Hessian based on
                    #      full Hessian errors in fullhessian_error_estimate
                    #      Now just taken errors for hess_ee which is incorrect.
                    hessian_error_estimate = info.error_estimate[0:6, 0:6]
                    break
                except Exception as e:
                    print (e)
                    success = False
                    return e,'_', success

        elif method == "energy-condensed":
            # Compute stiffness matrix using condensed energy that is
            # relaxed with respect to the internal atom positions
            func = self.get_energy_from_strain
            for i in range(len(sg)):
                hess = ndt.Hessian(func, step=sg[i], full_output=True)
                try:
                    hessian, info = hess(np.zeros(6, dtype=float))
                    hessian_error_estimate = info.error_estimate
                    break
                except Exception as e:
                    print (e) 
                    success = False
                    return e,'_', success

        elif method == "stress-condensed" or method == "stress-condensed-fast":
            # Compute stiffness matrix using condensed stress that is
            # relaxed with respect to the internal atom positions

            func = self.get_stress_from_strain
            for i in range(len(sg)):
                jac = ndt.Jacobian(func, step=sg[i], full_output=True)
                try:
                    hessian, info = jac(np.zeros(6, dtype=float))
                    hessian_error_estimate = info.error_estimate
                    break
                except Exception as e:
                    print (e)
                    success = False
                    return e,'_', success

        else:
            raise RuntimeError("Unknown method in results: method = " + method)

        #if not success:
        #    raise RuntimeError("Unable to compute Hessian for method = ", method)

        elastic_constants = np.zeros(shape=(6, 6))
        error_estimate = np.zeros(shape=(6, 6))
        if method == "energy-condensed" or method == "energy-full":
            # Hessian was computed as d^2 W / d eps_m d eps_n.
            # As noted in the doc string, a prefactor is required for some terms
            # to obtain the elastic constants
            for m in range(6):
                if m < 3:
                    factm = 1.0
                else:
                    factm = 0.5
                for n in range(6):
                    if n < 3:
                        fact = factm
                    else:
                        fact = factm * 0.5
                    elastic_constants[m, n] = fact * hessian[m, n]
                    error_estimate[m, n] = fact * hessian_error_estimate[m, n]
        elif method == "stress-condensed" or method == "stress-condensed-fast":
            # Hessian was computed as d sig_m / d eps_n.
            # As noted in the doc string, a prefactor is required for some terms
            # to obtain the elastic constants
            for m in range(6):
                for n in range(6):
                    if n < 3:
                        fact = 1.0
                    else:
                        fact = 0.5
                    elastic_constants[m, n] = fact * hessian[m, n]
                    error_estimate[m, n] = fact * hessian_error_estimate[m, n]
            # The elastic constants matrix should be symmetric, however due to
            # numerical precision issues in the stress components, in general,
            # d sig m / d eps_n will not equal d sig_n / d eps_m.
            # To address this, symmetrize the elastic constants matrix.
            for m in range(5):
                for n in range(m + 1, 6):
                    con = 0.5 * (elastic_constants[m, n] + elastic_constants[n, m])
                    err = 0.5 * (error_estimate[m, n] + error_estimate[n, m])
                    elastic_constants[m, n] = con
                    elastic_constants[n, m] = con
                    error_estimate[m, n] = err
                    error_estimate[n, m] = err

        return elastic_constants, error_estimate, True


def calc_bulk(elastic_constants):
    """
    Compute the bulk modulus given the elastic constants matrix in
    Voigt ordering.

    Parameters:
        elastic_constants : float
            A 6x6 numpy array containing the elastic constants in
            Voigy ordering. The material can have arbitrary anisotropy.

    Returns:
        bulk : float
            The bulk modulus, defined as the ratio between the hydrostatic
            stress (negative of the pressure p) in hydrostatic loading and
            the diltation e (trace of the strain tensor), i.e. B = -p/e
    """
    # Compute bulk modulus, based on exercise 6.14 in Tadmor, Miller, Elliott,
    # Continuum Mechanics and Thermodynamics, Cambridge University Press, 2012.
    if (
        matrix_rank(elastic_constants[3:, 0:3]) == 0
    ):  # block diagonal matrix, only invert first 3x3 block
        compliance = inv(elastic_constants[0:3, 0:3])
    else:
        compliance = inv(elastic_constants)
    bulk = 1 / np.sum(compliance[0:3, 0:3])
    return bulk


def map_to_Kelvin(C):
    """Compute the Kelvin form of the input matrix"""
    Ch = C.copy()
    Ch[0:3, 3:6] *= math.sqrt(2.0)
    Ch[3:6, 0:3] *= math.sqrt(2.0)
    Ch[3:6, 3:6] *= 2.0
    return Ch


def function_of_matrix(A, f):
    """Compute the function of a matrix"""
    ev, R = eig(A)
    Dtilde = np.diag([f(e) for e in ev])
    return np.matmul(np.matmul(R, Dtilde), np.transpose(R))


def find_nearest_isotropy(elastic_constants):
    """
    Compute the distance between the provided matrix of elastic constants
    in Voigt notation, to the nearest matrix of elastic constants for an
    isotropic material. Return this distance, and the isotropic bulk and
    shear modulus.

    Ref: Morin, L; Gilormini, P and Derrien, K,
         "Generalized Euclidean Distances for Elasticity Tensors",
         Journal of Elasticity, Vol 138, pp. 221-232 (2020).

    Parameters:
        elastic_constants : float
            A 6x6 numpy array containing the elastic constants in
            Voigt ordering. The material can have arbitrary anisotropy.

    Returns:
        d : float
            Distance to the nearest elastic constants.
            log Euclidean metric.
        kappa : float
            Isotropic bulk modulus
        mu : float
            Isotropic shear modulus
    """
    E0 = 1.0  # arbitrary scaling constant (result unaffected by it)

    I3 = np.diag([1.0, 1.0, 1.0])

    JJ = np.zeros(shape=(6, 6))
    KK = np.zeros(shape=(6, 6))
    v = {0: [0, 0], 1: [1, 1], 2: [2, 2], 3: [1, 2], 4: [0, 2], 5: [0, 1]}
    for ii in range(6):
        for jj in range(6):
            # i j k l = v[ii][0] v[ii][1] v[jj][0] v[jj][1]
            JJ[ii][jj] = (1.0 / 3.0) * (v[ii][0] == v[ii][1]) * (v[jj][0] == v[jj][1])
            KK[ii][jj] = (1.0 / 2.0) * (
                (v[ii][0] == v[jj][0]) * (v[ii][1] == v[jj][1])
                + (v[ii][0] == v[jj][1]) * (v[ii][1] == v[jj][0])
            ) - JJ[ii][jj]
    Chat = map_to_Kelvin(elastic_constants)
    JJhat = map_to_Kelvin(JJ)
    KKhat = map_to_Kelvin(KK)

    # Eqn (49) in Morin et al.
    fCoverE0 = function_of_matrix(Chat / E0, math.log)
    kappa = (E0 / 3.0) * math.exp(np.einsum("ij,ij", fCoverE0, JJhat))
    mu = (E0 / 2.0) * math.exp(0.2 * np.einsum("ij,ij", fCoverE0, KKhat))

    # Eqn (47) in Morin et al.
    dmat = (
        fCoverE0 - math.log(3.0 * kappa / E0) * JJhat - math.log(2.0 * mu / E0) * KKhat
    )
    d = math.sqrt(np.einsum("ij,ij", dmat, dmat))

    # Return results
    return d, kappa, mu
