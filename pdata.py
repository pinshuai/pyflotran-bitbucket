""" Class for pyflotran data """

"""
PyFLOTRAN v1.0.0 LA-CC-14-094 

Copyright (c) 2014, Los Alamos National Security, LLC.  
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
   following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

__author__ = "Satish Karra, Cory Kitay"
__version__ = "1.0.0"
__maintainer__ = "Satish Karra"
__email__ = "satkarra@lanl.gov"

from copy import copy
import subprocess
import matplotlib.pyplot as plt
import itertools as it
from matplotlib import rc

rc('text', usetex=True)

from ptool import *
from pdflt import *

dflt = pdflt()

WINDOWS = platform.system() == 'Windows'
if WINDOWS:
    copyStr = 'copy'
    delStr = 'del'
    slash = '\\'
else:
    copyStr = 'cp'
    delStr = 'rm'
    slash = '/'

# Multiple classes/key words - allowed strings
time_units_allowed = ['s', 'sec', 'm', 'min', 'h', 'hr', 'd', 'day', 'w', 'week', 'mo', 'month', 'y']
solver_names_allowed = ['transport', 'tran', 'flow']  # newton and linear
# simulation type - allowed strings
simulation_types_allowed = ['subsurface', 'surface_subsurface', 'hydroquake']
# mode - allowed strings
mode_names_allowed = ['richards', 'mphase', 'mph', 'flash2', 'th no_freezing', 'th freezing', 'immis']

# grid - allowed strings
grid_types_allowed = ['structured', 'structured_mimetic', 'unstructured', 'amr']
grid_symmetry_types_allowed = ['cartesian', 'cylindrical', 'spherical']  # cartesian is default in pflotran

# output - allowed strings
output_formats_allowed = ['TECPLOT BLOCK', 'TECPLOT POINT', 'HDF5', 'HDF5 MULTIPLE_FILES', 'MAD', 'VTK']

output_variables_allowed = ['liquid_pressure', 'liquid_saturation', 'liquid_density', 'liquid_mobility',
                            'liquid_energy', 'liquid_mole_fractions', 'gas_pressure', 'gas_saturation', 'gas_density',
                            'gas_mobility', 'gas_mole_fractions', 'air_pressure', 'capillary_pressure',
                            'thermodynamic_state', 'temperature', 'residual', 'porosity', 'mineral_porosity',
                            'permeability', 'mineral_porosity']

# saturation_function - allowed strings
saturation_function_types_allowed = ['VAN_GENUCHTEN', 'BROOKS_COREY', 'THOMEER_COREY', 'NMT_EXP', 'PRUESS_1']

permeability_function_types_allowed = ['VAN_GENUCHTEN', 'MUALEM', 'BURDINE', 'NMT_EXP', 'PRUESS_1']

# characteristic_curves - allowed strings - saturation & permeability functions
characteristic_curves_saturation_function_types_allowed = ['VAN_GENUCHTEN', 'BROOKS_COREY']
characteristic_curves_gas_permeability_function_types_allowed = ['MAULEM_VG_GAS', 'BURDINE_BC_GAS']
characteristic_curves_liquid_permeability_function_types_allowed = ['MAULEM', 'BURDINE', 'MUALEM_VG_LIQ']

# material_property, region, initial_condition, boundary_condition, 
# source_sink, stratigraphy_couplers - manual does not appear to document 
# all valid entries

# flow_conditions - allowed strings
flow_condition_type_names_allowed = ['PRESSURE', 'RATE', 'FLUX', 'TEMPERATURE', 'CONCENTRATION', 'SATURATION', 'WELL','ENTHALPY']
pressure_types_allowed = ['dirichlet', 'heterogeneous_dirichlet', 'hydrostatic', 'zero_gradient', 'conductance',
                          'seepage']
rate_types_allowed = ['mass_rate', 'volumetric_rate', 'scaled_volumetric_rate']
well_types_allowed = ['well']
flux_types_allowed = ['dirichlet', 'neumann', 'mass_rate', 'hydrostatic, conductance', 'zero_gradient',
                      'production_well', 'seepage', 'volumetric', 'volumetric_rate', 'equilibrium']
temperature_types_allowed = ['dirichlet', 'hydrostatic', 'zero_gradient', 'neumann']
concentration_types_allowed = ['dirichlet', 'hydrostatic', 'zero_gradient']
saturation_types_allowed = ['dirichlet']
enthalpy_types_allowed = ['dirichlet', 'hydrostatic', 'zero_gradient']

# transport_condition - allowed strings
transport_condition_types_allowed = ['dirichlet', 'dirichlet_zero_gradient', 'equilibrium', 'neumann', 'mole',
                                     'mole_rate', 'zero_gradient']

cards = ['co2_database', 'uniform_velocity', 'nonuniform_velocity', 'simulation', 'regression', 'checkpoint', 'restart',
         'dataset', 'chemistry', 'grid', 'timestepper', 'material_property', 'time', 'linear_solver', 'newton_solver',
         'output', 'fluid_property', 'saturation_function', 'characteristic_curves', 'region', 'observation',
         'flow_condition', 'transport_condition', 'initial_condition', 'boundary_condition', 'source_sink', 'strata',
         'constraint', 'hydroquake']

headers = ['co2 database path', 'uniform velocity', 'nonuniform velocity', 'simulation', 'regression', 'checkpoint',
           'restart', 'dataset', 'chemistry', 'grid', 'time stepping', 'material properties', 'time', 'linear solver',
           'newton solver', 'output', 'fluid properties', 'saturation functions', 'characteristic curves', 'regions',
           'observation', 'flow conditions', 'transport conditions', 'initial condition', 'boundary conditions',
           'source sink', 'stratigraphy couplers', 'constraints', 'hydroquake']

headers = dict(zip(cards, headers))

build_warnings = []


class puniform_velocity(Frozen):
    """
    Class for specifiying uniform velocity with transport. Optional with transport problem when not coupling with
    any flow mode.  If not specified, assumes diffusion transport only.

    :param value_list: List of variables of uniform_velocity. First 3 variables are vlx, vly, vlz in unit [m/s]. 4th
    variable specifies unit. e.g., [14.4e0, 0.e0, 0.e0, 'm/yr']
    :type value_list: [float,float,float,str]
    """

    def __init__(self, value_list=None):
        if value_list is None:
            value_list = []
        self.value_list = value_list
        self._freeze()


class pnonuniform_velocity(Frozen):
    """
    Class for specifiying nonuniform velocity with transport. Optional
    with transport problem when not coupling with any flow mode. If not specified, assumes diffusion transport only.

    :param filename: Filename of the non uniform velocities
    :type filename: str
    """

    def __init__(self, filename=''):
        self.filename = filename
        self._freeze()


class pmaterial(Frozen):
    """
    Class for defining a material property.
    Multiple material property objects can be created.

    :param id: Unique identifier of material property.
    :type id: int
    :param name: Name of material property. e.g., 'soil1'.
    :type name: str
    :param characteristic_curves: Unique identifier of material characteristic curves
    :type characteristic_curves: str
    :param porosity: Porosity of material.
    :type porosity: float
    :param tortuosity: Tortuosity of material.
    :type tortuosity: float
    :param density: Rock density of material in kg/m^3.
    :type density: float
    :param specific_heat: Specific heat of material in J/kg/K.
    :type specific_heat: float
    :param cond_dry: Thermal dry conductivity of material in W/m/K.
    :type cond_dry: float
    :param cond_wet: Thermal wet conductivity of material in W/m/K.
    :type cond_wet: float
    :param saturation: Saturation function of material property. e.g., 'sf2'
    :type saturation: str
    :param permeability: Permeability of material property. Input is a list of 3 floats. Uses diagonal permeability in
    unit order: k_xx [m^2], k_yy [m^2], k_zz [m^2]. e.g., [1.e-15,1.e-15,1.e-17].
    :type permeability: [float]*3
    :param longitudinal_dispersivity: Longitudinal dispersion coefficient 
    :type longitudinal_dispersivity: float
    :param transverse_dispersivity_h: Transverse dispersion coefficient horizontal
    :type transverse_dispersivity_h: float
    :param transverse_dispersivity_v: Transverse dispersion coefficient vertical
    :type transverse_dispersivity_v: float


    """

    # definitions are put on one line to work better with rst/latex/sphinx.
    def __init__(self, id=None, name='', characteristic_curves='', porosity=None, tortuosity=None, density=None,
                 specific_heat=None, cond_dry=None, cond_wet=None, saturation='', permeability=None,
                 permeability_power='', permeability_critical_porosity='', permeability_min_scale_factor='',
		 longitudinal_dispersivity='',transverse_dispersivity_h='',transverse_dispersivity_v =''):
        if permeability is None:
            permeability = []

        self.id = id
        self.name = name
        self.characteristic_curves = characteristic_curves
        self.porosity = porosity
        self.tortuosity = tortuosity
        self.density = density
        self.specific_heat = specific_heat
        self.cond_dry = cond_dry
        self.cond_wet = cond_wet
        self.saturation = saturation
        self.permeability = permeability
        self.permeability_power = permeability_power
        self.permeability_critical_porosity = permeability_critical_porosity
        self.permeability_min_scale_factor = permeability_min_scale_factor
	self.longitudinal_dispersivity = longitudinal_dispersivity
	self.transverse_dispersivity_h = transverse_dispersivity_h
	self.transverse_dispersivity_v = transverse_dispersivity_v
        self._freeze()


class ptime(Frozen):
    """
    Class for time. Used to specify final time of simulation,
    initial timestep size, maximum timestep size (throughout the
    simulation or a particular instant of time). Time values and
    units need to be specified. Acceptable time units are: (s, m, h, d, mo, y).

    :param tf: final tim. 1st variable is time value. 2nd variable specifies time unit. e.g., [0.25e0, 'y']
    :type tf: [float, str]
    :param dti: delta (change) time initial a.k.a. initial timestep size. 1st variable is time value. 2nd variable
    specifies time unit. e.g., [0.25e0, 'y']
    :type dti: [float, str]
    :param dtf: delta (change) time final a.k.a. maximum timestep size. 1st variable is time value. 2nd variable
    specifies time unit. e.g., [50.e0, 'y']
    :type dtf: [float, str]
    :param dtf_list: delta (change) time starting at a given time instant.  Input is a list that can have multiple lists
    appended to it. e.g., time.dtf_list.append([1.e2, 's', 5.e3, 's'])
    :type dtf_list: [ [float, str, float, str] ]
    :param steady_state: Run as steady state.
    :type steady_state: Bool
    """

    # definitions are put on one line to work better with rst/latex/sphinx.
    def __init__(self, tf=None, dti=None, dtf=None, steady_state=False, dtf_list=None):
        if tf is None:
            tf = []
        if dti is None:
            dti = []
        if dtf is None:
            dtf = []
        if dtf_list is None:
            dtf_list = []

        self.tf = tf  # Final Time, 2nd parameter is unit, same for all other
        # variables except dtf_i
        self.dti = dti  # Initial Timestep Size
        self.dtf = dtf  # Maximum Timestep Size
        self.dtf_list = dtf_list  # Maximum Timestep Size using keyword 'at'
        # Lists (manually) are assigned to lists
        self.steady_state = steady_state
        self._freeze()


class pgrid(Frozen):
    """
    Class for defining a grid. Used to define type, resolution and geometry of the gird

    :param type: Grid type. Valid entries include: 'structured', 'unstructured'.
    :type type: str
    :param lower_bounds: Lower/Minimum 3D boundaries coordinates in order of x_min, y_min, z_min. Input is a list of 3
    floats. e.g., [0.e0, 0.e0, 0.e0].
    :type lower_bounds: [float]*3
    :param upper_bounds: Upper/Maximum 3D boundaries coordinates in order of x_max, y_max, z_max. Input is a list of 3
    floats. e.g., [321.e0, 1.e0, 51.e0].
    :type lower_bounds: [float]*3
    :param origin: Coordinates of grid origin. Optional. Input is a list of 3 floats. Default: [0.e0, 0.e0, 0.e0].
    :type origin: [float]*3
    :param nxyz: Number of grid cells in x,y,z directions. Only works with type='structured'. Input is a list of 3
    floats. e.g., [107, 1, 51].
    :type nxyz: [float]*3
    :param dx: Specifies grid spacing of structured cartesian grid in the x-direction. e.g., [0.1, 0.2, 0.3, 0.4, 1, 1,
    1, 1].
    :type dx: [float]
    :param dy: Specifies grid spacing of structured cartesian grid in the y-direction.
    :type dy: [float]
    :param dz: Specifies grid spacing of structured cartesian grid in the z-direction
    :type dz: [float]
    :param gravity: Specifies gravity vector in m/s^2. Input is a list of 3 floats.
    :type gravity: [float]*3
    :param filename: Specify name of file containing grid information. Only works with type='unstructured'.
    :type filename: str
    """

    # definitions are put on one line to work better with rst/latex/sphinx.
    def __init__(self, type='structured', lower_bounds=None, upper_bounds=None,
                 origin=None, nxyz=None, dx=None, dy=None, dz=None, gravity=None, filename=''):
        if lower_bounds is None:
            lower_bounds = [0.0, 0.0, 0.0]
        if upper_bounds is None:
            upper_bounds = [1.0, 1.0, 1.0]
        if origin is None:
            origin = []
        if nxyz is None:
            nxyz = [10, 10, 10]
        if dx is None:
            dx = []
        if dy is None:
            dy = []
        if dz is None:
            dz = []
        if gravity is None:
            gravity = []

        self.type = type
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.origin = origin
        self.nxyz = nxyz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.gravity = gravity
        self.filename = filename
        self._freeze()


class psimulation(Frozen):
    """
    Class for specifying simulation type and simulation mode.

    :param simulation_type: Specify simulation type. Options include: 'surface','subsurface.
    :type simulation_type: str
    :param subsurface_flow: Specify the process model.
    :type subsurface_flow: str
    :param subsurface_transport: Specify the process model.
    :type subsurface_transport: str
    :param mode: Specify the mode for the subsurface flow model
    :type mode: str
    :param flowtran_coupling: Specify the type for the flow transport coupling
    :type mode: str
    """

    def __init__(self, simulation_type='subsurface', subsurface_flow='', subsurface_transport='', mode='richards',
                 flowtran_coupling=''):
        self.simulation_type = simulation_type
        self.subsurface_flow = subsurface_flow
        self.subsurface_transport = subsurface_transport
        self.flowtran_coupling = flowtran_coupling
        self.mode = mode
        self._freeze()


class pregression(Frozen):
    """
    Class for specifying regression details.

    :param cells: Specify cells for regression.
    :type cells: list of int
    :param cells_per_process: Specify the number cells per process.
    :type cells_per_process: int
    """

    def __init__(self, cells=None, cells_per_process=''):
        if cells is None:
            cells = []
        self.cells = cells
        self.cells_per_process = cells_per_process
        self._freeze()


class ptimestepper(Frozen):
    """
    Class for controling time stepping.
        
    :param ts_mode: FLOW or TRAN mode
    :type ts_mode: string
    :param ts_acceleration: Integer for time step acceleration ramp.
    :type ts_acceleration: int
    :param num_steps_after_cut: Number of time steps after a time step cut that the time step size is held constant.
    :type num_steps_after_cut: int
    :param max_steps: Maximum time step after which the simulation will be terminated.
    :type max_steps: int
    :param max_ts_cuts: Maximum number of consecutive time step cuts before the simulation is terminated.
    :type max_ts_cuts: int
    :param cfl_limiter: CFL number for transport.
    :type cfl_limiter: float
    :param initialize_to_steady_state: Boolean flag to initialize a simulation to steady state
    :type initialize_to_steady_state: bool - True or False
    :param uun_as_steady_state: Boolean flag to run a simulation to steady state
    :type run_as_steady_state: bool - True or False
    :param max_pressure_change: Maximum change in pressure for a time step. Default: 5.d4 Pa.
    :type max_pressure_change: float
    :param max_temperature_change: Maximum change in temperature for a time step. Default: 5 C.
    :type max_temperature_change: float
    :param max_concentration_change: Maximum change in pressure for a time step. Default: 1. mol/L.
    :type max_concentration_change: float
    :param max_saturation_change: Maximum change in saturation for a time step. Default: 0.5.
    :type max_saturation_change: float
    """

    # definitions are put on one line to work better with rst/latex/sphinx.
    def __init__(self, ts_mode='flow', ts_acceleration=None, num_steps_after_cut=None, max_steps=None,
                 max_ts_cuts=None, cfl_limiter=None, initialize_to_steady_state=False,
                 run_as_steady_state=False, max_pressure_change=None, max_temperature_change=None,
                 max_concentration_change=None, max_saturation_change=None):
        self.ts_mode = ts_mode
        self.ts_acceleration = ts_acceleration
        self.num_steps_after_cut = num_steps_after_cut
        self.max_steps = max_steps
        self.max_ts_cuts = max_ts_cuts
        self.cfl_limiter = cfl_limiter
        self.initialize_to_steady_state = initialize_to_steady_state
        self.run_as_steady_state = run_as_steady_state
        self.max_pressure_change = max_pressure_change
        self.max_temperature_change = max_temperature_change
        self.max_concentration_change = max_concentration_change
        self.max_saturation_change = max_saturation_change
        self._freeze()


class plsolver(Frozen):
    """
    Class for specifying linear solver. Multiple linear solver Frozens
    can be created one for flow and one for transport.

    :param name: Specify name of the physics for which the linear solver is
     being defined. Options include: 'tran', 'transport','flow'.
    :type name: str
    :param solver: Specify solver type: Options include: 'solver', 'krylov_type', 'krylov', 'ksp', 'ksp_type'
    :type solver: str
    :param preconditioner: Specify preconditioner type: Options include: 'ilu'
    :type solver: str
    """

    def __init__(self, name='', solver='', preconditioner=''):
        self.name = name  # TRAN, TRANSPORT / FLOW
        self.solver = solver  # Solver Type
        self.preconditioner = preconditioner
        self._freeze()


class pnsolver(Frozen):
    """
    Class for newton solver card. Multiple newton solver objects
    can be created, one for flow and one for transport.

    :param name: Specify newton solver to use: Options include: 'tran',
     'transport', 'tran_solver', 'flow_solver'. Default: 'flow_solver'
    :type name: str
    :param atol: Absolute tolerance.
    :type atol: float
    :param rtol: Relative tolerance w.r.t previous iteration.
    :type rtol: float
    :param stol: Relative tolerance of the update w.r.t previous iteration.
    :type stol: float
    :param dtol: Divergence tolerance.
    :type dtol: float
    :param itol: Tolerance compared to infinity norm.
    :type itol: float
    :param max_it: Cuts time step if the number of iterations exceed this value.
    :type max_it: int
    :param max_f: Maximum function evaluations (useful with linesearch methods.)
    :type max_f: int
    """

    # definitions are put on one line to work better with rst/latex/sphinx.
    def __init__(self, name='', atol=None, rtol=None, stol=None, dtol=None, itol=None, max_it=None, max_f=None):
        self.name = name  # Indicates Flow or Tran for Transport
        self.atol = atol
        self.rtol = rtol
        self.stol = stol
        self.dtol = dtol
        self.itol = itol
        self.max_it = max_it
        self.max_f = max_f
        self._freeze()


class poutput(Frozen):
    """
    Class for dumping simulation output.
    Acceptable time units (units of measurements) are: 's', 'min', 'h', 'd', 'w', 'mo', 'y'.

    :param time_list: List of time values. 1st variable specifies time unit to be used. Remaining variable(s) are floats
    :type time_list: [str, float*]
    :param print_column_ids: Flag to indicate whether to print column numbers in observation
     and mass balance output files. Default: False
    :type print_column_ids: bool - True or False
    :param screen_output: Turn the screen output on/off.
    :type screen_periodic: bool
    :param screen_periodic: Print to screen every <integer> time steps.
    :type screen_periodic: int
    :param periodic_time: 1st variable is value, 2nd variable is time unit.
    :type periodic_time: [float, str]
    :param periodic_timestep: 1st variable is value, 2nd variable is time unit.
    :type periodic_timestep: [float, str]
    :param periodic_observation_time: Output the results at observation points and mass balance
     output at specified output time. 1st variable is value, 2nd variable is time unit.
    :type periodic_observation_time: [float, str]
    :param periodic_observation_timestep: Outputs the results at observation points and mass
    balance output at specified time steps.
    :type periodic_observation_timestep: int
    :param format_list: Specify the file format for time snapshot of the simulation in
     time file type. Input is a list of strings. Multiple formats can be specified.
     File format options include: 'TECPLOT BLOCK' - TecPlot block format, 'TECPLOT POINT' -
     TecPlot point format (requires a single processor), 'HDF5' - produces single HDF5 file
     and xml for unstructured grids,  'HDF5 MULTIPLE_FILES' - produces a separate HDF5 file
     at each output time, 'VTK' - VTK format.
    :type format_list: [str]
    :param velocities: Turn velocity output on/off.
    :type velocities: bool - True or False
    :param velocity_at_center: Turn velocity output on/off.
    :type velocity_at_center: bool - True or False
    :param velocity_at_face: Turn velocity output at face on/off.
    :type velocity_at_face: bool - True or False
    :param mass_balance: Flag to indicate whether to output the mass balance of the system.
    :type mass_balance: bool - True or False
    :param variables_list: List of variables to be printed in the output file
    :type variables_list: [str]
    """

    # definitions are put on one line to work better with rst/latex/sphinx.
    def __init__(self, time_list=None, print_column_ids=False, screen_periodic=None, screen_output=True,
                 periodic_time=None, periodic_timestep=None, periodic_observation_time=None,
                 periodic_observation_timestep=None, format_list=None, permeability=False, porosity=False,
                 velocities=False, velocity_at_center=False, velocity_at_face=False, mass_balance=False,
                 variables_list=None):
        if time_list is None:
            time_list = []
        if periodic_time is None:
            periodic_time = []
        if periodic_observation_time is None:
            periodic_observation_time = []
        if format_list is None:
            format_list = []
        if variables_list is None:
            variables_list = []

        self.time_list = time_list
        self.print_column_ids = print_column_ids
        self.screen_output = screen_output  # Bool
        self.screen_periodic = screen_periodic  # int
        self.periodic_time = periodic_time  # [float, str]
        self.periodic_timestep = periodic_timestep  # [float, str]
        self.periodic_observation_time = periodic_observation_time  # [float, str]
        self.periodic_observation_timestep = periodic_observation_timestep  # int
        self.format_list = format_list
        self.permeability = permeability
        self.porosity = porosity
        self.velocities = velocities
        self.mass_balance = mass_balance
        self.variables_list = variables_list
        self.velocity_at_center = velocity_at_center
        self.velocity_at_face = velocity_at_face
        self._freeze()


class pfluid(Frozen):
    """
    Class for specifying fluid properties.

    :param diffusion_coefficient: Unit of measurement is [m^2/s]. Default: 1e-09
    :type diffusion_coefficient: float
    """

    def __init__(self, diffusion_coefficient=1.e-9):
        self.diffusion_coefficient = diffusion_coefficient
        self._freeze()


class psaturation(Frozen):
    """
    Class for specifying saturation functions.

    :param name: Saturation function name. e.g., 'sf2'
    :type name: str
    :param permeability_function_type: Options include: 'VAN_GENUCHTEN', 'MUALEM', 'BURDINE', 'NMT_EXP', 'PRUESS_1'.
    :type permeability_function_type: str
    :param saturation_function_type: Options include: 'VAN_GENUCHTEN', 'BROOKS_COREY', 'THOMEER_COREY', 'NMT_EXP',
    'PRUESS_1'.
    :type saturation_function_type: str
    :param residual_saturation: MODES: RICHARDS, TH, THC
    :type residual_saturation: float
    :param residual_saturation_liquid: MODES: MPHASE
    :type residual_saturation_liquid: float
    :param residual_saturation_gas: MODES: MPHASE
    :type residual_saturation_gas: float
    :param a_lambda: lambda
    :type a_lambda: float
    :param alpha: Pa^-1
    :type alpha: float
    :param max_capillary_pressure: Pa
    :type max_capillary_pressure: float
    :param betac:
    :type betac: float
    :param power:
    :type power: float
    """

    # definitions are put on one line to work better with rst/latex/sphinx.
    def __init__(self, name='', permeability_function_type=None, saturation_function_type=None,
                 residual_saturation=None, residual_saturation_liquid=None, residual_saturation_gas=None, a_lambda=None,
                 alpha=None, max_capillary_pressure=None, betac=None, power=None):
        self.name = name
        self.permeability_function_type = permeability_function_type
        self.saturation_function_type = saturation_function_type
        self.residual_saturation = residual_saturation  # float
        self.residual_saturation_liquid = residual_saturation_liquid  # float
        self.residual_saturation_gas = residual_saturation_gas  # float
        self.a_lambda = a_lambda
        self.alpha = alpha
        self.max_capillary_pressure = max_capillary_pressure
        self.betac = betac
        self.power = power
        self._freeze()


class ppoint(Frozen):
    """
    Class for a point.

    :param name: point name
    :type name: str
    :param coordinate: Coordinate of the point
    :type coordinate: [float]*3
    """

    def __init__(self, name='', coordinate=None):
        if coordinate is None:
            coordinate = [0.0, 0.0, 0.0]
        self.name = name.lower()
        self.coordinate = coordinate
        self._freeze()


class pcharacteristic_curves(Frozen):
    """
    Class for specifying characteristic curves. This card is used only in GENERAL mode; the SATURATION_FUNCTION card
    should be used in RICHARDS mode.

    :param name: Characteristic curve name. e.g., 'cc1'
    :param saturation_function_type: Options include: 'VAN_GENUCHTEN', 'BROOKS_COREY'.
    :type saturation_function_type: str
    :param sf_alpha: Pa^-1
    :type sf_alpha: float
    :param sf_m: Van Genutchen m
    :type sf_m: float
    :param sf_lambda: lambda: Brooks Corey lambda
    :type sf_lambda: float
    :param sf_liquid_residual_saturation: Residual saturation for liquid phase
    :type sf_liquid_residual_saturation: float
    :param sf_gas_residual_saturation: Residual saturation for gas phase
    :type sf_gas_residual_saturation: float
    :param max_capillary_pressure: Pa
    :type max_capillary_pressure: float
    :param smooth: Recommended for BROOKS_COREY
    :type smooth: No value, just a flag. Input 1 to turn flag on
    :param power: Placeholder. Currently not used
    :type power: float
    :param default: sets up dummy saturation and permeability functions for saturated single phase flow
    :type default: No value, just a flag. Input 1 to turn flag on
    :param liquid_permeability_function_type: Options include: 'MAULEM', 'BURDINE'.
    :type liquid_permeability_function_type: str
    :param lpf_m: Van Genutchen m
    :type lpf_m: float
    :param lpf_lambda: lambda: Brooks Corey lambda
    :type lpf_lambda: float
    :param lpf_liquid_residual_saturation: Residual saturation for liquid phase
    :type lpf_liquid_residual_saturation: float
    :param gas_permeability_function_type: Options include: 'MAULEM_VG_GAS', 'BURDINE_BC_GAS'.
    :type gas_permeability_function_type: str
    :param gpf_m: Van Genutchen m
    :type gpf_m: float
    :param gpf_lambda: lambda: Brooks Corey lambda
    :type gpf_lambda: float
    :param gpf_liquid_residual_saturation: Residual saturation for liquid phase
    :type gpf_liquid_residual_saturation: float
    :param gf_gas_residual_saturation: Residual saturation for gas phase
    :type gf_gas_residual_saturation: float
    """

    # definitions are put on one line to work better with rst/latex/sphinx.
    def __init__(self, name='', saturation_function_type=None, sf_alpha=None, sf_m=None, sf_lambda=None,
                 sf_liquid_residual_saturation=None, sf_gas_residual_saturation=None, max_capillary_pressure=None,
                 smooth='', power=None, default=None, liquid_permeability_function_type=None, lpf_m=None,
                 lpf_lambda=None, lpf_liquid_residual_saturation=None, gas_permeability_function_type=None, gpf_m=None,
                 gpf_lambda=None, gpf_liquid_residual_saturation=None, gpf_gas_residual_saturation=None):
        self.name = name
        self.saturation_function_type = saturation_function_type
        self.sf_alpha = sf_alpha
        self.sf_m = sf_m
        self.sf_lambda = sf_lambda
        self.sf_liquid_residual_saturation = sf_liquid_residual_saturation  # float
        self.sf_gas_residual_saturation = sf_gas_residual_saturation  # float
        self.max_capillary_pressure = max_capillary_pressure
        self.smooth = smooth
        self.power = power
        self.default = default
        self.liquid_permeability_function_type = liquid_permeability_function_type
        self.lpf_m = lpf_m
        self.lpf_lambda = lpf_lambda
        self.lpf_liquid_residual_saturation = lpf_liquid_residual_saturation
        self.gas_permeability_function_type = gas_permeability_function_type
        self.gpf_m = gpf_m
        self.gpf_lambda = gpf_lambda
        self.gpf_liquid_residual_saturation = gpf_liquid_residual_saturation
        self.gpf_gas_residual_saturation = gpf_gas_residual_saturation
        self._freeze()


class pregion(Frozen):
    """Class for specifying a PFLOTRAN region. Multiple region objects can be created.

    :param name: Region name.
    :type name: str
    :param coordinates_lower: Lower/minimum 3D coordinates for defining a volumetric,
     planar, or point region between two points in space in order of x1, y1, z1. e.g., [0.e0, 0.e0, 0.e0]
    :type coordinates_lower: [float]*3
    :param coordinates_upper: Upper/maximum 3D coordinates for defining a volumetric,
     planar, or point region between two points in space in order of x2, y2, z2. e.g., [321.e0, 1.e0,  51.e0]
    :type coordinates_upper: [float]*3
    :param face: Defines the face of the grid cell to which boundary conditions are connected.
     Options include: 'west', 'east', 'north', 'south', 'bottom', 'top'. (structured grids only).
    :type face: str
    """

    def __init__(self, name='', coordinates_lower=None, coordinates_upper=None, face=None):
        if coordinates_lower is None:
            coordinates_lower = [0.0, 0.0, 0.0]
        if coordinates_upper is None:
            coordinates_upper = [0.0, 0.0, 0.0]
        self.name = name.lower()
        self.coordinates_lower = coordinates_lower  # 3D coordinates
        self.coordinates_upper = coordinates_upper  # 3D coordinates
        self.face = face
        self.point_list = []
        self._freeze()


class pobservation(Frozen):
    """
    Class for specifying an observation region. Multiple observation objects may be added.
    Currently, only region is supported in PyFLOTRAN.

    :param region: Defines the name of the region to which the observation object is linked.
    :type region: str
    """

    def __init__(self, region=None):
        self.region = region
        self._freeze()


class pflow(Frozen):
    """
    Class for specifying a PFLOTRAN flow condition. There can be multiple flow condition objects.

    :param name: Name of the flow condition.
    :type name: str
    :param units_list: Not currently supported.
    :type units_list: [str]
    :param iphase:
    :type iphase: int
    :param sync_timestep_with_update: Flag that indicates whether to use sync_timestep_with_update. Default: False.
    :type sync_timestep_with_update: bool - True or False
    :param datum: Input is either a list of [d_dx, d_dy, d_dz] OR a 'file_name'
     with a list of [d_dx, d_dy, d_dz]. Choose one format type or the other, not both.
     If both are used, then only the file name will be written to the input deck.
    :type datum: Multiple [float, float, float] or str.
    :param datum_type: file or dataset
    :type datum_type: str
    :param varlist: Input is a list of pflow_variable objects. Sub-class of pflow.
     It is recommended to use dat.add(obj=pflow_variable) for easy appending.
     Use dat.add(index='pflow_variable.name' or dat.add(index=pflow_variable) to
     specify pflow object to add pflow_variable to. If no pflow object is specified,
     pflow_variable will be appended to the last pflow object appended to pdata.
     E.g., dat.add(variable, 'initial') if variable = pflow_variable and pflow.name='initial'.
    :type varlist: [pflow_variable]
    """

    def __init__(self, name='', units_list=None, iphase=None, sync_timestep_with_update=False, datum=None,
                 datum_type='', varlist=None, gradient=None):

        if datum is None:
            datum = []
        if varlist is None:
            varlist = []

        self.name = name.lower()  # Include initial, top, source
        self.units_list = units_list  # Specify type of units to display such as
        # time,length,rate,pressure,velocity, temperature,
        # concentration, and enthalpy.
        # May be used to determine each variable unit
        self.iphase = iphase  # Holds 1 int
        self.sync_timestep_with_update = sync_timestep_with_update  # Boolean
        self.datum = datum  # x, y, z, and a file name. [float,float,float,str]
        self.varlist = varlist
        self.datum_type = datum_type
        self.gradient = gradient
        self._freeze()


class pflow_variable(Frozen):
    """
    Sub-class of pflow for each kind of variable (includes type and value) such as
    pressure, temperature, etc. There can be multiple pflow_variable objects appended to a single pflow object.

    :param name: Indicates name of the flow variable. Options include: ['PRESSURE', 'RATE', 'FLUX', 'TEMPERATURE',
    'CONCENTRATION', 'SATURATION', 'ENTHALPY'].
    :type name: str
    :param type: Indicates type that is associated with name under keyword TYPE.
     Options for PRESSURE include: 'dirichlet', 'hydrostatic', 'zero_gradient', 'conductance',
     'seepage'. Options for RATE include: 'mass_rate', 'volumetric_rate', 'scaled_volumetric_rate'.
     Options for FLUX include: 'dirichlet', 'neumann, mass_rate', 'hydrostatic, conductance',
     'zero_gradient', 'production_well', 'seepage', 'volumetric', 'volumetric_rate', 'equilibrium'.
     Options for TEMPERATURE include: 'dirichlet', 'hydrostatic', 'zero_gradient'.
     Options for CONCENTRATION include: 'dirichlet', 'hydrostatic', 'zero_gradient'.
     Options for SATURATION include: 'dirichlet'. Options for ENTHALPY include: 'dirichlet', 'hydrostatic',
     'zero_gradient'
    :type type: str
    :param valuelist: Provide one or two values associated with a single Non-list alternative, do not use with list
    alternative. The 2nd float is optional and is needed for multiphase simulations.
    :type valuelist: [float, float]
    :param unit: Non-list alternative, do not use with list alternative. Specify unit of measurement.
    :type unit: str
    :param time_unit_type: List alternative, do not use with non-list alternative attributes/parameters.
    :type time_unit_type: str
    :param list: List alternative, do not use with non-list alternative attributes/parameters. Input is a list of
    pflow_variable_list objects. Sub-class of pflow_variable. The add function currently does not support
    adding pflow_variable_list to pflow_variable objects. Appending to can be done manually.
    e.g., variable.list.append(var_list) if var_list=pflow_variable_list.
    :type list: [pflow_variable_list]
    """

    def __init__(self, name='', type=None, valuelist=None, unit='', time_unit_type='', data_unit_type='', plist=None):
        if valuelist is None:
            valuelist = []
        if plist is None:
            plist = []
        self.name = name.lower()  # Pressure,temp., concen.,enthalpy...(String)
        self.type = type  # hydrostatic, zero_gradient, dirichlet ...(String)

        # The Following attributes are a stand alone single plist w/out lists
        # (e.g., Rate instead of Rate List)
        self.valuelist = valuelist  # Holds 2 floats - 2nd is optional
        self.unit = unit  # Possible to overide Parent class? - sorda?

        # Following attributes are used with lists (eg. Rate Lists instead of Rate)
        self.time_unit_type = time_unit_type  # e.g., 'y'
        self.data_unit_type = data_unit_type  # e.g., 'kg/s'
        self.list = plist  # Holds a plist of pflow_variable_lists objects
        self._freeze()


class pflow_variable_list(Frozen):
    """
    Sub-class of pflow_variable.
    Used for pflow_variables that are lists (as function of time) instead of a single value. Each of these list
    objects can hold multiple lines (from a Python input file) with each line holding one time_unit_value and a
    data_unit_value_list that can hold multiple values.

    :param time_unit_value:
    :type time_unit_value: float
    :param data_unit_value_list:
    :type data_unit_value_list: [float]
    """

    def __init__(self, time_unit_value=None, data_unit_value_list=None):
        if data_unit_value_list is None:
            data_unit_value_list = []
        self.time_unit_value = time_unit_value  # 1 float
        self.data_unit_value_list = data_unit_value_list  # 2 floats? (2nd optional?)
        self._freeze()


class pinitial_condition(Frozen):
    """
    Class for initial condition - a coupler between regions and initial flow and transport conditions.

    :param flow: Specify flow condition name
    :type flow: str
    :param transport: Specify transport condition name
    :type transport: str
    :param region: Specify region to apply the above specified flow and transport conditions as initial conditions.
    :type region: str
    """

    def __init__(self, flow=None, transport=None, region=None, name=''):
        self.name = name
        self.flow = flow  # Flow Condition (e.g., initial)
        self.transport = transport
        self.region = region  # Define region (e.g., west, east, well)
        self._freeze()


class pboundary_condition(Frozen):
    """
    Class for boundary conditions - performs coupling between a region and a flow/transport condition which are to be
    set as boundary conditions to that region. Multiple objects can be created.

    :param name: Name of boundary condition. (e.g., west, east)
    :type name: str
    :param flow: Defines the name of the flow condition to be linked to this boundary condition.
    :type flow: str
    :param transport: Defines the name of the transport condition to be linked to this boundary condition
    :type transport: str
    :param region: Defines the name of the region to which the conditions are linked
    :type region: str
    """

    def __init__(self, name='', flow='', transport='', region=''):
        self.name = name  # Name of boundary condition. (e.g., west, east)
        self.flow = flow  # Flow Condition (e.g., initial)
        self.transport = transport  # Transport Condition (e.g., river_chemistry)
        self.region = region  # Define region (e.g., west, east, well)
        self._freeze()


class psource_sink(Frozen):
    """
    Class for specifying source sink - this is also a condition coupler that links a region to the source sink condition

    :param flow: Name of the flow condition the source/sink term is applied to.
    :type flow: str
    :param region: Name of the region the source/sink term is applied to.
    :type region: str
    """

    def __init__(self, flow='', transport='', region='', name=''):
        self.name = name  # Name of source sink
        self.flow = flow  # Flow Condition (e.g., initial)
        self.transport = transport  # Flow Condition (e.g., initial)
        self.region = region  # Define region (e.g., west, east, well)
        self._freeze()


class pstrata(Frozen):
    """
    Class for specifying stratigraphy coupler. Multiple stratigraphy couplers can be created. Couples material
    properties with a region.

    :param region: Name of the material property to be associated with a region.
    :type region: str
    :param material: Name of region associated with a material property.
    :type material: str
    """

    def __init__(self, region=None, material=None):
        self.region = region
        self.material = material
        self._freeze()


class pcheckpoint(Frozen):
    """
    Class for specifying checkpoint options.

    :param frequency: Checkpoint dump frequency.
    :type frequency: int
    :param overwrite: Intended to be used for the PFLOTRAN keyword OVERWRITE_RESTART_FLOW_PARAMS.
    :type overwrite: bool - True or False
    """

    def __init__(self, frequency=None, overwrite=False):
        self.frequency = frequency  # int
        self.overwrite = overwrite  # Intended for OVERWRITE_RESTART_FLOW_PARAMS, incomplete, uncertain how to write it.
        self._freeze()


class prestart(Frozen):
    """
    Class for restarting a simulation.

    :param file_name: Specify file path and name for restart.chk file.
    :type file_name: str
    :param time_value: Specify time value.
    :type time_value: float
    :param time_unit: Specify unit of measurement to use for time. Options include: 's', 'sec','m', 'min', 'h', 'hr',
    'd', 'day', 'w', 'week', 'mo', 'month', 'y'.
    :type time_unit: str
    """

    def __init__(self, file_name='', time_value=None, time_unit=''):
        self.file_name = file_name  # restart.chk file name
        self.time_value = time_value  # float
        self.time_unit = time_unit  # unit of measurement to use for time - str
        self._freeze()


class pdataset(Frozen):
    """
    Class for incorporating data within a model.

    :param dataset_name: Opens the card block with the name of the data set in the string. I name is not given the NAME
    entry is required.
    :type dataset_name: str
    :param dataset_mapped_name: Adds the MAPPED flag to the DATASET and allows for the dataset to be named.
    :type dataset_name: str
    :param name: Name of the data set if not included with DATASET card. Note: this string overwrites the name specified
    with DATASET
    :type name: str
    :param file_name: Name of the file containing the data
    :type file_name: str
    :param hdf5_dataset_name: Name of the group within the hdf5 file where the data resides
    :type hdf5_dataset_name: str
    :param map_hdf5_dataset_name: Name of the group within the hdf5 file where the map information for the data resides
    :type map_hdf5_dataset_name: str
    :param max_buffer_size: size of internal buffer for storing transient data
    :type max_buffer_size: float
    :param realization_dependent: Add when doing stochastic multiple realizations 
    :type realization_dependent: bool 
   """

    def __init__(self, dataset_name='', dataset_mapped_name='', name='', file_name='', hdf5_dataset_name='',
                 map_hdf5_dataset_name='', max_buffer_size='',realization_dependent=''):
        self.dataset_name = dataset_name  # name of dataset
        self.dataset_mapped_name = dataset_mapped_name
        self.name = name  # name of dataset (overwrites dataset_name)
        self.file_name = file_name  # name of file containing the data
        self.hdf5_dataset_name = hdf5_dataset_name  # name of hdf5 group
        self.map_hdf5_dataset_name = map_hdf5_dataset_name
        self.max_buffer_size = max_buffer_size
        self.realization_dependent = realization_dependent
        self._freeze()


class pchemistry(Frozen):
    """
    Class for specifying chemistry.

    :param pspecies_list: List of primary species that fully describe the chemical composition of the fluid. The set of
    primary species must form an independent set of species in terms of which all homogeneous aqueous equilibrium
    reactions can be expressed.
    :type pspecies_list: [str]
    :param sec_species_list: List of aqueous species in equilibrium with primary species.
    :type sec_species_list: [str]
    :param gas_species_list: List of gas species.
    :type gas_species_list: [str]
    :param minerals_list: List of mineral names.
    :type minerals_list: [str]
    :param m_kinetics_list: List of pchemistry_m_kinetic objects. Holds kinetics information about a specified mineral
    name. Works with add function so that m_kinetics_list does not need to be remembered. e.g., dat.add(mineral_kinetic)
    :type m_kinetics_list: [pchemistry_m_kinetic]
    :param log_formulation:
    :type log_formulation: bool - True or False
    :param update_porosity:
    :type update_porosity: bool - True or False
    :param update_permeability:
    :type update_permeability: bool - True or False
    :param database:
    :type database: str
    :param activity_coefficients: Options include: 'LAG', 'NEWTON', 'TIMESTEP', 'NEWTON_ITERATION'.
    :type activity_coefficients: str
    :param molal:
    :type molal: bool - True or False
    :param output_list: To print secondary aqueous complex concentrations, either add the names of the secondary species
    of interest or the keyword 'SECONDARY_SPECIES' for all secondary species to the CHEMISTRY OUTPUT card.
    e.g., output_list = 'SECONDARY_SPECIES' or output_list = ['CO2(aq), 'PH']. By default, if ALL or MINERALS are
    listed under CHEMISTRY OUTPUT, the volume fractions and rates of kinetic minerals are printed. To print out
    the saturation indices of minerals listed under the MINERAL keyword, add the name of the
    mineral to the OUTPUT specification.
    :type output_list: [str]
    """

    def __init__(self, pspecies_list=None, sec_species_list=None, gas_species_list=None, minerals_list=None,
                 m_kinetics_list=None, log_formulation=False, database=None, activity_coefficients=None, molal=False,
                 output_list=None, update_permeability=False, update_porosity=False):
        if pspecies_list is None:
            pspecies_list = []
        if sec_species_list is None:
            sec_species_list = []
        if gas_species_list is None:
            gas_species_list = []
        if minerals_list is None:
            minerals_list = []
        if m_kinetics_list is None:
            m_kinetics_list = []
        if output_list is None:
            output_list = []
        self.pspecies_list = pspecies_list  # primary_species (eg. 'A(aq') - string
        self.sec_species_list = sec_species_list  # Secondary_species (E.g. 'OH-' - string
        self.gas_species_list = gas_species_list  # E.g. 'CO2(g)'
        self.minerals_list = minerals_list  # E.g. 'Calcite'
        self.m_kinetics_list = m_kinetics_list  # has pchemistry_m_kinetic assigned to it
        self.log_formulation = log_formulation
        self.update_permeability = update_permeability
        self.update_porosity = update_porosity
        self.database = database  # Database path (String)
        self.activity_coefficients = activity_coefficients
        self.molal = molal  # boolean
        self.output_list = output_list  # incl. molarity/all, species and mineral names - string
        self._freeze()


class pchemistry_m_kinetic(Frozen):
    """
    Sub-class of pchemistry. Mineral kinetics are assigned to m_kinetics_list in pchemistry. The add function can do
    this automatically. e.g., dat.add(mineral_kinetic).

    :param name: Mineral name.
    :type name: str
    :param rate_constant_list: Value, Unit of Measurement. e.g., rate_constant_list=[1.e-6, 'mol/m^2-sec']
    :type rate_constant_list: [float, str]
    """

    def __init__(self, name=None, rate_constant_list=None):
        if rate_constant_list is None:
            rate_constant_list = []
        self.name = name
        self.rate_constant_list = rate_constant_list
        self._freeze()


class ptransport(Frozen):
    """
    Class for specifying a transport condition. Multiple transport objects can be created.
    Specifies a transport condition based on various user defined constraints with minerals, gases, pH, charge balance,
    free ion, and total concentrations.

    :param name: Transport condition name.
    :type name: str
    :param type: Options include: 'dirichlet', 'dirichlet_zero_gradient', 'equilibrium', 'neumann', 'mole', 'mole_rate',
    'zero_gradient'.
    :type type: str
    :param constraint_list_value: List of constraint values. The position of each value in the list correlates with the
    position of each type in constraint_list_type.
    :type constraint_list_value: [float]
    :param constraint_list_type: List of constraint types. The position of each value in the list correlates with
    the position of each value in constraint_list_value. E.g., 'initial_constraint', 'inlet_constraint'.
    :type constraint_list_type: [str]
    """

    def __init__(self, name='', tran_type='', constraint_list_value=None, constraint_list_type=None):
        if constraint_list_value is None:
            constraint_list_value = []
        if constraint_list_type is None:
            constraint_list_type = []
        self.name = name  # e.g., initial, west, east
        self.type = tran_type  # e.g., dirichlet, zero_gradient
        self.constraint_list_value = constraint_list_value
        self.constraint_list_type = constraint_list_type
        self._freeze()


class pconstraint(Frozen):
    """
    Class for specifying a transport constraint.  Multiple constraint objects can be created.

    :param name: Constraint name.
    :type name: str
    :param concentration_list: List of pconstraint_concentration objects.
    :type concentration_list: [pconstraint_concentration]. Works with add function so that concentration_list does
    not need to be remembered. e.g., dat.add(concentration). Used for key word CONC or CONCENTRATIONS
    :param mineral_list: List of pconstraint_mineral objects. Currently does not work with add function. Used for
    keyword MNRL OR MINERALS.
    :type mineral_list: [pconstraint_mineral]
    """

    def __init__(self, name='', concentration_list=None, mineral_list=None):
        if concentration_list is None:
            concentration_list = []
        if mineral_list is None:
            mineral_list = []
        self.name = name.lower()
        self.concentration_list = concentration_list  # Composed of pconstraint_concentration objects
        self.mineral_list = mineral_list  # list of minerals
        self._freeze()


class pconstraint_concentration(Frozen):
    """
    Concentration unit, Sub-class for constraint. There can be multiple pconstraint_concentration objects appended to a
    single pconstraint object. Works with add function so that concentration_list in pconstraint does not need to be
    remembered. e.g., dat.add(concentration) instead of dat.constraint.concentration_list.append(concentration).

    :param pspecies: Primary species name for concentration.
    :type pspecies: str
    :param value: Concentration value.
    :type value: float
    :param constraint: Constraint name for concentration. Options include: 'F', 'FREE', 'T', 'TOTAL', 'TOTAL_SORB', 'P',
    'pH', 'L', 'LOG', 'M', 'MINERAL', 'MNRL', 'G', 'GAS', 'SC', 'CONSTRAINT_SUPERCRIT_CO2
    :type constraint: str
    :param element: Name of mineral or gas.
    :type element: str
    """

    def __init__(self, pspecies='', value=None, constraint='', element=''):
        self.pspecies = pspecies  # Primary Species Name (H+, O2(aq), etc.)
        self.value = value
        self.constraint = constraint  # (F, T, TOTAL_SORB, SC, etc.)
        self.element = element  # mineral or gas
        self._freeze()


class pconstraint_mineral(Frozen):
    """
    Class for mineral in a constraint with vol. fraction and surface area. There can be multiple
    pconstraint_concentration objects appended to a single pconstraint object. Currently does not work with add
    function. pconstraint_mineral can be manually appended to minerals_list in a pconstraint object.
    e.g., 'constraint.mineral_list.append(mineral)'.

    :param name: Mineral name.
    :type name: str
    :param volume_fraction: Volume fraction. [--]
    :type volume_fraction: float
    :param surface_area: Surface area. [m^-1]
    :type surface_area: float
    """

    def __init__(self, name='', volume_fraction=None, surface_area=None):
        self.name = name
        self.volume_fraction = volume_fraction
        self.surface_area = surface_area
        self._freeze()


class pdata(object):
    """
    Class for pflotran data file. Use 'from pdata import*' to access pdata library
    """

    # definitions are put on one line to work better with rst/latex/sphinx.
    def __init__(self, filename='', work_dir=''):
        # Note that objects need to be instantiated when hard-coded when it's set to
        # None here.
        self.co2_database = ''
        self.uniform_velocity = puniform_velocity()
        self.nonuniform_velocity = pnonuniform_velocity()
        self.overwrite_restart_flow_params = False
        self.regression = pregression()
        self.simulation = psimulation()
        self.checkpoint = pcheckpoint()
        self.restart = prestart()
        self.datasetlist = []
        self.chemistry = None
        self.grid = pgrid()
        self.timestepper = None
        self.proplist = []
        self.time = ptime()
        self.lsolverlist = []  # Possible to have 1 or 2 lsolver lists. FLOW/TRAN
        self.nsolverlist = []  # Possible to have 1 or 2 nsolver lists. FLOW/TRAN
        self.output = poutput()
        self.fluid = pfluid()
        self.saturationlist = []
        self.regionlist = []  # There are multiple regions
        self.charlist = []
        self.regionlist = []  # There are multiple regions
        self.observation_list = []
        self.flowlist = []
        self.transportlist = []
        self.initial_condition_list = []
        self.boundary_condition_list = []
        self.source_sink_list = []
        self.strata_list = []
        self.constraint_list = []
        self.filename = filename
        self.hydroquake = pquake()

        # run object
        self._path = ppath(parent=self)
        self._running = False  # boolean indicating whether a simulation is in progress
        self.work_dir = work_dir
        if self.work_dir:
            try:
                os.makedirs(self.work_dir)
            except:
                pass

        # OPTIONS
        temp_path = ppath()
        temp_path.filename = filename
        if temp_path.filename:
            if temp_path.absolute_to_file != os.getcwd():
                self.work_dir = temp_path.absolute_to_file
            self._path.filename = filename
            self.read(filename)
        else:
            return

    def run(self, input='', input_prefix='', num_procs=1, exe=pdflt().pflotran_path,silent=False,num_realizations='', num_groups=''):
        """
        Run a pflotran simulation for a given input file with specified number of processors.

        :param input: Name of input file. Uses default -pflotranin flag
        :type input: str
        :param input_prefix: Name of input file prefix. Uses the -input_prefix flag.
        :type input_prefix: str
        :param exe: Path to PFLOTRAN executable.
        :type exe: str
        :param num_procs: Number of processors
        :type num_procs: int
        :param num_realizations: Number of realizations 
        :type num_realizations: int
        :param num_groups: Number of groups 
        :type num_groups: int
        :param silent: Hide screen output
        :type silent: bool
        """

        num_realizations = 1
        num_groups = 1

        # set up and check path to executable
        exe_path = ppath()
        exe_path.filename = exe

        if not os.path.isfile(exe_path.full_path):  # if can't find the executable, halt
            raise PyFLOTRAN_ERROR('Default location is' + exe + '. No executable at location ' + exe)

        # option to write input file to new name
        if input:
            self._path.filename = input
        else:
            self._path.filename = self._path.filename[:-3] + '_new.in'

        if input_prefix:
            self._path.filename = input_prefix
        # ASSEMBLE FILES IN CORRECT DIRECTORIES
        if self.work_dir:
            wd = self.work_dir + os.sep
        else:
            wd = os.getcwd() + os.sep
        # print wd # testing?
        # print self._path.filename # testing?
        return_flag = self.write(wd + self._path.filename)  # ALWAYS write input file

        if return_flag:
            raise PyFLOTRAN_ERROR('Writing files')

        # RUN SIMULATION
        cwd = os.getcwd()
        if self.work_dir:
            os.chdir(self.work_dir)
        if input and input_prefix:
            raise PyFLOTRAN_ERROR('Cannot specify both input and input_prefix')

        def run_popen(cmd):
            process = subprocess.Popen(cmd.split(' '), shell=False, stdout=subprocess.PIPE, stderr=sys.stderr)
            while True:
                out = process.stdout.read(1)
                if ('ERROR' or 'WARNING') in out:
                    raise PyFLOTRAN_ERROR(out)

                if out == '' and process.poll() is not None:
                    break
                if out != '' and silent == False:
                    sys.stdout.write(out)
                    sys.stdout.flush()

        if num_procs == 1:
            arg = exe_path.full_path + ' -pflotranin ' + self._path.filename
            run_popen(arg)
        else:
            if num_realizations > 1:
                arg = 'mpirun -np ' + str(num_procs) + ' ' +  exe_path.full_path + ' -pflotranin ' + self._path.filename + ' -stochastic -num_realizations ' + str(num_realizations) + ' -num_groups ' + str(num_groups)  
            else:
                arg = 'mpirun -np ' + str(num_procs) + ' ' +  exe_path.full_path + ' -pflotranin ' + self._path.filename
            run_popen(arg)

        if input_prefix:
            if num_procs == 1:
                arg = exe_path.full_path + ' -input_prefix ' + self._path.filename
                run_popen(arg)
            else:
                arg = 'mpirun -np ' + str(num_procs) + ' ' + exe_path.full_path + ' -input_prefix ' + \
                      self._path.filename
                run_popen(arg)

        # After executing simulation, go back to the parent directory
        if self.work_dir:
            os.chdir(cwd)

    def __repr__(self):
        return self.filename  # print to screen when called

    def plot_data_from_tec(self, direction='X', variable_list=None, tec_filenames=None, legend_list=None,
                           plot_filename='', fontsize=10, x_label='', y_label_list=None, x_type='linear',
                           y_type='linear', x_range=(), y_range=(), x_factor=1.0, y_factor=1.0):

        if variable_list is None:
            variable_list = []
        if tec_filenames is None:
            tec_filenames = []
        if legend_list is None:
            legend_list = []
        if y_label_list is None:
            y_label_list = []

        for var, y_label in zip(variable_list, y_label_list):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_xscale(x_type)
            ax.set_yscale(y_type)
            if x_range:
                ax.set_xlim(x_range)
            if y_range:
                ax.set_ylim(y_range)
            lns = []
            for FILE in tec_filenames:
                variable = []
                with open(FILE, 'r') as f:
                    f.readline()
                    title = f.readline()
                    title = title.split(',')
                    for i in title:
                        variable.append(i.strip('"'))
                    data = np.genfromtxt(FILE, skip_header=3)
                    data = data.T.tolist()
                    var_values_dict = dict(zip(variable, data))
                    found = False
                    for key in var_values_dict.keys():
                        if direction.upper() in key:
                            xval = [val * x_factor for val in var_values_dict[key]]
                        if var in key:
                            dat = [val * y_factor for val in var_values_dict[key]]
                            found = True
                    if not found:
                        print 'Variable ' + var + ' not found in ' + FILE
                    try:
                        ln, = ax.plot(xval, dat)
                        lns.append(ln)
                    except UnboundLocalError:
                        pass
            ax.legend(lns, legend_list, ncol=1, fancybox=True, shadow=False, prop={'size': str(fontsize)}, loc='best')
            if '.pdf' in plot_filename:
                plot_filename = plot_filename.replace(".pdf", "")
            if ' ' in var:
                var = var.replace(" ", "_")
            if found: print 'Plotting variable [' + var + '] in [' + direction + '] direction'
            fig.savefig(plot_filename + '_' + var + '.pdf')

        return 0

    def plot_observation(self, variable_list=None, observation_list=None, observation_filenames=None, plot_filename='',
                         legend_list=None, fontsize=10, x_label='', y_label='', x_type='linear', y_type='linear',
                         x_range=(), y_range=(), x_factor=1.0, y_factor=1.0):
        """
        Plot time-series data from observation files at a given set of observation points.

        :param variable_list: List of the variables to be plotted
        :type variable_list: [str]
        :param observation_list: List of observation names to be plotted
        :type observation_list: [str]
        :param observation_filenames: List of observation filenames that are to be used for extracting data
        :type observation_filenames: [str]
        :param plot_filename: Name of the file to which the plot is saved
        :type plot_filename: str
        :param legend_list: List of legend
        :type legend_list: [str]
        :param fontsize: size of the legend font
        :type fontsize: float
        :param x_label: label on the x-axis
        :type x_label: str
        :param y_label: label on the y-axis
        :type y_label: str
        :param x_type: type of plot in the x-direction, e.g., 'log', 'linear', 'symlog'
        :type x_type: str
        :param y_type: type of plot in the y-direction, e.g., 'log', 'linear', 'symlog'
        :type y_type: str
        :param x_range: limits on the x-axis range, e.g., (0,100)
        :type x_range: (float,float)
        :param y_range: limits on the y-axis range, e.g., (0,100)
        :type y_range: (float,float)
        """

        if variable_list is None:
            variable_list = []
        if observation_list is None:
            observation_list = []
        if observation_filenames is None:
            observation_filenames = []
        if legend_list is None:
            legend_list = []
        combined_dict = {}
        for FILE in observation_filenames:
            variable = []
            f = open(FILE, 'r')
            title = f.readline()
            title = title.split(',')
            for i in title:
                variable.append(i.strip('"'))
            data = np.genfromtxt(FILE, skip_header=1)
            data = data.T.tolist()
            var_values_dict = dict(zip(variable, data))
            combined_dict.update(var_values_dict)

        for key in combined_dict.keys():
            if 'Time' in key:
                time = combined_dict[key]

        combined_var_obs_list = [variable_list, observation_list]
        combined_var_obs_list = list(it.product(*combined_var_obs_list))

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xscale(x_type)
        ax.set_yscale(y_type)
        if x_range:
            ax.set_xlim(x_range)
        if y_range:
            ax.set_ylim(y_range)
        lns = []
        for item in combined_var_obs_list:
            for key in combined_dict.keys():
                if item[0] in key and item[1] in key:
                    time_new = [t * x_factor for t in time]
                    var_new = [v * y_factor for v in combined_dict[key]]
                    ln, = ax.plot(time_new, var_new)
                    lns.append(ln)
        ax.legend(lns, legend_list, ncol=1, fancybox=True, shadow=False, prop={'size': str(fontsize)}, loc='best')
        fig.savefig(plot_filename)

        return 0

    def read(self, filename=''):
        """
        Read a given PFLOTRAN input file. This method is useful for reading an existing a PFLOTRAN input deck and all
        the corresponding PyFLOTRAN objects and data structures are autmatically created.

        :param filename: Name of input file.
        :type filename: str
        """
        if not os.path.isfile(filename):
            raise IOError(filename + ' not found...')
        self.filename = filename  # assign filename attribute
        read_fn = dict(zip(cards,
                           [self._read_co2_database,
                            self._read_uniform_velocity,
                            self._read_nonuniform_velocity,
                            self._read_simulation,
                            self._read_regression,
                            self._read_checkpoint,
                            self._read_restart,
                            self._read_dataset,
                            self._read_chemistry,
                            self._read_grid,
                            self._read_timestepper,
                            self._read_prop,
                            self._read_time,
                            self._read_lsolver,
                            self._read_nsolver,
                            self._read_output,
                            self._read_fluid,
                            self._read_saturation,
                            self._read_characteristic_curves,
                            self._read_region,
                            self._read_observation,
                            self._read_flow,
                            self._read_transport,
                            self._read_initial_condition,
                            self._read_boundary_condition,
                            self._read_source_sink,
                            self._read_strata,
                            self._read_constraint],
                           ))  # associate each card name with a read function, defined further below

        skip_readline = False
        p_line = ''  # Memorizes the most recent line read in.

        def get_next_line(skip_readline=skip_readline, line=p_line):
            """
            Used by read function to avoid skipping a line in cases where a particular read function might read an
            extra line.
            """

            if skip_readline:
                skip_readline = False
                return line
            else:
                line = infile.readline()
                return line

        with open(self.filename, 'r') as infile:
            keep_reading = True
            while keep_reading:
                p_line = get_next_line()
                if not p_line:
                    keep_reading = False
                if len(p_line.strip()) == 0:
                    continue
                card = p_line.split()[0].lower()  # make card lower case
                if card == 'overwrite_restart_flow_params':
                    self.overwrite_restart_flow_params = True
                if card == 'skip':
                    keep_reading_1 = True
                    while keep_reading_1:
                        line1 = get_next_line()
                        if not line1:
                            keep_reading_1 = False
                        if len(line1.strip()) == 0:
                            continue
                        card1 = line1.split()[0].lower()
                        if card1 == 'noskip':
                            keep_reading_1 = False

                if card in cards:  # check if a valid card name
                    if card in ['co2_database', 'checkpoint', 'restart', 'dataset', 'material_property', 'simulation',
                                'regression', 'grid', 'timestepper', 'linear_solver', 'newton_solver',
                                'saturation_function', 'region', 'flow_condition', 'boundary_condition', 'source_sink',
                                'initial_condition', 'transport_condition', 'constraint', 'uniform_velocity',
                                'nonuniform_velocity']:
                        read_fn[card](infile, p_line)
                    else:
                        read_fn[card](infile)

    def write(self, filename=''):
        """
        Write pdata object to PFLOTRAN input file. Does not execute the input file - only writes a corresponding
        PFLOTRAN input file.

        :param filename: Name of PFLOTRAN input file.
        :type filename: str
        """

        if filename:
            self.filename = filename
        outfile = open(self.filename, 'w')

        # Presumes simulation.simulation_type is required
        if self.simulation.simulation_type:
            self._write_simulation(outfile)
        else:
            raise PyFLOTRAN_ERROR('simulation is required, it is currently reading as empty')

        if self.simulation.subsurface_flow or self.simulation.subsurface_transport:
            self._write_subsurface_simulation_begin(outfile)

        if self.regression.cells or self.regression.cells_per_process:
            self._write_regression(outfile)

        if self.uniform_velocity.value_list:
            self._write_uniform_velocity(outfile)

        if self.nonuniform_velocity.filename:
            self._write_nonuniform_velocity(outfile)

        if self.co2_database:
            self._write_co2_database(outfile)

        if self.overwrite_restart_flow_params:
            self._write_overwrite_restart(outfile)

        if self.checkpoint.frequency:
            self._write_checkpoint(outfile)
        # else: print 'info: checkpoint not detected\n'

        if self.restart.file_name:
            self._write_restart(outfile)
        # else: print 'info: restart not detected\n'

        if self.datasetlist:
            self._write_dataset(outfile)
        # else: print 'info: dataset name not detected\n'

        if self.chemistry:
            self._write_chemistry(outfile)
        # else: print 'info: chemistry not detected\n'

        if self.grid:
            self._write_grid(outfile)
        else:
            raise PyFLOTRAN_ERROR('grid is required, it is currently reading as empty!')

        if self.timestepper:
            self._write_timestepper(outfile)
        # else: print 'info: timestepper not detected\n'

        if self.time:
            self._write_time(outfile)
        else:
            raise PyFLOTRAN_ERROR('time is required, it is currently reading as empty!')

        if self.proplist:
            self._write_prop(outfile)
        else:
            raise PyFLOTRAN_ERROR('proplist is required, it is currently reading as empty!')

        if self.lsolverlist:
            self._write_lsolver(outfile)
        # else: print 'info: lsolverlist (linear solver list) not detected\n'

        if self.nsolverlist:
            self._write_nsolver(outfile)
        # else: print 'info: nsolverlist (newton solver list) not detected\n'

        if self.output:
            self._write_output(outfile)
        else:
            raise PyFLOTRAN_ERROR('output is required, it is currently reading as empty!')

        if self.fluid:
            self._write_fluid(outfile)
        else:
            raise PyFLOTRAN_ERROR('fluid is required, it is currently reading as empty!')

        if self.saturationlist:
            self._write_saturation(outfile)

        if self.charlist:
            self._write_characteristic_curves(outfile)

        # if (not self.charlist and not self.saturationlist  self.simulation.subsurface_flow=''):
        # raise PyFLOTRAN_ERROR('either saturation or characteristic curves need to be defined!')

        if self.regionlist:
            self._write_region(outfile)
        else:
            raise PyFLOTRAN_ERROR('regionlist is required, it is currently reading as empty!')

        if self.observation_list:
            self._write_observation(outfile)

        if self.flowlist:
            self._write_flow(outfile)

        if self.transportlist:
            self._write_transport(outfile)

        if self.initial_condition_list:
            self._write_initial_condition(outfile)
        else:
            raise PyFLOTRAN_ERROR('initial_condition_list is required, it is currently reading as empty!')

        if self.boundary_condition_list:
            self._write_boundary_condition(outfile)

        if self.source_sink_list:
            self._write_source_sink(outfile)

        if not (self.boundary_condition_list or self.source_sink_list):
            raise PyFLOTRAN_ERROR('source_sink_list or boundary_condition_list is required, it is currently reading' +
                                  'as empty!')

        if self.strata_list:
            self._write_strata(outfile)
        else:
            raise PyFLOTRAN_ERROR('(stratigraphy_coupler) strata is required, it is currently reading as empty!')

        if self.constraint_list:
            self._write_constraint(outfile)

        if self.simulation.subsurface_flow or self.simulation.subsurface_transport:
            self._write_subsurface_simulation_end(outfile)

        if self.simulation.simulation_type == 'hydroquake':
            self._write_hydroquake(outfile)

        outfile.close()

    def add(self, obj, index='', overwrite=False):  # Adds a new object to the file
        """
        Attach an object associated w/ a list (e.g., pregion) that belongs to a pdata object.

        :param obj: Object to be added to the data file.
        :type obj: object(e.g., pregion)
        :param index: (Optional) Used to find an object that is using a string as an index in a dictionary. Intended for the super class object. (E.g. Index represents flow.name if instance is pflow_variable.) Default if not specified is to use the last super-class object added to pdata.
        :type index: String
        :param overwrite: Flag to overwrite an object if it already exists in a pdata object.
        :type overwrite: bool
        """

        add_checklist = [pmaterial, pdataset, psaturation, pcharacteristic_curves, pchemistry_m_kinetic, plsolver,
                         pnsolver, pregion, pobservation, pflow, pflow_variable, pinitial_condition,
                         pboundary_condition, psource_sink, pstrata, ptransport, pconstraint, pconstraint_concentration]

        # Check if obj first is an object that belongs to add_checklist
        checklist_bool = [isinstance(obj, item) for item in add_checklist]
        if True not in checklist_bool:
            raise PyFLOTRAN_ERROR('pdata.add used incorrectly! Cannot use pdata.add with one of the specified object.')

        # Always make index lower case if it is being used as a string
        if isinstance(index, str):
            index = index.lower()
        if isinstance(obj, pmaterial):
            self._add_prop(obj, overwrite)
        if isinstance(obj, pdataset):
            self._add_dataset(obj, overwrite)
        if isinstance(obj, psaturation):
            self._add_saturation(obj, overwrite)
        if isinstance(obj, pcharacteristic_curves):
            self._add_characteristic_curves(obj, overwrite)
        if isinstance(obj, pchemistry_m_kinetic):
            self._add_chemistry_m_kinetic(obj, overwrite)
        if isinstance(obj, plsolver):
            self._add_lsolver(obj, overwrite)
        if isinstance(obj, pnsolver):
            self._add_nsolver(obj, overwrite)
        if isinstance(obj, pregion):
            self._add_region(obj, overwrite)
        if isinstance(obj, pobservation):
            self._add_observation(obj, overwrite)
        if isinstance(obj, pflow):
            self._add_flow(obj, overwrite)
        if isinstance(obj, pflow_variable):
            self._add_flow_variable(obj, index, overwrite)
        if isinstance(obj, pinitial_condition):
            self._add_initial_condition(obj, overwrite)
        if isinstance(obj, pboundary_condition):
            self._add_boundary_condition(obj, overwrite)
        if isinstance(obj, psource_sink):
            self._add_source_sink(obj, overwrite)
        if isinstance(obj, pstrata):
            self._add_strata(obj, overwrite)
        if isinstance(obj, ptransport):
            self._add_transport(obj, overwrite)
        if isinstance(obj, pconstraint):
            self._add_constraint(obj, overwrite)
        if isinstance(obj, pconstraint_concentration):
            self._add_constraint_concentration(obj, index, overwrite)

    def delete(self, obj, super_obj=None):  # Deletes an object from the file
        """
        Delete an object that is assigned to a list of objects belong to a pdata object, e.g., pregion.

        :param obj: Object to be deleted from the data file. Can be a list of objects.
        :type obj: Object (e.g., pregion), list
        """

        if isinstance(obj, pmaterial):
            self._delete_prop(obj)
        elif isinstance(obj, list):
            for obji in copy(obj):  # obji = object index
                if isinstance(obji, pmaterial):
                    self._delete_prop(obji)

        if isinstance(obj, pcharacteristic_curves):
            self._delete_char(obj)
        elif isinstance(obj, list):
            for obji in copy(obj):  # obji = object index
                if isinstance(obji, pcharacteristic_curves):
                    self._delete_char(obji)

        if isinstance(obj, pchemistry_m_kinetic):
            self._delete_pchemistry_m_kinetic(obj)
        elif isinstance(obj, list):
            for obji in copy(obj):
                if isinstance(obji, pchemistry_m_kinetic):
                    self._delete_pchemistry_m_kinetic(obji)

        if isinstance(obj, plsolver):
            self._delete_lsolver(obj)
        elif isinstance(obj, list):
            for obji in copy(obj):
                if isinstance(obji, plsolver):
                    self._delete_lsolver(obji)

        if isinstance(obj, pnsolver):
            self._delete_nsolver(obj)
        elif isinstance(obj, list):
            for obji in copy(obj):
                if isinstance(obji, pnsolver):
                    self._delete_nsolver(obji)

        if isinstance(obj, pobservation):
            self._delete_observation(obj)
        elif isinstance(obj, list):
            for obji in copy(obj):
                if isinstance(obji, pobservation):
                    self._delete_observation(obji)

        if isinstance(obj, pregion):
            self._delete_region(obj)
        elif isinstance(obj, list):
            for obji in copy(obj):
                if isinstance(obji, pregion):
                    self._delete_region(obji)

        if isinstance(obj, pflow):
            self._delete_flow(obj)
        elif isinstance(obj, list):
            for obji in copy(obj):
                if isinstance(obji, pflow):
                    self._delete_flow(obji)

        if isinstance(obj, pflow_variable):  # Flow object needs to be specified
            self._delete_flow_variable(obj, super_obj)
        elif isinstance(obj, list):  # Condition not tested
            for obji in copy(obj):
                if isinstance(obji, pflow_variable):
                    self._delete_flow_variable(obji)

        if isinstance(obj, pinitial_condition):
            self._delete_initial_condition(obj)
        elif isinstance(obj, list):
            for obji in copy(obj):
                if isinstance(obji, pinitial_condition):
                    self._delete_initial_condition(obji)

        if isinstance(obj, pboundary_condition):
            self._delete_boundary_condition(obj)
        elif isinstance(obj, list):
            for obji in copy(obj):
                if isinstance(obji, pboundary_condition):
                    self._delete_boundary_condition(obji)

        if isinstance(obj, psource_sink):
            self._delete_source_sink(obj)
        elif isinstance(obj, list):
            for obji in copy(obj):
                if isinstance(obji, psource_sink):
                    self._delete_source_sink(obji)

        if isinstance(obj, pstrata):
            self._delete_strata(obj)
        elif isinstance(obj, list):
            for obji in copy(obj):
                if isinstance(obji, pstrata):
                    self._delete_strata(obji)

        if isinstance(obj, ptransport):
            self._delete_transport(obj)
        elif isinstance(obj, list):
            for obji in copy(obj):
                if isinstance(obji, ptransport):
                    self._delete_transport(obji)

        if isinstance(obj, pconstraint):
            self._delete_constraint(obj)
        elif isinstance(obj, list):
            for obji in copy(obj):
                if isinstance(obji, pconstraint):
                    self._delete_constraint(obji)

        if isinstance(obj, pconstraint_concentration):  # Constraint object needs to be specified
            self._delete_constraint_concentration(obj, super_obj)
        elif isinstance(obj, list):  # Condition not tested
            for obji in copy(obj):
                if isinstance(obji, pconstraint_concentration):
                    self._delete_constraint_concentration(obji)

    @staticmethod
    def splitter(a_line):
        return a_line.split()[-1]

    def _read_uniform_velocity(self, infile, line):
        np_value_list = []
        tstring = line.split()[1:]  # Convert to list, ignore 1st word

        i = 0  # index/count
        while i < len(tstring):
            try:
                np_value_list.append(floatD(tstring[i]))
            except ValueError:
                np_value_list.append(tstring[i])
            i += 1

        new_uniform_velocity = puniform_velocity(np_value_list)
        self.uniform_velocity = new_uniform_velocity

    def _write_uniform_velocity(self, outfile):
        self._header(outfile, headers['uniform_velocity'])
        outfile.write('UNIFORM_VELOCITY ')
        for v in self.uniform_velocity.value_list:  # value in value_list
            outfile.write(strD(v) + ' ')
        outfile.write('\n\n')

    def _read_nonuniform_velocity(self, infile, line):
        filename = ''
        tstring = line.split()
        filename = tstring[1]

        new_nonuniform_velocity = pnonuniform_velocity(filename)
        self.nonuniform_velocity = new_nonuniform_velocity

    def _write_nonuniform_velocity(self, outfile):
        self._header(outfile, headers['nonuniform_velocity'])
        outfile.write('NONUNIFORM_VELOCITY ')
        outfile.write(str(self.nonuniform_velocity.filename))
        outfile.write('\n\n')

    def _read_simulation(self, infile, line):
        simulation = psimulation()
        keep_reading = True
        key_bank = []
        while keep_reading:  # Read through all cards
            line = infile.readline()  # get next line
            key = line.strip().split()[0].lower()  # take first key word
            if key == 'simulation_type':
                simulation.simulation_type = self.splitter(line)
            elif key == 'subsurface_flow':
                simulation.subsurface_flow = self.splitter(line)
                keep_reading_1 = True
                while keep_reading_1:
                    line = infile.readline()
                    key1 = line.strip().split()[0].lower()
                    if key1 == 'mode':
                        simulation.mode = self.splitter(line).lower()
                    elif key1 in ['/', 'end']:
                        keep_reading_1 = False
                    else:
                        raise PyFLOTRAN_ERROR('mode is missing!')
                key_bank.append(key)
            elif key == 'subsurface_transport':
                simulation.subsurface_transport = self.splitter(line)
                keep_reading_2 = True
                while keep_reading_2:
                    line1 = infile.readline()
                    key1 = line1.strip().split()[0].lower()
                    if key1 == 'global_implicit':
                        simulation.flowtran_coupling = key1
                    elif key1 in ['/', 'end']:
                        keep_reading_2 = False
                # else:
                # raise PyFLOTRAN_ERROR('flow tran coupling type missing!')
                key_bank.append(key)
            elif key in ['/', 'end']:
                keep_reading = False
        if not ('subsurface_flow' in key_bank) and ('subsurface_transport' in key_bank):
            simulation.subsurface_flow = ''
            simulation.mode = ''
        self.simulation = simulation

    def _write_simulation(self, outfile):
        self._header(outfile, headers['simulation'])
        simulation = self.simulation
        # Write out simulation header
        outfile.write('SIMULATION' + '\n')
        if simulation.simulation_type.lower() in simulation_types_allowed:
            outfile.write('  SIMULATION_TYPE ' + simulation.simulation_type.upper() + '\n')
        else:
            print '       valid simulation.simulation_type:', simulation_types_allowed, '\n'
            raise PyFLOTRAN_ERROR('simulation.simulation_type: \'' + simulation.simulation_type + '\' is invalid!')

        if simulation.subsurface_flow and simulation.subsurface_transport:
            outfile.write('  PROCESS_MODELS' + '\n')
            outfile.write('    SUBSURFACE_FLOW ' + simulation.subsurface_flow + '\n')
            if simulation.mode in mode_names_allowed:
                outfile.write('      MODE ' + simulation.mode + '\n')
            else:
                print '       valid simulation.mode:', mode_names_allowed, '\n'
                raise PyFLOTRAN_ERROR('simulation.mode: \'' + simulation.mode + '\' is invalid!')

            outfile.write('    / ' + '\n')
            outfile.write('    SUBSURFACE_TRANSPORT ' + simulation.subsurface_transport + '\n')
            if simulation.flowtran_coupling:
                outfile.write('      ' + simulation.flowtran_coupling.upper() + '\n')
            outfile.write('    / ' + '\n')
            outfile.write('  / ' + '\n')
            outfile.write('END' + '\n\n')
        elif simulation.subsurface_flow:
            outfile.write('  PROCESS_MODELS' + '\n')
            outfile.write('    SUBSURFACE_FLOW ' + simulation.subsurface_flow + '\n')
            if simulation.mode in mode_names_allowed:
                outfile.write('      MODE ' + simulation.mode + '\n')
            else:
                print('simulation.mode: \'' + simulation.mode + '\' is invalid!')
                print '       valid simulation.mode:', mode_names_allowed, '\n'
            outfile.write('    / ' + '\n')
            outfile.write('  / ' + '\n')
            outfile.write('END' + '\n\n')
        elif simulation.subsurface_transport:
            outfile.write('  PROCESS_MODELS' + '\n')
            outfile.write('    SUBSURFACE_TRANSPORT ' + simulation.subsurface_transport + '\n')
            outfile.write('    / ' + '\n')
            outfile.write('  / ' + '\n')
            outfile.write('END' + '\n\n')

    def _write_subsurface_simulation_begin(self, outfile):
        if self.simulation.subsurface_flow or self.simulation.subsurface_transport:
            outfile.write('SUBSURFACE\n\n')

    def _write_subsurface_simulation_end(self, outfile):
        if self.simulation.subsurface_flow or self.simulation.subsurface_transport:
            outfile.write('END_SUBSURFACE\n\n')

    def _read_co2_database(self, infile, line):
        self.co2_database = del_extra_slash(self.splitter(line))

    def _write_overwrite_restart(self, outfile):
        outfile.write('OVERWRITE_RESTART_FLOW_PARAMS' + '\n\n')

    def _write_co2_database(self, outfile):
        self._header(outfile, headers['co2_database'])
        outfile.write('CO2_DATABASE ' + self.co2_database + '\n\n')

    def _read_regression(self, infile, line):
        regression = pregression()
        keep_reading = True
        while keep_reading:  # Read through all cards
            line = infile.readline()  # get next line
            key = line.strip().split()[0].lower()  # take first key word

            if key == 'cells':
                keep_reading_2 = True
                cell_list = []
                while keep_reading_2:
                    for i in range(100):
                        line1 = infile.readline()
                        if line1.strip().split()[0].lower() in ['/', 'end']:
                            keep_reading_2 = False
                            break
                        cell_list.append(int(line1))
                regression.cells = cell_list
            elif key == 'cells_per_process':
                regression.cells_per_process = self.splitter(line)
            elif key in ['/', 'end']:
                keep_reading = False

        self.regression = regression

    def _write_regression(self, outfile):
        self._header(outfile, headers['regression'])
        regression = self.regression
        outfile.write('REGRESSION' + '\n')
        if regression.cells and regression.cells[0] != '':
            outfile.write('  CELLS' + '\n')
            for cell in regression.cells:
                outfile.write('    ' + str(cell) + '\n')
            outfile.write('  /' + '\n')
        if regression.cells_per_process:
            outfile.write('  CELLS_PER_PROCESS' + ' ' + str(regression.cells_per_process) + '\n')
        outfile.write('END' + '\n\n')

    def _read_grid(self, infile, line):
        grid = pgrid()  # assign defaults before reading in values

        keep_reading = True
        bounds_key = False
        while keep_reading:
            line = infile.readline()  # get next line
            key = line.strip().split()[0].lower()  # take first keyword
            if key in ['#']:
                pass
            if key == 'type':
                grid.type = self.splitter(line)
            elif key == 'bounds':
                keep_reading_2 = True
                while keep_reading_2:
                    line1 = infile.readline()
                    grid.lower_bounds[0] = floatD(line1.split()[0])
                    grid.lower_bounds[1] = floatD(line1.split()[1])
                    grid.lower_bounds[2] = floatD(line1.split()[2])
                    line2 = infile.readline()
                    grid.upper_bounds[0] = floatD(line2.split()[0])
                    grid.upper_bounds[1] = floatD(line2.split()[1])
                    grid.upper_bounds[2] = floatD(line2.split()[2])
                    line3 = infile.readline()
                    if line3.strip().split()[0].lower() in ['/', 'end']: keep_reading_2 = False
            elif key == 'origin':
                grid.origin.append(floatD(line.strip().split()[1]))
                grid.origin.append(floatD(line.strip().split()[2]))
                grid.origin.append(floatD(line.strip().split()[3]))
            elif key == 'nxyz':
                grid.nxyz[0] = int(line.split()[1])
                grid.nxyz[1] = int(line.split()[2])
                grid.nxyz[2] = int(line.split()[3])
            elif key == 'gravity':
                grid.gravity.append(floatD(line.split()[1]))
                grid.gravity.append(floatD(line.split()[2]))
                grid.gravity.append(floatD(line.split()[3]))
            elif key == 'filename':
                if grid.type != 'unstructured':
                    raise PyFLOTRAN_ERROR('filename not need with structure grid!')
                grid.filename = self.splitter(line)
            elif key == 'dxyz':
                if bounds_key:
                    raise PyFLOTRAN_ERROR('specify either bounds of dxyz!')
                keep_reading_2 = True
                count = 0
                while keep_reading_2:
                    line = infile.readline()
                    if line.strip().split()[0].lower() in ['/', 'end']:
                        keep_reading_2 = False
                    else:
                        grid.dxyz[count] = floatD(line.strip().split()[0])
                        count += 1
            elif key in ['/', 'end']:
                keep_reading = False
        self.grid = grid

    def _write_grid(self, outfile):
        self._header(outfile, headers['grid'])
        grid = self.grid
        outfile.write('GRID\n')
        if grid.type in grid_types_allowed:
            outfile.write('  TYPE ' + grid.type + '\n')
        else:
            print '       valid grid.types:', grid_types_allowed
            raise PyFLOTRAN_ERROR('grid.type: \'' + grid.type + '\' is invalid!')
        if grid.lower_bounds:
            outfile.write('  BOUNDS\n')
            outfile.write('    ')
            for i in range(3):
                outfile.write(strD(grid.lower_bounds[i]) + ' ')
            outfile.write('\n    ')
            for i in range(3):
                outfile.write(strD(grid.upper_bounds[i]) + ' ')
            outfile.write('\n  /\n')  # / marks end of writing out bounds
        else:  # DXYZ is only written if no bounds are provided
            outfile.write('  DXYZ\n')
            for j in range(len(grid.dx)):
                outfile.write('    ' + strD(grid.dx[j]))
                if j % 5 == 4:
                    outfile.write('   ' + '\\' + '\n')
            outfile.write('\n')
            for j in range(len(grid.dy)):
                outfile.write('    ' + strD(grid.dy[j]))
                if j % 5 == 4:
                    outfile.write('   ' + '\\' + '\n')
            outfile.write('\n')
            for j in range(len(grid.dz)):
                outfile.write('    ' + strD(grid.dz[j]))
                if j % 5 == 4:
                    outfile.write('   ' + '\\' + '\n')
            outfile.write('\n')
            outfile.write('  END\n')
        if grid.origin:
            outfile.write('  ORIGIN' + ' ')
            for i in range(3):
                outfile.write(strD(grid.origin[i]) + ' ')
            outfile.write('\n')
        outfile.write('  NXYZ' + ' ')
        for i in range(3):
            outfile.write(strD(grid.nxyz[i]) + ' ')
        outfile.write('\n')
        if grid.gravity:
            outfile.write('  GRAVITY' + ' ')
            for i in range(3):
                outfile.write(strD(grid.gravity[i]) + ' ')
            outfile.write('\n')
        if grid.type == 'unstructured':
            outfile.write('  FILENAME' + grid.filename + '\n')
        outfile.write('END\n\n')

    def _read_timestepper(self, infile, line):
        p = ptimestepper()
        np_ts_mode = p.ts_mode
        np_ts_acceleration = p.ts_acceleration
        np_num_steps_after_cut = p.num_steps_after_cut
        np_max_steps = p.max_steps
        np_max_ts_cuts = p.max_ts_cuts
        np_cfl_limiter = p.cfl_limiter
        np_initialize_to_steady_state = p.initialize_to_steady_state
        np_run_as_steady_state = p.run_as_steady_state
        np_max_pressure_change = p.max_pressure_change
        np_max_temperature_change = p.max_temperature_change
        np_max_concentration_change = p.max_concentration_change
        np_max_saturation_change = p.max_saturation_change

        keep_reading = True

        while keep_reading:  # read through all cards
            line = infile.readline()  # get next line
            key = line.strip().split()[0].lower()  # take first keyword
            if key == 'ts_mode':
                np_ts_mode = str(self.splitter(line))
            if key == 'ts_acceleration':
                np_ts_acceleration = int(self.splitter(line))
            elif key == 'num_steps_after_cut':
                np_num_steps_after_cut = int(self.splitter(line))
            elif key == 'max_steps':
                np_max_steps = int(self.splitter(line))
            elif key == 'max_ts_cuts':
                np_max_ts_cuts = int(self.splitter(line))
            elif key == 'cfl_limiter':
                np_cfl_limiter = floatD(self.splitter(line))
            elif key == 'initialize_to_steady_state':
                np_initialize_to_steady_state = True
            elif key == 'run_as_steady_state':
                np_run_as_steady_state = True
            elif key == 'max_pressure_change':
                np_max_pressure_change = floatD(self.splitter(line))
            elif key == 'max_temperature_change':
                np_max_temperature_change = floatD(self.splitter(line))
            elif key == 'max_concentration_change':
                np_max_concentration_change = floatD(self.splitter(line))
            elif key == 'max_saturation_change':
                np_max_saturation_change = floatD(self.splitter(line))
            elif key in ['/', 'end']:
                keep_reading = False

        new_timestep = ptimestepper(np_ts_mode, np_ts_acceleration, np_num_steps_after_cut, np_max_steps,
                                    np_max_ts_cuts, np_cfl_limiter, np_initialize_to_steady_state,
                                    np_run_as_steady_state, np_max_pressure_change, np_max_temperature_change,
                                    np_max_concentration_change, np_max_saturation_change)

        self.timestepper = new_timestep

    def _write_timestepper(self, outfile):
        self._header(outfile, headers['timestepper'])
        outfile.write('TIMESTEPPER ' + self.timestepper.ts_mode + '\n')
        if self.timestepper.ts_acceleration:
            outfile.write('  ' + 'TS_ACCELERATION ' + str(self.timestepper.ts_acceleration) + '\n')
        if self.timestepper.num_steps_after_cut:
            outfile.write('  ' + 'NUM_STEPS_AFTER_CUT ' + str(self.timestepper.num_steps_after_cut) + '\n')
        if self.timestepper.max_ts_cuts:
            outfile.write('  ' + 'MAX_TS_CUTS ' + str(self.timestepper.max_ts_cuts) + '\n')
        if self.timestepper.max_steps:
            outfile.write('  ' + 'MAX_STEPS ' + str(self.timestepper.max_steps) + '\n')
        if self.timestepper.cfl_limiter:
            outfile.write('  ' + 'CFL_LIMITER ' + strD(self.timestepper.cfl_limiter) + '\n')
        if self.timestepper.initialize_to_steady_state:
            outfile.write('  ' + 'INITIALIZE_TO_STEADY_STATE ' + '\n')
        if self.timestepper.run_as_steady_state:
            outfile.write('  ' + 'RUN_AS_STEADY_STATE ' + '\n')
        if self.timestepper.max_pressure_change:
            outfile.write('  ' + 'MAX_PRESSURE_CHANGE ' + strD(self.timestepper.max_pressure_change) + '\n')
        if self.timestepper.max_temperature_change:
            outfile.write('  ' + 'MAX_TEMPERATURE_CHANGE ' + strD(self.timestepper.max_temperature_change) + '\n')
        if self.timestepper.max_concentration_change:
            outfile.write('  ' + 'MAX_CONCENTRATION_CHANGE ' + strD(self.timestepper.max_concentration_change) + '\n')
        if self.timestepper.max_saturation_change:
            outfile.write('  ' + 'MAX_SATURATION_CHANGE ' + strD(self.timestepper.max_saturation_change) + '\n')
        outfile.write('END\n\n')

    def _read_prop(self, infile, line):

        np_name = self.splitter(line)  # property name
        np_id = None
        p = pmaterial(0, '')  # assign defaults before reading in values
        np_porosity = p.porosity
        np_characteristic_curves = p.characteristic_curves
        np_tortuosity = p.tortuosity
        np_density = p.density
        np_specific_heat = p.specific_heat
        np_cond_dry = p.cond_dry
        np_cond_wet = p.cond_wet
        np_saturation = p.saturation
        np_permeability = []
        np_permeability_critical_porosity = p.permeability_critical_porosity
        np_permeability_power = p.permeability_power
        np_permeability_min_scale_factor = p.permeability_min_scale_factor
	np_longitudinal_dispersivity = p.longitudinal_dispersivity
	np_transverse_dispersivity_h = p.transverse_dispersivity_h
	np_transverse_dispersivity_v = p.transverse_dispersivity_v
        keep_reading = True

        while keep_reading:  # read through all cards
            line = infile.readline()  # get next line
            key = line.strip().split()[0].lower()  # take first keyword
            if key == 'id':
                np_id = int(self.splitter(line))
            elif key == 'characteristic_curves':
                np_characteristic_curves = self.splitter(line)
            elif key == 'porosity':
                if line.split()[1].lower() == 'dataset':
                    np_porosity = self.splitter(line)
                else:
                    np_porosity = floatD(self.splitter(line))
            elif key == 'tortuosity':
                np_tortuosity = floatD(self.splitter(line))
            elif key == 'rock_density':
                np_density = floatD(self.splitter(line))
            elif key == 'specific_heat':
                np_specific_heat = floatD(self.splitter(line))
            elif key == 'thermal_conductivity_dry':
                np_cond_dry = floatD(self.splitter(line))
            elif key == 'thermal_conductivity_wet':
                np_cond_wet = floatD(self.splitter(line))
            elif key == 'saturation_function':
                np_saturation = self.splitter(line)
            elif key == 'permeability_power':
                np_permeability_power = self.splitter(line)
            elif key == 'permeability_critical_porosity':
                np_permeability_critical_porosity = self.splitter(line)
            elif key == 'permeability_min_scale_factor':
                np_permeability_min_scale_factor = self.splitter(line)
            elif key == 'longitudinal_dispersivity':
                np_longitudinal_dispersivity = self.splitter(line)
            elif key == 'transverse_dispersivity_h':
                np_transverse_dispersivity_h = self.splitter(line)
            elif key == 'transverse_dispersivity_v':
                np_transverse_dispersivity_v = self.splitter(line)
            elif key == 'permeability':
                keep_reading_2 = True
                while keep_reading_2:
                    line = infile.readline()  # get next line
                    key = line.split()[0].lower()  # take first keyword
                    if key == 'perm_iso':
                        np_permeability.append(floatD(self.splitter(line)))
                    elif key == 'perm_x':
                        np_permeability.append(floatD(self.splitter(line)))
                    elif key == 'perm_y':
                        np_permeability.append(floatD(self.splitter(line)))
                    elif key == 'perm_z':
                        np_permeability.append(floatD(self.splitter(line)))
                    elif key in ['/', 'end']:
                        keep_reading_2 = False
            elif key in ['/', 'end']:
                keep_reading = False

        # create an empty material property
        new_prop = pmaterial(np_id, np_name, np_characteristic_curves, np_porosity, np_tortuosity, np_density,
                             np_specific_heat, np_cond_dry, np_cond_wet, np_saturation, np_permeability,
                             np_permeability_power, np_permeability_critical_porosity, np_permeability_min_scale_factor,
			     np_longitudinal_dispersivity,np_transverse_dispersivity_h,np_transverse_dispersivity_v)

        self.add(new_prop)

    def _add_prop(self, prop=pmaterial(), overwrite=False):  # Adds a prop object.
        # check if prop already exists
        if isinstance(prop, pmaterial):
            if prop.id in self.prop.keys():
                if not overwrite:
                    warning = 'WARNING: A Material Property with id \'' + str(prop.id) + '\' already exists. Prop' + \
                              'will not be defined, use overwrite = True in add() to overwrite the old prop.'
                    print warning,
                    build_warnings.append(warning)
                    return
                else:  # Executes if overwrite = True
                    self.delete(self.prop[prop.id])

        if prop not in self.proplist:
            self.proplist.append(prop)

    def _delete_prop(self, prop=pmaterial()):
        self.proplist.remove(prop)

    def _write_prop(self, outfile):
        self._header(outfile, headers['material_property'])
        for prop in self.proplist:
            if prop.name:
                outfile.write('MATERIAL_PROPERTY ' + prop.name + '\n')
            if prop.id:
                outfile.write('  ID ' + str(prop.id) + '\n')
            if prop.characteristic_curves:
                outfile.write('  CHARACTERISTIC_CURVES ' + prop.characteristic_curves + '\n')
            if prop.porosity:
                if type(prop.porosity) is str:
                    outfile.write('  POROSITY DATASET ' + prop.porosity + '\n')
                else:
                    outfile.write('  POROSITY ' + strD(prop.porosity) + '\n')
            if prop.tortuosity:
                outfile.write('  TORTUOSITY ' + strD(prop.tortuosity) + '\n')
            if prop.density:
                outfile.write('  ROCK_DENSITY ' + strD(prop.density) + '\n')
            if prop.specific_heat:
                outfile.write('  SPECIFIC_HEAT ' + strD(prop.specific_heat) + '\n')
            if prop.cond_dry:
                outfile.write('  THERMAL_CONDUCTIVITY_DRY ' + strD(prop.cond_dry) + '\n')
            if prop.cond_wet:
                outfile.write('  THERMAL_CONDUCTIVITY_WET ' + strD(prop.cond_wet) + '\n')
            if prop.saturation:
                outfile.write('  SATURATION_FUNCTION ' + prop.saturation + '\n')
            if prop.permeability_power:
                outfile.write('  PERMEABILITY_POWER ' + prop.permeability_power + '\n')
            if prop.permeability_critical_porosity:
                outfile.write('  PERMEABILITY_CRITICAL_POROSITY ' + prop.permeability_critical_porosity + '\n')
            if prop.permeability_min_scale_factor:
                outfile.write('  PERMEABILITY_MIN_SCALE_FACTOR ' + prop.permeability_min_scale_factor + '\n')
            if prop.longitudinal_dispersivity:
                outfile.write('  LONGITUDINAL_DISPERSIVITY ' + strD(prop.longitudinal_dispersivity) + '\n')
            if prop.transverse_dispersivity_h:
                outfile.write('  TRANSVERSE_DISPERSIVITY_H ' + strD(prop.transverse_dispersivity_h) + '\n')
            if prop.transverse_dispersivity_v:
                outfile.write('  TRANSVERSE_DISPERSIVITY_V ' + strD(prop.transverse_dispersivity_v) + '\n')



            if prop.permeability:
                outfile.write('  PERMEABILITY\n')
		if type(prop.permeability) is str:
                    outfile.write('    DATASET ' + prop.permeability + '\n')
                elif len(prop.permeability) == 1:
                    outfile.write('    PERM_ISO ' + strD(prop.permeability[0]) + '\n')
                else:
                    outfile.write('    PERM_X ' + strD(prop.permeability[0]) + '\n')
                    outfile.write('    PERM_Y ' + strD(prop.permeability[1]) + '\n')
                    outfile.write('    PERM_Z ' + strD(prop.permeability[2]) + '\n')
                outfile.write('  /\n')
            outfile.write('END\n\n')

    def _read_time(self, infile):
        time = ptime()
        time.dtf_list = []

        keep_reading = True
        while keep_reading:
            line = infile.readline()  # get next line
            key = line.split()[0].lower()  # take first keyword
            if key == 'final_time':
                tstring = line.split()[1:]  # temp list of strings,
                # do not include 1st sub-string
                if len(tstring) == 2:  # Do this if there is a time unit to read
                    time.tf.append(floatD(tstring[0]))
                    time.tf.append(tstring[-1])
                else:  # No time unit being read in
                    time.tf.append(floatD(tstring[0]))
            elif key == 'initial_timestep_size':
                tstring = line.split()[1:]
                if len(tstring) == 2:
                    time.dti.append(floatD(tstring[0]))
                    time.dti.append(tstring[-1])
                else:
                    time.dti.append(floatD(tstring[0]))
            elif key == 'maximum_timestep_size':
                if 'at' not in line:
                    tstring = line.split()[1:]
                    if len(tstring) == 2:
                        time.dtf.append(floatD(tstring[0]))
                        time.dtf.append(tstring[-1])
                    else:
                        time.dtf.append(floatD(tstring[0]))
                elif 'at' in line:
                    # Read maximum_timestep_size with AT keyword
                    if key == 'maximum_timestep_size':

                        # temporary variable
                        dtf_more = []

                        # Read before AT
                        tstring = line.split()[1:]
                        if len(tstring) >= 2:
                            # assign 1st value
                            dtf_more.append(floatD(tstring[0]))

                            # assign 1st unit
                            dtf_more.append(tstring[1])

                        # Read after AT
                        at_i = tstring.index('at')  # Find index # in list (Not string)
                        tstring = line.split()[at_i + 2:]  # Use string only after 'at'

                        if len(tstring) == 2:
                            # assign 2nd value (increment)
                            dtf_more.append(floatD(tstring[0]))

                            # assign 2nd unit (increment)
                            dtf_more.append(tstring[1])

                        time.dtf_list.append(dtf_more)

            elif key in ['/', 'end']:
                keep_reading = False

        self.time = time

    def _write_time(self, outfile):
        self._header(outfile, headers['time'])
        time = self.time
        outfile.write('TIME\n')
        if time.steady_state:
            outfile.write('  STEADY_STATE\n')

        # write FINAL_TIME statement (tf)
        if time.tf:
            try:
                outfile.write('  FINAL_TIME ' + strD(time.tf[0]))  # Write value
                if time.tf[1].lower() in time_units_allowed:
                    outfile.write(' ' + time.tf[1].lower() + '\n')  # Write time unit
                else:
                    print '       valid time.units', time_units_allowed, '\n'
                    raise PyFLOTRAN_ERROR('PyFLOTRAN ERROR: time.tf[1]: \'' + time.tf[1] + '\' is invalid!')
            except:
                raise PyFLOTRAN_ERROR(
                    'time.tf (final time) input is invalid. Format should be a list: [number, string]')

        # write INITIAL_TIMESTEP_SIZE statement (dti)
        if time.dti:
            try:
                outfile.write('  INITIAL_TIMESTEP_SIZE ' + strD(time.dti[0]))  # Write value
                if time.dti[1].lower() in time_units_allowed:
                    outfile.write(' ' + time.dti[1] + '\n')  # Write time unit
                else:
                    print '       valid time.units', time_units_allowed, '\n'
                    raise PyFLOTRAN_ERROR('time.dti[1]: \'' + time.dti[1] + '\' is invalid.')
            except:
                raise PyFLOTRAN_ERROR(
                    'time.dti (initial timestep size) input is invalid. Format should be a list: [number, string]')

        # write MAXIMUM_TIMESTEP_SIZE statement	dtf
        if time.dtf:
            try:
                outfile.write('  MAXIMUM_TIMESTEP_SIZE ' + strD(time.dtf[0]))
                if time.dtf[1].lower() in time_units_allowed:
                    outfile.write(' ' + time.dtf[1] + '\n')
                else:
                    print '       valid time.units', time_units_allowed, '\n'
                    raise PyFLOTRAN_ERROR('time.dtf[1]: \'' + time.dtf[1] + '\' is invalid.')
            except:
                raise PyFLOTRAN_ERROR(
                    'time.dtf (maximum timestep size) input is invalid. Format should be a list: [number, string]')

        # Write more MAXIMUM_TIME_STEP_SIZE statements if applicable
        for dtf in time.dtf_list:
            outfile.write('  MAXIMUM_TIMESTEP_SIZE ')

            try:
                # Write 1st value before 'at'
                if isinstance(dtf[0], float):
                    outfile.write(strD(dtf[0]) + ' ')
                else:
                    raise PyFLOTRAN_ERROR('The 1st variable in a dtf_list is not recognized as a float.')

                # Write 1st time unit before 'at'
                if isinstance(dtf[1], str):
                    outfile.write((dtf[1]) + ' ')
                else:
                    raise PyFLOTRAN_ERROR('The 2nd variable in a dtf_list is not recognized as a str (string).')

                outfile.write('at ')

                # Write 2nd value after 'at'
                if isinstance(dtf[2], float):
                    outfile.write(strD(dtf[2]) + ' ')
                else:
                    raise PyFLOTRAN_ERROR('The 3rd variable in a dtf_list is not recognized as a float.')

                # Write 2nd time unit after 'at'
                if isinstance(dtf[3], str):
                    outfile.write((dtf[3]))
                else:
                    raise PyFLOTRAN_ERROR('PyFLOTRAN ERROR: The 4th variable in a dtf_list is not recognized as a '
                                          'str (string).')
            except:
                raise PyFLOTRAN_ERROR(
                    'PyFLOTRAN ERROR: time.dtf_list (maximum timestep size with \'at\') is invalid. Format should be a '
                    'list: [float, str, float, str]')
            outfile.write('\n')
        '''
        # Determine dtf_i size, the length of the smallest sized list being used
        # with MAXIMUM_TIMESTEP_SIZE with key word  'at'.
        # Displays a warning if the lists are not all of equal length.
        if time.dtf_i == 0:	# Checks to see if user manually specified length so that
                    # it does not re-assign user input
            # Assign minimum value
            time.dtf_i = min(len(time.dtf_lv), len(time.dtf_li),
                     len(time.dtf_lv_unit), len(time.dtf_li_unit))

            # Display warning if lists are not all of equal length
            # This check may not be needed.
            if not all(i == time.dtf_i for i in (len(time.dtf_lv), len(time.dtf_li),
                                 len(time.dtf_lv_unit),
                                 len(time.dtf_li_unit))):
                print 'WARNING: The lengths of time.dtf_lv, time.dtf_li, time.dtf_lv, and time.dtf_li are not all of
                equal length.\n\tSome values assigned will be missing.\n'

        # Write more MAXIMUM_TIMESTEP_SIZE statements if applicable
        for i in range(0, time.dtf_i):
            try:
                # write before key word 'AT'
                time.dtf_lv_unit[i] = time.dtf_lv_unit[i].lower()# lower capitalization
                outfile.write('  MAXIMUM_TIMESTEP_SIZE ')
                outfile.write(strD(time.dtf_lv[i]) + ' ') # Write Value
                if time.dtf_lv_unit[i] in time_units_allowed:
                    outfile.write(time.dtf_lv_unit[i])# Write Time Unit
                else:
                    print 'PyFLOTRAN ERROR: time.dtf_lv_unit: \'' + time.dtf_lv_unit[i] + '\' is invalid.'
                    print '       valid time.units', time_units_allowed, '\n'

                # write after key word 'AT'
                time.dtf_li_unit[i] = time.dtf_li_unit[i].lower()# lower capitalization
                outfile.write(' at ')
                outfile.write(strD(time.dtf_li[i]) + ' ') # Write Value
                if time.dtf_li_unit[i] in time_units_allowed:
                    outfile.write(time.dtf_li_unit[i]) # Write Time Unit
                else:
                    print 'PyFLOTRAN ERROR: time.dtf_li_unit: \'' + time.dtf_li_unit[i] + '\' is invalid.'
                    print '       valid time.units', time_units_allowed, '\n'
                outfile.write('\n')
            except:
                print 'PyFLOTRAN ERROR: Invalid input at maximum_time_step_size with key word \'at\'. time.dtf_lv and
                time.dtf_li should be a list of floats. time_dtf_lv_unit and time_dtf_li_unit should be a list of
                strings. All lists should be of equal length.\n'
        '''
        outfile.write('END\n\n')

    def _read_lsolver(self, infile, line):
        lsolver = plsolver()  # temporary object while reading
        lsolver.name = self.splitter(line).lower()  # solver type - tran_solver or flow_solver

        keep_reading = True

        while keep_reading:  # Read through all cards
            line = infile.readline()  # get next line
            key = line.strip().split()[0].lower()  # take first key word

            if key == 'solver':
                lsolver.solver = self.splitter(line)  # Assign last word
            if key == 'preconditioner':
                lsolver.preconditioner = self.splitter(line)
            elif key in ['/', 'end']:
                keep_reading = False

        self.add(lsolver)  # Assign object

    def _add_lsolver(self, lsolver=plsolver(), overwrite=False):  # Adds a Linear Solver object.
        # check if lsolver already exists
        if isinstance(lsolver, plsolver):
            if lsolver.name in self.lsolver.keys():
                if not overwrite:
                    warning = 'WARNING: A linear solver with name \'' + str(
                        lsolver.name) + '\' already exists. lsolver will not be defined, use overwrite = True in add() ' \
                                        'to overwrite the old lsolver.'
                    print warning,
                    build_warnings.append(warning)
                    return
                else:
                    self.delete(self.lsolver[lsolver.name])

        if lsolver not in self.lsolverlist:
            self.lsolverlist.append(lsolver)

    def _delete_lsolver(self, lsolver=plsolver()):
        self.lsolverlist.remove(lsolver)

    def _write_lsolver(self, outfile):
        self._header(outfile, headers['linear_solver'])

        for lsolver in self.lsolverlist:
            if lsolver.name.lower() in solver_names_allowed:
                outfile.write('LINEAR_SOLVER ' + lsolver.name.lower() + '\n')
            else:
                print '       valid solver.names', solver_names_allowed, '\n'
                raise PyFLOTRAN_ERROR('lsolver.name: \'' + lsolver.name + '\' is invalid.')
            if lsolver.solver:
                outfile.write('  SOLVER ' + lsolver.solver.upper() + '\n')
            if lsolver.preconditioner:
                outfile.write('  PRECONDITIONER ' + lsolver.preconditioner.upper() + '\n')
            outfile.write('END\n\n')

    def _read_nsolver(self, infile, line):

        nsolver = pnsolver('')  # Assign Defaults

        nsolver.name = self.splitter(line).lower()  # newton solver type - tran_solver or flow_solver

        keep_reading = True

        while keep_reading:  # Read through all cards
            line = infile.readline()  # get next line
            key = line.strip().split()[0].lower()  # take first key word

            if key == 'atol':
                nsolver.atol = floatD(self.splitter(line))
            if key == 'rtol':
                nsolver.rtol = floatD(self.splitter(line))
            if key == 'stol':
                nsolver.stol = floatD(self.splitter(line))
            if key == 'dtol':
                nsolver.dtol = floatD(self.splitter(line))
            if key == 'itol':
                nsolver.itol = floatD(self.splitter(line))
            if key == 'maxit':
                nsolver.max_it = int(self.splitter(line))
            if key == 'maxf':
                nsolver.max_f = int(self.splitter(line))
            elif key in ['/', 'end']:
                keep_reading = False
        self.add(nsolver)  # Assign

    def _add_nsolver(self, nsolver=pnsolver(), overwrite=False):  # Adds a Newton Solver object.
        # check if nsolver already exists
        if isinstance(nsolver, pnsolver):
            if nsolver.name in self.nsolver.keys():
                if not overwrite:
                    warning = 'WARNING: A newton solver with name \'' + str(nsolver.name) + '\' already exists. ' + \
                              'nsolver will not be defined, use overwrite = True in add() to overwrite the old nsolver.'
                    print warning,
                    build_warnings.append(warning)
                    return
                else:
                    self.delete(self.nsolver[nsolver.name])

        if nsolver not in self.nsolverlist:
            self.nsolverlist.append(nsolver)

    def _delete_nsolver(self, nsolver=pnsolver()):
        self.nsolverlist.remove(nsolver)

    def _write_nsolver(self, outfile):
        self._header(outfile, headers['newton_solver'])

        for nsolver in self.nsolverlist:
            if nsolver.name.lower() in solver_names_allowed:
                outfile.write('NEWTON_SOLVER ' + nsolver.name.lower() + '\n')
            else:
                print '       valid solver.names', solver_names_allowed, '\n'
                raise PyFLOTRAN_ERROR('nsolver.name: \'' + nsolver.name + '\' is invalid.')
            if nsolver.atol:
                outfile.write('  ATOL ' + strD(nsolver.atol) + '\n')
            if nsolver.rtol:
                outfile.write('  RTOL ' + strD(nsolver.rtol) + '\n')
            if nsolver.stol:
                outfile.write('  STOL ' + strD(nsolver.stol) + '\n')
            if nsolver.dtol:
                outfile.write('  DTOL ' + strD(nsolver.dtol) + '\n')
            if nsolver.itol:
                outfile.write('  ITOL ' + strD(nsolver.itol) + '\n')
            if nsolver.max_it:
                outfile.write('  MAXIT ' + str(nsolver.max_it) + '\n')
            if nsolver.max_f:
                outfile.write('  MAXF ' + str(nsolver.max_f) + '\n')
            outfile.write('END\n\n')

    def _read_output(self, infile):
        output = poutput()
        output.time_list = []
        output.format_list = []
        output.variables_list = []

        keep_reading = True

        while keep_reading:  # Read through all cards
            line = infile.readline()  # get next line
            key = line.strip().split()[0].lower()  # take first key word

            if key == 'times':
                tstring = line.split()[1:]  # Turn into list, exempt 1st word
                for t in tstring:
                    try:
                        output.time_list.append(floatD(t))
                    except:
                        output.time_list.append(t)

            elif key == 'screen':
                tstring = line.strip().split()[1].lower()  # Read the 2nd word
                if tstring == 'periodic':
                    output.screen_periodic = int(self.splitter(line))
            elif key == 'periodic':
                tstring = line.strip().split()[1].lower()  # Read the 2nd word
                if tstring == 'time':
                    output.periodic_time.append(floatD(line.split()[-2]))  # 2nd from last word.
                    output.periodic_time.append(self.splitter(line))  # last word
                elif tstring == 'timestep':
                    output.periodic_timestep.append(floatD(line.split()[-2]))  # 2nd from last word.
                    output.periodic_timestep.append(self.splitter(line))  # last word
            elif key == 'periodic_observation':
                tstring = line.strip().split()[1].lower()  # Read the 2nd word
                if tstring == 'time':
                    output.periodic_observation_time.append(floatD(line.split()[-2]))  # 2nd from last word.
                    output.periodic_observation_time.append(self.splitter(line))  # last word
                elif tstring == 'timestep':
                    output.periodic_observation_timestep = int(self.splitter(line))
            elif key == 'print_column_ids':
                output.print_column_ids = True
            elif key == 'format':
                tstring = (line.strip().split()[1:])  # Do not include 1st sub-string
                tstring = ' '.join(tstring).lower()  # Convert list into a string seperated by a space
                output.format_list.append(tstring)  # assign
            elif key == 'velocities':
                output.velocities = True
            elif key == 'velocity_at_center':
                output.velocity_at_center = True
            elif key == 'velocity_at_face':
                output.velocity_at_face = True
            elif key == 'mass_balance':
                output.mass_balance = True
            elif key == 'variables':
                keep_reading_1 = True
                while keep_reading_1:
                    line1 = infile.readline()
                    key1 = line1.strip().split()[0].lower()
                    if key1 in output_variables_allowed:
                        output.variables_list.append(key1)
                    elif key1 in ['/', 'end']:
                        keep_reading_1 = False
                    else:
                        raise PyFLOTRAN_ERROR('variable ' + str(key1) + ' cannot be an output variable.')
            elif key in ['/', 'end']:
                keep_reading = False

        self.output = output

    def _write_output(self, outfile):
        self._header(outfile, headers['output'])
        output = self.output

        # Write Output - if used so null/None entries are not written
        outfile.write('OUTPUT\n')

        if output.time_list:
            # Check if 1st variable in list a valid time unit
            if output.time_list[0].lower() in time_units_allowed:
                outfile.write('  TIMES ')
                # Write remaining number(s) after time unit is specified
                for value in output.time_list:
                    outfile.write(' ' + strD(value).lower())
            else:
                print '       valid time.units', time_units_allowed, '\n'
                raise PyFLOTRAN_ERROR('output.time_list[0]: \'' + output.time_list[0] + '\' is invalid.')
            outfile.write('\n')

        # This is here on purpose - Needed later
        # if output.periodic_observation_time:
        # outfile.write('  PERIODIC_OBSERVATION TIME  '+
        # str(output.periodic_observation_time)+'\n')
        if not output.screen_output:
            try:  # Error checking to ensure screen_output is Bool.
                output.screen_output = bool(output.screen_output)
                outfile.write('  ' + 'SCREEN OFF' + '\n')
            except ValueError:
                raise PyFLOTRAN_ERROR('output.screen_output: \'' + str(output.screen_output) + '\' is not bool.')

        if output.screen_periodic:
            try:  # Error checking to ensure screen_periodic is int (integer).
                output.screen_periodic = int(output.screen_periodic)
                outfile.write('  ' + 'SCREEN PERIODIC ' + str(output.screen_periodic) + '\n')
            except ValueError:
                raise PyFLOTRAN_ERROR(
                    'output.screen_periodic: \'' + str(output.screen_periodic) + '\' is not int (integer).')
        if output.periodic_time:
            try:  # Error checking to ensure periodic_time is [float, str].
                output.periodic_time[0] = floatD(output.periodic_time[0])
                if output.periodic_time[1].lower() in time_units_allowed:
                    output.periodic_time[1] = str(output.periodic_time[1].lower())
                else:
                    output.periodic_time[1] = str(output.periodic_time[1].lower())
                    raise PyFLOTRAN_ERROR('time unit in output.periodic_time[1] is invalid. Valid time units are:',
                                          time_units_allowed)
                outfile.write('  ' + 'PERIODIC TIME ')
                outfile.write(strD(output.periodic_time[0]) + ' ')
                outfile.write(output.periodic_time[1] + '\n')
            except:
                raise PyFLOTRAN_ERROR('output.periodic_time: \'' + str(output.periodic_time) +
                                      '\' is not [float, str].')
        if output.periodic_timestep:
            try:  # Error checking to ensure periodic_timestep is [float].
                output.periodic_timestep = int(output.periodic_timestep)
                outfile.write('  ' + 'PERIODIC TIMESTEP ')
                outfile.write(strD(output.periodic_timestep) + '\n')
            except:
                raise PyFLOTRAN_ERROR('output.periodic_timestep: \'' + str(output.periodic_timestep) +
                                      '\' is not [float].')
        if output.periodic_observation_time:
            try:
                # Error checking to ensure periodic_observation_time is [float, str].
                output.periodic_observation_time[0] = floatD(output.periodic_observation_time[0])
                if output.periodic_observation_time[1].lower() in time_units_allowed:
                    output.periodic_observation_time[1] = str(output.periodic_observation_time[1].lower())
                else:
                    output.periodic_observation_time[1] = str(output.periodic_observation_time[1].lower())
                    raise PyFLOTRAN_ERROR('time unit in output.periodic_observation_time[1] is invalid. Valid time '
                                          'units are:', time_units_allowed)

                # Writing out results
                outfile.write('  ' + 'PERIODIC_OBSERVATION TIME ')
                outfile.write(strD(output.periodic_observation_time[0]) + ' ')
                outfile.write(output.periodic_observation_time[1] + '\n')
            except:
                raise PyFLOTRAN_ERROR('output.periodic_observation_time: \'' + str(output.periodic_observation_time) +
                                      '\' is not [float, str].')
        if output.periodic_observation_timestep:
            outfile.write('  PERIODIC_OBSERVATION TIMESTEP ' + str(output.periodic_observation_timestep) + '\n')
        if output.print_column_ids:
            outfile.write('  ' + 'PRINT_COLUMN_IDS' + '\n')
        for out_format in output.format_list:
            if out_format.upper() in output_formats_allowed:
                outfile.write('  FORMAT ')
                outfile.write(out_format.upper() + '\n')
            else:
                print '       valid output.format:', output_formats_allowed, '\n'
                raise PyFLOTRAN_ERROR('output.format: \'' + out_format + '\' is invalid.')
        if output.velocities:
            outfile.write('  ' + 'VELOCITIES' + '\n')
        if output.velocity_at_center:
            outfile.write('  ' + 'VELOCITY_AT_CENTER' + '\n')
        if output.velocity_at_face:
            outfile.write('  ' + 'VELOCITY_AT_FACE' + '\n')
        if output.mass_balance:
            outfile.write('  ' + 'MASS_BALANCE' + '\n')
        if output.variables_list:
            outfile.write('  VARIABLES \n')
            for variable in output.variables_list:
                if variable.lower() in output_variables_allowed:
                    outfile.write('    ' + variable.upper() + '\n')
                else:
                    print '       valid output.variable:', output_variables_allowed, '\n'
                    raise PyFLOTRAN_ERROR('output.variable: \'' + variable + '\' is invalid.')
            outfile.write('  /\n')
        outfile.write('END\n\n')

    def _read_fluid(self, infile):
        p = pfluid()
        np_diffusion_coefficient = p.diffusion_coefficient

        keep_reading = True

        while keep_reading:  # Read through all cards
            line = infile.readline()  # get next line
            key = line.strip().split()[0].lower()  # take first

            if key == 'diffusion_coefficient':
                np_diffusion_coefficient = floatD(self.splitter(line))  # Read last entry
            elif key in ['/', 'end']:
                keep_reading = False

        # Create new employ fluid properties object and assign read in values to it
        new_fluid = pfluid(np_diffusion_coefficient)
        self.fluid = new_fluid

    def _write_fluid(self, outfile):
        self._header(outfile, headers['fluid_property'])
        fluid = self.fluid
        outfile.write('FLUID_PROPERTY\n')

        # Write out requested (not null) fluid properties
        if fluid.diffusion_coefficient:
            outfile.write('  DIFFUSION_COEFFICIENT ' + strD(fluid.diffusion_coefficient) + '\n')  # Read last entry
        outfile.write('END\n\n')

    def _read_saturation(self, infile, line):

        saturation = psaturation()  # assign defaults before reading in values
        saturation.name = self.splitter(line).lower()  # saturation function name, passed in.

        keep_reading = True

        while keep_reading:  # Read through all cards
            line = infile.readline()  # get next line
            key = line.strip().split()[0].lower()  # take first  key word

            if key == 'permeability_function_type':
                saturation.permeability_function_type = self.splitter(line)
            elif key == 'saturation_function_type':
                saturation.saturation_function_type = self.splitter(line)
            elif key == 'residual_saturation_liquid':
                saturation.residual_saturation_liquid = floatD(self.splitter(line))
            elif key == 'residual_saturation_gas':
                saturation.residual_saturation_gas = floatD(self.splitter(line))
            elif key == 'residual_saturation':  # Alternative to check
                tstring = line.strip().split()[1].lower()  # take 2nd key word
                if tstring == 'liquid_phase':
                    saturation.residual_saturation_liquid = floatD(self.splitter(line))
                elif tstring == 'gas_phase':
                    saturation.residual_saturation_gas = floatD(self.splitter(line))
                else:  # if no 2nd word exists
                    saturation.residual_saturation = floatD(self.splitter(line))
            elif key == 'lambda':
                saturation.a_lambda = floatD(self.splitter(line))
            elif key == 'alpha':
                saturation.alpha = floatD(self.splitter(line))
            elif key == 'max_capillary_pressure':
                saturation.max_capillary_pressure = floatD(self.splitter(line))
            elif key == 'betac':
                saturation.betac = floatD(self.splitter(line))
            elif key == 'power':
                saturation.power = floatD(self.splitter(line))
            elif key in ['/', 'end']:
                keep_reading = False

        # Create an empty saturation function and assign the values read in
        self.add(saturation)

    def _add_saturation(self, sat=psaturation(), overwrite=False):  # Adds a saturation object.
        # check if saturation already exists
        if isinstance(sat, psaturation):
            if sat.name in self.saturation.keys():
                if not overwrite:
                    warning = 'WARNING: A saturation function with name \'' + str(
                        sat.name) + '\' already exists. Use overwrite = True in add() to overwrite the old saturation ' \
                                    'function.'
                    print warning,
                    build_warnings.append(warning)
                    return
                else:  # Executes if overwrite = True
                    self.delete(self.sat[sat.name])

        if sat not in self.saturationlist:
            self.saturationlist.append(sat)

    def _delete_saturation(self, sat=psaturation()):
        self.saturationlist.remove(sat)

    def _write_saturation(self, outfile):
        self._header(outfile, headers['saturation_function'])
        for sat in self.saturationlist:
            # Write out saturation properties that exist
            outfile.write('SATURATION_FUNCTION')
            if sat.name:
                outfile.write('  ' + sat.name + '\n')
            else:
                outfile.write('\n')
            if sat.permeability_function_type:
                if sat.permeability_function_type in permeability_function_types_allowed:
                    outfile.write('  PERMEABILITY_FUNCTION_TYPE ' + sat.permeability_function_type + '\n')
                else:
                    print('valid saturation.permeability_function_types', saturation_function_types_allowed, '\n')
                    raise PyFLOTRAN_ERROR(
                        'saturation.saturation_function_type: \'' + sat.saturation_function_type + '\' is invalid.')
            if sat.saturation_function_type:
                if sat.saturation_function_type in saturation_function_types_allowed:
                    outfile.write('  SATURATION_FUNCTION_TYPE ' +
                                  sat.saturation_function_type + '\n')
            if sat.residual_saturation or sat.residual_saturation == 0:
                outfile.write('  RESIDUAL_SATURATION ' + strD(sat.residual_saturation) + '\n')
            if sat.residual_saturation_liquid or sat.residual_saturation_liquid == 0:
                outfile.write('  RESIDUAL_SATURATION LIQUID_PHASE ' + strD(sat.residual_saturation_liquid) + '\n')
            if sat.residual_saturation_gas or sat.residual_saturation_gas == 0:
                outfile.write('  RESIDUAL_SATURATION GAS_PHASE ' + strD(sat.residual_saturation_gas) + '\n')
            if sat.a_lambda:
                outfile.write('  LAMBDA ' + strD(sat.a_lambda) + '\n')
            if sat.alpha:
                outfile.write('  ALPHA ' + strD(sat.alpha) + '\n')
            if sat.max_capillary_pressure:
                outfile.write('  MAX_CAPILLARY_PRESSURE ' + strD(sat.max_capillary_pressure) + '\n')
            if sat.betac:
                outfile.write('  BETAC ' + strD(sat.betac) + '\n')
            if sat.power:
                outfile.write('  POWER ' + strD(sat.power) + '\n')
            outfile.write('END\n\n')

    def _read_characteristic_curves(self, infile, line):

        characteristic_curves = pcharacteristic_curves()  # assign defaults before reading in values
        characteristic_curves.name = self.splitter(line).lower()  # Characteristic curve name, passed in.

        keep_reading = True

        while keep_reading:  # Read through all cards
            line = infile.readline()  # get next line
            key = line.strip().split()[0].lower()  # take first  key word

            if key == 'saturation_function_type':
                characteristic_curves.saturation_function_type = self.splitter(line)
            elif key == 'sf_alpha':
                characteristic_curves.sf_alpha = floatD(self.splitter(line))
            elif key == 'sf_m':
                characteristic_curves.sf_m = floatD(self.splitter(line))
            elif key == 'sf_lambda':
                characteristic_curves.sf_lambda = floatD(self.splitter(line))
            elif key == 'sf_liquid_residual_saturation':
                characteristic_curves.sf_liquid_residual_saturation = floatD(self.splitter(line))
            elif key == 'sf_gas_residual_saturation':
                characteristic_curves.sf_gas_residual_saturation = floatD(self.splitter(line))
            elif key == 'max_capillary_pressure':
                characteristic_curves.max_capillary_pressure = floatD(self.splitter(line))
            elif key == 'smooth':
                characteristic_curves.smooth = floatD(self.splitter(line))
            elif key == 'power':
                characteristic_curves.power = floatD(self.splitter(line))
            elif key == 'default':
                characteristic_curves.default = floatD(self.splitter(line))
            elif key == 'liquid_permeability_function_type':
                characteristic_curves.liquid_permeability_function_type = self.splitter(line)
            elif key == 'lpf_m':
                characteristic_curves.lpf_m = floatD(self.splitter(line))
            elif key == 'lpf_lambda':
                characteristic_curves.lpf_lambda = floatD(self.splitter(line))
            elif key == 'lpf_liquid_residual_saturation':
                characteristic_curves.lpf_liquid_residual_saturation = floatD(self.splitter(line))
            elif key == 'gas_permeability_function_type':
                characteristic_curves.gas_permeability_function_type = self.splitter(line)
            elif key == 'gpf_m':
                characteristic_curves.gpf_m = floatD(self.splitter(line))
            elif key == 'gpf_lambda':
                characteristic_curves.gpf_lambda = floatD(self.splitter(line))
            elif key == 'gpf_liquid_residual_saturation':
                characteristic_curves.gpf_liquid_residual_saturation = floatD(self.splitter(line))
            elif key == 'gpf_gas_residual_saturation':
                characteristic_curves.gpf_gas_residual_saturation = floatD(self.splitter(line))
            elif key in ['/', 'end']:
                keep_reading = False

        new_cc = pcharacteristic_curves(characteristic_curves.name, characteristic_curves.saturation_function_type,
                                        characteristic_curves.sf_alpha, characteristic_curves.sf_m,
                                        characteristic_curves.sf_lambda,
                                        characteristic_curves.sf_liquid_residual_saturation,
                                        characteristic_curves.sf_gas_residual_saturation,
                                        characteristic_curves.max_capillary_pressure, characteristic_curves.smooth,
                                        characteristic_curves.power, characteristic_curves.default,
                                        characteristic_curves.liquid_permeability_function_type,
                                        characteristic_curves.lpf_m, characteristic_curves.lpf_lambda,
                                        characteristic_curves.lpf_liquid_residual_saturation,
                                        characteristic_curves.gas_permeability_function_type,
                                        characteristic_curves.gpf_m, characteristic_curves.gpf_lambda,
                                        characteristic_curves.gpf_liquid_residual_saturation,
                                        characteristic_curves.gpf_gas_residual_saturation)

        self.add(new_cc)

    def _add_characteristic_curves(self, char=pcharacteristic_curves(), overwrite=False):  # Adds a char object.
        # check if char already exists
        if isinstance(char, pcharacteristic_curves):
            if char.name in self.char.keys():
                if not overwrite:
                    warning = 'WARNING: A Characteristic Curve with name \'' + str(char.name) + '\' already exists.' + \
                              'Characteristic curve will not be defined, use overwrite = True ' 'in add() to' + \
                              'overwrite the old characteristic curve.'
                    print warning,
                    build_warnings.append(warning)
                    return
                else:  # Executes if overwrite = True
                    self.delete(self.char[char.name])

        if char not in self.charlist:
            self.charlist.append(char)

    def _delete_char(self, char=pcharacteristic_curves()):
        self.charlist.remove(char)

    def _write_characteristic_curves(self, outfile):

        self._header(outfile, headers['characteristic_curves'])

        for char in self.charlist:
            # Write out characteristic curve properties that exist
            if char.name:
                outfile.write('CHARACTERISTIC_CURVES ' + char.name + '\n')
            if char.saturation_function_type:
                if char.saturation_function_type in characteristic_curves_saturation_function_types_allowed:
                    outfile.write('  SATURATION_FUNCTION ' + char.saturation_function_type + '\n')
                else:
                    print '       valid  char.saturation_function_types', \
                        characteristic_curves_saturation_function_types_allowed, '\n'
                    raise PyFLOTRAN_ERROR(
                        'char.saturation_function_type: \'' + char.saturation_function_type + '\' is invalid.')
                if char.sf_alpha:
                    outfile.write('   ALPHA ' + strD(char.sf_alpha) + '\n')
                if char.sf_m:
                    outfile.write('   M ' + strD(char.sf_m) + '\n')
                if char.sf_lambda:
                    outfile.write('   LAMBDA ' + strD(char.sf_lambda) + '\n')
                if char.sf_liquid_residual_saturation or char.sf_liquid_residual_saturation == 0:
                    outfile.write('   LIQUID_RESIDUAL_SATURATION ' + strD(char.sf_liquid_residual_saturation) + '\n')
                if char.sf_gas_residual_saturation or char.sf_gas_residual_saturation == 0:
                    outfile.write('   GAS_RESIDUAL_SATURATION ' + strD(char.sf_gas_residual_saturation) + '\n')
                if char.max_capillary_pressure:
                    outfile.write('   MAX_CAPILLARY_PRESSURE ' + strD(char.max_capillary_pressure) + '\n')
                if char.smooth:
                    outfile.write('   SMOOTH ' + '\n')  # This just prints the SMOOTH flag
                outfile.write('  / ' + '\n')

            if char.power:
                outfile.write('  POWER ' + strD(char.power) + '\n')
            if char.default:
                outfile.write('  DEFAULT ' + '\n')  # This just prints the DEFAULT flag
            if char.liquid_permeability_function_type:
                if char.liquid_permeability_function_type in \
                        characteristic_curves_liquid_permeability_function_types_allowed:
                    outfile.write('  PERMEABILITY_FUNCTION ' + char.liquid_permeability_function_type + '\n')
                # outfile.write('   PHASE LIQUID' + '\n')
                else:
                    print '       valid  char.liquid_permeability_function_types', \
                        characteristic_curves_liquid_permeability_function_types_allowed, '\n'
                    raise PyFLOTRAN_ERROR('char.liquid_permeability_function_type: \'' +
                                          char.liquid_permeability_function_type + '\' is invalid.')
                if char.lpf_m:
                    outfile.write('   M ' + strD(char.lpf_m) + '\n')
                if char.lpf_lambda:
                    outfile.write('   LAMBDA ' + strD(char.lpf_lambda) + '\n')
                if char.lpf_liquid_residual_saturation or char.lpf_liquid_residual_saturation == 0:
                    outfile.write('   LIQUID_RESIDUAL_SATURATION ' + strD(char.lpf_liquid_residual_saturation) + '\n')
                outfile.write('  / ' + '\n')

            if char.gas_permeability_function_type:
                if char.gas_permeability_function_type in characteristic_curves_gas_permeability_function_types_allowed:
                    outfile.write('  PERMEABILITY_FUNCTION ' + char.gas_permeability_function_type + '\n')
                    outfile.write('   PHASE GAS' + '\n')
                else:

                    print '       valid  char.gas_permeability_function_types', \
                        characteristic_curves_gas_permeability_function_types_allowed, '\n'
                    raise PyFLOTRAN_ERROR(
                        'char.gas_permeability_function_type: \'' + char.gas_permeability_function_type +
                        '\' is invalid.')
                if char.gpf_m:
                    outfile.write('   M ' + strD(char.gpf_m) + '\n')
                if char.gpf_lambda:
                    outfile.write('   LAMBDA ' + strD(char.gpf_lambda) + '\n')
                if char.gpf_liquid_residual_saturation or char.gpf_liquid_residual_saturation == 0:
                    outfile.write('   LIQUID_RESIDUAL_SATURATION ' + strD(char.gpf_liquid_residual_saturation) + '\n')
                if char.gpf_gas_residual_saturation or char.gpf_gas_residual_saturation == 0:
                    outfile.write('   GAS_RESIDUAL_SATURATION ' + strD(char.gpf_gas_residual_saturation) + '\n')
                outfile.write('  / ' + '\n')

            outfile.write('END\n\n')

    def _read_region(self, infile, line):

        region = pregion()
        region.coordinates_lower = [None] * 3
        region.coordinates_upper = [None] * 3

        region.name = self.splitter(line).lower()

        keep_reading = True
        while keep_reading:  # Read through all cards
            line = infile.readline()  # get next line
            key = line.strip().split()[0].lower()  # take first keyword
            if key == 'coordinates':
                keep_reading_2 = True
                while keep_reading_2:
                    line1 = infile.readline()
                    region.coordinates_lower[0] = floatD(line1.split()[0])
                    region.coordinates_lower[1] = floatD(line1.split()[1])
                    region.coordinates_lower[2] = floatD(line1.split()[2])
                    line2 = infile.readline()
                    region.coordinates_upper[0] = floatD(line2.split()[0])
                    region.coordinates_upper[1] = floatD(line2.split()[1])
                    region.coordinates_upper[2] = floatD(line2.split()[2])
                    line3 = infile.readline()
                    if line3.strip().split()[0].lower() in ['/', 'end']:
                        keep_reading_2 = False
            elif key == 'face':
                region.face = line.strip().split()[-1].lower()
            elif key == 'coordinate':
                line1 = line.split()[1::]
                point = ppoint()
                point.name = region.name
                point.coordinate[0] = floatD(line1[0])
                point.coordinate[1] = floatD(line1[1])
                point.coordinate[2] = floatD(line1[2])
                region.point_list.append(point)
            elif key in ['/', 'end']:
                keep_reading = False

        self.add(region)

    def _add_region(self, region=pregion(), overwrite=False):  # Adds a Region object.
        # check if region already exists
        if isinstance(region, pregion):
            if region.name in self.region.keys():
                if not overwrite:
                    warning = 'WARNING: A region with name \'' + str(region.name) + '\' already exists. Region will' + \
                              'not be defined, use overwrite = True in add() to overwrite the old region.'
                    print warning,
                    build_warnings.append(warning)
                    return
                else:
                    self.delete(self.region[region.name])

        if region not in self.regionlist:
            self.regionlist.append(region)

    def _delete_region(self, region=pregion()):
        self.regionlist.remove(region)

    def _write_region(self, outfile):
        self._header(outfile, headers['region'])

        # Write out all valid region object entries with Region as Key word
        for region in self.regionlist:
            outfile.write('REGION ')
            outfile.write(region.name.lower() + '\n')
            if region.face:
                outfile.write('  FACE ' + region.face.lower() + '\n')
            # no if statement below to ensure 0's are accepted for coordinates
            if region.point_list:
                for point in region.point_list:
                    outfile.write('  COORDINATE ')
                    for i in range(3):
                        outfile.write(strD(point.coordinate[i]) + ' ')
                    outfile.write('\n')
            else:
                outfile.write('  COORDINATES\n')
                outfile.write('    ')
                for i in range(3):
                    outfile.write(strD(region.coordinates_lower[i]) + ' ')
                outfile.write('\n    ')
                for i in range(3):
                    outfile.write(strD(region.coordinates_upper[i]) + ' ')
                outfile.write('\n')
                outfile.write('  END\n')
            outfile.write('END\n\n')

    def _read_observation(self, infile):
        observation = pobservation()

        keep_reading = True

        while keep_reading:
            line = infile.readline()  # get next line
            key = line.strip().split()[0].lower()  # take first keyword
            if key == 'region':
                observation.region = self.splitter(line)
            elif key in ['/', 'end']:
                keep_reading = False

        self.observation_list.append(observation)

    def _add_observation(self, observation=pobservation(), overwrite=False):  # Adds a Observation object.
        # check if observation already exists
        if isinstance(observation, pobservation):
            if observation.region in self.observation.keys():
                if not overwrite:
                    warning = 'WARNING: A observation with region \'' + str(observation.region) + '\' already ' \
                                                                                                  'exists. Observation will not be defined, use overwrite = True in add() to overwrite' \
                                                                                                  'the old observation.'
                    print warning,
                    build_warnings.append(warning)
                    return
                else:
                    self.delete(self.observation[observation.region])

        if observation not in self.observation_list:
            self.observation_list.append(observation)

    def _delete_observation(self, observation=pobservation()):
        self.observation_list.remove(observation)

    def _write_observation(self, outfile):
        self._header(outfile, headers['observation'])

        for observation in self.observation_list:
            outfile.write('OBSERVATION\n')
            if observation.region:
                outfile.write('  REGION ' + observation.region.lower() + '\n')
            outfile.write('END\n\n')

    def _read_flow(self, infile, line):
        flow = pflow()
        flow.datum = []
        flow.varlist = []
        flow.datum_type = ''
        flow.name = self.splitter(line).lower()  # Flow Condition name passed in.

        keep_reading = True
        is_valid = False  # Used so that entries outside flow conditions are ignored
        end_count = 0
        total_end_count = 1
        while keep_reading:  # Read through all cards

            line = infile.readline()  # get next line
            key = line.strip().split()[0].lower()  # take first keyword
            if key == 'type':
                total_end_count = 2  # Basically ensures that both read ifs for
                # the varlist will execute
                # This # indicates how many time a / or 'end'
                # can be read before loop terminates.

            elif key == 'rate' or key == 'pressure' or key == 'temperature' or key == 'concentration' or key == \
                    'enthalpy' or key == 'flux':
                if end_count == 0:
                    '''
                    Appending and instantiation of new flow_variables occur here. Only two entries are filled,
                    the rest are assigned in the elif code block where end_count == 1
                    '''
                    var = pflow_variable()
                    var.name = key
                    var.type = line.strip().split()[-1].lower()
                    var.valuelist = []
                    var.list = []

                    is_valid = True  # Indicates the entries read here should be written so that entries outside
                    # flow conditions are ignored.
                    flow.varlist.append(var)

                elif end_count == 1:
                    tstring2name = line.strip().split()[0]  # Assigns the 1st word on a line
                    tstring2 = line.split()[1:]  # Assigns the rest of the line
                    # #2 used because this is the 2nd reading of the variables

                    # Deterine if variable is a list or stand-alone
                    if tstring2[0].lower() == 'list':  # Executes only if 2nd word on line == 'list'

                        # for each list in a pflow_variable object, check all
                        # pflow_variable objects by name to determine correct assignment
                        # before assigning in values from a list
                        keep_reading_list = True
                        while keep_reading_list:

                            line = infile.readline()  # get next line
                            tstring2 = line.split()[:]  # split the whole string/line
                            for var in flow.varlist:  # var represents a pflow_variable object
                                if tstring2name.lower() == var.name.lower():
                                    if line[0] == ':' or line[0] == '#' or line[0] == '/':
                                        pass  # ignore a commented line
                                    # line[0] == '/' is a temporary fix
                                    elif tstring2[0].lower() == 'time_units':
                                        var.time_unit_type = tstring2[1]
                                    elif tstring2[0].lower() == 'data_units':
                                        var.data_unit_type = tstring2[1]
                                    elif line.split()[0] in ['/', 'end']:
                                        keep_reading_list = False
                                    else:
                                        tvarlist = pflow_variable_list()
                                        tvarlist.time_unit_value = floatD(tstring2[0])
                                        tvarlist.data_unit_value_list = []
                                        tvarlist.data_unit_value_list.append(floatD(tstring2[1]))
                                        if len(tstring2) > 2:
                                            tvarlist.data_unit_value_list.append(floatD(tstring2[2]))
                                        var.list.append(tvarlist)
                            if line.split()[0] in ['/', 'end']:
                                keep_reading_list = False
                    else:
                        # for each single variable in a pflow_variable object, check all
                        # pflow_variable object by name to determine correct assignment
                        for substring in tstring2:  # Checks all values/types on this line
                            for var in flow.varlist:  # var represents a pflow_variable object
                                if tstring2name.lower() == var.name.lower():
                                    try:
                                        var.valuelist.append(floatD(substring))
                                    # If a string (e.g., C for temp.), assign to unit
                                    except ValueError:
                                        var.unit = substring
            elif key == 'iphase':
                flow.iphase = int(self.splitter(line))
            elif key == 'sync_timestep_with_update':
                flow.sync_timestep_with_update = True
            elif key == 'datum':
                # Assign file_name with list of d_dx, d_dy, d_dz values.
                if line.strip().split()[1].upper() == 'FILE':
                    flow.datum_type = 'file'
                    flow.datum = line.split()[1]
                if line.strip().split()[1].upper() == '':
                    flow.datum_type = 'DATASET'
                    flow.datum = line.split()[1]
                # Assign d_dx, d_dy, d_dz values
                else:
                    temp_list = [floatD(line.split()[1]), floatD(line.split()[2]), floatD(line.split()[3])]
                    flow.datum.append(temp_list)

            # Detect if there is carriage return after '/' or 'end' to end loop
            # Alternative method of count implemented by Satish
            elif key in ['/', 'end']:
                end_count = end_count + 1
                if end_count == total_end_count:
                    keep_reading = False

        if is_valid:
            self.add(flow)

    def _add_flow(self, flow=pflow(), overwrite=False):  # Adds a Flow object.
        # check if flow already exists
        if isinstance(flow, pflow):
            if flow.name in self.flow.keys():
                if not overwrite:
                    warning = 'WARNING: A flow with name \'' + str(flow.name) + '\' already exists. Flow will not be ' +\
                              'defined, use overwrite = True in add() to overwrite the old flow.'
                    print warning,
                    build_warnings.append(warning)
                    return
                else:  # Executes if overwrite = True
                    self.delete(self.flow[flow.name])

        if flow not in self.flowlist:
            self.flowlist.append(flow)

    def _delete_flow(self, flow=pflow()):
        self.flowlist.remove(flow)

    '''
    Automate adding the sub-class flow_variable to a flow object. The flow object
    can be specified by name. If flow object name is not specified, the function
    will append pflow_variable to the last flow object added to the list.
    Function will provided a warning if a flow_variable.name already exists
    in the flow object it is trying to add it to.
    '''

    def _add_flow_variable(self, flow_variable=pflow_variable(), index='', overwrite=False):
        # check if flow.name was specified
        if index:
            if isinstance(index, str):
                flow = self.flow.get(index)  # Assign flow object to existing flow object with string type name/index
                if not flow:  # Occurs if index/string is not found in flow object
                    print 'WARNING: a flow object with flow.name', index, 'was not found. Current found entries are:', \
                        self.flow.keys(), 'pflow_variable was not added.\n'
                    return

            elif isinstance(index, pflow):
                flow = index  # Assigns if index is the flow object reference
        else:  # Set flow to last flow object in list
            flow = self.flowlist[-1]

        # check if flow_variable already exists
        if isinstance(flow_variable, pflow_variable):
            if flow_variable.name in self.flow_variable(flow).keys():
                if not overwrite:
                    warning = 'WARNING: A flow_variable with name \'' + str(
                        flow_variable.name) + '\' already exists in flow with name \'' + str(
                        flow.name) + '\'. Flow_variable will not be defined, use overwrite = True in add() to ' \
                                     'overwrite the old flow_variable. Use flow=\'name\' if you want to specify the ' \
                                     'flow object to add flow_variable to.'
                    print warning,
                    build_warnings.append(warning)
                    return
                else:  # Executes if overwrite = True
                    self.delete(self.flow_variable(flow)[flow_variable.name], flow)

        # Add flow_variable to flow (as a sub-class) if flow_variable does
        # not exist in specified flow object
        if flow_variable not in flow.varlist:
            flow.varlist.append(flow_variable)

    def _delete_flow_variable(self, flow_variable=pflow_variable(), flow=pflow()):
        flow.varlist.remove(flow_variable)

    def _write_flow(self, outfile):
        self._header(outfile, headers['flow_condition'])

        # Function is used to determine which flow_condition type allowed list
        # to check depending on the flow_condition name specified.
        # Also does the work of writing or error reporting
        def check_condition_type(condition_name, condition_type):
            if condition_name.upper() == 'PRESSURE':
                if condition_type.lower() in pressure_types_allowed:
                    outfile.write(condition_type.lower())
                else:
                    print '       valid flow_condition pressure_types_allowed:', pressure_types_allowed, '\n'
                    raise PyFLOTRAN_ERROR('flow.varlist.type: \'' + condition_type + '\' is invalid.')
                return 0  # Break out of function
            elif condition_name.upper() == 'FLUX':
                if condition_type.lower() in flux_types_allowed:
                    outfile.write(condition_type.lower())
                else:
                    print '       valid flow_condition flux_types_allowed:', flux_types_allowed, '\n'
                    raise PyFLOTRAN_ERROR('flow.varlist.type: \'' + condition_type + '\' is invalid.')
                return 0  # Break out of function
            elif condition_name.upper() == 'RATE':
                if condition_type.lower() in rate_types_allowed:
                    outfile.write(condition_type.lower())
                else:
                    print '       valid flow_condition rate_types_allowed:', rate_types_allowed, '\n'
                    raise PyFLOTRAN_ERROR('flow.varlist.type: \'' + condition_type + '\' is invalid.')
                return 0  # Break out of function
            elif condition_name.upper() == 'WELL':
                if condition_type.lower() in well_types_allowed:
                    outfile.write(condition_type.lower())
                else:
                    print '       valid well_conditions well_types_allowed:', well_types_allowed, '\n'
                    raise PyFLOTRAN_ERROR('flow.varlist.type: \'' + condition_type + '\' is invalid.')
                return 0  # Break out of function
            elif condition_name.upper() == 'TEMPERATURE':
                if condition_type.lower() in temperature_types_allowed:
                    outfile.write(condition_type.lower())
                else:
                    print '       valid flow_condition temperature_types_allowed:', temperature_types_allowed, '\n'
                    raise PyFLOTRAN_ERROR('flow.varlist.type: \'' + condition_type + '\' is invalid.')
                return 0  # Break out of function
            elif condition_name.upper() == 'CONCENTRATION':
                if condition_type.lower() in concentration_types_allowed:
                    outfile.write(condition_type.lower())
                else:
                    print '       valid flow_condition concentration_types_allowed:', concentration_types_allowed, '\n'
                    raise PyFLOTRAN_ERROR('flow.varlist.type: \'' + condition_type + '\' is invalid.')
                return 0  # Break out of function
            elif condition_name.upper() == 'SATURATION':
                if condition_type.lower() in saturation_types_allowed:
                    outfile.write(condition_type.lower())
                else:
                    print 'PyFLOTRAN ERROR: flow.varlist.type: \'' + condition_type + '\' is invalid.'
                    print '       valid flow_condition saturation_types_allowed:', saturation_types_allowed, '\n'
                return 0  # Break out of function
            elif condition_name.upper() == 'ENTHALPY':
                if condition_type.lower() in enthalpy_types_allowed:
                    outfile.write(condition_type.lower())
                else:
                    print '       valid flow_condition enthalpy_types_allowed:', enthalpy_types_allowed, '\n'
                    raise PyFLOTRAN_ERROR('flow.varlist.type: \'' + condition_type + '\' is invalid.')
                return 0  # Break out of function
            else:
                pass  # Error reporting for flow_condition.name is done elsewhere
                # name should be validated before this function is called.

        # Write out all valid flow_conditions objects with FLOW_CONDITION as keyword
        for flow in self.flowlist:
            outfile.write('FLOW_CONDITION  ' + flow.name.lower() + '\n')

            if flow.sync_timestep_with_update:
                outfile.write('  SYNC_TIMESTEP_WITH_UPDATE\n')

            if flow.datum:  # error-checking not yet added
                outfile.write('  DATUM')

                if isinstance(flow.datum, str):
                    if flow.datum_type == 'file':
                        outfile.write(' FILE ')
                    if flow.datum_type == 'dataset':
                        outfile.write(' DATASET ')
                    outfile.write(flow.datum)
                else:  # Applies if datum is a list of [d_dx, d_dy, d_dz]
                    # write out d_dx, d_dy, d_dz
                    outfile.write(' ')
                    outfile.write(strD(flow.datum[0][0]) + ' ')
                    outfile.write(strD(flow.datum[0][1]) + ' ')
                    outfile.write(strD(flow.datum[0][2]))
                outfile.write('\n')
                if flow.gradient:
                    outfile.write('    GRADIENT\n')
                    outfile.write('      ' + flow.gradient[0].upper() + ' ' + str(flow.gradient[1]) + ' ' + str(flow.gradient[2]) + ' ' +  str(flow.gradient[3]) + '\n')
                    outfile.write('    /\n')
            outfile.write('  TYPE\n')  # Following code is paired w/ this statement.
            # variable name and type from lists go here
            for a_flow in flow.varlist:
                if a_flow.name.upper() in flow_condition_type_names_allowed:
                    outfile.write('    ' + a_flow.name.upper() + '  ')
                else:
                    print '       valid flow_condition.names:', flow_condition_type_names_allowed, '\n'
                    raise PyFLOTRAN_ERROR('flow.varlist.name: \'' + a_flow.name + '\' is invalid.')

                # Checks a_flow.type and performs write or error reporting
                check_condition_type(a_flow.name, a_flow.type)
                outfile.write('\n')

            outfile.write('  END\n')
            if flow.iphase:
                outfile.write('  IPHASE ' + str(flow.iphase) + '\n')

            # variable name and values from lists along with units go here
            for a_flow in flow.varlist:
                if a_flow.valuelist:
                    outfile.write('    ' + a_flow.name.upper())
                    if isinstance(a_flow.valuelist[0], str):
                        if a_flow.valuelist[0] == 'file':
                            outfile.write(' FILE ' + a_flow.valuelist[1])
                        else:
                            outfile.write(' DATASET ' + a_flow.valuelist[0])
                    else:
                        for flow_val in a_flow.valuelist:
                            outfile.write(' ' + strD(flow_val))
                    if a_flow.unit:
                        outfile.write(' ' + a_flow.unit.lower())
                    outfile.write('\n')
                elif a_flow.list:
                    outfile.write('    ' + a_flow.name.upper() + ' LIST' + '\n')
                    if a_flow.time_unit_type:
                        outfile.write('      TIME_UNITS ' + a_flow.time_unit_type + '\n')
                    if a_flow.data_unit_type:
                        outfile.write('      DATA_UNITS ' + a_flow.data_unit_type + '\n')
                    for k in a_flow.list:
                        outfile.write('        ' + strD(k.time_unit_value))
                        for p in range(len(k.data_unit_value_list)):
                            outfile.write('  ' + strD(k.data_unit_value_list[p]))
                        outfile.write('\n')
                    outfile.write('    /\n')

            outfile.write('END\n\n')

    def _read_initial_condition(self, infile, line):
        if len(line.split()) > 1:
            np_name = self.splitter(line).lower()  # Flow Condition name passed in.
        else:
            np_name = None
        p = pinitial_condition()
        np_flow = p.flow
        np_transport = p.transport
        np_region = p.region

        keep_reading = True

        while keep_reading:  # Read through all cards
            line = infile.readline()  # get next line
            key = line.strip().split()[0].lower()  # take first  key word

            if key == 'flow_condition':
                np_flow = self.splitter(line)
            elif key == 'transport_condition':
                np_transport = self.splitter(line)
            elif key == 'region':
                np_region = self.splitter(line)
            elif key in ['/', 'end']:
                keep_reading = False

        # Create an empty initial condition and assign the values read in
        new_initial_condition = pinitial_condition(np_flow, np_transport, np_region, np_name)
        self.add(new_initial_condition)

    def _add_initial_condition(self, initial_condition=pinitial_condition(),
                               overwrite=False):  # Adds a initial_condition object.
        # check if flow already exists
        if isinstance(initial_condition, pinitial_condition):
            if initial_condition.region in self.initial_condition.keys():
                if not overwrite:
                    warning = 'WARNING: A initial_condition with region \'' + str(
                        initial_condition.region) + '\' already exists. initial_condition will not be defined, use ' \
                                                    'overwrite = True in add() to overwrite the old initial_condition.'
                    print warning,
                    build_warnings.append(warning)
                    return
                else:
                    self.delete(self.initial_condition[initial_condition.region])

        if initial_condition not in self.initial_condition_list:
            self.initial_condition_list.append(initial_condition)

    def _delete_initial_condition(self, initial_condition=pinitial_condition()):
        self.initial_condition_list.remove(initial_condition)

    def _write_initial_condition(self, outfile):
        self._header(outfile, headers['initial_condition'])
        # Write all initial conditions to file
        try:
            for b in self.initial_condition_list:  # b = initial_condition
                if b.name:
                    outfile.write('INITIAL_CONDITION ' + b.name.lower() + '\n')
                else:
                    raise PyFLOTRAN_ERROR('Give a name for initial condition!')
                if b.flow:
                    outfile.write('  FLOW_CONDITION ' + b.flow.lower() + '\n')
                if b.transport:
                    outfile.write('  TRANSPORT_CONDITION ' + b.transport.lower() + '\n')
                if b.region:
                    outfile.write('  REGION ' + b.region.lower() + '\n')
                else:
                    raise PyFLOTRAN_ERROR('initial_condition.region is required')
                outfile.write('END\n\n')
        except:
            raise PyFLOTRAN_ERROR('At least one initial condition with valid attributes is required')

    def _read_boundary_condition(self, infile, line):
        if len(line.split()) > 1:
            np_name = self.splitter(line).lower()  # Flow Condition name passed in.
        else:
            np_name = None
        p = pboundary_condition('')
        np_flow = p.flow
        np_transport = p.transport
        np_region = p.region

        keep_reading = True

        while keep_reading:  # Read through all cards
            line = infile.readline()  # get next line
            key = line.split()[0].lower()  # take first key word

            if key == 'flow_condition':
                np_flow = self.splitter(line)  # take last word
            elif key == 'transport_condition':
                np_transport = self.splitter(line)
            elif key == 'region':
                np_region = self.splitter(line)
            elif key in ['/', 'end']:
                keep_reading = False

        # Create an empty boundary condition and assign the values read in
        new_boundary_condition = pboundary_condition(np_name, np_flow, np_transport, np_region)
        self.add(new_boundary_condition)

    def _add_boundary_condition(self, boundary_condition=pboundary_condition(),
                                overwrite=False):  # Adds a boundary_condition object.
        # check if flow already exists
        if isinstance(boundary_condition, pboundary_condition):
            if boundary_condition.region in self.boundary_condition.keys():
                if not overwrite:
                    warning = 'WARNING: A boundary_condition with region \'' + str(boundary_condition.region) + '\'' + \
                              ' already exists. boundary_condition will not be defined, use overwrite = True in add()' + \
                              'to overwrite the old boundary_condition.'
                    print warning,
                    build_warnings.append(warning)
                    return
                else:
                    self.delete(self.boundary_condition[boundary_condition.region])

        if boundary_condition not in self.boundary_condition_list:
            self.boundary_condition_list.append(boundary_condition)

    def _delete_boundary_condition(self, boundary_condition=pboundary_condition()):
        self.boundary_condition_list.remove(boundary_condition)

    def _write_boundary_condition(self, outfile):
        self._header(outfile, headers['boundary_condition'])

        # Write all boundary conditions to file
        try:
            for b in self.boundary_condition_list:  # b = boundary_condition
                if b.name:
                    outfile.write('BOUNDARY_CONDITION ' + b.name.lower() + '\n')
                else:
                    raise PyFLOTRAN_ERROR('Give a name for boundary condition!')
                if b.flow:
                    outfile.write('  FLOW_CONDITION ' + b.flow.lower() + '\n')
                if b.transport:
                    outfile.write('  TRANSPORT_CONDITION ' + b.transport.lower() + '\n')
                if b.region:
                    outfile.write('  REGION ' + b.region.lower() + '\n')
                outfile.write('END\n\n')
        except:
            raise PyFLOTRAN_ERROR('At least one boundary_condition with valid attributes is required')

    def _read_source_sink(self, infile, line):
        p = psource_sink()
        np_flow = p.flow
        np_region = p.region

        if len(line.split()) > 1:
            np_name = self.splitter(line).lower()  # Flow Condition name passed in.
        else:
            np_name = None

        keep_reading = True

        while keep_reading:  # Read through all cards
            line = infile.readline()  # get next line
            key = line.strip().split()[0].lower()  # take first key word

            if key == 'flow_condition':
                np_flow = self.splitter(line)  # take last word
            elif key == 'region':
                np_region = self.splitter(line)
            elif key in ['/', 'end']:
                keep_reading = False

        # Create an empty source sink and assign the values read in
        new_source_sink = psource_sink(np_flow, np_region, np_name)
        self.add(new_source_sink)

    def _add_source_sink(self, source_sink=psource_sink(), overwrite=False):  # Adds a source_sink object.
        # check if flow already exists
        if isinstance(source_sink, psource_sink):
            if source_sink.region in self.source_sink.keys():
                if not overwrite:
                    warning = 'WARNING: A source_sink with region \'' + str(
                        source_sink.region) + '\' already exists. source_sink will not be defined, use overwrite = True ' \
                                              'in add() to overwrite the old source_sink.'
                    print warning,
                    build_warnings.append(warning)
                    return
                else:
                    self.delete(self.source_sink[source_sink.region])

        if source_sink not in self.source_sink_list:
            self.source_sink_list.append(source_sink)

    def _delete_source_sink(self, source_sink=psource_sink()):
        self.source_sink_list.remove(source_sink)

    def _write_source_sink(self, outfile):
        self._header(outfile, headers['source_sink'])

        # Write all source_sink conditions to file
        for b in self.source_sink_list:  # b = source_sink
            if b.name:
                outfile.write('SOURCE_SINK ' + b.name.lower() + '\n')
            else:
                outfile.write('SOURCE_SINK\n')
            if b.flow:
                outfile.write('  FLOW_CONDITION ' + b.flow.lower() + '\n')
            if b.transport:
                outfile.write('  TRANSPORT_CONDITION ' + b.transport.lower() + '\n')
            if b.region:
                outfile.write('  REGION ' + b.region.lower() + '\n')
            else:
                raise PyFLOTRAN_ERROR('source_sink.region is required')
            outfile.write('END\n\n')

    def _delete_strata(self, strata=pstrata()):
        self.strata_list.remove(strata)

    def _read_strata(self, infile):
        strata = pstrata()
        keep_reading = True

        while keep_reading:  # Read through all cards
            line = infile.readline()  # get next line
            key = line.strip().split()[0].lower()  # take first key word

            if key == 'region':
                strata.region = self.splitter(line)  # take last word
            elif key == 'material':
                strata.material = self.splitter(line)  # take last word
            elif key in ['/', 'end']:
                keep_reading = False

        # Create an empty source sink and assign the values read in
        self.add(strata)

    def _add_strata(self, strata=pstrata(), overwrite=False):  # Adds a strata object.
        # check if stratigraphy coupler already exists
        if isinstance(strata, pstrata):
            if strata.region in self.strata.keys():
                if not overwrite:
                    warning = 'WARNING: A strata with name \'' + str(strata.region) + '\' already exists. strata will' + \
                              'not be defined, use overwrite = True in add() to overwrite the old strata.'
                    print warning,
                    build_warnings.append(warning)
                    return
                else:
                    self.delete(self.strata[strata.region])

        if strata not in self.strata_list:
            self.strata_list.append(strata)

    def _write_strata(self, outfile):
        self._header(outfile, headers['strata'])
        # strata = self.strata

        # Write out strata condition variables
        for strata in self.strata_list:

            outfile.write('STRATA\n')
            if strata.region:
                outfile.write('  REGION ' + strata.region.lower() + '\n')
            # else:
            #    raise PyFLOTRAN_ERROR('strata.region is required')
            if strata.material:
                outfile.write('  MATERIAL ' + strata.material.lower() + '\n')
            else:
                raise PyFLOTRAN_ERROR('strata.material is required')
            outfile.write('END\n\n')

    def _read_checkpoint(self, infile, line):
        checkpoint = pcheckpoint()

        checkpoint.frequency = self.splitter(line).lower()  # checkpoint int passed in.

        self.checkpoint = checkpoint

    def _write_checkpoint(self, outfile):
        self._header(outfile, headers['checkpoint'])
        checkpoint = self.checkpoint

        try:
            # error-checking to make sure checkpoint.frequency is int (integer)
            checkpoint.frequency = int(checkpoint.frequency)

            # write results
            outfile.write('CHECKPOINT ')
            outfile.write(str(checkpoint.frequency))
            outfile.write('\n')
        except ValueError:
            # write results
            outfile.write('CHECKPOINT ')
            outfile.write(str(checkpoint.frequency))
            outfile.write('\n')

            raise PyFLOTRAN_ERROR('checkpoint.frequency is not int (integer).')

        outfile.write('\n')

    def _read_restart(self, infile, line):
        restart = prestart()

        tstring = line.split()[1:]  # 1st line for restart passed in

        restart.file_name = tstring[0]
        if len(tstring) > 1:
            restart.time_value = floatD(tstring[1])
        elif len(tstring) > 2:
            restart.time_unit = tstring[2]

        self.restart = restart

    def _write_restart(self, outfile):
        self._header(outfile, headers['restart'])
        restart = self.restart

        # write file name
        outfile.write('RESTART ' + str(restart.file_name) + ' ')

        # Write time value
        if restart.time_value:
            try:
                # error-checking
                restart.time_value = floatD(restart.time_value)

                # writing
                outfile.write(strD(restart.time_value) + ' ')
            except:
                # writing
                outfile.write(strD(restart.time_value) + ' ')

                raise PyFLOTRAN_ERROR('restart.time_value is not float.')

        # Write time unit of measurement
        if restart.time_unit:
            restart.time_unit = str(restart.time_unit).lower()
            if restart.time_unit in time_units_allowed:
                outfile.write(restart.time_unit)
            else:
                outfile.write(restart.time_unit)

                raise PyFLOTRAN_ERROR('restart.time_unit \'', restart.time_unit,
                                      '\' is invalid. Valid times units are:', time_units_allowed, '\n')

        outfile.write('\n\n')

    def _read_dataset(self, infile, line):
        dataset = pdataset()
        keep_reading = True
        dataset.dataset_name = self.splitter(line)
        while keep_reading:  # Read through all cards
            line = infile.readline()  # get next line
            key = line.strip().split()[0].lower()  # take first  key word
            if key == 'dataset_mapped_name':
                dataset.dataset_mapped_name = self.splitter(line)
            elif key == 'name':
                dataset.name = self.splitter(line)
            elif key == 'filename':
                dataset.file_name = self.splitter(line)
            elif key == 'hdf5_dataset_name':
                dataset.hdf5_dataset_name = self.splitter(line)
            elif key == 'map_hdf5_dataset_name':
                dataset.map_hdf5_dataset_name = self.splitter(line)
            elif key == 'max_buffer_size':
                dataset.max_buffer_size = floatD(self.splitter(line))
            elif key == 'realization_dependent':
                dataset.realization_dependent = True
            elif key in ['/', 'end']:
                keep_reading = False

        self.add(dataset)

    def _write_dataset(self, outfile):
        self._header(outfile, headers['dataset'])
        for dataset in self.datasetlist:
            # Write out dataset variables
            if dataset.dataset_name:
                outfile.write('DATASET ' + dataset.dataset_name + '\n')
            if dataset.dataset_mapped_name:
                outfile.write('DATASET MAPPED ' + dataset.dataset_mapped_name + '\n')
            if dataset.dataset_name and dataset.dataset_mapped_name:
                raise PyFLOTRAN_ERROR('Cannot use both DATASET and DATASET MAPPED')
            if dataset.name:
                outfile.write('  NAME ' + dataset.name + '\n')
            if dataset.file_name:
                outfile.write('  FILENAME ' + dataset.file_name + '\n')
            if dataset.hdf5_dataset_name:
                outfile.write('  HDF5_DATASET_NAME ' + dataset.hdf5_dataset_name + '\n')
            if dataset.map_hdf5_dataset_name:
                outfile.write('  MAP_HDF5_DATASET_NAME ' + dataset.map_hdf5_dataset_name + '\n')
            if dataset.max_buffer_size:
                outfile.write('  MAX_BUFFER_SIZE ' + strD(dataset.max_buffer_size) + '\n')
            if dataset.realization_dependent:
                outfile.write('  REALIZATION_DEPENDENT ' + '\n')
            outfile.write('END\n\n')

    def _add_dataset(self, dat=pdataset(), overwrite=False):  # Adds a dataset object.
        # check if dataset already exists
        if isinstance(dat, pdataset):
            if dat.name in self.dataset.keys():
                if not overwrite:
                    warning = 'WARNING: A dataset with name \'' + str(
                        dat.name) + '\' already exists. Use overwrite = True in add() to overwrite the old dataset.'
                    print warning,
                    build_warnings.append(warning)
                    return
                else:  # Executes if overwrite = True
                    self.delete(self.dat[dat.name])

        if dat not in self.datasetlist:
            self.datasetlist.append(dat)

    def _delete_dataset(self, dat=pdataset()):
        self.datasetlist.remove(dat)

    def _read_chemistry(self, infile):
        chem = pchemistry()

        keep_reading = True

        while keep_reading:  # Read through all cards
            line = infile.readline()  # get next line
            try:
                key = line.strip().split()[0].lower()  # take first key word
            except IndexError:
                continue  # Read the next line if line is empty.
            if key == 'primary_species':
                while True:
                    line = infile.readline()  # get next line
                    if line.strip() in ['/', 'end']:
                        break
                    chem.pspecies_list.append(line.strip())
            elif key == 'skip':
                keep_reading_1 = True
                while keep_reading_1:
                    line1 = infile.readline()
                    if line1.strip().split()[0].lower() == 'noskip':
                        keep_reading_1 = False
            elif key == 'secondary_species':
                while True:
                    line = infile.readline()  # get next line
                    if line.strip() in ['/', 'end']:
                        break
                    chem.sec_species_list.append(line.strip())
            elif key == 'gas_species':
                while True:
                    line = infile.readline()  # get next line
                    if line.strip() in ['/', 'end']:
                        break
                    chem.gas_species_list.append(line.strip())
            elif key == 'minerals':
                while True:
                    line = infile.readline()  # get next line
                    if line.strip() in ['/', 'end']:
                        break
                    chem.minerals_list.append(line.strip())
            elif key == 'mineral_kinetics':
                while True:
                    line = infile.readline()  # get next line
                    if line.strip() in ['/', 'end']:
                        break

                    mkinetic = pchemistry_m_kinetic()  # temporary object
                    mkinetic.rate_constant_list = []

                    # assign kinetic mineral name
                    mkinetic.name = line.strip()

                    # Write mineral attributes here
                    while True:
                        line = infile.readline()  # get next line
                        if line.strip().lower() in ['/', 'end']:
                            break

                        # key is a kinetic mineral attribute here
                        key = line.strip().split()[0].lower()  # take 1st

                        tstring = line.split()[1:]  # Assigns the rest of the line

                        # assign kinetic mineral attributes
                        if key == 'rate_constant':
                            for substring in tstring:
                                try:
                                    mkinetic.rate_constant_list.append(floatD(substring))
                                except ValueError:
                                    mkinetic.rate_constant_list.append(substring)

                    chem.m_kinetics_list.append(mkinetic)  # object assigned
            elif key == 'database':
                chem.database = self.splitter(line)  # take last word
            elif key == 'log_formulation':
                chem.log_formulation = True
            elif key == 'update_porosity':
                chem.update_porosity = True
            elif key == 'update_permeability':
                chem.update_permeability = True
            elif key == 'activity_coefficients':
                chem.activity_coefficients = self.splitter(line)
            elif key == 'molal':
                chem.molal = True
            elif key == 'output':
                while True:
                    line = infile.readline()  # get next line
                    if line.strip() in ['/', 'end']:
                        break
                    chem.output_list.append(line.strip())
            elif key in ['/', 'end']:
                keep_reading = False

        # Create an empty chemistry object and assign the values read in
        self.chemistry = chem

    '''
    Automate adding the sub-class m_kinetic to chemistry object.
    Function will provide a warning if a m_kinetic.name already exists
    in the chemistry object it is trying to add it to.
    '''

    def _add_chemistry_m_kinetic(self, m_kinetic=pchemistry_m_kinetic(),
                                 overwrite=False):  # Adds a mineral_kinetic object

        chemistry = self.chemistry

        # check if m_kinetic already exists
        if isinstance(m_kinetic, pchemistry_m_kinetic):
            if m_kinetic.name in self.m_kinetic.keys():
                if not overwrite:
                    warning = 'WARNING: A m_kinetic with name \'' + str(m_kinetic.name) + '\' already exists in ' \
                                                                                          'chemistry. Mineral_Kinetic ' \
                                                                                          'will not be defined, use ' \
                                                                                          'overwrite = True in add() ' \
                                                                                          'to overwrite the old ' \
                                                                                          'm_kinetic.'
                    print warning
                    print
                    build_warnings.append(warning)
                    return  # exit function
                else:  # Executes if overwrite = True
                    self.delete(self.m_kinetic[m_kinetic.name])

        # Add m_kinetic to chemistry (as a sub-class) if that specific
        # m_kinetic does not exist in chemistry object
        if m_kinetic not in chemistry.m_kinetics_list:
            chemistry.m_kinetics_list.append(m_kinetic)

    def _delete_pchemistry_m_kinetic(self, m_kinetic=pchemistry_m_kinetic()):
        self.chemistry._m_kinetics_list.remove(m_kinetic)

    def _write_chemistry(self, outfile):
        self._header(outfile, headers['chemistry'])
        c = self.chemistry
        outfile.write('CHEMISTRY\n')

        # Write out chemistry variables
        if c.pspecies_list:
            outfile.write('  PRIMARY_SPECIES\n')
            for p in c.pspecies_list:  # p = primary_specie in primary_species_list
                outfile.write('    ' + p + '\n')
            outfile.write('  /\n')
        if c.sec_species_list:
            outfile.write('  SECONDARY_SPECIES\n')
            for s in c.sec_species_list:  # s = secondary_specie
                outfile.write('    ' + s + '\n')
            outfile.write('  /\n')
        if c.gas_species_list:
            outfile.write('  GAS_SPECIES\n')
            for g in c.gas_species_list:  # s = gas_specie
                outfile.write('    ' + g + '\n')
            outfile.write('  /\n')
        if c.minerals_list:
            outfile.write('  MINERALS\n')
            for m in c.minerals_list:  # m = mineral
                outfile.write('    ' + m + '\n')
            outfile.write('  /\n')
        if c.m_kinetics_list:
            outfile.write('  MINERAL_KINETICS\n')
            for mk in c.m_kinetics_list:  # mk = mineral_kinetics
                outfile.write('    ' + mk.name + '\n')

                if mk.rate_constant_list:
                    outfile.write('      RATE_CONSTANT ')
                for rate in mk.rate_constant_list:
                    try:
                        outfile.write(strD(rate) + ' ')
                    except TypeError:
                        outfile.write(rate + ' ')
                outfile.write('\n    /\n')  # marks end for mineral name
            outfile.write('  /\n')  # marks end for mineral_kinetics
        if c.database:
            outfile.write('  DATABASE ' + c.database + '\n')
        if c.log_formulation:
            outfile.write('  LOG_FORMULATION\n')
        if c.update_permeability:
            outfile.write('  UPDATE_PERMEABILITY\n')
        if c.update_porosity:
            outfile.write('  UPDATE_POROSITY\n')
        if c.activity_coefficients:
            outfile.write('  ACTIVITY_COEFFICIENTS ' + c.activity_coefficients.upper() + '\n')
        if c.molal:
            outfile.write('  MOLAL\n')
        if c.output_list:
            outfile.write('  OUTPUT\n')
            for o in c.output_list:  # o = output in in output_list
                outfile.write('    ' + o + '\n')
            outfile.write('  /\n')
        outfile.write('END\n\n')

    def _read_transport(self, infile, line):
        p = ptransport('')
        np_name = self.splitter(line).lower()  # Transport Condition name passed in.
        np_type = p.type
        np_constraint_list_value = []
        np_constraint_list_type = []

        keep_reading = True

        while keep_reading:  # Read through all cards
            line = infile.readline()  # get next line
            key = line.split()[0].lower()  # take first key word

            if key == 'type':
                if len(line.split()) == 2:  # Only Assign if 2 words are on the line
                    np_type = self.splitter(line)  # take last word
            elif key == 'constraint_list':
                keep_reading_2 = True
                line = infile.readline()
                while keep_reading_2:
                    try:
                        # print np_constraint_list_value,line.split()[0].lower()
                        np_constraint_list_value.append(floatD(line.split()[0].lower()))  # Read 1st word online
                        np_constraint_list_type.append(line.split()[1].lower())  # Read 2nd word on line
                    except:
                        raise PyFLOTRAN_ERROR('constraint_list_value and constraint_list_type requires at least one' +
                                              'value. Value should = Number and type should = String\n')

                    line = infile.readline()  # get next line
                    key = line.split()[0].lower()  # Used to stop loop when / or end is read
                    if key in ['/', 'end']: keep_reading_2 = False
            elif key in ['/', 'end']:
                keep_reading = False

        # Create an empty transport condition and assign the values read in
        new_transport = ptransport(np_name, np_type, np_constraint_list_value,
                                   np_constraint_list_type)
        self.add(new_transport)

    def _add_transport(self, transport=ptransport(), overwrite=False):  # Adds a transport object.
        # check if transport already exists
        if isinstance(transport, ptransport):
            if transport.name in self.transport.keys():
                if not overwrite:
                    warning = 'WARNING: A transport with name \'' + str(transport.name) + '\' already exists.' + \
                              'transport will not be defined, use overwrite = True in add() to overwrite the' + \
                              'old transport.'
                    print warning,
                    build_warnings.append(warning)
                    return
                else:
                    self.delete(self.transport[transport.name])

        if transport not in self.transportlist:
            self.transportlist.append(transport)

    def _delete_transport(self, transport=ptransport()):
        self.transportlist.remove(transport)

    def _write_transport(self, outfile):
        self._header(outfile, headers['transport_condition'])
        tl = self.transportlist
        for t in tl:  # t for transport
            if t.name:
                outfile.write('TRANSPORT_CONDITION ' + t.name.lower() + '\n')
            else:
                raise PyFLOTRAN_ERROR('transport_condition[' + str(tl.index(t)) + '].name is required.\n')
            if t.type.lower() in transport_condition_types_allowed:
                outfile.write('  TYPE ' + t.type.lower() + '\n')
            else:
                print '       valid transport_condition.types:', transport_condition_types_allowed, '\n'
                raise PyFLOTRAN_ERROR('transport.type: \'' + t.type + '\' is invalid.')
            try:
                outfile.write('  CONSTRAINT_LIST\n')

                clv = t.constraint_list_value
                clt = t.constraint_list_type

                for i, a_clv in enumerate(clv):
                    if a_clv is not None:
                        outfile.write('    ' + strD(a_clv))
                    if clt[i] is not None:
                        if i == len(clv) - 1:
                            outfile.write('  ' + str(clt[i]).lower())
                        else:
                            outfile.write('  ' + str(clt[i]).lower() + '\n')
                    else:
                        raise PyFLOTRAN_ERROR('transport[' + str(tl.index(t)) + '].constraint_list_type[' +
                                              str(clt.index(i)) + '] is required to have a value when'
                                                                  ' transport.constraint_list_value does.')
            except:
                raise PyFLOTRAN_ERROR('transport.constraint_list_value and transport.constraint_list_type should be' +
                                      'in list format, be equal in length, and have at least one value.\n')
            outfile.write('\n  END\n')  # END FOR CONSTRAINT_LIST
            outfile.write('END\n\n')  # END FOR TRANSPORT_CONDITION

    def _read_constraint(self, infile, line):
        constraint = pconstraint()
        constraint.name = self.splitter(line).lower()  # constraint name passed in.
        constraint.concentration_list = []
        constraint.mineral_list = []

        keep_reading = True

        while keep_reading:  # Read through all cards
            line = infile.readline()  # get next line
            key = line.split()[0].lower()  # take first key word

            if key == 'concentrations':
                while True:
                    line = infile.readline()  # get next line
                    tstring = line.split()  # Convert line to temporary list of strings

                    if line.strip().lower() in ['/', 'end']:
                        break  # Stop loop if line is a / or 'end'

                    concentrations = pconstraint_concentration()

                    # Assign concentrations - one line
                    try:
                        concentrations.pspecies = tstring[0]
                        concentrations.value = floatD(tstring[1])
                        concentrations.constraint = tstring[2]
                        concentrations.element = tstring[3]
                    except IndexError:
                        pass  # No assigning is done if a value doesn't exist while being read in.
                    constraint.concentration_list.append(concentrations)

            elif key == 'minerals':
                while True:
                    line = infile.readline()
                    tstring = line.split()
                    if line.strip().lower() in ['/', 'end']:
                        break
                    mineral = pconstraint_mineral()

                    try:
                        mineral.name = tstring[0]
                        if tstring[1].lower() == 'dataset':
                            mineral.volume_fraction = tstring[2]
                            if tstring[3].lower() == 'dataset':
                                mineral.surface_area = tstring[4]
                            else:
                                mineral.surface_area = floatD(tstring[3])
                        else:
                            mineral.volume_fraction = floatD(tstring[1])
                            if tstring[2].lower() == 'dataset':
                                mineral.surface_area = tstring[3]
                            else:
                                mineral.surface_area = floatD(tstring[2])

                    except IndexError:
                        pass  # No assigning is done if a value doesn't exist while being read in.

                    constraint.mineral_list.append(mineral)

            elif key in ['/', 'end']:
                keep_reading = False

        self.add(constraint)

    def _add_constraint(self, constraint=pconstraint(), overwrite=False):  # Adds a constraint object.
        # check if constraint already exists
        if isinstance(constraint, pconstraint):
            if constraint.name in self.constraint.keys():
                if not overwrite:
                    warning = 'WARNING: A constraint with name \'' + str(constraint.name) + '\' already exists. ' + \
                              'constraint will not be defined, use overwrite = True in add() to overwrite the old ' + \
                              'constraint.'
                    print warning,
                    build_warnings.append(warning)
                    return
                else:
                    self.delete(self.constraint[constraint.name])

        if constraint not in self.constraint_list:
            self.constraint_list.append(constraint)

    def _delete_constraint(self, constraint=pconstraint()):
        self.constraint_list.remove(constraint)

    # Adds a constraint_concentration object
    def _add_constraint_concentration(self, constraint_concentration=pconstraint_concentration(), index='',
                                      overwrite=False):

        # check if constraint.name was specified
        if index:
            if isinstance(index, str):
                constraint = self.constraint.get(index)  # Assign constraint object to existing constraint object with string type name/index
                if not constraint:  # Occurs if index/string is not found in constraint object
                    print 'WARNING: a constraint object with constraint.name', index, 'was not found. Current found ' \
                                                                                      'entries are:', \
                        self.constraint.keys(), 'pconstraint_concentration was not added.\n'
                    return

            elif isinstance(index, pconstraint):
                constraint = index  # Assigns if index is the constraint object reference
        else:  # Set constraint to last constraint object in list
            constraint = self.constraint_list[-1]

        # check if constraint_concentration already exists
        if isinstance(constraint_concentration, pconstraint_concentration):
            if constraint_concentration.pspecies in self.constraint_concentration(constraint).keys():
                if not overwrite:
                    warning = 'WARNING: A constraint_concentration with pspecies \'' + str(
                        constraint_concentration.pspecies) + '\' already exists in constraint with name \'' + str(
                        constraint.name) + '\'. constraint_concentration will not be defined, use overwrite = True in ' \
                                           'add() to overwrite the old constraint_concentration. Use constraint=\'name\' ' \
                                           'if you want to specify the constraint object to ' \
                                           'add constraint_concentration to.'
                    print warning,
                    build_warnings.append(warning)
                    return
                else:  # Executes if overwrite = True
                    self.delete(self.constraint_concentration(constraint)[constraint_concentration.pspecies],
                                constraint)

        # Add constraint_concentration to constraint (as a sub-class) if constraint_concentration does not exist in
        # specified constraint object
        if constraint_concentration not in constraint.concentration_list:
            constraint.concentration_list.append(constraint_concentration)

    def _delete_constraint_concentration(self,
                                         constraint_concentration=pconstraint_concentration(),
                                         constraint=pconstraint()):
        constraint.concentration_list.remove(constraint_concentration)

    def _write_constraint(self, outfile):
        self._header(outfile, headers['constraint'])
        cl = self.constraint_list

        for c in cl:  # c = constraint, cl = constraint_list
            if c.name:
                outfile.write('CONSTRAINT ' + c.name.lower() + '\n')
            else:
                raise PyFLOTRAN_ERROR('constraint_list[' + str(cl.index(c)) + '].name is required.')

            outfile.write('  CONCENTRATIONS\n')

            for concn in c.concentration_list:  # concn = concentration, c = constraint
                if concn.pspecies:
                    outfile.write('    ' + concn.pspecies)
                if concn.value:
                    outfile.write('  ' + strD(concn.value))
                else:
                    raise PyFLOTRAN_ERROR('invalid concentration value!')
                if concn.constraint:
                    outfile.write('  ' + concn.constraint)
                if concn.element:
                    outfile.write('  ' + concn.element)
                outfile.write('\n')

            outfile.write('  /\n')  # END for concentrations
            if c.mineral_list:
                outfile.write('  MINERALS\n')
                for mineral in c.mineral_list:
                    if mineral.name:
                        outfile.write('    ' + mineral.name)
                    if type(mineral.volume_fraction) is str:
                        outfile.write('  ' + 'DATASET ' + mineral.volume_fraction)
                    else:
                        outfile.write('  ' + strD(mineral.volume_fraction))
                    if type(mineral.surface_area) is str:
                        outfile.write('  ' + 'DATASET ' + mineral.surface_area)
                    else:
                        outfile.write('  ' + strD(mineral.surface_area))
                    outfile.write('\n')
                outfile.write('  /\n')  # END for concentrations
            outfile.write('END\n\n')  # END for constraint

    def _header(self, outfile, header):
        if not header:
            return
        ws = '# '
        pad = int(np.floor((80 - len(header) - 4) / 2))
        for i in range(pad):
            ws += '='
        ws += ' ' + header + ' '
        for i in range(pad):
            ws += '='
        ws += '\n'

    def _write_hydroquake(self, outfile):
        self._header(outfile, headers['hydroquake'])
        outfile.write('HYDROQUAKE\n')
        if self.hydroquake.mapping_file:
            outfile.write('  HYDROQUAKE_MAPPING_FILE ' + self.hydroquake.mapping_file + '\n')
        if self.hydroquake.time_scaling:
            outfile.write('  TIME_SCALING ' + strD(self.hydroquake.time_scaling) + '\n')
        if self.hydroquake.pressure_scaling:
            outfile.write('  PRESSURE_SCALING ' + strD(self.hydroquake.pressure_scaling) + '\n')
        outfile.write('END_HYDROQUAKE')

    @property
    def prop(self):
        return dict([(p.id, p) for p in self.proplist] + [(p.id, p) for p in self.proplist])

    @property
    def dataset(self):
        return dict([(p.dataset_name, p) for p in self.datasetlist])

    @property
    def saturation(self):
        return dict([(p.name, p) for p in self.saturationlist])

    @property
    def lsolver(self):
        return dict([lsolv.name, lsolv] for lsolv in self.lsolverlist if lsolv.name)

    @property
    def nsolver(self):
        return dict([nsolv.name, nsolv] for nsolv in self.nsolverlist if nsolv.name)

    @property
    def char(self):
        return dict(
            [(characteristic_curves.name.lower(), characteristic_curves) for characteristic_curves in self.charlist] +
            [(characteristic_curves.name.lower(), characteristic_curves) for characteristic_curves in self.charlist])

    @property
    def region(self):
        return dict([region.name.lower(), region] for region in self.regionlist if region.name)

    @property
    def observation(self):
        return dict(
            [observation.region.lower(), observation] for observation in self.observation_list if observation.region)

    @property
    def flow(self):
        return dict([flow.name.lower(), flow] for flow in self.flowlist if flow.name.lower)

    def flow_variable(self, flow=pflow()):
        return dict([flow_variable.name.lower(), flow_variable] for flow_variable in flow.varlist
                    if flow_variable.name.lower())

    @property
    def initial_condition(self):
        return dict([initial_condition.region, initial_condition] for initial_condition in self.initial_condition_list
                    if initial_condition.region)

    @property
    def boundary_condition(self):
        return dict(
            [boundary_condition.region, boundary_condition] for boundary_condition in self.boundary_condition_list if
            boundary_condition.region)

    @property
    def source_sink(self):
        return dict([source_sink.region, source_sink] for source_sink in self.source_sink_list if source_sink.region)

    @property
    def strata(self):
        return dict([strata.region, strata] for strata in self.strata_list if strata.region)

    @property
    def m_kinetic(self):
        chemistry = self.chemistry
        return dict([m_kinetic.name, m_kinetic] for m_kinetic in chemistry.m_kinetics_list if m_kinetic.name)

    @property
    def transport(self):
        return dict([transport.name, transport] for transport in self.transportlist if transport.name)

    @property
    def constraint(self):
        return dict(
            [constraint.name.lower(), constraint] for constraint in self.constraint_list if constraint.name.lower())

    def constraint_concentration(self, constraint=pconstraint()):
        return dict([constraint_concentration.pspecies, constraint_concentration] for constraint_concentration in
                    constraint.concentration_list if constraint_concentration.pspecies)

    @staticmethod
    def paraview(vtk_filepath_list=None):
        if vtk_filepath_list is not None:
            imports = 'from paraview import simple'
            legacy_reader = ''
            for vtk_filepath in vtk_filepath_list:
                if not os.path.isfile(vtk_filepath):
                    raise PyFLOTRAN_ERROR(vtk_filepath + ' is not a valid filepath!')
                elif vtk_filepath[-3:] != 'vtk':
                    raise PyFLOTRAN_ERROR(vtk_filepath + ' does not have a valid extension (.vtk)!')
            legacy_reader += 'simple.LegacyVTKReader(FileNames=' + str(vtk_filepath_list).replace(' ', '\n') + ')\n'
            with open('paraview-script.py', 'w+') as f:
                f.write(imports + '\n')
                f.write(legacy_reader + '\n')
                f.write('simple.Show()\nsimple.Render()')
        process = subprocess.Popen('paraview --script=' + os.path.dirname(vtk_filepath_list[0]) + '/paraview-script.py',
                                   shell=True, stdout=subprocess.PIPE, stderr=sys.stderr)
        while True:
                out = process.stdout.read(1)
                if out == '' and process.poll() is not None:
                    break
                if out != '':
                    sys.stdout.write(out)
                    sys.stdout.flush()


class pquake(Frozen):
    """
    Class for specifying pflotran-qk3 related information

    :param name: Specify name of the physics for which the linear solver is
     being defined. Options include: 'tran', 'transport','flow'.
    :type name: str
    :param solver: Specify solver type: Options include: 'solver', 'krylov_type', 'krylov', 'ksp', 'ksp_type'
    :type solver: str
    :param preconditioner: Specify preconditioner type: Options include: 'ilu'
    :type solver: str
    """

    def __init__(self, mapping_file='mapping.dat', time_scaling=1.0, pressure_scaling=1.0):
        self.mapping_file = mapping_file
        self.time_scaling = time_scaling
        self.pressure_scaling = pressure_scaling
        self._freeze()
