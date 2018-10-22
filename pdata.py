from copy import copy
import subprocess
import itertools as it
from matplotlib import rc
import os
import sys
from ptool import *
from pdflt import *
import math
import shutil
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
  import h5py
  __h5_on__ = True
  __h5_idx__ = 2
except ImportError:
  PyFLOTRAN_WARNING("Could not import h5py. DBASE parsing is unsupported.")
  __h5_on__ = False

""" Class for pyflotran data """

"""
PyFLOTRAN v1.0.0 LA-CC-14-094

Copyright (c) 2014, Los Alamos National Security, LLC.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the
following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

__author__ = "Satish Karra, Cory Kitay"
__version__ = "1.0.0"
__maintainer__ = "Satish Karra"
__email__ = "satkarra@lanl.gov"

__enable_hdf5__ = True

if __enable_hdf5__:
  import h5py

try:
    pflotran_dir = os.environ['PFLOTRAN_DIR']
except KeyError:
    print('PFLOTRAN_DIR must point to PFLOTRAN installation' +
          'directory and be defined in system environment variables.')
    sys.exit(1)

rc('text', usetex=True)
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Gill Sans']

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
time_units_allowed = ['s', 'sec', 'm', 'min', 'h', 'hr', 'd', 'day', 'w',
                      'week', 'mo', 'month', 'y', 'yr']
solver_names_allowed = ['transport', 'tran', 'flow']  # newton and linear
# simulation type - allowed strings
simulation_types_allowed = ['subsurface', 'surface_subsurface', 'hydroquake',
                            'geomechanics_subsurface', 'hydrogeophysics']
# mode - allowed strings
mode_names_allowed = ['richards', 'mphase', 'mph', 'flash2', 'th no_freezing',
                      'th freezing', 'immis', 'general', 'th', 'thc',
                      'toil_ims', 'todd_longstaff', 'solvent_tl',
                      'wipp_flow', 'richards_ts']

# checkpoint - allowed formats
checkpoint_formats_allowed = ['hdf5', 'binary']

gradient_types_allowed = ['pressure', 'temperature']

# grid - allowed strings
grid_types_allowed = ['structured', 'unstructured_explicit',
                      'unstructured_implicit', 'unstructured',
                      'unstructured_polyhedra']

# cartesian is default in pflotran
grid_symmetry_types_allowed = ['cartesian', 'cylindrical', 'spherical', '']

# output - allowed strings
output_formats_allowed = ['TECPLOT BLOCK', 'TECPLOT POINT', 'HDF5',
                          'HDF5 MULTIPLE_FILES', 'MAD', 'VTK']

velocity_units_allowed = ['m/s', 'm/yr', 'cm/s', 'cm/yr', 'm/y']

pvt_types_allowed = ['pvds','pvdg','pvco','pvdo']

output_variables_allowed = ['permeability',
                            'permeability_x',
                            'permeability_y',
                            'permeability_z',
                            'liquid_pressure',
                            'liquid_saturation',
                            'liquid_density',
                            'liquid_head',
                            'liquid_mobility',
                            'liquid_energy',
                            'liquid_mole_fractions',
                            'liquid_mass_fractions',
                            'gas_pressure',
                            'gas_saturation',
                            'gas_density',
                            'gas_mobility',
                            'gas_energy',
                            'gas_mole_fractions',
                            'gas_mass_fractions',
                            'air_pressure',
                            'capillary_pressure',
                            'vapor_pressure',
                            'saturation_pressure',
                            'thermodynamic_state',
                            'temperature',
                            'residual',
                            'porosity',
                            'effective_porosity',
                            'tortuosity',
                            'mineral_porosity',
                            'maximum_pressure',
                            'oil_pressure',
                            'oil_saturation',
                            'oil_density',
                            'oil_mobility',
                            'oil_energy',
                            'soil_compressibility',
                            'soil_reference_pressure',
                            'process_id',
                            'volume',
                            'material_id',
                            'natural_id',
                            'material_id_kludge_for_visit']


# saturation_function - allowed strings
saturation_function_types_allowed = ['VAN_GENUCHTEN', 'BROOKS_COREY',
                                     'THOMEER_COREY', 'NMT_EXP',
                                     'PRUESS_1']
lower_list = [sat.lower() for sat in saturation_function_types_allowed]

saturation_function_types_allowed = \
    list(set(saturation_function_types_allowed +
             lower_list))

permeability_function_types_allowed = ['VAN_GENUCHTEN', 'MUALEM', 'BURDINE',
                                       'NMT_EXP', 'PRUESS_1']

# characteristic_curves - allowed strings - saturation & permeability functions
characteristic_curves_saturation_function_types_allowed = ['VAN_GENUCHTEN',
                                                           'BROOKS_COREY',
                                                           'LINEAR',
                                                           'CONSTANT',
                                                           'BRAGFLO_KRP9',
                                                           'BRAGFLO_KRP4',
                                                           'SATURATION_FUNCTION',
                                                           'BRAGFLO_KRP12',
                                                           'BRAGFLO_KRP11']
lower_list = [sat.lower() for sat in
              characteristic_curves_saturation_function_types_allowed]

characteristic_curves_saturation_function_types_allowed = list(set(
    characteristic_curves_saturation_function_types_allowed + lower_list))

characteristic_curves_gas_permeability_function_types_allowed = [
    'MUALEM_VG_GAS', 'BURDINE_BC_GAS']

lower_list = [sat.lower() for sat in
              characteristic_curves_gas_permeability_function_types_allowed]

characteristic_curves_gas_permeability_function_types_allowed = list(set(
    characteristic_curves_gas_permeability_function_types_allowed +
    lower_list))

characteristic_curves_liquid_permeability_function_types_allowed = [
    'MUALEM', 'BURDINE', 'MUALEM_VG_LIQ', 'MUALEM_BC_LIQ',
    'TOUGH2_LINEAR_OIL', 'BURDINE_BC_LIQ', 'BURDINE_LINEAR_LIQ',
    'BRAGFLO_KRP9_LIQ', 'BRAGFLO_KRP4_LIQ', 'MOD_BC_OIL', 
    'BURDINE_VG_LIQ', 'BRAGFLO_KRP12_LIQ', 'BRAGFLO_KRP11_LIQ']

lower_list = [sat.lower() for sat in
              characteristic_curves_liquid_permeability_function_types_allowed]

characteristic_curves_liquid_permeability_function_types_allowed = list(set(
    characteristic_curves_liquid_permeability_function_types_allowed +
    lower_list))

allowed_compressibility_functions = ['linear_model', 'bandis', 'turner']

allowed_soil_compressibility_functions = ['LINEAR', 'LEIJNSE', 'DEFAULT',
                                          'BRAGFLO', 'WIPP', 'QUADRATIC',
                                          'BANDIS']

flow_condition_type_names_allowed = ['PRESSURE', 'RATE', 'FLUX', 'TEMPERATURE',
                                     'CONCENTRATION', 'SATURATION', 'WELL',
                                     'ENTHALPY',
                                     'LIQUID_PRESSURE', 'GAS_PRESSURE',
                                     'LIQUID_SATURATION', 'GAS_SATURATION',
                                     'MOLE_FRACTION', 'RELATIVE_HUMIDITY',
                                     'LIQUID_FLUX', 'GAS_FLUX', 'ENERGY_FLUX']

geomech_condition_type_names_allowed = ['DISPLACEMENT_X', 'DISPLACEMENT_Y',
                                        'DISPLACEMENT_Z', 'FORCE_X', 'FORCE_Y',
                                        'FORCE_Z']

pressure_types_allowed = ['dirichlet', 'heterogeneous_dirichlet',
                          'hydrostatic', 'zero_gradient', 'conductance',
                          'seepage', 'heterogeneous_conductance',
                          'heterogeneous_seepage',
                          'neumann', 'surface_dirichlet', 'surface_spillover',
                          'surface_zero_gradheight',
                          'heterogeneous_surface_seepage']

rate_types_allowed = ['mass_rate', 'volumetric_rate', 'scaled_volumetric_rate',
                      'scaled_mass_rate', 'heterogeneous_mass_rate']

scaling_options_allowed = ['perm', 'volume', 'neighbor_perm']

well_types_allowed = ['well']

flux_types_allowed = ['dirichlet', 'neumann', 'mass_rate', 'hydrostatic',
                      'conductance', 'zero_gradient',
                      'production_well', 'seepage', 'volumetric',
                      'volumetric_rate', 'equilibrium']

temperature_types_allowed = ['dirichlet', 'hydrostatic', 'zero_gradient',
                             'neumann', 'heterogeneous_dirichlet']

concentration_types_allowed = ['dirichlet', 'hydrostatic', 'zero_gradient']

saturation_types_allowed = ['dirichlet']

mole_fraction_types_allowed = ['dirichlet']

liquid_pressure_types_allowed = pressure_types_allowed

gas_pressure_types_allowed = ['dirichlet']

liquid_flux_types_allowed = ['neumann']

gas_flux_types_allowed = ['neumann']

gas_saturation_types_allowed = ['dirichlet']

enthalpy_types_allowed = ['dirichlet',
                          'hydrostatic', 'zero_gradient', 'temperature']

transport_condition_types_allowed = ['dirichlet', 'dirichlet_zero_gradient',
                                     'equilibrium', 'neumann', 'mole',
                                     'mole_rate', 'zero_gradient']

geomech_subsurface_coupling_types_allowed = ['two_way_coupled',
                                             'one_way_coupled']

eos_fluid_names_allowed = ['water', 'gas', 'oil', 'solvent']

eos_density_types_allowed = ['constant', 'exponential', 'default', 'ideal',
                             'rks', 'batzle_and_wang', 'linear',
                             'trangenstein', 'inverse_linear', 'pr_methane',
                             'duanmo', 'bragflo']

eos_enthalpy_types_allowed = ['constant', 'ideal', 'default', 'linear_temp',
                              'quadratic_temp']

eos_viscosity_types_allowed = ['constant', 'default', 'batzle_and_wang',
                               'quadratic', 'grabowski']

eos_henrys_types_allowed = ['constant', 'default', 'duanmao']

cards = ['co2_database', 'uniform_velocity', 'nonuniform_velocity',
         'simulation', 'regression', 'restart',
         'dataset', 'chemistry', 'grid', 'timestepper', 'material_property',
         'time', 'linear_solver', 'newton_solver',
         'output', 'fluid_property', 'saturation_function',
         'characteristic_curves', 'region', 'observation',
         'flow_condition', 'transport_condition', 'initial_condition',
         'boundary_condition', 'source_sink', 'strata',
         'constraint', 'hydroquake', 'multiple_continuum',
         'secondary_continuum', 'geomechanics', 'geomechanics_regression',
         'geomechanics_grid', 'geomechanics_subsurface_coupling',
         'geomechanics_time', 'geomechanics_region', 'geomechanics_condition',
         'geomechanics_boundary_condition', 'geomechanics_strata',
         'geomechanics_time', 'geomechanics_material_property',
         'geomechanics_output', 'eos', 'integral_flux', 'ufd_decay',
         'ufd_biosphere', 'source_sink_sandbox', 'waste_form_general',
         'wipp_source_sink']

headers = ['co2 database path', 'uniform velocity', 'nonuniform velocity',
           'simulation', 'regression',
           'restart', 'dataset', 'chemistry', 'grid', 'time stepping',
           'material properties', 'time', 'linear solver',
           'newton solver', 'output', 'fluid properties',
           'saturation functions', 'characteristic curves', 'regions',
           'observation', 'flow conditions', 'transport conditions',
           'initial condition', 'boundary conditions',
           'source sink', 'stratigraphy couplers', 'constraints',
           'hydroquake', 'multiple continuum',
           'secondary continuum', 'geomechanics', 'geomechanics regression',
           'geomechanics grid', 'geomechanics subsurface coupling',
           'geomechanics time', 'geomechanics region',
           'geomechanics condition', 'geomechanics boundary condition',
           'geomechanics strata', 'geomechanics time',
           'geomechanics material property', 'geomechanics output', 'eos',
           'integral_flux', 'ufd decay', 'ufd biosphere', 'source_sink_sandbox',
           'waste form general','wipp source sink']

read_cards = ['co2_database', 'uniform_velocity', 'nonuniform_velocity',
              'simulation', 'regression',
              'dataset', 'chemistry', 'grid', 'timestepper',
              'material_property',
              'time', 'linear_solver', 'newton_solver',
              'output', 'fluid_property', 'saturation_function',
              'characteristic_curves', 'region', 'observation',
              'flow_condition', 'transport_condition', 'initial_condition',
              'boundary_condition', 'source_sink', 'strata',
              'constraint', 'geomechanics_regression', 'geomechanics_grid',
              'geomechanics_subsurface_coupling', 'geomechanics_time',
              'geomechanics_region', 'geomechanics_condition',
              'geomechanics_boundary_condition', 'geomechanics_strata',
              'geomechanics_material_property', 'geomechanics_output',
              'eos','specified_velocity','reference_liquid_density',
              'minimum_hydrostatic_pressure','update_flow_permeability',
              'ufd_decay', 'ufd_biosphere', 'source_sink_sandbox',
              'waste_form_general','wipp_source_sink']

headers = dict(zip(cards, headers))

build_warnings = []

class Coeff():
    '''
    Class to store values and their corresponding units and types.
    '''
    def __init__(self,value,unit=None,dtype=None):
        self._value = value
        self._unit = unit
        
        if dtype is None:
            dtype = type(value)
            
        self._dtype = dtype

    def __str__(self):
        if isinstance(self._value,list):
            _tmp = ' '.join(strD(_x) for _x in self._value)
        else:
            _tmp = strD(self._value)
        
        if self._unit:
            return '%s %s' % (_tmp,self._unit)
        return '%s' % (_tmp)
    
    def __add__(self,other):
        
        if isinstance(self._value,list):
            self._value = [_x+other for _x in self._value]
            return [self._dtype(_x) for _x in self._value]
        else:
            self._value += float(other)
            return self._dtype(self._value)
    
    def __radd__(self, other):
            return self.__add__(other)
    
    def __sub__(self,other):
        
        if isinstance(self._value,list):
            self._value = [_x-other for _x in self._value]
            return [self._dtype(_x) for _x in self._value]
        else:
            self._value -= float(other)
            return self._dtype(self._value)
        
    def __rsub__(self,other):
        
        if isinstance(self._value,list):
            self._value = [other-_x for _x in self._value]
            return [self._dtype(_x) for _x in self._value]
        else:
            self._value = float(other) - self._value
            return self._dtype(self._value)

class pspecified_velocity(Frozen):
    '''
    Defines Darcy flow velocities to be used when there is no flow process model employed.

    :param uniform: Specifies whether Darcy velocities are uniform or not.
    :type uniform: bool
    :param dataset: Darcy velocities (defined with list, file, or singlely)
    :type dataset: list or np.ndarray
    :param time_units: Units for time [T].
    :type time_units: str
    :param data_units: Units for Darcy velocity [L/T].
    :type data_units: str
    :param interpolation: Type of interpolation [LINEAR, STEP].
    :type interpolation: str
    '''

    def __init__(self,uniform=True,dataset=None,time_units=None,data_units=None,
                 interpolation=None):
        self.uniform = uniform
        self.dataset = dataset
        self.time_units = time_units
        self.data_units = data_units
        self.interpolation = interpolation
        self._freeze()

class puniform_velocity(Frozen):
    """
    Class for specifiying uniform velocity with transport. Optional with
    transport problem when not coupling with
    any flow mode.  If not specified, assumes diffusion transport only.

    :param value_list: List of variables of uniform_velocity. First 3
     variables are vlx, vly, vlz in unit [m/s]. 4th variable specifies
     unit. e.g., [14.4e0, 0.e0, 0.e0, 'm/yr']
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
    with transport problem when not coupling with any flow mode.
    If not specified, assumes diffusion transport only.

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
    :param characteristic_curves: Unique identifier of material
     characteristic curves
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
    :param permeability: Permeability of material property. Input is a list
     of 3 floats. Uses diagonal permeability in
     unit order: k_xx [m^2], k_yy [m^2], k_zz [m^2].
     e.g., [1.e-15,1.e-15,1.e-17].
    :type permeability: [float]*3
    :param longitudinal_dispersivity: Longitudinal dispersion coefficient
    :type longitudinal_dispersivity: float
    :param transverse_dispersivity_h: Transverse dispersion coefficient
     horizontal
    :type transverse_dispersivity_h: float
    :param transverse_dispersivity_v: Transverse dispersion coefficient
     vertical
    :type transverse_dispersivity_v: float
    :param anisotropic: Turn this on if permeability is anisotropic
    :type anisotropic: Bool
    :param soil_compressibility_function: Selects the
     compressibility function if geomechanics is not use (e.g. LEIJNSE)
    :type soil_compressibility_function: String
    :param soil_compressibility: Compressibility of soil matrix (Pa^-1)
    :type soil_compressibility: Float
    :param compressibility_function: Inserts the GEOMECHANICS_SUBSURFACE_PROPS
     keyword and points to a compressibility function (e.g. 'LINEAR_MODEL',
     'BANDIS', 'TURNER')
    :type compressibility_function: String
    :param bandis_A: Used with GEOMECHANICS_SUBSURFACE_PROPS
     keyword and BANDIS model.
    :type bandis_A: float
    :param bandis_B: Used with GEOMECHANICS_SUBSURFACE_PROPS
     keyword and BANDIS model.
    :type bandis_B: float
    :param maximum_aperture: Used with GEOMECHANICS_SUBSURFACE_PROPS
     keyword and BANDIS model.
    :type maximum_aperture: float
    :param normal_vector: Defines fracture normal vector for
     GEOMECHANICS_SUBSURFACE_PROPS keyword and BANDIS model.
    :type normal_vector: [float, float, float]
    :type A_Bandis: float
    :param A_Bandis: Used with SOIL_COMPRESSIBILITY_FUNCTION BANDIS in
     MATERIAL_PROPERTY
    :type B_Bandis: float
    :param B_Bandis: Used with SOIL_COMPRESSIBILITY_FUNCTION BANDIS in
     MATERIAL_PROPERTY
    :type b_min_Bandis: float
    :param b_min_Bandis: Used with SOIL_COMPRESSIBILITY_FUNCTION BANDIS in
     MATERIAL_PROPERTY
    :type b_max_Bandis: float
    :param b_max_Bandis: Used with SOIL_COMPRESSIBILITY_FUNCTION BANDIS in
     MATERIAL_PROPERTY
    :type sigma_x_Bandis: float
    :param sigma_x_Bandis: Used with SOIL_COMPRESSIBILITY_FUNCTION BANDIS in
     MATERIAL_PROPERTY
    :type normal_x_Bandis: float
    :param normal_x_Bandis: Used with SOIL_COMPRESSIBILITY_FUNCTION BANDIS in
     MATERIAL_PROPERTY
    :type friction_Bandis: float
    :param friction_Bandis: Used with SOIL_COMPRESSIBILITY_FUNCTION BANDIS in
     MATERIAL_PROPERTY
    :type perm_multiplier_Bandis: float
    :param perm_multiplier_Bandis: Used with SOIL_COMPRESSIBILITY_FUNCTION
     BANDIS in MATERIAL_PROPERTY
    :type frac_present_Bandis: float
    :param frac_present_Bandis: Used with SOIL_COMPRESSIBILITY_FUNCTION
     BANDIS in MATERIAL_PROPERTY
    :type b_p_Bandis: float
    :param b_p_Bandis: Used with SOIL_COMPRESSIBILITY_FUNCTION BANDIS in
     MATERIAL_PROPERTY

    """

    # definitions are put on one line to work better with rst/latex/sphinx.
    def __init__(self, id=1, name='default', characteristic_curves='default',
                 porosity=0.25, tortuosity=1.0, density=None,
                 specific_heat=None, cond_dry=None,
                 cond_wet=None, saturation='', permeability=[1e-15],
                 permeability_power='', permeability_critical_porosity='',
                 permeability_min_scale_factor='',
                 longitudinal_dispersivity='', transverse_dispersivity_h='',
                 transverse_dispersivity_v='',
                 secondary_continuum='',
                 soil_compressibility_function='',
                 soil_reference_pressure=None,
                 soil_compressibility=None, compressibility_function='',
                 bandis_A=None, bandis_B=None, maximum_aperture=None,
                 normal_vector=None, B_Bandis=None,
                 A_Bandis=None, b_min_Bandis=None, b_max_Bandis=None,
                 sigma_n_Bandis=None, sigma_x_Bandis=None, sigma_y_Bandis=None,
                 sigma_z_Bandis=None, normal_x_Bandis=None,
                 normal_y_Bandis=None,
                 normal_z_Bandis=None, friction_Bandis=None,
                 perm_multiplier_Bandis=None,
                 frac_present_Bandis=None,
                 b_p_Bandis=None, density_unit='', cond_wet_unit='',
                 cond_dry_unit='', specific_heat_unit='',
                 heat_capacity='', heat_capacity_unit='',
                 perm_factor=None,
                 tortuosity_function_of_porosity=None,
                 inactive=False,permeability_anisotropic=None,
                 permeability_isotropic=None):

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
        self.permeability_isotropic = permeability_isotropic
        self.permeability_anisotropic = permeability_anisotropic
        self.longitudinal_dispersivity = longitudinal_dispersivity
        self.transverse_dispersivity_h = transverse_dispersivity_h
        self.transverse_dispersivity_v = transverse_dispersivity_v
        self.secondary_continuum = secondary_continuum
        self.soil_compressibility_function = soil_compressibility_function
        self.soil_reference_pressure = soil_reference_pressure
        self.soil_compressibility = soil_compressibility
        self.compressibility_function = compressibility_function
        self.perm_factor = perm_factor
        self.tortuosity_function_of_porosity = tortuosity_function_of_porosity
        self.inactive = inactive

        if bandis_A is None:
            bandis_A = []
        if bandis_B is None:
            bandis_B = []
        if maximum_aperture is None:
            maximum_aperture = []
        if normal_vector is None:
            normal_vector = []

        if B_Bandis is None:
            B_Bandis = []
        self.B_Bandis = B_Bandis
        if A_Bandis is None:
            A_Bandis = []
        self.A_Bandis = A_Bandis
        if b_max_Bandis is None:
            b_max_Bandis = []
        self.b_max_Bandis = b_max_Bandis
        if b_min_Bandis is None:
            b_min_Bandis = []
        self.b_min_Bandis = b_min_Bandis
        if sigma_n_Bandis is None:
            sigma_n_Bandis = []
        self.sigma_n_Bandis = sigma_n_Bandis

        if friction_Bandis is None:
            friction_Bandis = []
        self.friction_Bandis = friction_Bandis

        if perm_multiplier_Bandis is None:
            perm_multiplier_Bandis = []
        self.perm_multiplier_Bandis = perm_multiplier_Bandis

        if sigma_x_Bandis is None:
            sigma_x_Bandis = []
        self.sigma_x_Bandis = sigma_x_Bandis
        if sigma_y_Bandis is None:
            sigma_y_Bandis = []
        self.sigma_y_Bandis = sigma_y_Bandis
        if sigma_z_Bandis is None:
            sigma_z_Bandis = []
        self.sigma_z_Bandis = sigma_z_Bandis

        if normal_x_Bandis is None:
            normal_x_Bandis = []
        self.normal_x_Bandis = normal_x_Bandis
        if normal_y_Bandis is None:
            normal_y_Bandis = []
        self.normal_y_Bandis = normal_y_Bandis
        if normal_z_Bandis is None:
            normal_z_Bandis = []
        self.normal_z_Bandis = normal_z_Bandis

        if frac_present_Bandis is None:
            frac_present_Bandis = []
        self.frac_present_Bandis = frac_present_Bandis
        if b_p_Bandis is None:
            b_p_Bandis = []
        self.b_p_Bandis = b_p_Bandis

        self.bandis_A = bandis_A
        self.bandis_B = bandis_B
        self.maximum_aperture = maximum_aperture
        self.normal_vector = normal_vector
        self.density_unit = density_unit
        self.cond_wet_unit = cond_wet_unit
        self.cond_dry_unit = cond_dry_unit
        self.specific_heat_unit = specific_heat_unit
        self.heat_capacity_unit = heat_capacity_unit
        self.heat_capacity = heat_capacity
        self._freeze()

    class perm_factor(Frozen):
        def __init__(self,min_pressure=None,max_pressure=None,
                     max_permfactor=None):
            self.min_pressure = min_pressure
            self.max_pressure = max_pressure
            self.max_permfactor = max_permfactor
            self._freeze()

    class secondary_continuum(Frozen):
        def __init__(self,continuum_type=None,log_grid_spacing=False,
                     radius=None,num_cells=None,outer_spacing=None,
                     epsilon=None,temperature=None,porosity=None):

            self.continuum_type = continuum_type
            self.log_grid_spacing = log_grid_spacing
            self.radius = radius
            self.num_cells = num_cells
            self.outer_spacing = outer_spacing
            self.epsilon = epsilon
            self.temperature = temperature
            self.porosity = porosity

        def _write(self,outfile):
            outfile.write('  SECONDARY_CONTINUUM\n')
            if self.continuum_type:
                outfile.write('    CONTINUUM_TYPE %s\n' % self.continuum_type)
            if self.log_grid_spacing:
                outfile.write('    LOG_GRID_SPACING\n')
            if self.radius:
                outfile.write('    RADIUS %s\n' % strD(self.radius))
            if self.num_cells:
                outfile.write('    NUM_CELLS %s\n' % strD(self.num_cells))
            if self.outer_spacing:
                outfile.write('    OUTER_SPACING %s\n' % strD(self.outer_spacing))
            if self.epsilon:
                outfile.write('    EPSILON %s\n' % strD(self.epsilon))
            if self.temperature:
                outfile.write('    TEMPERATURE %s\n' % strD(self.temperature))
            if self.porosity:
                outfile.write('    POROSITY %s\n' % strD(self.porosity))
            outfile.write('  /\n')

class psecondary_continuum(Frozen):
    """
    Class for defining a secondary continuum material property.

    :param type: Type of secondary continuum material, e.g., 'nested_spheres',
     'nested_cubes', 'slab'.
    :type name: str
    :param log_spacing: Turn this on if you need log spacing
    :type log_spacing: bool
    :param outer_spacing: Specify the outer spacing for log spacing
    :type outer_spacing: float
    :param fracture_spacing: Specify the spacing between fratures in
     secondary continuum for nested cubes
    :type fracture_spacing: float
    :param num_cells: Specify number of grid cells in the secondary continuum
    :type num_cells: int
    :param epsilon: Specify the volume fraction of the secondary continuum
    :type epsilon: float
    :param aperture: Specify the aperture of the secondary continuum
    :type aperture: float
    :param temperature: Initial temperature in the secondary continuum
    :type temperature: float
    :param diffusion_coefficient: Specify the diffusion coefficient in the
     secondary continuum for transport
    :type diffusion_coefficient: float
    :param porosity: Specify the porosity of the secondary continuum
    :type porosity: float


    """

    # definitions are put on one line to work better with rst/latex/sphinx.
    def __init__(self, id=None, type=None, log_spacing=False,
                 outer_spacing=None, fracture_spacing=None, num_cells=None,
                 epsilon=None, temperature=None, diffusion_coefficient=None,
                 porosity=None, aperture=None):
        self.id = id
        self.type = type
        self.log_spacing = log_spacing
        self.outer_spacing = outer_spacing
        self.fracture_spacing = fracture_spacing
        self.num_cells = num_cells
        self.epsilon = epsilon
        self.aperture = aperture
        self.temperature = temperature
        self.diffusion_coefficient = diffusion_coefficient
        self.porosity = porosity

        self._freeze()


class ptime(Frozen):
    """
    Class for time. Used to specify final time of simulation,
    initial timestep size, maximum timestep size (throughout the
    simulation or a particular instant of time). Time values and
    units need to be specified. Acceptable time units are: (s, m, h, d, mo, y).

    :param tf: final tim. 1st variable is time value. 2nd variable specifies
     time unit. e.g., [0.25e0, 'y']
    :type tf: [float, str]
    :param dti: delta (change) time initial a.k.a. initial timestep size.
     1st variable is time value. 2nd variable
     specifies time unit. e.g., [0.25e0, 'y']
    :type dti: [float, str]
    :param dtf: delta (change) time final a.k.a. maximum timestep size.
     1st variable is time value. 2nd variable
     specifies time unit. e.g., [50.e0, 'y']
    :type dtf: [float, str]
    :param dtf_list: delta (change) time starting at a given time instant.
     Input is a list that can have multiple lists
     appended to it. e.g., time.dtf_list.append([1.e2, 's', 5.e3, 's'])
    :type dtf_list: [ [float, str, float, str] ]
    :param steady_state: Run as steady state.
    :type steady_state: Bool
    """

    # definitions are put on one line to work better with rst/latex/sphinx.
    def __init__(self, tf=[1.0, 'd'], dti=[1e-3, 'd'], dtf=[1.0, 'd'],
                 steady_state=False, dtf_list=None):
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
    Class for defining a grid. Used to define type, resolution and
    geometry of the grid

    :param type: Grid type. Valid entries include: 'structured',
     'unstructured'.
    :type type: str
    :param symmetry_type: Grid symmetry type. Valid entries include:
     'cartesian' and 'cylindrical'.  Blank entries are equivalent
     to cartesian.  Only used for structured grids.  When 'cylindrical'
     is used in conjunction with the lower_bounds and upper_bounds keywords,
     the first entry is for radius, the third is for z, and the second is
     a non-used dummy value.
    :type type: str
    :param lower_bounds: Lower/Minimum 3D boundaries coordinates in
     order of x_min, y_min, z_min. Input is a list of 3
     floats. e.g., [0.e0, 0.e0, 0.e0].
    :type lower_bounds: [float]*3
    :param upper_bounds: Upper/Maximum 3D boundaries coordinates in
     order of x_max, y_max, z_max. Input is a list of 3
     floats. e.g., [321.e0, 1.e0, 51.e0].
    :type lower_bounds: [float]*3
    :param origin: Coordinates of grid origin. Optional. Input is a list
     of 3 floats. Default: [0.e0, 0.e0, 0.e0].
    :type origin: [float]*3
    :param nxyz: Number of grid cells in x,y,z directions. Only works with
     type='structured'. Input is a list of 3
     floats. e.g., [107, 1, 51].  Default is [10,10,10]
    :type nxyz: [float]*3
    :param dx: Specifies grid spacing of structured cartesian grid in the
     x-direction. e.g., [0.1, 0.2, 0.3, 0.4, 1, 1,
     1, 1].
    :type dx: [float]
    :param dy: Specifies grid spacing of structured cartesian grid in the
     y-direction.
    :type dy: [float]
    :param dz: Specifies grid spacing of structured cartesian grid in the
     z-direction
    :type dz: [float]
    :param gravity: Specifies gravity vector in m/s^2. Input is a list of
     3 floats.
    :type gravity: [float]*3
    :param filename: Specify name of file containing grid information.
     Only works with type='unstructured'.
    :type filename: str
    """

    # definitions are put on one line to work better with rst/latex/sphinx.
    def __init__(self, type='structured', symmetry_type='cartesian',
                 lower_bounds=None, upper_bounds=None,
                 origin=None, nxyz=None, dx=None, dy=None, dz=None,
                 gravity=None, filename=''):
        if dx is None:
            if lower_bounds is None:
                lower_bounds = [None, None, None]
            if upper_bounds is None:
                upper_bounds = [None, None, None]
            if nxyz is None:
                nxyz = [None, None, None]
        else:
            nxyz = [len(dx), len(dy), len(dz)]
        if origin is None:
            origin = []
        if dx is None:
            dx = []
        if dy is None:
            dy = []
        if dz is None:
            dz = []
        if gravity is None:
            gravity = []

        self.type = type
        self.symmetry_type = symmetry_type
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.origin = origin
        self.nxyz = nxyz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.gravity = gravity
        self.filename = filename
        self._nodelist = []
        self._celllist = []
        self._connectivity = []
        self._parent = None
        self._path = ppath(parent=self)
        self._freeze()
        self._parent = None
        self.read_grid()

    @property
    def xmin(self):
        if self.type == 'structured':
            if self.lower_bounds[0] is not None:
                return self.lower_bounds[0]
            else:
                raise PyFLOTRAN_ERROR('grid lower bounds not set')
        elif self.type == 'unstructured_explicit':
            return min([cell[0] for cell in self._celllist])
        else:
            print("property xmin not implemented for"
                  " unstructured_implicit type!")

    @property
    def ymin(self):
        if self.type == 'structured':
            if self.lower_bounds[1] is not None:
                return self.lower_bounds[1]
            else:
                raise PyFLOTRAN_ERROR('grid lower bounds not set')
        elif self.type == 'unstructured_explicit':
            return min([cell[1] for cell in self._celllist])
        else:
            print("property ymin not implemented for"
                  " unstructured_implicit type!")

    @property
    def zmin(self):
        if self.type == 'structured':
            if self.lower_bounds[2] is not None:
                return self.lower_bounds[2]
            else:
                raise PyFLOTRAN_ERROR('grid lower bounds not set')
        elif self.type == 'unstructured_explicit':
            return min([cell[2] for cell in self._celllist])
        else:
            print("property zmin not implemented for"
                  " unstructured_implicit type!")

    @property
    def xmax(self):
        if self.type == 'structured':
            if self.upper_bounds[0] is not None:
                return self.upper_bounds[0]
            else:
                raise PyFLOTRAN_ERROR('grid upper bounds not set')
        elif self.type == 'unstructured_explicit':
            return max([cell[0] for cell in self._celllist])
        else:
            print("property xmax not implemented for"
                  " unstructured_implicit type!")

    @property
    def ymax(self):
        if self.type == 'structured':
            if self.upper_bounds[1] is not None:
                return self.upper_bounds[1]
            else:
                raise PyFLOTRAN_ERROR('grid upper bounds not set')
        elif self.type == 'unstructured_explicit':
            return max([cell[1] for cell in self._celllist])
        else:
            print("property ymax not implemented for"
                  " unstructured_implicit type!")

    @property
    def zmax(self):
        if self.type == 'structured':
            if self.upper_bounds[2] is not None:
                return self.upper_bounds[2]
            else:
                raise PyFLOTRAN_ERROR('grid upper bounds not set')
        elif self.type == 'unstructured_explicit':
            return max([cell[2] for cell in self._celllist])
        else:
            print("property zmax not implemented for"
                  " unstructured_implicit type!")

    @property
    def nodelist(self):
        if self._nodelist == []:
            if self.type == 'structured':
                nx = self.nxyz[0]
                ny = self.nxyz[1]
                nz = self.nxyz[2]

                x_vert = np.linspace(self.xmin, self.xmax, num=nx + 1)
                y_vert = np.linspace(self.ymin, self.ymax, num=ny + 1)
                z_vert = np.linspace(self.zmin, self.zmax, num=nz + 1)

                nodes = [(i, j, k)
                         for k in z_vert for j in y_vert for i in x_vert]
                self._nodelist = nodes
            else:
                print("pgrid nodelist not implemented for unstructured yet!")
        return self._nodelist

    @nodelist.setter
    def nodelist(self, value):
        self._nodelist = value

    @property
    def connectivity(self):
        if self.type == 'unstructured_explicit':
            return self._connectivity
        else:
            print("property connectivity only implemented fo"
                  " unstructured_explicit type!")
            return

    @property
    def celllist(self):
        if self.type == 'structured':
            nx = self.nxyz[0]
            ny = self.nxyz[1]
            nz = self.nxyz[2]

            x_vert = np.linspace(self.xmin, self.xmax, num=nx + 1)
            x_cell = [np.mean([x_vert[i], x_vert[i + 1]]) for i in
                      range(len(x_vert) - 1)]
            y_vert = np.linspace(self.ymin, self.ymax, num=ny + 1)
            y_cell = [np.mean([y_vert[i], y_vert[i + 1]]) for i in
                      range(len(y_vert) - 1)]
            z_vert = np.linspace(self.zmin, self.zmax, num=nz + 1)
            z_cell = [np.mean([z_vert[i], z_vert[i + 1]]) for i in
                      range(len(z_vert) - 1)]

            cells = list(it.product(x_cell, y_cell, z_cell))
            cells = [list(cell) for cell in cells]
            self._celllist = cells
        elif self.type == 'unstructured_explicit':
            return self._celllist
            # do nothing
        else:
            print("pgrid celllist not implemented for"
                  " unstructured_implicit type yet!")
            cells = []
            self._celllist = cells
        return self._celllist

    def read_grid(self):
        if self.type == 'structured':
            return
        elif self.type == 'unstructured_explicit':
            filename = self.filename
            cells = []
            conn = []
            with open(filename, 'r') as f:
                line = get_next_line(infile)
                num_cells = int(line.split()[1])
                for i in range(num_cells):
                    line = get_next_line(infile)
                    x_cell = float(line.split()[1])
                    y_cell = float(line.split()[2])
                    z_cell = float(line.split()[3])
                    cells.append([x_cell, y_cell, z_cell])
                self._celllist = cells
                self.nxyz = num_cells
                line = get_next_line(infile)
                num_conn = int(line.split()[1])
                for i in range(num_conn):
                    line = get_next_line(infile)
                    id_up = int(line.split()[0])
                    id_dn = int(line.split()[1])
                    x_conn = float(line.split()[2])
                    y_conn = float(line.split()[3])
                    z_conn = float(line.split()[4])
                    area_conn = float(line.split()[5])
                    conn.append([id_up, id_dn, x_conn, y_conn, z_conn,
                                 area_conn])
                self._connectivity = conn
        else:
            print("read_grid not implemented for unstructured_implicit type!")
            return

    def plot(self, filename='', angle=[45, 45], color='k', connections=False,
             equal_axes=True, xlabel='x [m]', ylabel='y [m]', zlabel='z [m]',
             title='', font_size='small',
             cutaway=[]):  # generates a 3-D plot of the zone.
        """
        Generates and saves a 3-D plot of the grid.

        :param filename: Name of saved zone file.
        :type filename: str
        :param angle:   View angle of zone. First number is tilt angle
         in degrees, second number is azimuth. Alternatively, if angle
         is 'x', 'y', 'z', view is aligned along the corresponding axis.
        :type angle: [fl64,fl64], str
        :param color: Colour of zone.
        :type color: str, [fl64,fl64,fl64]
        :param connections: Plot connections. If ``True`` all connections
         plotted. If between 0 and 1, random proportion plotted.
         If greater than 1, specified number plotted.
        :type connections: bool
        :param equal_axes: Force plotting with equal aspect ratios for
         all axes.
        :type equal_axes: bool

        :param xlabel: Label on x-axis.
        :type xlabel: str
        :param ylabel: Label on y-axis.
        :type ylabel: str
        :param zlabel: Label on z-axis.
        :type zlabel: str
        :param title: Title of plot.
        :type title: str

        :param font_size: Size of text on plot.
        :type font_size: str, int

        :param cutaway: Coordinate from which cutaway begins.
         Alternatively, specifying 'middle','center' with choose
         the center of the grid as the cutaway point.
        :type cutaway: [fl64,fl64,fl64], str

        """

        if cutaway in ['middle', 'center', 'center', 'mid']:
            cutaway = [(self.xmin + self.xmax) / 2,
                       (self.ymin + self.ymax) / 2,
                       (self.zmin + self.zmax) / 2]
        if isinstance(angle, str):
            if angle == 'x':
                angle = [0, 0]
            elif angle == 'y':
                angle = [0, 90]
            elif angle == 'z':
                angle = [90, 90]
            else:
                return
            face1 = True
            face2 = True
            face3 = True
            face4 = True
            face5 = True
            face6 = True
        else:
            while angle[0] < -90:
                angle[0] += 180
            while angle[0] > 90:
                angle[0] -= 180
            while angle[1] < 0:
                angle[1] += 180
            while angle[1] > 360:
                angle[1] -= 180
            if angle[0] > 0:
                face1 = True
                face2 = False
            else:
                face1 = False
                face2 = True
            if angle[1] > 270 or angle[1] <= 90:
                face3 = True
                face4 = False
            else:
                face3 = False
                face4 = True
            if angle[1] > 0 and angle[1] <= 180:
                face5 = True
                face6 = False
            else:
                face5 = False
                face6 = True
        # plot bounding box
        plt.clf()
        fig = plt.figure(figsize=[10.5, 8.275])
        ax = plt.axes(projection='3d')
        ax.set_aspect('equal', 'datalim')

        ax.set_xlabel(xlabel, size=font_size)
        ax.set_ylabel(ylabel, size=font_size)
        ax.set_zlabel(zlabel, size=font_size)
        ax.set_title(title, size=font_size)

        for t in ax.get_xticklabels():
            t.set_fontsize(font_size)
        for t in ax.get_yticklabels():
            t.set_fontsize(font_size)
        for t in ax.get_zticklabels():
            t.set_fontsize(font_size)

        xmin, xmax = self.xmin, self.xmax
        ymin, ymax = self.ymin, self.ymax
        zmin, zmax = self.zmin, self.zmax

        if equal_axes:
            MAX = np.max([xmax - xmin, ymax - ymin, zmax - zmin]) / 2
            C = np.array([xmin + xmax, ymin + ymax, zmin + zmax]) / 2
            for direction in (-1, 1):
                for point in np.diag(direction * MAX * np.array([1, 1, 1])):
                    ax.plot([point[0] + C[0]], [point[1] + C[1]],
                            [point[2] + C[2]], 'w')
        ax.view_init(angle[0], angle[1])

        if cutaway:
            xmid, ymid, zmid = cutaway
        else:
            if face1:
                if face5:
                    if face3:
                        xmid, ymid, zmid = xmax, ymax, zmax
                    else:
                        xmid, ymid, zmid = xmin, ymax, zmax
                else:
                    if face3:
                        xmid, ymid, zmid = xmax, ymin, zmax
                    else:
                        xmid, ymid, zmid = xmin, ymin, zmax
            else:
                if face5:
                    if face3:
                        xmid, ymid, zmid = xmax, ymax, zmin
                    else:
                        xmid, ymid, zmid = xmin, ymax, zmin
                else:
                    if face3:
                        xmid, ymid, zmid = xmax, ymin, zmin
                    else:
                        xmid, ymid, zmid = xmin, ymin, zmin

        p13 = [xmid, ymid, zmid]
        if face1:
            if face5:
                if face3:
                    p1 = [xmin, ymin, zmax]
                    p2 = [xmin, ymax, zmax]
                    p3 = [xmin, ymax, zmin]
                    p4 = [xmax, ymax, zmin]
                    p5 = [xmax, ymin, zmin]
                    p6 = [xmax, ymin, zmax]
                    p7 = [xmax, ymid, zmax]
                    p8 = [xmid, ymid, zmax]
                    p9 = [xmid, ymax, zmax]
                    p10 = [xmid, ymax, zmid]
                    p11 = [xmax, ymax, zmid]
                    p12 = [xmax, ymid, zmid]
                else:
                    p1 = [xmax, ymin, zmax]
                    p2 = [xmax, ymax, zmax]
                    p3 = [xmax, ymax, zmin]
                    p4 = [xmin, ymax, zmin]
                    p5 = [xmin, ymin, zmin]
                    p6 = [xmin, ymin, zmax]
                    p7 = [xmin, ymid, zmax]
                    p8 = [xmid, ymid, zmax]
                    p9 = [xmid, ymax, zmax]
                    p10 = [xmid, ymax, zmid]
                    p11 = [xmin, ymax, zmid]
                    p12 = [xmin, ymid, zmid]
            else:
                if face3:
                    p1 = [xmin, ymax, zmax]
                    p2 = [xmin, ymin, zmax]
                    p3 = [xmin, ymin, zmin]
                    p4 = [xmax, ymin, zmin]
                    p5 = [xmax, ymax, zmin]
                    p6 = [xmax, ymax, zmax]
                    p7 = [xmax, ymid, zmax]
                    p8 = [xmid, ymid, zmax]
                    p9 = [xmid, ymin, zmax]
                    p10 = [xmid, ymin, zmid]
                    p11 = [xmax, ymin, zmid]
                    p12 = [xmax, ymid, zmid]
                else:
                    p1 = [xmax, ymax, zmax]
                    p2 = [xmax, ymin, zmax]
                    p3 = [xmax, ymin, zmin]
                    p4 = [xmin, ymin, zmin]
                    p5 = [xmin, ymax, zmin]
                    p6 = [xmin, ymax, zmax]
                    p7 = [xmin, ymid, zmax]
                    p8 = [xmid, ymid, zmax]
                    p9 = [xmid, ymin, zmax]
                    p10 = [xmid, ymin, zmid]
                    p11 = [xmin, ymin, zmid]
                    p12 = [xmin, ymid, zmid]
        else:
            if face5:
                if face3:
                    p1 = [xmin, ymin, zmin]
                    p2 = [xmin, ymax, zmin]
                    p3 = [xmin, ymax, zmax]
                    p4 = [xmax, ymax, zmax]
                    p5 = [xmax, ymin, zmax]
                    p6 = [xmax, ymin, zmin]
                    p7 = [xmax, ymid, zmin]
                    p8 = [xmid, ymid, zmin]
                    p9 = [xmid, ymax, zmin]
                    p10 = [xmid, ymax, zmid]
                    p11 = [xmax, ymax, zmid]
                    p12 = [xmax, ymid, zmid]
                else:
                    p1 = [xmax, ymin, zmin]
                    p2 = [xmax, ymax, zmin]
                    p3 = [xmax, ymax, zmax]
                    p4 = [xmin, ymax, zmax]
                    p5 = [xmin, ymin, zmax]
                    p6 = [xmin, ymin, zmin]
                    p7 = [xmin, ymid, zmin]
                    p8 = [xmid, ymid, zmin]
                    p9 = [xmid, ymax, zmin]
                    p10 = [xmid, ymax, zmid]
                    p11 = [xmin, ymax, zmid]
                    p12 = [xmin, ymid, zmid]
            else:
                if face3:
                    p1 = [xmin, ymax, zmin]
                    p2 = [xmin, ymin, zmin]
                    p3 = [xmin, ymin, zmax]
                    p4 = [xmax, ymin, zmax]
                    p5 = [xmax, ymax, zmax]
                    p6 = [xmax, ymax, zmin]
                    p7 = [xmax, ymid, zmin]
                    p8 = [xmid, ymid, zmin]
                    p9 = [xmid, ymin, zmin]
                    p10 = [xmid, ymin, zmid]
                    p11 = [xmax, ymin, zmid]
                    p12 = [xmax, ymid, zmid]
                else:
                    p1 = [xmax, ymax, zmin]
                    p2 = [xmax, ymin, zmin]
                    p3 = [xmax, ymin, zmax]
                    p4 = [xmin, ymin, zmax]
                    p5 = [xmin, ymax, zmax]
                    p6 = [xmin, ymax, zmin]
                    p7 = [xmin, ymid, zmin]
                    p8 = [xmid, ymid, zmin]
                    p9 = [xmid, ymin, zmin]
                    p10 = [xmid, ymin, zmid]
                    p11 = [xmin, ymin, zmid]
                    p12 = [xmin, ymid, zmid]
        pt1 = p1
        pt2 = p2
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'k-')
        pt1 = p9
        pt2 = p2
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'k-')
        pt1 = p3
        pt2 = p2
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'k-')
        pt1 = p3
        pt2 = p4
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'k-')
        pt1 = p11
        pt2 = p4
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'k-')
        pt1 = p5
        pt2 = p4
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'k-')
        pt1 = p5
        pt2 = p6
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'k-')
        pt1 = p1
        pt2 = p6
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'k-')
        pt1 = p7
        pt2 = p6
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'k-')
        pt1 = p7
        pt2 = p8
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'k-')
        pt1 = p7
        pt2 = p12
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'k-')
        pt1 = p11
        pt2 = p12
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'k-')
        pt1 = p13
        pt2 = p12
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'k-')
        pt1 = p13
        pt2 = p8
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'k-')
        pt1 = p13
        pt2 = p10
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'k-')
        pt1 = p9
        pt2 = p8
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'k-')
        pt1 = p9
        pt2 = p10
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'k-')
        pt1 = p11
        pt2 = p10
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'k-')

        # minor lines
        xs = np.unique([nd[0] for nd in self.nodelist])
        ys = np.unique([nd[1] for nd in self.nodelist])
        zs = np.unique([nd[2] for nd in self.nodelist])

        for x in xs:
            if x >= np.min([p2[0], p9[0]]) and x <= np.max([p2[0], p9[0]]):
                ax.plot([x, x], [p1[1], p2[1]], [p2[2], p2[2]],
                        color=color, linewidth=0.5)
                ax.plot([x, x], [p2[1], p2[1]], [p2[2], p3[2]],
                        color=color, linewidth=0.5)
            else:
                ax.plot([x, x], [p12[1], p2[1]], [p10[2], p10[2]],
                        color=color, linewidth=0.5)
                ax.plot([x, x], [p12[1], p6[1]], [p2[2], p2[2]],
                        color=color, linewidth=0.5)
                ax.plot([x, x], [p7[1], p7[1]], [p7[2], p11[2]],
                        color=color, linewidth=0.5)
                ax.plot([x, x], [p11[1], p11[1]], [p11[2], p3[2]],
                        color=color, linewidth=0.5)
        for y in ys:
            if y >= np.min([p6[1], p7[1]]) and y <= np.max([p6[1], p7[1]]):
                ax.plot([p6[0], p1[0]], [y, y], [p2[2], p2[2]],
                        color=color, linewidth=0.5)
                ax.plot([p6[0], p6[0]], [y, y], [p2[2], p3[2]],
                        color=color, linewidth=0.5)
            else:
                ax.plot([p10[0], p11[0]], [y, y], [p10[2], p10[2]],
                        color=color, linewidth=0.5)
                ax.plot([p9[0], p2[0]], [y, y], [p2[2], p2[2]],
                        color=color, linewidth=0.5)
                ax.plot([p4[0], p4[0]], [y, y], [p4[2], p11[2]],
                        color=color, linewidth=0.5)
                ax.plot([p10[0], p10[0]], [y, y], [p10[2], p9[2]],
                        color=color, linewidth=0.5)
        for z in zs:
            if z >= np.min([p4[2], p11[2]]) and z <= np.max([p4[2], p11[2]]):
                ax.plot([p4[0], p4[0]], [p5[1], p4[1]], [z, z],
                        color=color, linewidth=0.5)
                ax.plot([p4[0], p3[0]], [p4[1], p4[1]], [z, z],
                        color=color, linewidth=0.5)
            else:
                ax.plot([p4[0], p4[0]], [p6[1], p7[1]], [z, z],
                        color=color, linewidth=0.5)
                ax.plot([p10[0], p10[0]], [p7[1], p11[1]], [z, z],
                        color=color, linewidth=0.5)
                ax.plot([p2[0], p8[0]], [p4[1], p4[1]], [z, z],
                        color=color, linewidth=0.5)
                ax.plot([p7[0], p8[0]], [p7[1], p7[1]], [z, z],
                        color=color, linewidth=0.5)

        extension, save_fname = save_name(filename, variable='grid', time=1)
        if self._parent:
            if self._parent.work_dir and not os.path.isdir(
                    self._parent.work_dir):
                os.makedirs(self._parent.work_dir)
            if self._parent.work_dir:
                plt.savefig(self._parent.work_dir + slash + save_fname,
                            dpi=200, facecolor='w', edgecolor='w',
                            orientation='portrait',
                            format=extension, transparent=True,
                            bbox_inches=None, pad_inches=0.1)
            else:
                plt.savefig(save_fname, dpi=200, facecolor='w',
                            edgecolor='w', orientation='portrait',
                            format=extension, transparent=True,
                            bbox_inches=None, pad_inches=0.1)
        else:
            plt.savefig(save_fname, dpi=200, facecolor='w', edgecolor='w',
                        orientation='portrait',
                        format=extension, transparent=True,
                        bbox_inches=None, pad_inches=0.1)

    def rotate(self, angle=0., center=[0., 0.]):
        """
        Rotates the grid by some angle about a specified vertical axis.

        :param angle: Clockwise angle by which to rotate grid.
        :type angle: fl64
        :param center: x and y coordinates of vertical axis about which
         to rotate. Alternatively, the center of the computational domain
         can be specified by passing 'mid','middle','center', or 'center'.
        :type center: [fl64,fl64], str
        """
        if center in ['middle', 'mid', 'center', 'center']:
            center = [(self.xmin + self.xmax) / 2.,
                      (self.ymin + self.ymax) / 2.]
        nodes = []
        for nd in self.nodelist:
            # position relative to center of rotation
            old_pos = np.array(nd[0:2]) - np.array(center)
            theta_f = math.atan2(old_pos[1], old_pos[
                0]) + angle / 180. * math.pi
            dist = np.sqrt(np.dot(old_pos, old_pos))
            new_pos = [dist * math.cos(theta_f), dist * math.sin(theta_f)]
            nd[0] = new_pos[0] + center[0]
            nd[1] = new_pos[1] + center[1]
            nodes.append(nd)
            self._nodelist = nodes

    def dump_vtk(self, filename='mesh.vtk', format='ascii'):
        """
        Dumps vtk format of the mesh, currently only for structured grid

        :param filename: Name of the vtk file to be dumped.
        :type filename: str
        """
        try:
            import pyvtk
        except ImportError:
            print('\nThere was no pyvtk module installed')
        pp = self.nodelist
        vtk = pyvtk.VtkData(pyvtk.StructuredGrid(
            [self.nxyz[0] + 1, self.nxyz[1] + 1, self.nxyz[2] + 1], pp))
        vtk.tofile(filename, format)


class pcheckpoint(Frozen):
    """
    Class for specifying checkpoint options.

    :param time_list: List of times. e.g., [1,10,100]
    :type time_list: list
    :param time_unit: Unit of times in time_list
    :type time_unit: str
    :param periodic_time: checkpoint at every prescribed time.
    :type periodic_time: float
    :param periodic_time_unit: unit for periodic_time
    :type periodic_time_unit: str
    :param periodic_timestep: checkpoint at every n timesteps
    :type periodic_timestep: int
    :param format: specify 'binary' or 'hdf5'.
    :type format: str
    """

    def __init__(self, time_list=[], periodic_time=None,
                 periodic_timestep=None, format=None, periodic_time_unit=None,
                 time_unit=None):
        if time_list is None:
            time_list = []
        self.periodic_time = periodic_time
        self.time_list = time_list
        self.periodic_timestep = periodic_timestep
        self.periodic_time_unit = periodic_time_unit
        self.time_unit = time_unit
        self.format = format
        self._freeze()


class prestart(Frozen):
    """
    Class for restarting a simulation.

    :param filename: Specify file path and name for restart.chk file.
    :type filename: str
    :param time_value: Specify time value.
    :type time_value: float
    """

    def __init__(self, file_name='', time_value=None):
        self.file_name = file_name  # restart.chk file name
        self.time_value = time_value  # float
        self._freeze()

class pwipp_source_sink(Frozen):
    def __init__(self,brucitec=None,bruciteh=None,hymagcon=None,sat_wick=None,
                 salt_percent=None,gratmici=None,gratmich=None,corrmco2=None,
                 humcorr=None,asdrum=None,alpharxn=None,socmin=None,
                 biogenfc=None,probdeg=None,stoichiometric_matrix=None,
                 inventory=None,waste_panel=None):

        if waste_panel is None:
            waste_panel = []

        if inventory is None:
            inventory = []

        if stoichiometric_matrix is None:
            stoichiometric_matrix = []

        self.brucitec = brucitec
        self.bruciteh = bruciteh
        self.hymagcon = hymagcon
        self.sat_wick = sat_wick
        self.salt_percent = salt_percent
        self.gratmici = gratmici
        self.gratmich = gratmich
        self.corrmco2 = corrmco2
        self.humcorr = humcorr
        self.asdrum = asdrum
        self.alpharxn = alpharxn
        self.socmin = socmin
        self.biogenfc = biogenfc
        self.probdeg = probdeg
        self.stoichiometric_matrix = stoichiometric_matrix
        self.inventory = inventory
        self.waste_panel = waste_panel
        self._freeze()

    class inventory(Frozen):
        def __init__(self,name=None,vrepos=None,solids=None,aqueous=None):

            if solids is None:
                solids = []

            if aqueous is None:
                aqueous = []

            self.name = name
            self.vrepos = vrepos
            self.solids = solids
            self.aqueous = aqueous
            self._freeze()

        class solid(Frozen):
            def __init__(self,name=None,value=None):
                self.name = name
                self.value = value
                self._freeze()

        class aqueous(Frozen):
            def __init__(self,name=None,value=None):
                self.name = name
                self.value = value
                self._freeze()

    class waste_panel(Frozen):
        def __init__(self,name=None,region=None,inventory=None,
                     scale_by_volume=None):
            self.name = name
            self.region = region
            self.inventory = inventory
            self.scale_by_volume = scale_by_volume
            self._freeze()


class psource_sink_sandbox(Frozen):
    def __init__(self,mass_rate_downregulated=None):
        self.mass_rate_downregulated = mass_rate_downregulated
        self._freeze()

    class mass_rate_downregulated(Frozen):
        def __init__(self,rate=None,negative_reg_pressure=None,
                     positive_reg_pressure=None,coordinate=None):
            if coordinate is None:
                coordinate = []
            self.rate = rate
            self.negative_reg_pressure = negative_reg_pressure
            self.positive_reg_pressure = positive_reg_pressure
            self.coordinate = coordinate
            self._freeze()

        def _write(self,outfile):
            outfile.write('  MASS_RATE_DOWNREGULATED\n')
            if self.rate:
                outfile.write('    RATE %s\n' % strD(self.rate))
            if self.negative_reg_pressure:
                outfile.write('    NEGATIVE_REG_PRESSURE %s\n' % \
                               strD(self.negative_reg_pressure))
            if self.positive_reg_pressure:
                outfile.write('    POSITIVE_REG_PRESSURE %s\n' % \
                               strD(self.positive_reg_pressure))
            if self.coordinate:
                _coord = ' '.join([strD(x) for x in self.coordinate])
                outfile.write('    COORDINATE %s\n' % _coord)
            outfile.write('  /\n')

class pufd_decay(Frozen):
  '''
  The UFD Decay Process Model.
  Specifies the isotope decay, ingrowth, and phase partitioning model.

  :param element: Opens the block for defining elements.
  :type element: pufd_decay.element()
  :param isotope: Specifies each isotope.
  :type isotope: pufd_decay.isotope()
  :param implicit_solution: Applies implicit solution approach for isotope decay.
  :type implicit_solution: bool
  '''
  def __init__(self,elements=None,isotopes=None,implicit_solution=False):

      if elements is None:
          elements = []

      if isotopes is None:
          isotopes = []

      self.elements = elements
      self.isotopes = isotopes
      self.implicit_solution = implicit_solution
      self._freeze()
      
  class element(Frozen):
      '''
      Opens the block for defining elements, where <string> is the name of the element.

      :param name: Name of the element.
      :type name: str
      :param solubility: Specifies the solubility of the element in units of [M].
      :type solubility: float
      :param kd: List of Kds for each material property [name,value,...].
      :type kd: list<str,float,...>
      '''
      def __init__(self,name=None,solubility=None,kd=None):
          if kd is None:
              kd = []
          self.name = name
          self.solubility = solubility
          self.kd = kd
          self._freeze()
      def _write(self,outfile):
          outfile.write('  ELEMENT %s\n' % self.name)
          if self.solubility:
              outfile.write('    SOLUBILITY %s\n' % strD(self.solubility))
          if self.kd:
              assert len(self.kd) % 2 == 0, 'Missing values in KD'
              outfile.write('      KD\n')
              for i in range(0,len(self.kd),2):
                  outfile.write('        %s %s\n' % (self.kd[i],self.kd[i+1]))
              outfile.write('      /\n')
          outfile.write('    /\n')

  class isotope(Frozen):
      '''
      Specifies each isotope.

      :param name: Element name
      :type name: str
      :param element: Element group
      :type element: str
      :param decay_rate: Specifies the first-order decay rate in units [1/sec].
      :type decay_rate: float
      :param daughter: Name of the daughter isotope and stoichiometry.
      :type daughter: list<str,float>
      '''

      def __init__(self,name=None,element=None,decay_rate=None,daughter=None):
          if daughter is None:
              daughter = []
          self.name = name
          self.element = element
          self.decay_rate = decay_rate
          self.daughter = daughter
          self._freeze()

      def _write(self,outfile):
          outfile.write('  ISOTOPE %s\n' % self.name)
          if self.element:
              outfile.write('    ELEMENT %s\n' % self.element)
          if self.decay_rate is not None:
              outfile.write('    DECAY_RATE %s\n' % strD(self.decay_rate))
          if self.daughter:
              assert len(self.daughter) % 2 == 0, 'Missing values in Daughter'
              outfile.write('    DAUGHTER %s %s\n' % (self.daughter[0],
                                                    strD(self.daughter[1])))
          outfile.write('  /\n')

class pufd_biosphere(Frozen):
    '''
    Specifies the Example Reference Biosphere (ERB) Model 1,
    which calculates annual dose to an individual drinking radioactive water
    from a well (IAEA 2003).

    :param erb_1a: List of ERB Model 1A types.
    :type erb_1a: pufd_biosphere.erb_1a
    :param erb_1b: List of ERB Model 1B types.
    :type erb_1b: pufd_biosphere.erb_1b
    :param supported_radionuclides: Radionuclides in dose calculations
    :type supported_radionuclides:
    :param unsupported_radionuclides: Unsupported radionuclides in dose calculations
    :type unsupported_radionuclides:
    '''
    def __init__(self,erb_1a=None,erb_1b=None,supported_radionuclides=None,
                 unsupported_radionuclides=None,output_start_time=None):
        if erb_1a is None:
            erb_1a = []
        if erb_1b is None:
            erb_1b = []
        if supported_radionuclides is None:
            supported_radionuclides = []
        if unsupported_radionuclides is None:
            unsupported_radionuclides = []

        self.erb_1a = erb_1a
        self.erb_1b = erb_1b
        self.supported_radionuclides = supported_radionuclides
        self.unsupported_radionuclides = unsupported_radionuclides
        self.output_start_time = output_start_time
        self._freeze()

    class erb_1a(Frozen):
        def __init__(self,name=None,region=None,
                     individual_consumption_rate=None,
                     include_unsupported_rads=False):
            self.name = name
            self.region = region
            self.individual_consumption_rate = individual_consumption_rate
            self.include_unsupported_rads = include_unsupported_rads
            self._freeze()

        def _write(self,outfile):
            outfile.write('  ERB_1A %s\n' % self.name)
            if self.region is not None:
                outfile.write('    REGION %s\n' % self.region.lower())
            if self.individual_consumption_rate is not None:
                outfile.write('    INDIVIDUAL_CONSUMPTION_RATE %s\n' % \
                              str(self.individual_consumption_rate))
            if self.include_unsupported_rads:
                outfile.write('    INCLUDE_UNSUPPORTED_RADS\n')
            outfile.write('  /\n')

    class erb_1b(Frozen):
        def __init__(self,name=None,region=None,
                     individual_consumption_rate=None,
                     include_unsupported_rads=False,dilution_factor=None):
            self.name = name
            self.region = region
            self.individual_consumption_rate = individual_consumption_rate
            self.include_unsupported_rads = include_unsupported_rads
            self.dilution_factor = dilution_factor
            self._freeze()

        def _write(self,outfile):
            outfile.write('  ERB_1B %s\n' % self.name)
            if self.region is not None:
                outfile.write('    REGION %s\n' % self.region.lower())
            if self.individual_consumption_rate is not None:
                outfile.write('    INDIVIDUAL_CONSUMPTION_RATE %s\n' % \
                              str(self.individual_consumption_rate))
            if self.dilution_factor is not None:
                outfile.write('    DILUTION_FACTOR %s\n' % \
                              str(self.dilution_factor))
            if self.include_unsupported_rads:
                outfile.write('    INCLUDE_UNSUPPORTED_RADS\n')
            outfile.write('  /\n')

    class radionuclide(Frozen):
        def __init__(self,name=None,element_kd=None,decay_rate=None,
                     supported_parent=None,ingestion_dose_coef=None,
                     emanation_factor=None):

            self.name = name
            self.element_kd = element_kd
            self.decay_rate = decay_rate
            self.supported_parent = supported_parent
            self.ingestion_dose_coef = ingestion_dose_coef
            self.emanation_factor = emanation_factor

        def _write(self,outfile):
            outfile.write('      RADIONUCLIDE %s\n' % self.name)
            if self.element_kd is not None:
                outfile.write('        ELEMENT_KD %s\n' % strD(self.element_kd))
            if self.decay_rate is not None:
                outfile.write('        DECAY_RATE %s\n' % str(self.decay_rate))
            if self.ingestion_dose_coef is not None:
                outfile.write('        INGESTION_DOSE_COEF %s\n' % \
                                     strD(self.ingestion_dose_coef))
            if self.emanation_factor is not None:
                outfile.write('        EMANATION_FACTOR %s\n' % \
                              strD(self.emanation_factor))
            if self.supported_parent:
                outfile.write('        SUPPORTED_PARENT %s\n' % \
                              self.supported_parent)
            outfile.write('      /\n')


class psimulation(Frozen):
    """
    Class for specifying simulation type and simulation mode.

    :param simulation_type: Specify simulation type. Options include:
     'surface','subsurface.
    :type simulation_type: str
    :param subsurface_flow: Specify the process model.
    :type subsurface_flow: str
    :param subsurface_transport: Specify the process model.
    :type subsurface_transport: str
    :param mode: Specify the mode for the subsurface flow model
    :type mode: str
    :param flowtran_coupling: Specify the type for the flow transport coupling
    :type flowtran_coupling: str
    :param isothermal: Turn on isothermal case
    :type isothermal: bool
    :param max_saturation_change: Set maximum saturation change in timestepping
    :type max_saturation_change: float
    :param max_temperature_change: Set maximum temperature change in
     timestepping
    :type max_temperature_change: float
    :param max_pressure_change: Set maximum pressure change in
     timestepping
    :type max_pressure_change: float
    :param max_concentration_change: Set maximum concentration
     change in timestepping
    :type max_concentration_change: float
    :param max_cfl: Set maximum CFL number
    :type max_cfl: float
    :param numerical_derivatives: Turn this on if you want
     numerical derivatives in Jacobian
    :type numerical_derivatives: bool
    :param pressure_dampening_factor: Specify the dampening factor value
    :type pressure_dampening_factor: float
    :param max_volume_fraction_change: Specify the maximum volume fraction
     change
    :type max_volume_fraction_change: float
    """

    def __init__(self, simulation_type='subsurface', subsurface_flow='flow',
                 subsurface_transport='', mode='richards',
                 flowtran_coupling='', geomechanics_subsurface='',
                 isothermal='', max_pressure_change='',
                 max_saturation_change='',
                 max_temperature_change='', max_concentration_change='',
                 max_cfl='', numerical_derivatives='',
                 pressure_dampening_factor='',
                 restart='', checkpoint='',
                 max_volume_fraction_change='',freezing=False,
                 ice_model=None,itol_relative_update=None,
                 multiple_continuum=False,numerical_jacobian=False,
                 analytical_jacobian=False,
                 ufd_decay=None,ufd_biosphere=None,
                 auxiliary=None,
                 inline_surface_region=None,inline_surface_mannings_coeff=None,
                 waste_form=None,harmonic_permeability_only=None,
                 do_not_scale_jacobian=False,gas_component_formula_weight=None,
                 options=None):
        self.simulation_type = simulation_type
        self.subsurface_flow = subsurface_flow
        self.subsurface_transport = subsurface_transport
        self.flowtran_coupling = flowtran_coupling
        self.geomechanics_subsurface = geomechanics_subsurface
        self.mode = mode
        self.isothermal = isothermal
        self.max_saturation_change = max_saturation_change
        self.max_temperature_change = max_temperature_change
        self.max_pressure_change = max_pressure_change
        self.max_concentration_change = max_concentration_change
        self.max_cfl = max_cfl
        self.numerical_derivatives = numerical_derivatives
        self.pressure_dampening_factor = pressure_dampening_factor
        self.restart = restart
        self.checkpoint = checkpoint
        self.max_volume_fraction_change = max_volume_fraction_change
        self.freezing = freezing
        self.ice_model = ice_model
        self.itol_relative_update = itol_relative_update
        self.multiple_continuum = multiple_continuum
        self.numerical_jacobian = numerical_jacobian
        self.analytical_jacobian = analytical_jacobian
        self.ufd_decay = ufd_decay
        self.ufd_biosphere = ufd_biosphere
        self.auxiliary = auxiliary
        self.inline_surface_region = inline_surface_region
        self.inline_surface_mannings_coeff = inline_surface_mannings_coeff
        self.waste_form = waste_form
        self.options = options
        self._freeze()

    class auxiliary(Frozen):
        def __init__(self,aux_type=None,species=None):
            if species is None:
                species = []

            self.aux_type = aux_type
            self.species = species
            self._freeze()

        def _write(self,outfile):
            outfile.write('    AUXILIARY %s\n' % self.aux_type)
            if self.species:
                _name = self.species[0]
                _val = strD(self.species[1])
                outfile.write('      SPECIES %s %s\n' % (_name,_val))
            outfile.write('    /\n')

    class options(Frozen):
        '''
        Class to store simulation options.
        '''
        def __init__(self,isothermal=False,inline_surface_region=None,
                     inline_surface_mannings_coeff=None,
                     max_pressure_change=None,max_concentration_change=None,
                     max_temperature_change=None,max_saturation_change=None,
                     max_cfl=None,numerical_derivatives=None,
                     pressure_dampening_factor=None,freezing=False,
                     ice_model=None,multiple_continuum=False,
                     numerical_jacobian=False,analytical_jacobian=False,
                     harmonic_permeability_only=False,
                     do_not_scale_jacobian=False,
                     gas_component_formula_weight=None,
                     liquid_residual_infinity_tol=None,
                     gas_residual_infinity_tol=None,
                     max_allow_rel_liq_pres_chang_ni=None,
                     gas_sat_thresh_force_extra_ni=None,
                     gas_sat_thresh_force_ts_cut=None,
                     max_allow_gas_sat_change_ts=None,
                     gas_sat_change_ts_governor=None,
                     gas_sat_gov_switch_abs_to_rel=None,
                     min_liq_pres_force_ts_cut=None,
                     max_allow_liq_pres_change_ts=None,
                     liq_pres_change_ts_governor=None):

            self.isothermal = isothermal
            self.inline_surface_region = inline_surface_region
            self.inline_surface_mannings_coeff = inline_surface_mannings_coeff
            self.max_pressure_change = max_pressure_change
            self.max_concentration_change = max_concentration_change
            self.max_temperature_change = max_temperature_change
            self.max_saturation_change = max_saturation_change
            self.max_cfl = max_cfl
            self.numerical_derivatives = numerical_derivatives
            self.pressure_dampening_factor = pressure_dampening_factor
            self.freezing = freezing
            self.ice_model = ice_model
            self.multiple_continuum = multiple_continuum
            self.numerical_jacobian = numerical_jacobian
            self.analytical_jacobian = analytical_jacobian
            self.harmonic_permeability_only = harmonic_permeability_only
            self.do_not_scale_jacobian = do_not_scale_jacobian
            self.gas_component_formula_weight = gas_component_formula_weight
            self.liquid_residual_infinity_tol = liquid_residual_infinity_tol
            self.gas_residual_infinity_tol = gas_residual_infinity_tol
            self.max_allow_rel_liq_pres_chang_ni = max_allow_rel_liq_pres_chang_ni
            self.gas_sat_thresh_force_extra_ni = gas_sat_thresh_force_extra_ni
            self.gas_sat_thresh_force_ts_cut = gas_sat_thresh_force_ts_cut
            self.max_allow_gas_sat_change_ts = max_allow_gas_sat_change_ts
            self.gas_sat_change_ts_governor = gas_sat_change_ts_governor
            self.gas_sat_gov_switch_abs_to_rel = gas_sat_gov_switch_abs_to_rel
            self.min_liq_pres_force_ts_cut = min_liq_pres_force_ts_cut
            self.max_allow_liq_pres_change_ts = max_allow_liq_pres_change_ts
            self.liq_pres_change_ts_governor = liq_pres_change_ts_governor
            self._freeze()

        def _write(self,outfile):
            outfile.write('      OPTIONS\n')
            if self.isothermal:
                outfile.write('        ISOTHERMAL\n')
            if self.inline_surface_region:
                _val = self.inline_surface_region
                outfile.write('        INLINE_SURFACE_REGION %s\n' % _val)
            if self.inline_surface_mannings_coeff:
                _val = strD(self.inline_surface_mannings_coeff)
                outfile.write('        INLINE_SURFACE_MANNINGS_COEFF %s\n' % _val)
            if self.max_pressure_change:
                _val = strD(self.max_pressure_change)
                outfile.write('        MAX_PRESSURE_CHANGE %s\n' % _val)
            if self.max_concentration_change:
                _val = strD(self.max_concentration_change)
                outfile.write('        MAX_CONCENTRATION_CHANGE %s\n' % _val)
            if self.max_temperature_change:
                _val = strD(self.max_temperature_change)
                outfile.write('        MAX_TEMPERATURE_CHANGE %s\n' % _val)
            if self.max_saturation_change:
                _val = strD(self.max_saturation_change)
                outfile.write('        MAX_SATURATION_CHANGE %s\n' % _val)
            if self.max_cfl:
                outfile.write('        MAX_CFL ' + strD(self.max_cfl) + '\n')
            if self.numerical_derivatives:
                _val = strD(self.numerical_derivatives)
                outfile.write('        NUMERICAL_DERIVATIVES %s\n' % _val)
            if self.pressure_dampening_factor:
                _val = strD(self.pressure_dampening_factor)
                outfile.write('        PRESSURE_DAMPENING_FACTOR %s\n' % _val)
            if self.freezing:
                outfile.write('        FREEZING\n')
            if self.ice_model:
                outfile.write('        ICE_MODEL %s\n' % self.ice_model)
            if self.multiple_continuum:
                outfile.write('        MULTIPLE_CONTINUUM\n')
            if self.numerical_jacobian:
                outfile.write('        NUMERICAL_JACOBIAN\n')
            if self.analytical_jacobian:
                outfile.write('        ANALYTICAL_JACOBIAN\n')
            if self.harmonic_permeability_only:
                outfile.write('        HARMONIC_PERMEABILITY_ONLY\n')
            if self.do_not_scale_jacobian:
                outfile.write('        DO_NOT_SCALE_JACOBIAN\n')
            if self.gas_component_formula_weight:
                _val = strD(self.gas_component_formula_weight)
                outfile.write('        GAS_COMPONENT_FORMULA_WEIGHT %s\n' % _val)
            if self.liquid_residual_infinity_tol:
                _val = strD(self.liquid_residual_infinity_tol)
                outfile.write('        LIQUID_RESIDUAL_INFINITY_TOL %s\n' % _val)
            if self.gas_residual_infinity_tol:
                _val = strD(self.gas_residual_infinity_tol)
                outfile.write('        GAS_RESIDUAL_INFINITY_TOL %s\n' % _val)
            if self.max_allow_rel_liq_pres_chang_ni:
                _val = strD(self.max_allow_rel_liq_pres_chang_ni)
                outfile.write('        MAX_ALLOW_REL_LIQ_PRES_CHANG_NI %s\n' % _val)
            if self.gas_sat_thresh_force_extra_ni:
                _val = strD(self.gas_sat_thresh_force_extra_ni)
                outfile.write('        GAS_SAT_THRESH_FORCE_EXTRA_NI %s\n' % _val)
            if self.gas_sat_thresh_force_ts_cut:
                _val = strD(self.gas_sat_thresh_force_ts_cut)
                outfile.write('        GAS_SAT_THRESH_FORCE_TS_CUT %s\n' % _val)
            if self.max_allow_gas_sat_change_ts:
                _val = strD(self.max_allow_gas_sat_change_ts)
                outfile.write('        MAX_ALLOW_GAS_SAT_CHANGE_TS %s\n' % _val)
            if self.gas_sat_change_ts_governor:
                _val = strD(self.gas_sat_change_ts_governor)
                outfile.write('        GAS_SAT_CHANGE_TS_GOVERNOR %s\n' % _val)
            if self.gas_sat_gov_switch_abs_to_rel:
                _val = strD(self.gas_sat_gov_switch_abs_to_rel)
                outfile.write('        GAS_SAT_GOV_SWITCH_ABS_TO_REL %s\n' % _val)
            if self.min_liq_pres_force_ts_cut:
                _val = strD(self.min_liq_pres_force_ts_cut)
                outfile.write('        MIN_LIQ_PRES_FORCE_TS_CUT %s\n' % _val)
            if self.max_allow_liq_pres_change_ts:
                _val = strD(self.max_allow_liq_pres_change_ts)
                outfile.write('        MAX_ALLOW_LIQ_PRES_CHANGE_TS %s\n' % _val)
            if self.liq_pres_change_ts_governor:
                _val = strD(self.liq_pres_change_ts_governor)
                outfile.write('        LIQ_PRES_CHANGE_TS_GOVERNOR %s\n' % _val)
            outfile.write('      /\n')

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
    :param num_steps_after_cut: Number of time steps after a time step cut
     that the time step size is held constant.
    :type num_steps_after_cut: int
    :param max_steps: Maximum time step after which the simulation will be
     terminated.
    :type max_steps: int
    :param max_ts_cuts: Maximum number of consecutive time step cuts before
     the simulation is terminated.
    :type max_ts_cuts: int
    :param initialize_to_steady_state: Boolean flag to initialize a simulation
     to steady state
    :type initialize_to_steady_state: bool - True or False
    :param run_as_steady_state: Boolean flag to run a simulation to steady
     state
    :type run_as_steady_state: bool - True or False
    """

    # definitions are put on one line to work better with rst/latex/sphinx.
    def __init__(self, ts_mode='flow', ts_acceleration=None,
                 num_steps_after_cut=None, max_steps=None,
                 max_ts_cuts=None,
                 initialize_to_steady_state=False,
                 run_as_steady_state=False):
        self.ts_mode = ts_mode
        self.ts_acceleration = ts_acceleration
        self.num_steps_after_cut = num_steps_after_cut
        self.max_steps = max_steps
        self.max_ts_cuts = max_ts_cuts
        self.initialize_to_steady_state = initialize_to_steady_state
        self.run_as_steady_state = run_as_steady_state
        self._freeze()


class plsolver(Frozen):
    """
    Class for specifying linear solver. Multiple linear solver Frozens
    can be created one for flow and one for transport.

    :param name: Specify name of the physics for which the linear solver is
     being defined. Options include: 'tran', 'transport','flow'.
    :type name: str
    :param solver: Specify solver type: Options include: 'solver',
     'krylov_type', 'krylov', 'ksp', 'ksp_type'
    :type solver: str
    :param preconditioner: Specify preconditioner type: Options include: 'ilu'
    :type solver: str
    """

    def __init__(self, name='', solver='', preconditioner='', ksp=''):
        self.name = name  # TRAN, TRANSPORT / FLOW
        self.solver = solver  # Solver Type
        self.preconditioner = preconditioner
        self.ksp = ksp
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
    :param itol_update: Tolerance for updates based on infinity norm
    :type itol_update: float
    :param max_it: Cuts time step if the number of iterations exceed
     this value.
    :type max_it: int
    :param max_f: Maximum function evaluations
     (useful with linesearch methods.)
    :type max_f: int
    """

    # definitions are put on one line to work better with rst/latex/sphinx.
    def __init__(self, name='', atol=None, rtol=None, stol=None, dtol=None,
                 itol=None, max_it=None, max_f=None, itol_update=None,
                 matrix_type=None, preconditioner_matrix_type=None,
                 no_infinity_norm=False):
        self.name = name  # Indicates Flow or Tran for Transport
        self.atol = atol
        self.rtol = rtol
        self.stol = stol
        self.dtol = dtol
        self.itol = itol
        self.max_it = max_it
        self.max_f = max_f
        self.itol_update = itol_update
        self.matrix_type = matrix_type
        self.preconditioner_matrix_type = preconditioner_matrix_type
        self.no_infinity_norm = False
        self._freeze()


class poutput_file(Frozen):
    """
    Class for output file type -- snapshot or observation or mass balance

    :param format: output format, works only with snapshot file, e.g., hdf5
    :type format: str
    :param times_per_file: specify the number of time snapshots that will go in
     one file
    :type times_per_file: int
    :param print_initial: set to False if you don't want to print initial state
    :type print_initial: bool
    :param print_final: set to False if you don't want to print final state
    :type print_final: bool
    :param periodic_timestep: output at every specified time step value
    :type periodic_timestep: float
    :param periodic_time: output at every specified value of time
    :type periodic_time: float
    :param periodic_time_unit: unit of the periodic time value specified
    :type periodic_time_unit: str
    :param time_list: list of times to print output
    :type time_list: list of floats
    :param time_unit: unit of the times list specified
    :type time_unit: str
    :param periodic_observation_timestep: output of observation points
      and mass balance at specified value of time step
    :type periodic_observation_timestep: int
    :param periodic_observation_time: output of observation points and
      mass balance at specified value of time
    :type periodic_observation_time: float
    :param periodic_observation_time_unit: unit of time specified in
      periodic_observation_time
    :type periodic_observation_time_unit: str
    :param variables_list: list of variables to be printed
    :type variables_list: list of string
    :param total_mass_regions: list of regions where total mass needs to be
      outputted
    :type total_mass_regions: list of string
    :param extend_hdf5_time_format: set to True to extend hdf5 time format
    :type extend_hdf5_time_format: bool
    """

    def __init__(self, format=None, times_per_file=None, print_initial=True,
                 print_final=True, time_list=None, time_unit=None,
                 periodic_timestep=None, periodic_time=None,
                 periodic_time_unit=None, periodic_observation_timestep=None,
                 periodic_observation_time=None,
                 periodic_observation_time_unit=None, variables_list=None,
                 total_mass_regions=None, extend_hdf5_time_format=False):

        if time_list is None:
            time_list = []
        if variables_list is None:
            variables_list = []
        if total_mass_regions is None:
            total_mass_regions = []

        self.time_list = time_list
        self.format = format
        self.times_per_file = times_per_file
        self.print_initial = print_initial
        self.print_final = print_final
        self.time_unit = time_unit
        self.periodic_timestep = periodic_timestep
        self.periodic_time_unit = periodic_time_unit
        self.periodic_time = periodic_time
        self.periodic_observation_time = periodic_observation_time
        self.periodic_observation_time_unit = periodic_observation_time_unit
        self.periodic_observation_timestep = periodic_observation_timestep
        self.variables_list = variables_list
        self.total_mass_regions = total_mass_regions
        self.extend_hdf5_time_format = extend_hdf5_time_format
        self._freeze()


class poutput(Frozen):
    """
    Class for dumping simulation output.
    Acceptable time units (units of measurements) are: 's', 'min', 'h',
    'd', 'w', 'mo', 'y'.

    :param time_list: List of time values. 1st variable specifies time unit
     to be used. Remaining variable(s) are floats
    :type time_list: [str, float*]
    :param print_column_ids: Flag to indicate whether to print column numbers
     in observation
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
    :param periodic_observation_time: Output the results at observation points
     and mass balance output at specified output time.
     1st variable is value, 2nd variable is time unit.
    :type periodic_observation_time: [float, str]
    :param periodic_observation_timestep: Outputs the results at observation
     points and mass balance output at specified time steps.
    :type periodic_observation_timestep: int
    :param format_list: Specify the file format for time snapshot of the
     simulation in time file type. Input is a list of strings.
     Multiple formats can be specified.
     File format options include: 'TECPLOT BLOCK' - TecPlot block format,
     'TECPLOT POINT' -- TecPlot point format (requires a single processor),
     'HDF5' -- produces single HDF5 file and xml for unstructured grids,
     'HDF5 MULTIPLE_FILES' -- produces a separate HDF5 file
     at each output time, 'VTK' - VTK format.
    :type format_list: [str]
    :param velocities: Turn velocity output on/off.
    :type velocities: bool - True or False
    :param velocity_at_center: Turn velocity output on/off.
    :type velocity_at_center: bool - True or False
    :param velocity_at_face: Turn velocity output at face on/off.
    :type velocity_at_face: bool - True or False
    :param mass_balance: Flag to indicate whether to output the mass
     balance of the system.
    :type mass_balance: bool - True or False
    :param variables_list: List of variables to be printed in the output file
    :type variables_list: [str]
    :param snapshot_file: details of snapshot file
    :type snapshot_file: poutput_file
    :param observation_file: details of observation file
    :type observation_file: poutput_file
    :param mass_balance_file: details of mass balance file
    :type mass_balance_file: poutput_file
    :param time_units: Units of time
    :type time_units: str
    """

    # definitions are put on one line to work better with rst/latex/sphinx.
    def __init__(self, time_list=None, print_column_ids=False,
                 screen_periodic=None, screen_output=True,
                 periodic_time=None, periodic_timestep=None,
                 periodic_observation_time=None,
                 periodic_observation_timestep=None, format_list=None,
                 permeability=False, porosity=False,
                 velocities=False, velocity_at_center=False,
                 velocity_at_face=False, mass_balance=False,
                 variables_list=None, snapshot_file=None,
                 observation_file=None,
                 mass_balance_file=None,
                 time_units=None):

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
        if periodic_timestep is None:
            periodic_timestep = []

        if observation_file is None:
            observation_file = poutput_file()

        if snapshot_file is None:
            snapshot_file = poutput_file()

        if mass_balance_file is None:
            mass_balance_file = poutput_file()

        self.time_list = time_list
        self.print_column_ids = print_column_ids
        self.screen_output = screen_output
        self.screen_periodic = screen_periodic
        self.periodic_time = periodic_time
        self.periodic_timestep = periodic_timestep
        self.periodic_observation_time = periodic_observation_time
        self.periodic_observation_timestep = periodic_observation_timestep
        self.format_list = format_list
        self.permeability = permeability
        self.porosity = porosity
        self.velocities = velocities
        self.mass_balance = mass_balance
        self.variables_list = variables_list
        self.velocity_at_center = velocity_at_center
        self.velocity_at_face = velocity_at_face
        self.observation_file = observation_file
        self.mass_balance_file = mass_balance_file
        self.snapshot_file = snapshot_file
        self.time_units = time_units

        self._freeze()


class pfluid(Frozen):
    """
    Class for specifying fluid properties.

    :param diffusion_coefficient: Unit of measurement is [m^2/s].
     Default: 1e-09
    :type diffusion_coefficient: float
    :param phase: liquid or gas phase
    :type: str
    """

    def __init__(self, diffusion_coefficient=1.e-9, phase=''):
        self.diffusion_coefficient = diffusion_coefficient
        self.phase = phase
        self._freeze()


class psaturation(Frozen):
    """
    Class for specifying saturation functions.

    :param name: Saturation function name. e.g., 'sf2'
    :type name: str
    :param permeability_function_type: Options include: 'VAN_GENUCHTEN',
     'MUALEM', 'BURDINE', 'NMT_EXP', 'PRUESS_1'.
    :type permeability_function_type: str
    :param saturation_function_type: Options include: 'VAN_GENUCHTEN',
     'BROOKS_COREY', 'THOMEER_COREY', 'NMT_EXP', PRUESS_1'.
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
    def __init__(self, name='', permeability_function_type=None,
                 saturation_function_type=None,
                 residual_saturation=None, residual_saturation_liquid=None,
                 residual_saturation_gas=None, a_lambda=None,
                 alpha=None, max_capillary_pressure=None, betac=None,
                 power=None):
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


class pwaste_form_general(Frozen):
    def __init__(self,print_mass_balance=False,implicit_solution=False,
                 waste_form=None,mechanism=None):

        if waste_form is None:
            waste_form = []

        if mechanism is None:
            mechanism = []

        self.print_mass_balance = print_mass_balance
        self.implicit_solution = implicit_solution
        self.waste_form = waste_form
        self.mechanism = mechanism

    class mechanism(Frozen):
        def __init__(self,name=None,mech_type=None,specific_surface_area=None,
                     matrix_density=None,kienzler_dissolution=None,
                     canister_degradation_mode=None,vitality_log10_mean=None,
                     vitality_log10_stdev=None,vitality_upper_truncation=None,
                     canister_material_constant=None,k0=None,species=None,
                     k_long=None,nu=None,ea=None,q=None,k=None,v=None,ph=None,
                     dissolution_rate=None,fractional_dissolution_rate=None,
                     fractional_dissolution_rate_vi=None):

            if species is None:
                species = []

            self.name = name
            self.mech_type = mech_type
            self.specific_surface_area = specific_surface_area
            self.matrix_density = matrix_density
            self.kienzler_dissolution = kienzler_dissolution
            self.canister_degradation_mode = canister_degradation_mode
            self.vitality_log10_mean = vitality_log10_mean
            self.vitality_log10_stdev = vitality_log10_stdev
            self.vitality_upper_truncation = vitality_upper_truncation
            self.canister_material_constant = canister_material_constant
            self.k0 = k0
            self.species = species
            self.k_long = k_long
            self.nu = nu
            self.ea = ea
            self.q = q
            self.k = k
            self.v = v
            self.ph = ph
            self.dissolution_rate = dissolution_rate
            self.fractional_dissolution_rate = fractional_dissolution_rate
            self.fractional_dissolution_rate_vi = fractional_dissolution_rate_vi

    class waste_form(Frozen):
        def __init__(self,coordinate=None,exposure_factor=None,volume=None,
                     mechanism_name=None,canister_vitality_rate=None,
                     canister_breach_time=None,decay_start_time=None):
            self.coordinate = coordinate
            self.exposure_factor = exposure_factor
            self.volume = volume
            self.mechanism_name = mechanism_name
            self.canister_vitality_rate = canister_vitality_rate
            self.canister_breach_time = canister_breach_time
            self.decay_start_time = decay_start_time

    def add_mechanism(self,**kwargs):
        mech = pwaste_form_general.mechanism(**kwargs)
        self.mechanism.append(mech)
        return mech

    def add_waste_form(self,**kwargs):
        wf = pwaste_form_general.waste_form(**kwargs)
        self.waste_form.append(wf)
        return wf


class pcharacteristic_curves(Frozen):
    """
    Class for specifying characteristic curves. This card is used only
     in GENERAL mode; the SATURATION_FUNCTION card
    should be used in RICHARDS mode.

    :param name: Characteristic curve name. e.g., 'cc1'
    :param saturation_function_type: Options include: 'VAN_GENUCHTEN',
     'BROOKS_COREY'.
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
    :param default: sets up dummy saturation and permeability functions for
     saturated single phase flow
    :type default: Bool
    :param liquid_permeability_function_type: Options include: 'MAULEM',
     'BURDINE'.
    :type liquid_permeability_function_type: str
    :param lpf_m: Van Genutchen m
    :type lpf_m: float
    :param lpf_lambda: lambda: Brooks Corey lambda
    :type lpf_lambda: float
    :param lpf_liquid_residual_saturation: Residual saturation for liquid phase
    :type lpf_liquid_residual_saturation: float
    :param gas_permeability_function_type: Options include: 'MAULEM_VG_GAS',
     'BURDINE_BC_GAS'.
    :type gas_permeability_function_type: str
    :param gpf_m: Van Genutchen m
    :type gpf_m: float
    :param gpf_lambda: lambda: Brooks Corey lambda
    :type gpf_lambda: float
    :param gpf_liquid_residual_saturation: Residual saturation for liquid phase
    :type gpf_liquid_residual_saturation: float
    :param gf_gas_residual_saturation: Residual saturation for gas phase
    :type gf_gas_residual_saturation: float
    :param phase: gas or liquid phase
    :type phase: str
    """

    # definitions are put on one line to work better with rst/latex/sphinx.
    def __init__(self, name='default',
                 saturation_function_type='van_genuchten', sf_alpha=1.e-4,
                 sf_m=0.5, sf_lambda=None,
                 sf_liquid_residual_saturation=0.1,
                 sf_gas_residual_saturation=None, max_capillary_pressure=None,
                 smooth='', power=None, default=False,
                 liquid_permeability_function_type='mualem_vg_liq',
                 lpf_m=0.1, lpf_lambda=None,
                 lpf_liquid_residual_saturation=0.1,
                 gas_permeability_function_type=None, gpf_m=None,
                 gpf_lambda=None, gpf_liquid_residual_saturation=None,
                 gpf_gas_residual_saturation=None, phase=None,
                 lpf_gas_residual_saturation=None):
        self.name = name
        self.saturation_function_type = saturation_function_type
        self.sf_alpha = sf_alpha
        self.sf_m = sf_m
        self.sf_lambda = sf_lambda
        self.sf_liquid_residual_saturation = sf_liquid_residual_saturation
        self.sf_gas_residual_saturation = sf_gas_residual_saturation
        self.max_capillary_pressure = max_capillary_pressure
        self.smooth = smooth
        self.power = power
        self.default = default
        self.liquid_permeability_function_type = \
            liquid_permeability_function_type
        self.lpf_m = lpf_m
        self.lpf_lambda = lpf_lambda
        self.lpf_gas_residual_saturation = lpf_gas_residual_saturation
        self.lpf_liquid_residual_saturation = lpf_liquid_residual_saturation
        self.gas_permeability_function_type = gas_permeability_function_type
        self.gpf_m = gpf_m
        self.gpf_lambda = gpf_lambda
        self.gpf_liquid_residual_saturation = gpf_liquid_residual_saturation
        self.gpf_gas_residual_saturation = gpf_gas_residual_saturation
        self.phase = phase
        self._freeze()


class pregion(Frozen):
    """
    Class for specifying a PFLOTRAN region. Multiple region objects
    can be created.

    :param name: Region name.
    :type name: str
    :param coordinates_lower: Lower/minimum 3D coordinates for defining a
     volumetric, planar, or point region between two points in space in
     order of x1, y1, z1. e.g., [0.e0, 0.e0, 0.e0]
    :type coordinates_lower: [float]*3
    :param coordinates_upper: Upper/maximum 3D coordinates for defining a
     volumetric, planar, or point region between two points in space in order
     of x2, y2, z2. e.g., [321.e0, 1.e0,  51.e0]
    :type coordinates_upper: [float]*3
    :param face: Defines the face of the grid cell to which boundary conditions
     are connected.
     Options include: 'west', 'east', 'north', 'south', 'bottom', 'top'.
     (structured grids only).
    :type face: str
    """

    def __init__(self, name='', coordinates_lower=None, coordinates_upper=None,
                 face=None, filename='', point_list=[], pm='', block=[]):
        if coordinates_lower is None:
            coordinates_lower = [0.0, 0.0, 0.0]
        if coordinates_upper is None:
            coordinates_upper = [0.0, 0.0, 0.0]
        self.name = name.lower()
        self.coordinates_lower = coordinates_lower  # 3D coordinates
        self.coordinates_upper = coordinates_upper  # 3D coordinates
        self.face = face
        self.point_list = []
        self.filename = filename
        self.pm = pm
        self.block = block
        self._freeze()


class pintegral_flux(Frozen):
    """
    Class for specifying INTEGRAL_FLUX card. Multiple objects
    can be created.

    :param name: Name of flux condition (e.g. outflow).
    :type name: str
    :param coordinates_lower: Lower/minimum 3D coordinates for defining a
     planar region between two points in space in
     order of x1, y1, z1. e.g., [0.e0, 0.e0, 0.e0]
    :type coordinates_lower: [float]*3
    :param coordinates_upper: Upper/maximum 3D coordinates for defining a
     planar region between two points in space in order
     of x2, y2, z2. e.g., [321.e0, 1.e0,  51.e0]
    :type coordinates_upper: [float]*3
    :param invert_direction: Will add INVERT_DIRECITON keyword.
    :type face: logical
    """

    def __init__(self, name='', coordinates_lower=None,
                 coordinates_upper=None):
        if coordinates_lower is None:
            coordinates_lower = [0.0, 0.0, 0.0]
        if coordinates_upper is None:
            coordinates_upper = [0.0, 0.0, 0.0]
        self.name = name.lower()
        self.coordinates_lower = coordinates_lower  # 3D coordinates
        self.coordinates_upper = coordinates_upper  # 3D coordinates
        self._freeze()


class pobservation(Frozen):
    """
    Class for specifying an observation region. Multiple observation objects
    may be added. Currently, only region is supported in PyFLOTRAN.

    :param region: Defines the name of the region to which the observation
     object is linked.
    :type region: str
    """

    def __init__(self, region=None, secondary_temperature=None,
                 secondary_concentration=None,
                 secondary_mineral_volfrac=None,
                 velocity=None, at_cell_center=False):
        self.region = region
        self.secondary_temperature = secondary_temperature
        self.secondary_concentration = secondary_concentration
        self.secondary_mineral_volfrac = secondary_mineral_volfrac
        self.velocity = velocity
        self.at_cell_center = at_cell_center
        self._freeze()


class pflow(Frozen):
    """
    Class for specifying a PFLOTRAN flow condition. There can be multiple
    flow condition objects.

    :param name: Name of the flow condition.
    :type name: str
    :param units_list: Not currently supported.
    :type units_list: [str]
    :param iphase:
    :type iphase: int
    :param sync_timestep_with_update: Flag that indicates whether to use
     sync_timestep_with_update. Default: False.
    :type sync_timestep_with_update: bool - True or False
    :param datum: Input is either a list of [d_dx, d_dy, d_dz] OR a 'file_name'
     with a list of [d_dx, d_dy, d_dz]. Choose one format type or the other,
     not both. If both are used, then only the file name will be written to
     the input deck.
    :type datum: Multiple [float, float, float] or str.
    :param datum_type: file or dataset
    :type datum_type: str
    :param varlist: Input is a list of pflow_variable objects.
     Sub-class of pflow. It is recommended to use dat.add(obj=pflow_variable)
     for easy appending.
     Use dat.add(index='pflow_variable.name' or dat.add(index=pflow_variable)
     to specify pflow object to add pflow_variable to. If no pflow object is
     specified, pflow_variable will be appended to the last pflow object
     appended to pdata. E.g., dat.add(variable, 'initial')
     if variable = pflow_variable and pflow.name='initial'.
    :type varlist: [pflow_variable]
    """

    def __init__(self, name='', units_list=None, iphase=None,
                 sync_timestep_with_update=False, datum=None,
                 datum_type='', datum_time_unit=None,
                 varlist=None, gradient=None, pm='',
                 gradient_type='', datum_data_unit=None,
                 interpolation=None, cyclic=False,
                 units=None):

        if datum is None:
            datum = []
        if varlist is None:
            varlist = []
        if gradient is None:
            gradient = []
        if units is None:
            units = []
        self.name = name.lower()  # Include initial, top, source
        self.units_list = units_list  # Specify type of units
        # to display such as
        # time,length,rate,pressure,velocity, temperature,
        # concentration, and enthalpy.
        # May be used to determine each variable unit
        self.iphase = iphase  # Holds 1 int
        self.sync_timestep_with_update = sync_timestep_with_update  # Boolean
        self.datum = datum  # x, y, z, and a file name. [float,float,float,str]
        self.varlist = varlist
        self.datum_type = datum_type
        self.gradient = gradient
        self.gradient_type = gradient_type
        self.datum_time_unit = datum_time_unit
        self.datum_data_unit = datum_data_unit
        self.pm = pm
        self.cyclic = cyclic
        self.interpolation = interpolation
        self.units = units
        self._freeze()


class pflow_variable(Frozen):
    """
    Class of pflow for each kind of variable (includes type and value)
    such as pressure, temperature, etc. There can be multiple pflow_variable
    objects appended to a single pflow object.

    :param name: Indicates name of the flow variable. Options include:
     ['PRESSURE', 'RATE', 'FLUX', 'TEMPERATURE', 'CONCENTRATION',
     'SATURATION', 'ENTHALPY'].
     :type name: str
    :param type: Indicates type that is associated with name under keyword
     TYPE. Options for PRESSURE include: 'dirichlet', 'hydrostatic',
     'zero_gradient', 'conductance', 'seepage'. Options for RATE include:
     'mass_rate', 'volumetric_rate', 'scaled_volumetric_rate'. Options for
     FLUX include: 'dirichlet', 'neumann, mass_rate', 'hydrostatic,
     'conductance','zero_gradient', 'production_well', 'seepage', 'volumetric',
     'volumetric_rate', 'equilibrium'. Options for TEMPERATURE include:
     'dirichlet', 'hydrostatic', 'zero_gradient'. Options for CONCENTRATION
     include: 'dirichlet', 'hydrostatic', 'zero_gradient'. Options for
     SATURATION include: 'dirichlet'. Options for ENTHALPY include:
     'dirichlet', 'hydrostatic', 'zero_gradient'
    :type type: str
    :param valuelist: Provide one or two values associated with a single
     Non-list alternative, do not use with list alternative. The 2nd float
     is optional and is needed for multiphase simulations.
    :type valuelist: [float, float]
    :param unit: Non-list alternative, do not use with list alternative.
     Specify unit of measurement.
    :type unit: str
    :param time_unit_type: List alternative, do not use with non-list
     alternative attributes/parameters.
    :type time_unit_type: str
    :param list: List alternative, do not use with non-list alternative
     attributes/parameters. Input is a list of
     pflow_variable_list objects. Sub-class of pflow_variable. The add function
     currently does not support
     adding pflow_variable_list to pflow_variable objects. Appending to can
     be done manually. E.g., variable.list.append(var_list) if
     var_list=pflow_variable_list.
    :type list: [pflow_variable_list]
    """

    def __init__(self, name='', type=None, valuelist=None, unit='',
                 time_unit_type='', data_unit_type='', plist=None,
                 subtype=None):
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

        # Following attributes are used with lists (eg. Rate Lists instead of
        # Rate)
        self.time_unit_type = time_unit_type  # e.g., 'y'
        self.data_unit_type = data_unit_type  # e.g., 'kg/s'
        self.list = plist  # Holds a plist of pflow_variable_lists objects
        self.subtype = subtype  # This is for rate subtypes
        self._freeze()


class pflow_variable_list(Frozen):
    """
    Sub-class of pflow_variable.
    Used for pflow_variables that are lists (as function of time) instead
    of a single value. Each of these list objects can hold multiple lines
    (from a Python input file) with each line holding one
    time_unit_value and a data_unit_value_list that can hold multiple values.

    :param time_unit_value:
    :type time_unit_value: float
    :param data_unit_value_list:
    :type data_unit_value_list: [float]
    """

    def __init__(self, time_unit_value=None, data_unit_value_list=None):
        if data_unit_value_list is None:
            data_unit_value_list = []
        self.time_unit_value = time_unit_value  # 1 float
        # 2 floats? (2nd optional?)
        self.data_unit_value_list = data_unit_value_list
        self._freeze()


class pinitial_condition(Frozen):
    """
    Class for initial condition - a coupler between regions and initial
    flow and transport conditions.

    :param flow: Specify flow condition name
    :type flow: str
    :param transport: Specify transport condition name
    :type transport: str
    :param region: Specify region to apply the above specified flow and
     transport conditions as initial conditions.
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
    Class for boundary conditions - performs coupling between a region and
    a flow/transport condition which are to be set as boundary conditions
    to that region. Multiple objects can be created.

    :param name: Name of boundary condition. (e.g., west, east)
    :type name: str
    :param flow: Defines the name of the flow condition to be linked to
     this boundary condition.
    :type flow: str
    :param transport: Defines the name of the transport condition to be
     linked to this boundary condition
    :type transport: str
    :param region: Defines the name of the region to which the conditions
     are linked
    :type region: str
    """

    def __init__(self, name='', flow='', transport='', region='', geomech=''):
        self.name = name  # Name of boundary condition. (e.g., west, east)
        self.flow = flow  # Flow Condition (e.g., initial)
        # Transport Condition (e.g., river_chemistry)
        self.transport = transport
        self.geomech = geomech
        self.region = region  # Define region (e.g., west, east, well)
        self._freeze()


class psource_sink(Frozen):
    """
    Class for specifying source sink - this is also a condition coupler that
    links a region to the source sink condition

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
    Class for specifying stratigraphy coupler. Multiple stratigraphy couplers
    can be created. Couples material properties with a region.

    :param region: Name of the material property to be associated with
     a region.
    :type region: str
    :param material: Name of region associated with a material property.
    :type material: str
    """

    def __init__(self, region='', material='', pm='', file=None, start_time=None,
                 final_time=None):
        self.region = region
        self.material = material
        self.pm = pm
        self.file = file
        self.start_time = start_time
        self.final_time = final_time
        self._freeze()


class pdbase(Frozen):

    def __init__(self, value):
        assert isinstance(value, str)
        self.value = value


class pdataset(Frozen):
    """
    Class for incorporating data within a model.

    :param dataset_name: Opens the card block with the name of the data
     set in the string. I name is not given the NAME entry is required.
    :type dataset_name: str
    :param dataset_mapped_name: Adds the MAPPED flag to the DATASET and
     allows for the dataset to be named.
    :type dataset_name: str
    :param name: Name of the data set if not included with DATASET card.
     Note: this string overwrites the name specified
     with DATASET
    :type name: str
    :param file_name: Name of the file containing the data
    :type file_name: str
    :param hdf5_dataset_name: Name of the group within the hdf5 file
     where the data resides
    :type hdf5_dataset_name: str
    :param map_hdf5_dataset_name: Name of the group within the hdf5 file
     where the map information for the data resides
    :type map_hdf5_dataset_name: str
    :param max_buffer_size: size of internal buffer for storing transient
     data
    :type max_buffer_size: float
    :param realization_dependent: Add when doing stochastic multiple
     realizations
    :type realization_dependent: bool
   """

    def __init__(self, dataset_name='', dataset_mapped_name='', name='',
                 file_name='', hdf5_dataset_name='',
                 map_hdf5_dataset_name='', max_buffer_size='',
                 realization_dependent=''):
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

    :param primary_species_list: List of primary species that fully describe
     the chemical composition of the fluid. The set of primary species must
     form an independent set of species in terms of which all homogeneous
     aqueous equilibrium reactions can be expressed.
    :type primar_species_list: [str]
    :param secondary_species_list: List of aqueous species in equilibrium with
     primary species.
    :type secondary_species_list: [str]
    :param gas_species_list: List of gas species.
    :type gas_species_list: [str]
    :param passive_gas_species_list: List of passive gas species.
    :type passive_gas_species: [str]
    :param active_gas_species: List of active gas species.
    :type active_gas_species: [str]
    :param minerals_list: List of mineral names.
    :type minerals_list: [str]
    :param m_kinetics_list: List of pchemistry_m_kinetic objects.
     Holds kinetics information about a specified mineral
     name. Works with add function so that m_kinetics_list does not
     need to be remembered. e.g., dat.add(mineral_kinetic)
    :type m_kinetics_list: [pchemistry_m_kinetic]
    :param log_formulation:
    :type log_formulation: bool - True or False
    :param use_full_geochemistry:
    :type use_full_geochemistry: bool - True or False
    :param update_porosity:
    :type update_porosity: bool - True or False
    :param update_permeability:
    :type update_permeability: bool - True or False
    :param database:
    :type database: str
    :param activity_coefficients: Options include: 'LAG', 'NEWTON',
     'TIMESTEP', 'NEWTON_ITERATION'.
    :type activity_coefficients: str
    :param molal:
    :type molal: bool - True or False
    :param output_list: To print secondary aqueous complex concentrations,
     either add the names of the secondary species of interest or the keyword
     'SECONDARY_SPECIES' for all secondary species to the CHEMISTRY OUTPUT
     card. E.g., output_list = 'SECONDARY_SPECIES' or output_list =
     ['CO2(aq), 'PH']. By default, if ALL or MINERALS are listed under
     CHEMISTRY OUTPUT, the volume fractions and rates of kinetic minerals
     are printed. To print out the saturation indices of minerals listed
     under the MINERAL keyword, add the name of the mineral to the OUTPUT
     specification.
    :type output_list: [str]
    """

    def __init__(self, primary_species_list=None, secondary_species_list=None,
                 gas_species_list=None, minerals_list=None,
                 m_kinetics_list=None, log_formulation=False, database=None,
                 activity_coefficients=None, molal=False,
                 output_list=None, update_permeability=False,
                 update_porosity=False, active_gas_species_list=None,
                 passive_gas_species_list=None, truncate_concentration=None,
                 use_full_geochemistry=False, max_dlnc=None,
                 max_residual_tolerance=None,
                 max_relative_change_tolerance=None, activity_water=False,
                 update_mineral_surface_area=False, no_bdot=False,
                 no_checkpoint_act_coefs=False, general_reaction=None,
                 sorption=None,
                 immobile_decay_reaction=None, radioactive_decay_reaction=None,
                 microbial_reaction=None, immobile_species_list=None):

        if primary_species_list is None:
            primary_species_list = []
        if secondary_species_list is None:
            secondary_species_list = []
        if gas_species_list is None:
            gas_species_list = []
        if passive_gas_species_list is None:
            passive_gas_species_list = []
        if active_gas_species_list is None:
            active_gas_species_list = []
        if minerals_list is None:
            minerals_list = []
        if m_kinetics_list is None:
            m_kinetics_list = []
        if output_list is None:
            output_list = []
        if microbial_reaction is None:
            microbial_reaction = []
        if radioactive_decay_reaction is None:
            radioactive_decay_reaction = []
        if immobile_decay_reaction is None:
            immobile_decay_reaction = []
        if immobile_species_list is None:
            immobile_species_list = []

        # primary_species (eg. 'A(aq') - string
        self.primary_species_list = primary_species_list

        # Secondary_species (E.g. 'OH-' - string
        self.secondary_species_list = secondary_species_list
        self.gas_species_list = gas_species_list  # E.g. 'CO2(g)'

        # E.g. 'CO2(g)'
        self.passive_gas_species_list = passive_gas_species_list
        self.active_gas_species_list = active_gas_species_list  # E.g. 'CO2(g)'
        self.minerals_list = minerals_list  # E.g. 'Calcite'

        # has pchemistry_m_kinetic assigned to it
        self.m_kinetics_list = m_kinetics_list
        self.log_formulation = log_formulation
        self.truncate_concentration = truncate_concentration
        self.activity_water = activity_water
        self.max_dlnc = max_dlnc
        self.max_relative_change_tolerance = max_relative_change_tolerance
        self.max_residual_tolerance = max_residual_tolerance
        self.update_mineral_surface_area = update_mineral_surface_area
        self.use_full_geochemistry = use_full_geochemistry
        self.update_permeability = update_permeability
        self.update_porosity = update_porosity
        self.no_bdot = no_bdot
        self.no_checkpoint_act_coefs = no_checkpoint_act_coefs
        self.general_reaction = general_reaction
        self.sorption = None
        self.microbial_reaction = microbial_reaction
        self.radioactive_decay_reaction = radioactive_decay_reaction
        self.immobile_decay_reaction = immobile_decay_reaction
        self.immobile_species_list = immobile_species_list

        if pflotran_dir:
            self.database = pflotran_dir + '/database/hanford.dat'
        else:
            self.database = database  # Database path (String)

        self.activity_coefficients = activity_coefficients
        self.molal = molal  # boolean
        # incl. molarity/all, species and mineral names - string
        self.output_list = output_list
        self._freeze()

    class pimmobile_decay_reaction(Frozen):
        '''
        Specifies parameters for first-order decay of an immobile species.

        :param species_name: Name of immobile species to undergo first-order
         decay.
        :type species_name: str
        :param rate_constant: First-order rate constant [1/sec].
         Can set rate_constant or half_life but not both.
        :type rate_constant: float
        :param half_life: Half life of species [sec].
         Can set rate_constant or half_life but not both.
        :type half_life: float
        '''

        def __init__(self, species_name=None, rate_constant=None,
                     half_life=None):

            assert isinstance(species_name, (str, type(None))
                              ), 'SPECIES_NAME must be a string'
            assert isinstance(rate_constant, (float, int, type(
                None))), 'RATE_CONSTANT must be a float'
            assert isinstance(half_life, (float, int, type(None))
                              ), 'HALF_LIFE must be a float'
            assert half_life is None or rate_constant is None, \
                'Only one of HALF_LIFE or RATE_CONSTANT can be set!'

            self.species_name = species_name
            self.rate_constant = rate_constant
            self.half_life = half_life

        def _write(self, outfile):
            if self.rate_constant is not None and self.half_life is not None:
                PyFLOTRAN_WARNING(
                    'Both RATE_CONSTANT and HALF_LIFE are defined. '
                    'Reverting to RATE_CONSTANT')

            if self.rate_constant is not None:
                rxn_type = 'RATE_CONSTANT'
                rxn_val = strD(self.rate_constant)
            elif self.half_life is not None:
                rxn_type = 'HALF_LIFE'
                rxn_val = strD(self.half_life)
            else:
                perror('Must define either RATE_CONSTANT or HALF_LIFE')

            outfile.write('  IMMOBILE_DECAY_REACTION\n')
            outfile.write('    SPECIES_NAME %s\n' % self.species_name)
            outfile.write('    %s %s\n' % (rxn_type, rxn_val))
            outfile.write('  /\n')

    class pradioactive_decay_reaction(Frozen):
        '''
        Specifies parameters for radioactive decay reaction.
        This reaction differs from the GENERAL_REACTION in that only one
        reactant species may be specified with a unit stoichiometry
        (i.e. the rate is always first order) and the reactant species is 
        decayed in both the aqueous and sorbed phases.

        :param reaction: Reaction equation. Only one reactant species may be
        listed on the left side of the equation
        (i.e. or on the right side with a negative stoichiometry).
        The reactant's stoichiometry is fixed at 1.0.
        The forward rate is applied to that one species as a first order rate
        constant [1/sec]. Multiple species are supported as daughter products
        on the right hand side and stoichiometries can be specified.
        :type reaction: str
        :param rate_constant: Rate constant for 1st-order decay reaction
         [1/sec, default units].
        The rate constant may be calculated from -ln(0.5) / half-life.
        :type rate_constant: float or list<float,str>.
        :param half_life: Half life of species [sec, default units].
        :type half_life: float or list<float,str>.
        '''

        def __init__(self, reaction=None, rate_constant=None,
                     half_life=None):

            assert isinstance(reaction, (str, type(None))
                              ), 'REACTION must be a string'
            assert isinstance(rate_constant, (float, int, list, type(
                None))), 'RATE_CONSTANT must be a float or a list: [float,str]'
            assert isinstance(half_life, (float, int, list, type(
                None))), 'HALF_LIFE must be a float or a list: [float,str]'
            assert half_life is None or rate_constant is None, \
            'Only one of HALF_LIFE or RATE_CONSTANT can be set!'

            self.reaction = reaction
            self.rate_constant = rate_constant
            self.half_life = half_life

        def _write(self, outfile):
            if self.rate_constant is not None and self.half_life is not None:
                PyFLOTRAN_WARNING(
                    'Both RATE_CONSTANT and HALF_LIFE are defined. '
                     'Reverting to RATE_CONSTANT')

            if self.rate_constant is not None:
                rxn_type = 'RATE_CONSTANT'
                rxn_val = self.rate_constant
            elif self.half_life is not None:
                rxn_type = 'HALF_LIFE'
                rxn_val = self.half_life
            else:
                perror('Must define either RATE_CONSTANT or HALF_LIFE')

            # Handle the case where: rxn_val -> [coeff., units]
            if isinstance(rxn_val, list):
                rxn_val = strD(rxn_val[0]) + ' ' + str(rxn_val[1])
            else:
                rxn_val = strD(rxn_val)

            outfile.write('  RADIOACTIVE_DECAY_REACTION\n')
            outfile.write('    REACTION %s\n' % self.reaction)
            outfile.write('    %s %s\n' % (rxn_type, rxn_val))
            outfile.write('  /\n')

    class pmicrobial_reaction(Frozen):
        '''
        Specifies parameters for microbially-mediated reactions.

        :param reaction: Reaction equation. The rate constant is
         multiplied by the Monod expressions for electron donor and acceptor
         for select species on the left
         side of the equation. The reaction may be inhibited by any
         species in the system.
        :type reaction: str
        :param rate_constant: Rate constant for the reaction, where
         the units are [mol/L-sec]
         if no biomass, or [mol-m3 bulk/(L water-mol biomass-sec)] if biomass.
        :type rate_constant: float
        :param monod: Specifies the Monod equation for the electron donor
         or acceptor.
        :type monod: pmicrobial_reaction.monod
        :param inhibition: Specifies inhibition based on species concentration
         and an
         inhibition constant(s). Three types of inhibition are
         currently supported:
         MONOD, INVERSE_MONOD, THRESHOLD.
        :type inhibition: pmicrobial_reaction.inhibition
        :param biomass: Specifies the immobile biomass species to be
         included in the rate expression.
        :type biomass: pmicrobial_reaction.biomass
        '''

        class monod(Frozen):

            def __init__(self, species_name=None,
                         half_saturation_constant=None,
                        threshold_concentration=None):
                self.species_name = species_name
                self.half_saturation_constant = half_saturation_constant
                self.threshold_concentration = threshold_concentration

        class inhibition(Frozen):

            def __init__(self, species_name=None,
                         inhibition_type=None,
                         inhibition_constant=None):
                self.species_name = species_name
                self.inhibition_type = inhibition_type
                self.inhibition_constant = inhibition_constant

        class biomass(Frozen):

            def __init__(self, species_name=None, biomass_yield=None):
                self.species_name = species_name
                self.biomass_yield = biomass_yield

        def __init__(self, reaction=None, rate_constant=None,
                     monod=None, inhibition=None,
                     biomass=None):

            assert isinstance(reaction, (str, type(None))
                              ), 'REACTION must be a string'
            assert isinstance(rate_constant, (float, int, type(
                None))), 'RATE_CONSTANT must be a float'

            if monod is None:
                monod = []
            if inhibition is None:
                inhibition = []
            if biomass is None:
                biomass = []

            self.reaction = reaction
            self.rate_constant = rate_constant
            self.monod = monod
            self.inhibition = inhibition
            self.biomass = biomass

        def add_monod(self, species_name=None,
                      half_saturation_constant=None,
                      threshold_concentration=None):
            mon = pchemistry.pmicrobial_reaction.monod(
                species_name=species_name,
                half_saturation_constant=half_saturation_constant,
                threshold_concentration=threshold_concentration)
            self.monod.append(mon)
            return mon

        def add_biomass(self, species_name=None, biomass_yield=None):
            biom = pchemistry.pmicrobial_reaction.biomass(
                species_name=species_name, biomass_yield=biomass_yield)
            self.biomass.append(biom)
            return biom

        def add_inhibition(self, species_name=None, inhibition_type=None,
                           inhibition_constant=None):
            inhibit = pchemistry.pmicrobial_reaction.inhibition(
                species_name=species_name, inhibition_type=inhibition_type,
                inhibition_constant=inhibition_constant)
            self.inhibition.append(inhibit)
            return inhibit

        def _write(self, outfile):
            if self.reaction is None and self.rate_constant is None:
                perror('Required MICROBIAL_REACTION cards are missing!')

            outfile.write('  MICROBIAL_REACTION\n')
            outfile.write('    REACTION %s\n' % self.reaction)
            outfile.write('    RATE_CONSTANT %s\n' % strD(self.rate_constant))

            if self.monod is not None:
                for mon in self.monod:
                    outfile.write('    MONOD\n')
                    outfile.write('      SPECIES_NAME %s\n' % mon.species_name)
                    if mon.half_saturation_constant is not None:
                        outfile.write('      HALF_SATURATION_CONSTANT %s\n' %
                                      strD(mon.half_saturation_constant))
                    if mon.threshold_concentration is not None:
                        outfile.write('      THRESHOLD_CONCENTRATION %s\n' %
                                      strD(mon.threshold_concentration))
                    outfile.write('    /\n')

            if self.inhibition is not None:
                for inhib in self.inhibition:
                    outfile.write('    INHIBITION\n')
                    outfile.write('      SPECIES_NAME %s\n' %
                                  inhib.species_name)
                    outfile.write('      TYPE %s\n' % inhib.inhibition_type)
                    outfile.write('      INHIBITION_CONSTANT %s\n' %
                                  strD(inhib.inhibition_constant))
                    outfile.write('    /\n')

            if self.biomass is not None:
                for biom in self.biomass:
                    outfile.write('    BIOMASS\n')
                    outfile.write('      SPECIES_NAME %s\n' %
                                  biom.species_name)
                    outfile.write('      YIELD %s\n' % biom.biomass_yield)
                    outfile.write('    /\n')

            outfile.write('  /\n')

    def add_immobile_decay_reaction(self, species_name=None,
                                    rate_constant=None, half_life=None):
        idr_rxn = pchemistry.pimmobile_decay_reaction(
            species_name=species_name, rate_constant=rate_constant,
            half_life=half_life)
        self.immobile_decay_reaction.append(idr_rxn)
        return idr_rxn

    def add_radioactive_decay_reaction(self,reaction=None,
                                       rate_constant=None,
                                       half_life=None):
        rad_rxn = pchemistry.pradioactive_decay_reaction(reaction=reaction,
                               rate_constant=rate_constant, half_life=half_life)
        self.radioactive_decay_reaction.append(rad_rxn)
        return rad_rxn

    def add_microbial_reaction(self,reaction=None,rate_constant=None,
                               monod_species_name=None,
                               monod_half_saturation_constant=None,
                               monod_threshold_concentration=None,
                               inhibition_species_name=None,
                               inhibition_type=None,inhibition_constant=None,
                               biomass_species_name=None,biomass_yield=None):

        microbe = pchemistry.pmicrobial_reaction(reaction=reaction,
                                                rate_constant=rate_constant)
        self.microbial_reaction.append(microbe)
        return microbe

    class psorption(Frozen):
        '''
        Specifies parameters for sorption reactions.

        :param ion_exchange_rxn: Sorption defined through ion exchange reactions
        :type ion_exchange_rxn: class pchemistry.psorption.pion_exchange_rxn
        :param isotherm_reactions: Sorption reactions defined by isotherms (e.g.
        linear, Langmuir, Freundlich).
        :type isotherm_reactions: class pchemistry.psorption.pisotherm_reactions
        :param surface_complexion_rxn: Opens surface complexation reaction block
        :type surface_complexion_rxn: psorption.psurface_complexion_rxn
        '''

        def __init__(self, ion_exchange_rxn=None, isotherm_reactions=None,
                    surface_complexation_rxn=None):

            assert isinstance(ion_exchange_rxn,\
              (pchemistry.psorption.pion_exchange_rxn,type(None))),\
              'Must be an instance of pion_exchange_rxn'
            #assert isinstance(isotherm_reactions,\
            #  (pchemistry.psorption.pisotherm_reactions, type(None))),\
            #  'Must be an instance of pisotherm_reactions'
            assert isinstance(surface_complexation_rxn,\
              (pchemistry.psorption.psurface_complexation_rxn, type(None))),\
              'Must be an instance of psurface_complexation_rxn'

            if isotherm_reactions is None:
                isotherm_reactions = []

            self.ion_exchange_rxn = ion_exchange_rxn
            self.isotherm_reactions = isotherm_reactions
            self.surface_complexation_rxn = surface_complexation_rxn

        def _write(self,outfile):

            outfile.write('  SORPTION\n')

            if self.ion_exchange_rxn is not None:
                outfile.write('    ION_EXCHANGE_RXN\n')
                if self.ion_exchange_rxn.mineral is not None:
                    outfile.write('      MINERAL %s\n' %
                                  self.ion_exchange_rxn.mineral)
                if self.ion_exchange_rxn.cec is not None:
                    if isinstance(self.ion_exchange_rxn.cec, list):
                        outfile.write('      CEC %s %s\n' % \
                                     (strD(self.ion_exchange_rxn.cec[0]),
                                      self.ion_exchange_rxn.cec[1]))
                    else:
                        outfile.write('      CEC %s \n' %
                                      strD(self.ion_exchange_rxn.cec))
                if self.ion_exchange_rxn.cations is not None:
                    outfile.write('      CATIONS\n')
                    for cat in self.ion_exchange_rxn.cations:
                        is_ref = 'REFERENCE' if cat.reference == True else ''
                        outfile.write('        %s %s %s\n' %
                                      (cat.name.ljust(8), cat.value, is_ref))
                    outfile.write('      /\n')
                outfile.write('    /\n')


            if self.isotherm_reactions is not None:
                outfile.write('    ISOTHERM_REACTIONS\n')
                for i_rxn in self.isotherm_reactions:
                    outfile.write('      %s\n' % i_rxn.name)

                    if i_rxn.ir_type is not None:
                        outfile.write('        TYPE %s\n' % i_rxn.ir_type)

                    if i_rxn.distribution_coefficient is not None:
                        _temp_real = i_rxn.distribution_coefficient
                        if isinstance(_temp_real,Coeff):
                            outfile.write('        DISTRIBUTION_COEFFICIENT %s\n'\
                                          % str(_temp_real))
                        else:
                            outfile.write('        DISTRIBUTION_COEFFICIENT %s\n'\
                                          % strD(i_rxn.distribution_coefficient))

                    if i_rxn.langmuir_b is not None:
                        outfile.write('        LANGMUIR_B %s\n' % \
                                      strD(i_rxn.langmuir_b))

                    if i_rxn.freundlich_n is not None:
                        outfile.write('        FREUNDLICH_N %s\n' % \
                                      strD(i_rxn.freundlich_n))

                    if i_rxn.kd_mineral_name is not None:
                        outfile.write('        KD_MINERAL_NAME %s\n' % \
                                      i_rxn.kd_mineral_name)

                    outfile.write('      /\n')
                outfile.write('    /\n')

            if self.surface_complexation_rxn is not None:
                self.surface_complexation_rxn.write(outfile)

            outfile.write('  /\n')


        def add_ion_exchange_rxn(self, cec=None, cations=None, mineral=None):
            '''
            Add an ION_EXCHANGE_RXN block to SORPTION.

            :param cec: Cation exchange capacity in (1) equivalents per volume
            of mineral [eq/m^3_mineral] or (2) equivalents per bulk volume
            [eq/m^3_bulk]
            :type cec: float
            :param cations: Opens the CATIONS block for listing cations 
            participating in the reaction.
            :type cations: list of <pchemistry.psorption.pion_exchange_rxn.cation>
            :param mineral: Name of the mineral to which the cations sorb.
            :type mineral: str
            '''

            self.ion_exchange_rxn = pchemistry.psorption.pion_exchange_rxn(
              cec=cec, cations=cations, mineral=mineral)
            return self.ion_exchange_rxn

        def add_isotherm_reactions(self, name=None,
                                   distribution_coefficient=None, ir_type=None,
                                   langmuir_b=None, freundlich_n=None,
                                   kd_mineral_name=None):
            '''
            Specifies parameters for a sorption reaction defined by an isotherm
            (e.g. linear, Langmuir, Freundlich).

            :param name: Name of primary species that sorbs.
            :type name: str
            :param distribution_coefficient: The value of K_D.
            :type distribution_coefficient: float
            :param ir_type: Type of isotherm, one of:
            ['linear','langmuir','freundlich']
            :type ir_type: str
            :param langmuir_b: b coefficient for Langmuir isotherm.
            Automatically sets the ir_type to langmuir.
            :type langmuir_b: float
            :param freundlich_n: n exponent in Freundlich isotherm.
            Automatically sets the ir_type to Freundlich.
            :type freundlich_n: float
            :param kd_mineral_name: Name of mineral. See PFLOTRAN documentation
            for more information.
            :type kd_mineral_name: str
            '''
            i_rxn = pchemistry.psorption.pisotherm_reactions(
                name=name,
                distribution_coefficient=distribution_coefficient,
                ir_type=ir_type, langmuir_b=langmuir_b,
                freundlich_n=freundlich_n,
                kd_mineral_name=kd_mineral_name)

            self.isotherm_reactions.append(i_rxn)
            return i_rxn

        def add_surface_complexation_rxn(self, sorption_type=None,
                                         complex_kinetics=None,rates=None,
                                         site_fraction=None, mineral=None,
                                         multirate_scale_factor=None,
                                         colloid=None, rock_density=None,
                                         site=None, complexes=None):
            self.surface_complexation_rxn = \
            pchemistry.psorption.psurface_complexation_rxn(
                sorption_type=sorption_type,
                complex_kinetics=complex_kinetics, rates=rates,
                site_fraction=site_fraction,
                mineral=mineral, multirate_scale_factor=multirate_scale_factor,
                colloid=colloid, rock_density=rock_density,
                site=site, complexes=complexes)
            return self.surface_complexation_rxn

        class pion_exchange_rxn(Frozen):
            '''
            Add an ION_EXCHANGE_RXN block to SORPTION.

            :param cec: Cation exchange capacity in (1) equivalents per volume
            of mineral [eq/m^3_mineral] or (2) equivalents per bulk volume
            [eq/m^3_bulk]
            :type cec: float
            :param cations: Opens the CATIONS block for listing cations
            participating in the reaction.
            :type cations: list <pchemistry.psorption.pion_exchange_rxn.cation>
            :param mineral: Name of the mineral to which the cations sorb.
            :type mineral: str
            '''

            def __init__(self, cec=None, cations=None, mineral=None):

                assert isinstance(cec, (int, float, type(None))), \
                                'ion_exchange_rxn.cec must be a number'
                assert isinstance(mineral, (str, type(None))), \
                                'ion_exchange_rxn.mineral must be a string'

                if cations is None:
                    cations = []

                self.cec = cec
                self.cations = cations
                self.mineral = mineral

            class cation(Frozen):
                '''
                Cation participating in an ION_EXCHANGE_RXN.

                :param name: Name of cation 
                :type name: str
                :param value: Associated selectivity coefficient
                :type value: float
                :param reference: Single reference cation with selectivity
                coeff. of 1 relative to other cations in rxn
                :type reference: bool
                '''

                def __init__(self, name=None, value=None, reference=False):
                    assert isinstance(name, (str, type(None))),\
                    'cation.name must be a string'
                    assert isinstance(value, (int, float, type(None))),\
                    'cation.value must be a number'
                    assert isinstance(reference, bool),\
                    'cation.reference must be a boolean'

                    self.name = name
                    self.value = value
                    self.reference = reference

            def add_cation(self, name=None, value=None, reference=False):
                '''
                Adds a new cation to an ion exchange reaction.
                '''
                new_cation = pchemistry.psorption.pion_exchange_rxn.cation(
                    name=name, value=value, reference=reference)
                self.cations.append(new_cation)
                return new_cation

        class pisotherm_reactions(Frozen):
            '''
            Specifies parameters for a sorption reaction defined by an isotherm
            (e.g. linear, Langmuir, Freundlich).

            :param name: Name of primary species that sorbs.
            :type name: str
            :param distribution_coefficient: The value of K_D.
            :type distribution_coefficient: float
            :param ir_type: Type of isotherm, where the options for <string>
            include: ['linear','langmuir','freundlich']
            :type ir_type: str
            :param langmuir_b: b coefficient for Langmuir isotherm.
            Automatically sets the ir_type to langmuir.
            :type langmuir_b: float
            :param freundlich_n: n exponent in Freundlich isotherm.
            Automatically sets the ir_type to Freundlich.
            :type freundlich_n: float
            :param kd_mineral_name: Name of mineral. See PFLOTRAN documentation
            for more information.
            :type kd_mineral_name: str
            '''

            def __init__(self,name=None,distribution_coefficient=None,
                          ir_type=None,langmuir_b=None,freundlich_n=None,
                          kd_mineral_name=None):

                self.name = name
                self.distribution_coefficient = distribution_coefficient
                self.ir_type = ir_type
                self.langmuir_b = langmuir_b
                self.freundlich_n = freundlich_n
                self.kd_mineral_name = kd_mineral_name

        class psurface_complexation_rxn(Frozen):
            '''
            Specifies parameters for a surface complexation reaction.

            :param sorption_type: One of:
            ['equilibrium','multirate_kinetic','kinetic']
            :type sorption_type: str
            :param complex_kinetics: Opens a block specifying forward and
            backward rate constants
            CLASS TYPE
            :type complex_kinetics: <UNDEFINED>
            :param rates: Specific kinetic rates associated with SITE_FRACTIONs.
            :type rates: list <float>
            :param site_fraction: Specifies site fractions associated with RATES
            for multirate kinetic sorption.
            :type site_fraction: list <float>
            :param mineral: The name of the mineral with which the sorption site
            density is associated
            :type mineral: str
            :param multirate_scale_factor: Floating point number that scales all
            rate constants for multirate kinetic sorption.
            :type multirate_scale_factor: float
            :param colloid: Name of the colloid associated with surface
            complexation reaction
            :type colloid: str
            :param rock_density: a flag which allows the calculation of surface
            site concentration based on rock density
            :type rock_density: bool
            :param site: name of site and site density [mol/m^3]
            :type site: list [str,float]
            :param complexes: opens a block listing the names of surface
            complexes associated with the surface complexation reaction and the
            surface site.
            :type complexes: 
            '''

            def __init__(self,sorption_type=None,complex_kinetics=None,
                         rates=None,site_fraction=None,mineral=None,
                         multirate_scale_factor=None,colloid=None,
                         rock_density=None,site=None,complexes=None,
                         equilibrium=False,multirate_kinetic=False):

                if complexes is None:
                    complexes = []

                #PyFLOTRAN_ERROR('Functionality not yet implemented!')
                self.sorption_type = sorption_type
                self.complex_kinetics = complex_kinetics
                self.rates = rates
                self.site_fraction = site_fraction
                self.mineral = mineral
                self.multirate_scale_factor = multirate_scale_factor
                self.colloid = colloid
                self.rock_density = rock_density
                self.site = site
                self.complexes = complexes
                self.equilibrium = equilibrium
                self.multirate_kinetic = multirate_kinetic
                self._freeze()

            def write(self,outfile):
                outfile.write('    SURFACE_COMPLEXATION_RXN\n')
                if self.multirate_kinetic:
                    outfile.write('      MULTIRATE_KINETIC\n')
                if self.site_fraction:
                    outfile.write('      SITE_FRACTION \\\n')
                    _v = self.site_fraction
                    s = ""
                    for i in range(0,len(_v),5):
                        if (i+4 < len(_v)):
                            s += ' '.join([str(x) for x in _v[i:i+4]])+" \\\n"
                        else:
                            s += ' '.join([str(x) for x in _v[i:]])+"\n"
                    outfile.write(s)
                if self.rates:
                    outfile.write('      RATES \\\n')
                    _v = self.rates
                    s = ""
                    for i in range(0,len(_v),5):
                        if (i+4 < len(_v)):
                            s += ' '.join([str(x) for x in _v[i:i+4]])+"\\\n"
                        else:
                            s += ' '.join([str(x) for x in _v[i:]])+"\n"
                    outfile.write(s)
                if self.multirate_scale_factor:
                    outfile.write('      MULTIRATE_SCALE_FACTOR %s\n' % strD(self.multirate_scale_factor))
                if self.mineral:
                    outfile.write('      MINERAL %s\n' % self.mineral)
                if self.site:
                    outfile.write('      SITE %s %s\n' % (self.site[0],strD(self.site[1])))
                if self.complexes:
                    outfile.write('      COMPLEXES\n')
                    for _complex in self.complexes:
                        outfile.write('        %s\n' % _complex)
                    outfile.write('      /\n')
                outfile.write('    /\n')

            class complex_kinetics(Frozen):

                def __init__(self,forward_rate_constant=None,backward_rate_constant=None):
                    self.forward_rate_constant = forward_rate_constant
                    self.backward_rate_constant = backward_rate_constant
                    self._freeze()

                def write(self,outfile):

                    outfile.write('    COMPLEX_KINETICS\n')

                    if forward_rate_constant is not None:
                        outfile.write('      FORWARD_RATE_CONSTANT %s\n' % strD(self.forward_rate_constant))
                    if backward_rate_constant is not None:
                        outfile.write('      FORWARD_RATE_CONSTANT %s\n' % strD(self.backward_rate_constant))

                    outfile.write('    /\n')

    def add_sorption(self, ion_exchange_rxn=None, isotherm_reactions=None,
                     surface_complexation_rxn=None):
        '''
        Adds a SORPTION block to CHEMISTRY.
        Use member functions add_ion_exchange_rxn, add_isotherm_reactions, and
        add_surface_complexation_rxn
        to fill out the SORPTION block.
        '''

        self.sorption = pchemistry.psorption(ion_exchange_rxn=ion_exchange_rxn,
          isotherm_reactions=isotherm_reactions,
          surface_complexation_rxn=surface_complexation_rxn)
        return self.sorption

    class pgeneral_reaction(Frozen):
        '''
        CHEMISTRY: GENERAL_REACTION
        Specifies parameters for general forward/reverse kinetic reaction.

        Example:

          >>> chem.add_general_reaction(reaction='Tracer <-> Tracer2',
                                      forward_rate=1.7584e-7,backward_rate=0.0)

        :param reaction: Reaction equation. The forward rate is applied to the
        reaction quotient
        of species on the left side of the reaction. The reverse or backward
        rate is applied to the right side.
        :type reaction: str
        :param forward_rate: Rate constant for nth-order forward reaction
        [kg-water(n-1)/mol(n-1) -sec]
        :type forward_rate: float or int
        :param backward_rate: Rate constant for nth-order reverse reaction
        [kg-water(n-1)/mol(n-1) -sec]
        :type backward_rate: float or int
        '''

        def __init__(self,reaction=None,forward_rate=None,backward_rate=None):

            assert isinstance(reaction, (str, type(None))),\
            'chem.general_reaction.reaction must be a string'
            assert isinstance(forward_rate, (int, float, type(None))),\
            'chem.general_reaction.forward_rate must be a float or int'
            assert isinstance(backward_rate, (int, float, type(None))),\
            'chem.general_reaction.backward_rate must be a float or int'

            self.reaction = reaction
            self.forward_rate = forward_rate
            self.backward_rate = backward_rate

            self._freeze()

    def add_general_reaction(self,reaction=None,forward_rate=None,
                             backward_rate=None):
        '''
        CHEMISTRY: GENERAL_REACTION
        Specifies parameters for general forward/reverse kinetic reaction.

        Example:

          >>> chem.add_general_reaction(reaction='Tracer <-> Tracer2',
                                      forward_rate=1.7584e-7,backward_rate=0.0)

        :param reaction: Reaction equation. The forward rate is applied to the
        reaction quotient
        of species on the left side of the reaction. The reverse or backward
        rate is applied to the right side.
        :type reaction: str
        :param forward_rate: Rate constant for nth-order forward reaction 
        [kg-water(n-1)/mol(n-1) -sec]
        :type forward_rate: float or int
        :param backward_rate: Rate constant for nth-order reverse reaction
        [kg-water(n-1)/mol(n-1) -sec]
        :type backward_rate: float or int

        Returns:

        :param self.general_reaction: instance of pchemistry_general_reaction
        :type self.general_reaction: pchemistry_general_reaction
        '''

        assert isinstance(reaction, (str, type(None))),\
        'chem.general_reaction.reaction must be a string'
        assert isinstance(forward_rate, (int, float, type(None))),\
        'chem.general_reaction.forward_rate must be a float or int'
        assert isinstance(backward_rate, (int, float, type(None))),\
        'chem.general_reaction.backward_rate must be a float or int'

        self.general_reaction = pchemistry.pgeneral_reaction()
        self.general_reaction.reaction = reaction
        self.general_reaction.forward_rate = forward_rate
        self.general_reaction.backward_rate = backward_rate

        return self.general_reaction


class pchemistry_m_kinetic(Frozen):
    """
    Class of pchemistry. Mineral kinetics are assigned
    to m_kinetics_list in pchemistry. The add function can do
    this automatically. e.g., dat.add(mineral_kinetic).

    :param name: Mineral name.
    :type name: str
    :param rate_constant_list: Value, Unit of Measurement. e.g.,
     rate_constant_list=[1.e-6, 'mol/m^2-sec']
    :type rate_constant_list: [float, str]
    """

    class prefactor(Frozen):
        """
        Class of pchemistry_m_kinetic. Contains PREFACTOR parameters for a
        MINERAL_KINETIC object.

        :param rate_constant: Kinetic rate constant [mol/m2/sec].
        :type rate_constant: [float, str]
        :param activation_energy: Activation energy for rate constant k_{ml}
        :type activation_energy: float
        :param pf_species: One or more PREFACTOR_SPECIES objects assigned to
        this PREFACTOR instance
        :type pf_species: list containing pchemistry_m_kinetic.prefactor_species
        objects
        """

        def __init__(self,rate_constant=None,activation_energy=None,
                     pf_species=None):

            # Ensure the types are correct
            assert isinstance(activation_energy, (int,float,type(None))),\
            'ACTIVATION_ENERGY must be a float'
            assert isinstance(rate_constant, (list, tuple, type(None))),\
            'RATE_CONSTANT must be a list of [float,string]'
            assert isinstance(pf_species, (list,\
              pchemistry_m_kinetic.prefactor_species,type(None))),\
            'PREFACTOR_SPECIES must be a list or instance of '+\
            'pchemistry_m_kinetic.prefactor.prefactor_species'

            # Assign class parameters
            self.rate_constant = rate_constant
            self.activation_energy = activation_energy
            self.pf_species = [prefactor_species] if isinstance(pf_species,\
              pchemistry_m_kinetic.prefactor_species) else pf_species

            if self.pf_species is None:
                self.pf_species = []
            if self.rate_constant is None:
                self.rate_constant = []

            # Freeze
            self._freeze()

        def add_species(self,name,alpha=None,beta=None,attenuation_coef=None):
            """
            Add a PREFACTOR_SPECIES instance to a PREFACTOR object.

            :param name: Prefactor species name
            :type name: str
            :param alpha: a_{jl}^{m} in mineral precipitation-dissolution
            reaction equation
            :type alpha: float
            :param beta: B_{jl}^{m} in mineral precipitation-dissolution
            reaction equation
            :type beta: float
            :param attenuation_coef: K_{jl} in mineral precipitation-dissolution
            reaction equation
            :type attenuation_coef: float
            """

            assert isinstance(
                alpha,(int,float,type(None))), 'ALPHA must be a float'
            assert isinstance(
                beta, (int,float,type(None))), 'BETA must be a float'
            assert isinstance(attenuation_coef, (int,float,type(
                None))), 'ATTENUATION_COEF must be a float'

            pf_spec = pchemistry_m_kinetic.prefactor_species(name=name,
              alpha=alpha,beta=beta,attenuation_coef=attenuation_coef)
            self.pf_species.append(pf_spec)
            return pf_spec

    class prefactor_species(Frozen):
        """
        Define a PREFACTOR_SPECIES to be used in conjunction with PREFACTOR for
        mineral precipitation-dissolution reactions.

        :param name: Prefactor species name
        :type name: str
        :param alpha: a_{jl}^{m} in mineral precipitation-dissolution reaction
        equation
        :type alpha: float
        :param beta: B_{jl}^{m} in mineral precipitation-dissolution reaction
        equation
        :type beta: float
        :param attenuation_coef: K_{jl} in mineral precipitation-dissolution
        reaction equation
        :type attenuation_coef: float
        """

        def __init__(self,name=None,alpha=None,beta=None,attenuation_coef=None):
            self.name = name
            self.alpha = alpha
            self.beta = beta
            self.attenuation_coef = attenuation_coef

    def __init__(self, name=None, rate_constant_list=None,
                 activation_energy=None, affinity_threshold=None,
                 rate_limiter=None, irreversible=None,
                 surface_area_porosity_power=None,
                 surface_area_vol_frac_power=None,
                 prefactors=None):

        # Verify correct input types
        assert isinstance(affinity_threshold, (int,float,type(None))),\
        'AFFINITY_THRESHOLD must be a float'
        assert isinstance(rate_constant_list, (list,tuple,type(None))),\
        'RATE_CONSTANT must be a list of [float,string]'
        assert isinstance(rate_limiter, (int,float,type(None))),\
        'RATE_LIMITER must be a float'
        assert isinstance(irreversible, (bool,type(None))),\
        'IRREVERSIBLE must be a boolean'
        assert isinstance(surface_area_porosity_power, \
          (int,float,type(None))),\
          'SURFACE_AREA_POROSITY_POWER must be a float'
        assert isinstance(surface_area_vol_frac_power, \
          (int,float,type(None))), \
          'SURFACE_AREA_VOL_FRAC_POWER must be a float'
        assert isinstance(prefactors, (list, self.prefactor, type(None))),\
          'PREFACTOR must be a list or instance of pchemistry_m_kinetic.prefactor'

        # Construct MINERAL_KINETICS object
        if rate_constant_list is None:
            rate_constant_list = []

        # Build MINERAL_KINETICS parameters
        self.name = name
        self.rate_constant_list = rate_constant_list
        self.activation_energy = activation_energy
        self.affinity_threshold = affinity_threshold
        self.rate_limiter = rate_limiter
        self.irreversible = irreversible
        self.surface_area_porosity_power = surface_area_porosity_power
        self.surface_area_vol_frac_power = surface_area_vol_frac_power

        # Turn PREFACTOR into a list if not already
        self.prefactors = [prefactors] if isinstance(
            prefactors, self.prefactor) else prefactors
        if prefactors is None:
            self.prefactors = []

        # Freeze
        self._freeze()

    def add_prefactor(self, rate_constant=None, activation_energy=None,
                      pf_species=None):
        """
        Add a new PREFACTOR instance to MINERAL_KINETICS object

        :param rate_constant: Kinetic rate constant [mol/m2/sec].
        :type rate_constant: [float, str]
        :param activation_energy: Activation energy for rate constant k_{ml}
        :type activation_energy: float
        :param pf_species: One or more PREFACTOR_SPECIES objects assigned to
        this PREFACTOR instance
        :type pf_species: list containing pchemistry_m_kinetic.prefactor_species
        objects
        """
        pf_instance = self.prefactor(rate_constant=rate_constant,
                      activation_energy=activation_energy,pf_species=pf_species)
        self.prefactors.append(pf_instance)
        return pf_instance

    def view(self):
        """
        View the metadata of a pchemistry_m_kinetic object.

        All attributes of the object, along with any prefactor species, are
        written to stdout.

        """

        # Compute a string full of all associated prefactors
        prefrac_str = ""
        for pf in self.prefactors:

            # Compute a string full of all associated PREFACTOR_SPECIES
            prefrac_spec_str = ""
            for pfs in pf.pf_species:
                smeta = '''
        ----------------------------------------
        PREFACTOR_SPECIES: {}
        ----------------------------------------
            Attenutation coefficient: {}
            Alpha: {}
            Beta: {}
            '''.format(pfs.name, pfs.attenuation_coef,
                       pfs.alpha, pfs.beta)

                prefrac_spec_str += smeta

            pfmeta = '''
        ========================================
        PREFACTOR:
            Prefactor rate constant: {}
            Prefactor activation energy: {}
            Prefactor species count: {}{}
        '''.format(pf.rate_constant, pf.activation_energy,
                   len(pf.pf_species), prefrac_spec_str)

            prefrac_str += pfmeta

        # Display object attributes, along with any PREFACTOR
        # or PREFACTOR_SPECIES attached to the object
        meta = '''
        ----------------------------------------
        MINERAL_KINETICS: {}
        ----------------------------------------
            Rate constant: {}
            Activation energy: {}
            Affinity threshold: {}
            Rate limiter: {}
            Irreversible: {}
            Surface area porosity power: {}
            Surface area vol frac power: {}{}
        ----------------------------------------
        '''.format(self.name, self.rate_constant_list, self.activation_energy,
                   self.affinity_threshold, self.rate_limiter, self.irreversible,
                   self.surface_area_porosity_power,
                   self.surface_area_vol_frac_power,
                   prefrac_str)
        print(meta)

        #print("Full object dump:")
        # print(self.__dict__)


class ptransport(Frozen):
    """
    Class for specifying a transport condition. Multiple transport objects
    can be created. Specifies a transport condition based on various
    user-defined constraints with minerals, gases, pH, charge balance,
    free ion, and total concentrations.

    :param name: Transport condition name.
    :type name: str
    :param type: Options include: 'dirichlet', 'dirichlet_zero_gradient',
     'equilibrium', 'neumann', 'mole', 'mole_rate', 'zero_gradient'.
    :type type: str
    :param constraint_list_value: List of constraint values.
     The position of each value in the list correlates with the
     position of each type in constraint_list_type.
    :type constraint_list_value: [float]
    :param constraint_list_type: List of constraint types. The position
     of each value in the list correlates with
     the position of each value in constraint_list_value. E.g.,
     'initial_constraint', 'inlet_constraint'.
    :type constraint_list_type: [str]
    """

    def __init__(self, name='', tran_type='', constraint_list_value=None,
                 constraint_list_type=None, time_units=''):
        if constraint_list_value is None:
            constraint_list_value = []
        if constraint_list_type is None:
            constraint_list_type = []
        self.name = name  # e.g., initial, west, east
        self.type = tran_type  # e.g., dirichlet, zero_gradient
        self.constraint_list_value = constraint_list_value
        self.constraint_list_type = constraint_list_type
        self.time_units = time_units
        self._freeze()


class pconstraint(Frozen):
    """
    Class for specifying a transport constraint.  Multiple constraint objects
    can be created.

    :param name: Constraint name.
    :type name: str
    :param concentration_list: List of pconstraint_concentration objects.
    :type concentration_list: [pconstraint_concentration]. Works with add
     function so that concentration_list does not need to be remembered.
     E.g., dat.add(concentration). Used for key word CONC or CONCENTRATIONS
    :param mineral_list: List of pconstraint_mineral objects.
     Currently does not work with add function. Used for keyword MNRL
     OR MINERALS.
    :type mineral_list: [pconstraint_mineral]
    """

    def __init__(self, name='', concentration_list=None, mineral_list=None,
                 secondary_continuum=False,free_ion_guess_list=None):
        if concentration_list is None:
            concentration_list = []
        if mineral_list is None:
            mineral_list = []
        if free_ion_guess_list is None:
            free_ion_guess_list = []
        self.name = name.lower()
        # Composed of pconstraint_concentration objects
        self.concentration_list = concentration_list
        self.mineral_list = mineral_list  # list of minerals
        self.secondary_continuum = secondary_continuum
        self.free_ion_guess_list = free_ion_guess_list
        self._freeze()

    class pfree_ion(Frozen):
        def __init__(self,name,value):
            self.name = name
            self.value = value
            self._freeze()


class pconstraint_concentration(Frozen):
    """
    Concentration unit, Sub-class for constraint. There can be multiple
    pconstraint_concentration objects appended to a
    single pconstraint object. Works with add function so that
    concentration_list in pconstraint does not need to be
    remembered. e.g., dat.add(concentration) instead of
    dat.constraint.concentration_list.append(concentration).

    :param pspecies: Primary species name for concentration.
    :type pspecies: str
    :param value: Concentration value.
    :type value: float
    :param constraint: Constraint name for concentration.
     Options include: 'F', 'FREE', 'T', 'TOTAL', 'TOTAL_SORB', 'P',
     'pH', 'L', 'LOG', 'M', 'MINERAL', 'MNRL', 'G', 'GAS', 'SC',
     'CONSTRAINT_SUPERCRIT_CO2
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
    Class for mineral in a constraint with vol. fraction and surface area.
    There can be multiple pconstraint_concentration objects appended to a
    single pconstraint object. Currently does not work with add function.
    pconstraint_mineral can be manually appended to minerals_list
    in a pconstraint object. E.g., 'constraint.mineral_list.append(mineral)'.

    :param name: Mineral name.
    :type name: str
    :param volume_fraction: Volume fraction. [--]
    :type volume_fraction: float
    :param surface_area: Surface area. [m^-1]
    :type surface_area: float
    :param surface_area_units: Surface area units. [m2/m3 or cm2/cm3]
    :type surface_area_units: str

    """

    def __init__(self, name='', volume_fraction=None, surface_area=None,
                 surface_area_units=None):
        self.name = name
        self.volume_fraction = volume_fraction
        self.surface_area = surface_area
        self.surface_area_units = surface_area_units
        self._freeze()


class pquake(Frozen):
    """
    Class for specifying pflotran-qk3 related information

    :param mapping_file: Name of the mapping file
    :type name: str
    :param time_scaling: Time scaling factor
    :type time_scaling: float
    :param pressure_scaling: Pressure scaling factor
    :type pressure_scaling: float

    """

    def __init__(self, mapping_file='mapping.dat', time_scaling=1.0,
                 pressure_scaling=1.0):
        self.mapping_file = mapping_file
        self.time_scaling = time_scaling
        self.pressure_scaling = pressure_scaling
        self._freeze()


class pgeomech_subsurface_coupling(Frozen):
    """
    Class for specifying geomechanics-flow coupling

    :param mapping_file: Name of the mapping file
    :type name: str
    :param coupling_type: time of coupling -- one way / two way
    :type coupling_type: str

    """

    def __init__(self, mapping_file='flow_geomech_mapping.dat',
                 coupling_type='two_way_coupled'):
        self.mapping_file = mapping_file
        self.coupling_type = coupling_type
        self._freeze()


class pgeomech_material(Frozen):
    """
    Class for defining geomechanics material property.

    :param id: Unique identifier of material property.
    :type id: int
    :param name: Name of material property. e.g., 'soil1'.
    :type name: str
    :param density: Rock density of material in kg/m^3.
    :type density: float
    :youngs_modulus: Young's modulus of the material
    :type youngs_modulus: float
    :poissons_ratio: Poisson's ratio of the material
    :type poissons_ratio: float
    :biot_coefficient: Biot coefficient of the material
    :type biot_coefficient: float
    :thermal_expansion_coefficient: Coefficient of thermal expansion
    :type thermal_expansion_coefficient: float

    """

    def __init__(self, id=1, name='default', density='', youngs_modulus='',
                 poissons_ratio='', biot_coefficient='',
                 thermal_expansion_coefficient=''):
        self.id = id
        self.name = name
        self.density = density
        self.youngs_modulus = youngs_modulus
        self.poissons_ratio = poissons_ratio
        self.biot_coefficient = biot_coefficient
        self.thermal_expansion_coefficient = thermal_expansion_coefficient
        self._freeze()


class pgeomech_grid(Frozen):
    """
    Class for defining a geomech grid.

    :param gravity: Specifies gravity vector in m/s^2. Input is a list of
     3 floats.
    :type gravity: [float]*3
    :param filename: Specify name of file containing geomech grid information.
    :type filename: str

    """

    # definitions are put on one line to work better with rst/latex/sphinx.
    def __init__(self, dirname='geomech_dat',
                 gravity=[0.0, 0.0, -9.81], grid_filename='usg.mesh'):
        self.gravity = gravity
        self.dirname = dirname
        self.grid_filename = grid_filename
        self._freeze()


class pgeomech_time(Frozen):
    """
    Class for geomechanics time. Essentially has times for coupling with flow

    :param coupling_timestep: Frequency of coupling. e.g., [0.25e0, 'y']
    :type coupling_timestep: [float, str]
    """

    # definitions are put on one line to work better with rst/latex/sphinx.
    def __init__(self, coupling_timestep=[0.1, 'd']):
        self.coupling_timestep = coupling_timestep
        self._freeze()


class pgeomech_output(Frozen):
    """
    Class for dumping geomechanics output.
    Acceptable time units (units of measurements) are: 's', 'min', 'h',
     'd', 'w', 'mo', 'y'.

    :param time_list: List of time values. 1st variable specifies time unit
     to be used. Remaining variable(s) are floats
    :type time_list: [str, float*]
    :param print_column_ids: Flag to indicate whether to print column numbers
     in observation
     and mass balance output files. Default: False
    :type print_column_ids: bool - True or False
    :param format_list: Specify the file format for time snapshot of the
     simulation in time file type. Input is a list of strings.
     Multiple formats can be specified.
     File format options include: 'TECPLOT BLOCK' - TecPlot block format,
     'TECPLOT POINT' -- TecPlot point format (requires a single processor),
     'HDF5' -- produces single HDF5 file and xml for unstructured grids,
     'HDF5 MULTIPLE_FILES' -- produces a separate HDF5 file
     at each output time, 'VTK' - VTK format.
    :type format_list: [str]
    """

    # definitions are put on one line to work better with rst/latex/sphinx.
    def __init__(self, time_list=None, print_column_ids=False,
                 format_list=None):
        if time_list is None:
            time_list = []
        if format_list is None:
            format_list = []

        self.time_list = time_list
        self.print_column_ids = print_column_ids
        self.format_list = format_list
        self._freeze()


class pgeomech_regression(Frozen):
    """
    Class for specifying geomechanics regression.

    :param vertices: Specify cells for regression.
    :type vertices: list of int
    :param vertices_per_process: Specify the number cells per process.
    :type vertices_per_process: int
    """

    def __init__(self, vertices=None, vertices_per_process='',
                 variables=['displacement_z', 'strain_zz', 'stress_zz']):
        if vertices is None:
            vertices = []
        if variables is None:
            variables = []
        self.vertices = vertices
        self.vertices_per_process = vertices_per_process
        self.variables = variables
        self._freeze()


class preference_stress_state(Frozen):
    """
    Class for specifiying uniform reference stress state used in
    conjunction with BANDIS_UNCOUPLED keyword.

    :param value_list: List of stress components [xx yy zz xy yz zx]
     e.g., [-25.E+06 -25.E+06 -25.E+06 -0.E+06 -0.E+06 -0.E+06]
    :type value_list: [float,float,float,float,float,float]
    """

    def __init__(self, value_list=None):
        if value_list is None:
            value_list = []
        self.value_list = value_list
        self._freeze()


class peos(Frozen):
    """
    Class for specifiying equation of state (EOS).

    :param fluid_name: Selects the type of fluid (either water or gas).
    :type fluid_name: string
    :param fluid_density: Specifies option for fluid density including
     "default", "constant", "linear" and "exponential" options with optional
     trailing floats.  (e.g. to specify constant density of 1000 use
     ['constant',1000]).
    :type fluid_density: list
    :param fluid_viscosity: Specifies option for fluid viscosity.
     "Constant" is currently supported.
    :type fluid_viscosity: list
    :param fluid_enthalpy: Specifies option for fluid viscosity.
     "Constant" is currently supported.
    :type fluid_enthalpy: list
    :param density_params: an optional explicit key/value pairing of
    density parameters
    :type density_params: dict{str,float}
    """

    def __init__(self, fluid_name=None, fluid_density=['DEFAULT'],
                 fluid_viscosity=None, fluid_enthalpy=None,
                 fluid_henrys_constant=None, fluid_test=None,
                 fluid_formula_weight=None, rks=None,
                 density_params=None):
        if fluid_density is None:
            fluid_density = []
        if fluid_viscosity is None:
            fluid_viscosity = []
        if fluid_enthalpy is None:
            fluid_enthalpy = []
        if fluid_test is None:
            fluid_test = []
        self.fluid_name = fluid_name
        self.fluid_density = fluid_density
        self.fluid_viscosity = fluid_viscosity
        self.fluid_enthalpy = fluid_enthalpy
        self.fluid_henrys_constant = fluid_henrys_constant
        self.fluid_test = fluid_test
        self.fluid_formula_weight = fluid_formula_weight
        self.rks = rks
        self.density_params = density_params
        self._freeze()

    class pvt(Frozen):
        def __init__(self,pvt_type=None,pressure_units=None,rs_units=None,
                     fvf_units=None,viscosity_units=None,
                     compressibility_units=None,temperature_data=None):
            
            if temperature_data is None:
                temperature_data = []

            self.pvt_type = pvt_type
            self.pressure_units = pressure_units
            self.rs_units = rs_units
            self.fvf_units = fvf_units
            self.viscosity_units = viscosity_units
            self.temperature_data = temperature_data
            self.compressibility_units = compressibility_units


class prks(Frozen):
    """
    Class for RKS density EOS.

    :param hydrogen: set to 'hydrogen' or 'non-hydrogen'.
    :type fluid_name: string
    :param tc: critical temperature
    :type tc: float
    :param pc: critical pressure
    :type pc: float
    :param ac: acentric factor
    :type ac: float
    :param a: omegaa value
    :type a: float
    :param b: omegab value
    :type b: float
    """

    def __init__(self, hydrogen=None, tc=None, pc=None, ac=None,
                 a=None, b=None):
        self.hydrogen = hydrogen
        self.tc = tc
        self.ac = ac
        self.pc = pc
        self.a = a
        self.b = b
        self._freeze()


class pdata(object):
    """
    Class for pflotran data file. Use 'from pdata import*' to
    access pdata library
    """

    # definitions are put on one line to work better with rst/latex/sphinx.
    def __init__(self, filename='', work_dir=''):
        if pflotran_dir:
            self.co2_database = pflotran_dir + '/database/co2data0.dat'
        else:
            self.co2_database = ''
        self.uniform_velocity = puniform_velocity()
        self.nonuniform_velocity = pnonuniform_velocity()
        self.overwrite_restart_flow_params = False
        self.overwrite_restart_transport = False
        self.initialize_flow_from_file = None
        self.initialize_transport_from_file = None
        self.isothermal = False
        self.multiple_continuum = False
        self.regression = pregression()
        self.simulation = psimulation()
        self.datasetlist = []
        self.chemistry = None
        self.grid = pgrid()
        self.timestepper_flow = None
        self.timestepper_transport = None
        self.proplist = []
        self.time = ptime()
        self.lsolverlist = []
        self.nsolverlist = []
        self.output = poutput()
        self.fluidlist = []
        self.saturationlist = []
        self.regionlist = []  # There are multiple regions
        self.integral_flux_list = []  # There are multiple integral_fluxes
        self.charlist = []
        self.observation_list = []
        self.flowlist = []
        self.transportlist = []
        self.initial_condition_list = []
        self.boundary_condition_list = []
        self.source_sink_list = []
        self.source_sink_sandbox_list = []
        self.strata_list = []
        self.constraint_list = []
        self.filename = filename
        self.hydroquake = pquake()
        self.reference_temperature = ''
        self.reference_porosity = ''
        self.geomech_grid = pgeomech_grid()
        self.geomech_proplist = []
        self.geomech_subsurface_coupling = pgeomech_subsurface_coupling()
        self.geomech_time = pgeomech_time()
        self.geomech_output = pgeomech_output()
        self.geomech_regression = pgeomech_regression()
        self.reference_stress_state = preference_stress_state()
        self.specified_velocity = None
        self.reference_liquid_density = None
        self.minimum_hydrostatic_pressure = None
        self.update_flow_permeability = False
        self.eoslist = []
        self.ufd_decay = None
        self.ufd_biosphere = None
        self.waste_form_general = None
        self.wipp_source_sink = None

        # run object
        self._path = ppath(parent=self)
        self._running = False  # boolean for simulation progress
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

    def run(self, input='', input_prefix='', num_procs=1,
            exe=pdflt().pflotran_path, silent=False, num_realizations=1,
            num_groups=1, commandline_options=''):
        """
        Run a pflotran simulation for a given input file with specified
        number of processors.

        :param input: Name of input file. Uses default -pflotranin flag
        :type input: str
        :param input_prefix: Name of input file prefix. Uses the
         -input_prefix flag.
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
        :param commandline_options: PFLOTRAN and PETSc commandline options
        :type commandline_options: str
        """

        # set up and check path to executable

        exe_path = ppath()

        if exe != pdflt().pflotran_path:
            exe_path.filename = exe
        else:
            exe_path.filename = pflotran_dir + '/src/pflotran/pflotran'

        # if can't find the executable, halt
        if not os.path.isfile(exe_path.full_path):
            raise PyFLOTRAN_ERROR('Default location is' +
                                  exe + '. No executable at location ' + exe)

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
        # ALWAYS write input file
        return_flag = self.write(wd + self._path.filename)

        if return_flag:
            raise PyFLOTRAN_ERROR('Writing files')

        # RUN SIMULATION
        cwd = os.getcwd()
        if self.work_dir:
            os.chdir(self.work_dir)
        if input and input_prefix:
            raise PyFLOTRAN_ERROR('Cannot specify both input and input_prefix')

        def run_popen(cmd):
            process = subprocess.Popen(
                cmd.split(' '), shell=False, stdout=subprocess.PIPE,
                stderr=sys.stderr)
            while True:
                out = decode(process.stdout.read(1))
                if ('ERROR' or 'WARNING') in out:
                    raise PyFLOTRAN_ERROR(out)

                if out == '' and process.poll() is not None:
                    break
                if out != '' and silent == False:
                    sys.stdout.write(out)
                    sys.stdout.flush()

        if num_procs == 1:
            arg = exe_path.full_path + ' -pflotranin ' + self._path.filename
            arg = arg + ' ' + commandline_options
            run_popen(arg)
        else:
            if num_realizations > 1:
                arg = 'mpirun -np ' + str(num_procs) + ' ' + \
                      exe_path.full_path + ' -pflotranin ' + \
                      self._path.filename + \
                      ' -stochastic -num_realizations ' + \
                      str(num_realizations) + \
                      ' -num_groups ' + str(num_groups)
            else:
                arg = 'mpirun -np ' + \
                      str(num_procs) + ' ' + exe_path.full_path + \
                      ' -pflotranin ' + self._path.filename

            arg = arg + ' ' + commandline_options
            run_popen(arg)

        if input_prefix:
            if num_procs == 1:
                arg = exe_path.full_path + ' -input_prefix ' + \
                    self._path.filename

            else:
                arg = 'mpirun -np ' + str(num_procs) + ' ' + \
                      exe_path.full_path + ' -input_prefix ' + \
                      self._path.filename

            arg = arg + ' ' + commandline_options
            run_popen(arg)

        # After executing simulation, go back to the parent directory
        if self.work_dir:
            os.chdir(cwd)

    def __repr__(self):
        return self.filename  # print to screen when called

    def plot_data_from_tec(self, direction='X', variable_list=None,
                           tec_filenames=None, legend_list=None,
                           plot_filename='', fontsize=10, x_label='',
                           y_label_list=None, x_type='linear',
                           y_type='linear', x_range=(), y_range=(),
                           x_factor=1.0, y_factor=1.0):

        if variable_list is None:
            variable_list = []
        if tec_filenames is None:
            tec_filenames = []
        if legend_list is None:
            legend_list = []
        if y_label_list is None:
            y_label_list = ['default y label']

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
                            xval = [
                                val * x_factor for val in var_values_dict[key]]
                        if var in key:
                            dat = [val * y_factor for val in
                                   var_values_dict[key]]
                            found = True
                    if not found:
                        print('Variable ' + var + ' not found in ' + FILE)
                    try:
                        ln, = ax.plot(xval, dat)
                        lns.append(ln)
                    except UnboundLocalError:
                        pass
            ax.legend(lns, legend_list, ncol=1, fancybox=True,
                      shadow=False, prop={'size': str(fontsize)}, loc='best')
            if '.pdf' in plot_filename:
                plot_filename = plot_filename.replace(".pdf", "")
            if ' ' in var:
                var = var.replace(" ", "_")
            if found:
                print('Plotting variable [' + var + '] in [' + \
                      direction + '] direction')
            fig.savefig(plot_filename + '_' + var + '.pdf')

        return 0

    def plot_observation(self, variable_list=None, observation_list=None,
                         observation_filenames=None,
                         plot_filename='plot.pdf',
                         legend_list=None, fontsize=10, x_label='', y_label='',
                         x_type='linear', y_type='linear',
                         x_range=(), y_range=(), x_factor=1.0, y_factor=1.0):
        """
        Plot time-series data from observation files at a given set
        of observation points.

        :param variable_list: List of the variables to be plotted
        :type variable_list: [str]
        :param observation_list: List of observation names to be plotted
        :type observation_list: [str]
        :param observation_filenames: List of observation filenames that
         are to be used for extracting data
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
        :param x_type: type of plot in the x-direction, e.g., 'log',
         'linear', 'symlog'
        :type x_type: str
        :param y_type: type of plot in the y-direction, e.g., 'log',
         'linear', 'symlog'
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
            combined_dict[FILE] = var_values_dict

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

        for file in combined_dict.keys():
            for key in combined_dict[file].keys():
                if 'Time' in key:
                    time = combined_dict[file][key]
                    time_new = [t * x_factor for t in time]
            for item in combined_var_obs_list:
                if item[0] == '' or item[1] == '':
                    print('Please provide a variable name' +
                          'and an observation name')
                else:
                    keys = [key for key in combined_dict[
                        file].keys() if item[0] in key and item[1] in key]
                    for key in keys:
                        var_new = [
                            v * y_factor for v in combined_dict[file][key]]
                        ln, = ax.plot(time_new, var_new)
                        lns.append(ln)

        ax.legend(lns, legend_list, ncol=1, fancybox=True,
                  shadow=False, prop={'size': str(fontsize)}, loc='best')

        fig.savefig(plot_filename)

        return 0

    def read(self, filename=''):
        """
        Read a given PFLOTRAN input file. This method is useful for
        reading an existing a PFLOTRAN input deck and all
        the corresponding PyFLOTRAN objects and data structures are
        autmatically created.

        :param filename: Name of input file.
        :type filename: str
        """
        if not os.path.isfile(filename):
            raise IOError(filename + ' not found...')
        self.filename = filename  # assign filename attribute
        read_fn = dict(zip(read_cards,
                           [self._read_co2_database,
                            self._read_uniform_velocity,
                            self._read_nonuniform_velocity,
                            self._read_simulation,
                            self._read_regression,
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
                            self._read_constraint,
                            self._read_geomechanics_regression,
                            self._read_geomechanics_grid,
                            self._read_geomechanics_subsurface_coupling,
                            self._read_geomechanics_time,
                            self._read_region,
                            self._read_flow,
                            self._read_boundary_condition,
                            self._read_strata,
                            self._read_geomechanics_prop,
                            self._read_geomechanics_output,
                            self._read_eos,
                            self._read_specified_velocity,
                            self._read_reference_liquid_density,
                            self._read_minimum_hydrostatic_pressure,
                            self._read_update_flow_permeability,
                            self._read_ufd_decay,
                            self._read_ufd_biosphere,
                            self._read_source_sink_sandbox,
                            self._read_waste_form_general,
                            self._read_wipp_source_sink],
                           ))

        # associate each card name with
        # a read function, defined further below

        skip_readline = False
        p_line = ''  # Memorizes the most recent line read in.

        def capture_external_file(cinfile):
            """
            Recursive function that is called with each capture of 
            EXTERNAL_FILE.
            """

            # Store the working file text to this variable.
            # It will get expanded as this function is
            # recursively called.
            filetxt = ''

            # Get the pathname - this may be relative, so
            # we must update it here
            cwd = os.path.dirname(cinfile)

            # Prepare for reading HDF5 files from DBASE_FILENAME
            open_files = []

            with open(cinfile, 'r') as child_file:
                for line in child_file:
                    if 'external_file' in line.lower():
                        exfile = line.split()[1]
                        expath = os.path.join(cwd, exfile)
                        filetxt += capture_external_file(expath)
                    elif 'dbase_filename' in line.lower():
                        ext = line.strip().split('.')[-1].lower()
                        _root_dir = os.path.dirname(cinfile)
                        _filepath = os.path.join(_root_dir,line.split()[-1])

                        if ext == 'txt':
                            with open(_filepath,'r') as f:
                                _full_text = f.read()
                            tmp_dict = dict()
                            for line in _full_text.split('\n'):
                                try:
                                    values = line.split()[1:]
                                    tmp_dict[line.split()[0]] = values
                                except IndexError:
                                    pass
                            open_files.append(tmp_dict)
                        elif ext == 'h5':
                            try:
                                open_files.append(h5py.File(_filepath, 'r'))
                            except:
                                pass

                    elif 'dbase_value' in line.lower():
                        _key = line.split()[-1]
                        for _h5file in open_files:
                            if _key in _h5file.keys():
                                _split_line = line.split()[:-2]
                                _split_line.append(strD(_h5file[_key][__h5_idx__]))
                                filetxt += ' '.join(_split_line) + "\n"
                                break
                    else:
                        filetxt += line

            for _h5file in open_files:
              try:
                  _h5file.close()
              except AttributeError:
                  pass

            # Return the file contents
            return filetxt

        def preprocess_file(pinfile, outfile='_pyflotran_preproc.in'):
            """
            Replaces all instances of EXTERNAL_FILE in a PFLOTRAN infile
            with the contents of that external file.
            """

            with open(outfile, 'w') as parent_file:
                parent_file.write(capture_external_file(pinfile))

            return outfile

        def get_next_line(skip_readline=skip_readline, line=p_line):
            """
            Used by read function to avoid skipping a line in cases where
            a particular read function might read an extra line.
            """

            if skip_readline:
                skip_readline = False
                return line
            else:
                line = infile.readline()
                return line

        # EXTERNAL_FILE contains blocks that read() will need to parse.
        # Check if read() has them.

        _temp_file = open(self.filename, 'r').read().lower()
        _should_preprocess = 'external_file' in _temp_file or \
                              (__h5_on__ and 'dbase_filename' in _temp_file)

        if _should_preprocess:
            infile_name = preprocess_file(
                self.filename, outfile='_pyflotran_preproc.in')
        else:
            infile_name = self.filename

        with open(infile_name, 'r') as infile:
            keep_reading = True
            while keep_reading:
                p_line = get_next_line()

                if not p_line:
                    keep_reading = False
                if len(p_line.strip()) == 0:
                    continue
                if list(p_line)[0] in ['#', '!']:
                    continue
                card = p_line.split()[0].lower()  # make card lower case
                if card == 'overwrite_restart_flow_params':
                    self.overwrite_restart_flow_params = True
                if card == 'overwrite_restart_transport':
                    self.overwrite_restart_transport = True
                if card == 'initialize_flow_from_file':
                    self.initialize_flow_from_file = p_line.split()[1]
                if card == 'initialize_transport_from_file':
                    self.initialize_transport_from_file = p_line.split()[1]
                if card == 'reference_temperature':
                    self.reference_temperature = floatD(p_line.split()[1])
                if card == 'reference_porosity':
                    self.reference_porosity = floatD(p_line.split()[1])
                if card == 'isothermal':
                    self.isothermal = True
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

                if card in read_cards:  # check if a valid card name
                    if card in ['co2_database',
                                'dataset', 'material_property', 'simulation',
                                'regression', 'grid', 'timestepper',
                                'linear_solver', 'newton_solver',
                                'saturation_function', 'region',
                                'flow_condition', 'boundary_condition',
                                'source_sink', 'initial_condition',
                                'transport_condition', 'constraint',
                                'uniform_velocity',
                                'nonuniform_velocity',
                                'characteristic_curves',
                                'geomechanics_regression',
                                'geomechanics_grid',
                                'geomechanics_subsurface_coupling',
                                'geomechanics_time',
                                'geomechanics_region',
                                'geomechanics_condition',
                                'geomechanics_boundary_condition',
                                'geomechanics_strata',
                                'strata', 'geomechanics_material_property',
                                'geomechanics_output',
                                'eos','reference_liquid_density',
                                'minimum_hydrostatic_pressure']:
                        read_fn[card](infile, p_line)
                    else:
                        read_fn[card](infile)

    def print_inputfile_to_screen(self, filename=''):
        """
        Print file to screen.

        :param filename: Name of the file to be printed to screen
        :type filename: str
        """
        if filename:
            file = filename
        elif self.filename:
            file = self.filename
        else:
            raise PyFLOTRAN_ERROR('PFLOTRAN input file name that is to be ' +
                                  'printed on screen needs to be specified!')
        outfile = open(file, 'r')
        print(outfile.read())

    def write(self, filename='pflotran.in'):
        """
        Write pdata object to PFLOTRAN input file. Does not execute
        the input file - only writes a corresponding
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
            raise PyFLOTRAN_ERROR(
                'simulation is required, it is currently reading as empty')

        if self.simulation.subsurface_flow or \
                self.simulation.subsurface_transport:
            self._write_subsurface_simulation_begin(outfile)

        if self.regression.cells or self.regression.cells_per_process:
            self._write_regression(outfile)

        if self.uniform_velocity.value_list:
            self._write_uniform_velocity(outfile)

        if self.specified_velocity:
            self._write_specified_velocity(outfile)

        if self.update_flow_permeability:
            self._write_update_flow_permeability(outfile)

        if self.reference_liquid_density:
            self._write_reference_liquid_density(outfile)

        if self.minimum_hydrostatic_pressure:
            self._write_minimum_hydrostatic_pressure(outfile)

        if self.reference_stress_state.value_list:
            self._write_reference_stress_state(outfile)

        if self.eoslist:
            self._write_eos(outfile)

        if self.nonuniform_velocity.filename:
            self._write_nonuniform_velocity(outfile)

        if self.simulation.mode.upper() == 'MPHASE' and self.co2_database:
            self._write_co2_database(outfile)

        if self.multiple_continuum:
            self._write_multiple_continuum(outfile)

        if self.overwrite_restart_flow_params:
            self._write_overwrite_restart_flow(outfile)

        if self.overwrite_restart_transport:
            self._write_overwrite_restart_transport(outfile)

        if self.initialize_flow_from_file is not None:
            self._write_initialize_flow_from_file(outfile)

        if self.initialize_transport_from_file is not None:
            self._write_initialize_transport_from_file(outfile)

        if self.isothermal:
            self._write_isothermal(outfile)

        if self.reference_temperature:
            self._write_reference_temperature(outfile)

        if self.reference_porosity:
            self._write_reference_porosity(outfile)

        if self.datasetlist:
            self._write_dataset(outfile)
        # else: print 'info: dataset name not detected\n'

        if self.chemistry:
            self._write_chemistry(outfile)
        # else: print 'info: chemistry not detected\n'

        if self.grid:
            self._write_grid(outfile)
        else:
            raise PyFLOTRAN_ERROR(
                'grid is required, it is currently reading as empty!')

        if self.timestepper_flow or self.timestepper_transport:
            self._write_timestepper(outfile)
        # else: print 'info: timestepper not detected\n'

        if self.time:
            self._write_time(outfile)
        else:
            raise PyFLOTRAN_ERROR(
                'time is required, it is currently reading as empty!')

        if self.proplist:
            self._write_prop(outfile)
        else:
            PyFLOTRAN_WARNING(
                'material property list is empty! ' +
                ' Using default material property settings')
            self.add(pmaterial())
            self._write_prop(outfile)

        if self.lsolverlist:
            self._write_lsolver(outfile)
        # else: print 'info: lsolverlist (linear solver list) not detected\n'

        if self.nsolverlist:
            self._write_nsolver(outfile)
        # else: print 'info: nsolverlist (newton solver list) not detected\n'

        if self.output:
            self._write_output(outfile)
        else:
            raise PyFLOTRAN_ERROR(
                'output is required, it is currently reading as empty!')

        if self.fluidlist:
            self._write_fluid(outfile)
        else:
            PyFLOTRAN_WARNING(
                'fluidlist is required, it is currently reading as empty!')

        if self.saturationlist:
            self._write_saturation(outfile)
        elif self.charlist:
            self._write_characteristic_curves(outfile)
        else:
            if self.simulation.subsurface_flow:
                self.add(pcharacteristic_curves())
                self._write_characteristic_curves(outfile)
                PyFLOTRAN_WARNING(
                    'characteristic_curves list or saturation list ' +
                    'is required, it is currently reading as empty! ' +
                    ' Using default characteristic_curves settings')

        if self.regionlist:
            self._write_region(outfile)
        else:
            raise PyFLOTRAN_ERROR(
                'regionlist is required, it is currently reading as empty!')

        if self.integral_flux_list:
            self._write_integral_flux(outfile)

        if self.observation_list:
            self._write_observation(outfile)

        if self.flowlist:
            self._write_flow(outfile)

        if self.transportlist:
            self._write_transport(outfile)

        if self.initial_condition_list:
            self._write_initial_condition(outfile)
        else:
            raise PyFLOTRAN_ERROR(
                'initial_condition_list is required,' +
                'it is currently reading as empty!')

        if self.boundary_condition_list:
            self._write_boundary_condition(outfile)

        if self.source_sink_list:
            self._write_source_sink(outfile)

        if self.source_sink_sandbox_list:
            self._write_source_sink_sandbox(outfile)

        if self.strata_list:
            self._write_strata(outfile)
        else:
            PyFLOTRAN_WARNING(
                'stratigraphy_coupler is required, ' +
                'it is currently reading as empty! ' +
                'Using default settings')
            self.add(pstrata())
            self._write_strata(outfile)

        if self.constraint_list:
            self._write_constraint(outfile)

        if self.simulation.subsurface_flow or \
                self.simulation.subsurface_transport:
            self._write_subsurface_simulation_end(outfile)

        if self.simulation.simulation_type == 'hydroquake':
            self._write_hydroquake(outfile)

        if self.simulation.simulation_type.lower() == 'geomechanics_subsurface':
            self._write_geomechanics(outfile)

        if self.waste_form_general:
            self._write_wasteform_general(outfile)

        if self.ufd_decay:
            self._write_ufd_decay(outfile)

        if self.ufd_biosphere:
            self._write_ufd_biosphere(outfile)

        if self.wipp_source_sink:
            self._write_wipp_source_sink(outfile)

        outfile.close()

    def add(self, obj, index='', overwrite=False):
        # Adds a new object to the file
        """
        Attach an object associated w/ a list (e.g., pregion)
        that belongs to a pdata object.

        :param obj: Object to be added to the data file.
        :type obj: object(e.g., pregion)
        :param index: (Optional) Used to find an object that is using a
         string as an index in a dictionary. Intended for the super class
         object. (E.g. Index represents flow.name if instance is
         pflow_variable.) Default if not specified is to use the last
         super-class object added to pdata.
        :type index: String
        :param overwrite: Flag to overwrite an object if it already
         exists in a pdata object.
        :type overwrite: bool
        """

        add_checklist = [pmaterial, pgeomech_material, pdataset, psaturation,
                         pcharacteristic_curves, pchemistry_m_kinetic,
                         plsolver, pnsolver, pregion, pobservation, pflow,
                         pflow_variable, pinitial_condition,
                         pboundary_condition, psource_sink, pstrata,
                         ptransport, pconstraint, pconstraint_concentration,
                         psecondary_continuum, pfluid, peos, pintegral_flux]

        # Check if obj first is an object that belongs to add_checklist
        checklist_bool = [isinstance(obj, item) for item in add_checklist]
        if True not in checklist_bool:
            raise PyFLOTRAN_ERROR(
                'pdata.add used incorrectly! ' +
                'Cannot use pdata.add with one of the specified object.')

        # Always make index lower case if it is being used as a string
        if isinstance(index, str):
            index = index.lower()
        if isinstance(obj, pmaterial):
            self._add_prop(obj, overwrite)
        if isinstance(obj, pgeomech_material):
            self._add_geomech_prop(obj, overwrite)
        if isinstance(obj, psecondary_continuum):
            self._add_sec(obj, overwrite)
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
        if isinstance(obj, pfluid):
            self._add_fluid(obj, overwrite)
        if isinstance(obj, peos):
            self._add_eos(obj, overwrite)
        if isinstance(obj, pregion):
            self._add_region(obj, overwrite)
        if isinstance(obj, pintegral_flux):
            self._add_integral_flux(obj, overwrite)
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
        Delete an object that is assigned to a list of objects
        belong to a pdata object, e.g., pregion.

        :param obj: Object to be deleted from the data file.
         Can be a list of objects.
        :type obj: Object (e.g., pregion), list
        """

        if isinstance(obj, pmaterial):
            self._delete_prop(obj)
        elif isinstance(obj, list):
            for obji in copy(obj):  # obji = object index
                if isinstance(obji, pmaterial):
                    self._delete_prop(obji)

        if isinstance(obj, pgeomech_material):
            self._delete_geomech_prop(obj)
        elif isinstance(obj, list):
            for obji in copy(obj):  # obji = object index
                if isinstance(obji, pgeomech_material):
                    self._delete_geomech_prop(obji)

        if isinstance(obj, psecondary_continuum):
            self._delete_prop(obj)
        elif isinstance(obj, list):
            for obji in copy(obj):  # obji = object index
                if isinstance(obji, psecondary_continuum):
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

        if isinstance(obj, pfluid):
            self._delete_fluid(obj)
        elif isinstance(obj, list):
            for obji in copy(obj):
                if isinstance(obji, pfluid):
                    self._delete_fluid(obji)

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

        if isinstance(obj, pintegral_flux):
            self._delete_integral_flux(obj)
        elif isinstance(obj, list):
            for obji in copy(obj):
                if isinstance(obji, pintegral_flux):
                    self._delete_integral_flux(obji)

        if isinstance(obj, pflow):
            self._delete_flow(obj)
        elif isinstance(obj, list):
            for obji in copy(obj):
                if isinstance(obji, pflow):
                    self._delete_flow(obji)

        if isinstance(obj, pflow_variable):
            # Flow object needs to be specified
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

        # Constraint object needs to be specified
        if isinstance(obj, pconstraint_concentration):
            self._delete_constraint_concentration(obj, super_obj)
        elif isinstance(obj, list):  # Condition not tested
            for obji in copy(obj):
                if isinstance(obji, pconstraint_concentration):
                    self._delete_constraint_concentration(obji)

    @staticmethod
    def splitter(a_line):
        """
        Returns the last element of the split list.

        :param a_line: Line input
        :type a_line: str
        """
        return a_line.split()[-1]

    def _read_uniform_velocity(self, infile, line):
        np_value_list = []
        # Convert to list, ignore 1st word
        tstring = filter_comment(line).split()[1:]
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
        if len(self.uniform_velocity.value_list) > 3:
            if self.uniform_velocity.value_list[3] in velocity_units_allowed:
                for val in self.uniform_velocity.value_list[:4]:
                    outfile.write(strD(val) + ' ')
        else:
            for val in self.uniform_velocity.value_list[:3]:
                outfile.write(strD(val) + ' ')
        outfile.write('\n\n')

    def _write_reference_stress_state(self, outfile):
        outfile.write('REFERENCE_STRESS_STATE ')
        i = 0
        for v in self.reference_stress_state.value_list:  # value in value_list
            outfile.write(strD(v) + ' ')
            i = i + 1
        if i != 6:
            raise PyFLOTRAN_ERROR('reference_stress_state' +
                                  ' must have 6 components')
        outfile.write('\n\n')

    def _read_update_flow_permeability(self,infile):
        self.update_flow_permeability = True

    def _write_update_flow_permeability(self,outfile):
        outfile.write('UPDATE_FLOW_PERMEABILITY\n')

    def _read_reference_liquid_density(self,infile,line):
        _split = line.strip().split()
        _value = floatD(_split[1])

        try:
            _units = _split[2]
        except IndexError:
            _units = None

        self.reference_liquid_density = Coeff(_value,unit=_units)

    def _write_reference_liquid_density(self,outfile):
        _value = str(self.reference_liquid_density)
        outfile.write('REFERENCE_LIQUID_DENSITY %s\n' % _value)

    def _read_minimum_hydrostatic_pressure(self,infile,line):
        _split = line.strip().split()
        _value = floatD(_split[1])

        try:
            _units = _split[2]
        except IndexError:
            _units = None

        self.minimum_hydrostatic_pressure = Coeff(_value,unit=_units)

    def _write_minimum_hydrostatic_pressure(self,outfile):
        _value = str(self.minimum_hydrostatic_pressure)
        outfile.write('MINIMUM_HYDROSTATIC_PRESSURE %s\n' % _value)

    def _read_eos(self, infile, line):
        keep_reading = True
        eos = peos(fluid_density=None, fluid_enthalpy=None,
                   fluid_henrys_constant=None)
        if line.strip().split()[0].lower() == 'eos':
            eos.fluid_name = line.strip().split()[-1].lower()

        if eos.fluid_name not in eos_fluid_names_allowed:
            PyFLOTRAN_ERROR('Unknown fluid under EOS!')

        while keep_reading:  # read through all cards
            line = get_next_line(infile)  # get next line
            key = line.strip().split()[0].lower()  # take first keyword

            if key == 'density':
                den_type = line.strip().split()[1].lower()
                if den_type in eos_density_types_allowed:
                    eos.fluid_density.append(den_type)
                    if den_type == 'rks':
                        eos.rks = prks()
                        keep_reading1 = True
                        while keep_reading1:
                            line1 = get_next_line(infile)
                            # take first keyword
                            key1 = line1.strip().split()[0].lower()
                            if key1 in ['hydrogen', 'non-hydrogen']:
                                eos.rks.hydrogen = key1
                            elif key1 in ['tc', 'critical_temperature']:
                                eos.rks.tc = floatD(line1.strip().split()[1])
                            elif key1 in ['pc', 'critical_pressure']:
                                eos.rks.pc = floatD(line1.strip().split()[1])
                            elif key1 in ['acentric_factor', 'ac']:
                                eos.rks.ac = floatD(line1.strip().split()[1])
                            elif key1 in ['omegaa', 'a']:
                                eos.rks.a = floatD(line1.strip().split()[1])
                            elif key1 in ['omegab', 'b']:
                                eos.rks.b = floatD(line1.strip().split()[1])
                            elif key1 in ['/', 'end']:
                                keep_reading1 = False
                    elif len(line.strip().split()) > 2:
                        for val in line.strip().split()[2:]:
                            eos.fluid_density.append(floatD(val))
                    elif den_type == 'linear' or den_type == 'inverse_linear':
                        # Here, we are assuming a multiline
                        # DENSITY LINEAR block.
                        eos.density_params = {}
                        while True:
                            subline = get_next_line(infile)
                            subkey = subline.strip().split()[0].lower()

                            if subkey in ['/', 'end']:
                                break

                            subvalue = subline.strip().split()[1].lower()
                            eos.density_params[subkey] = floatD(subvalue)
                else:
                    raise PyFLOTRAN_ERROR('Unknown EOS density type')
            elif key == 'enthalpy':
                enthalpy_type = line.strip().split()[1].lower()

                if enthalpy_type in eos_enthalpy_types_allowed:
                    eos.fluid_enthalpy.append(enthalpy_type)

                    if len(line.strip().split()) > 2:
                        for val in line.strip().split()[2:]:
                            eos.fluid_enthalpy.append(floatD(val))
                    else:
                        if enthalpy_type in ['quadratic_temp']:
                            while True:
                                subline = get_next_line(infile)
                                subkey = subline.strip().split()[0].lower()

                                if subkey in ['/', 'end']:
                                    break

                else:
                    raise PyFLOTRAN_ERROR('Unknown EOS enthalpy type')
            elif key == 'viscosity':
                visc_type = line.strip().split()[1].lower()
                if visc_type in eos_viscosity_types_allowed:

                    eos.fluid_viscosity.append(visc_type)
                    if len(line.strip().split()) > 2:
                        for val in line.strip().split()[2:]:
                            eos.fluid_viscosity.append(floatD(val))
                    elif visc_type == 'quadratic':
                        print('WARNING: QUADRATIC not fully supported.')
                        while True:
                            sline = get_next_line(infile).strip()
                            if sline in ['/','end']:
                                break
                else:
                    _err = visc_type + ' not recognized as a valid ' + \
                           'viscosity type.\n'
                    raise PyFLOTRAN_ERROR(_err)

            elif key == 'test':
                # Create a list storing the TEST card
                # Ex:
                #    TEST 10.d0 400.d0 1.d4 1.d8 10 10 uniform uniform
                eos_test = []
                for val in line.strip().split()[1:]:
                    try:
                        eos_test.append(floatD(val))
                    except ValueError:
                        eos_test.append(val)

                eos.fluid_test.append(eos_test)
            elif key == 'henrys_constant':
                if eos.fluid_name not in ['gas']:
                    raise PyFLOTRAN_ERROR(
                        'henrys_constant can only be set for fluid set ' +
                        'to gas!')
                henrys_type = line.strip().split()[1]
                if henrys_type in eos_henrys_types_allowed:
                    eos.fluid_henrys_constant.append(henrys_type)
                    if len(line.strip().split()) > 2:
                        for val in line.strip().split()[2:]:
                            eos.fluid_henrys_constant.append(floatD(val))
            elif key in pvt_types_allowed:
                pvt = peos.pvt(pvt_type=key)
                while True:
                    subline = get_next_line(infile)
                    subkey = subline.strip().split()[0].lower()

                    if subkey == 'data_units':
                        while True:
                            _line = get_next_line(infile)
                            _key = _line.strip().split()[0].lower()

                            if _key == 'pressure':
                                pvt.pressure_units = self.splitter(_line)
                            elif _key == 'rs':
                                pvt.rs_units = self.splitter(_line)
                            elif _key == 'fvf':
                                pvt.fvf_units = self.splitter(_line)
                            elif _key == 'viscosity':
                                pvt.viscosity_units = self.splitter(_line)
                            elif _key == 'compressibility':
                                pvt.compressibility_units = self.splitter(_line)
                            elif _key in ['/','end']:
                                break
                    elif subkey == 'data':
                        while True:
                            _line = get_next_line(infile)
                            _key = _line.strip().split()[0].lower()

                            if _key == 'temperature':
                                while True:
                                    _line2 = get_next_line(infile)
                                    _key2 = _line2.strip().split()[0].lower()
                                    if _key2 in ['/','end']:
                                        break
                                    else:
                                        pvt.temperature_data.append(\
                                                         _line2.strip().split())
                            elif _key in ['/','end']:
                                break
                    elif subkey in ['/','end']:
                        break
                    else:
                        raise PyFLOTRAN_ERROR('unknown key...')

            elif key == 'formula_weight':
                if eos.fluid_name not in ['gas','oil','solvent']:
                    raise PyFLOTRAN_ERROR(
                        'formula_weight can only be set for fluid set to gas!')

                eos.fluid_formula_weight = floatD(line.strip().split()[1])
            elif key in ['/', 'end']:
                keep_reading = False
        self.add(eos)

    def _write_eos(self, outfile):
        self._header(outfile, headers['eos'])
        for eos in self.eoslist:
            if eos.fluid_name.lower() in eos_fluid_names_allowed:
                outfile.write('EOS ' + eos.fluid_name.upper() + '\n')
                if eos.fluid_density:
                    if eos.fluid_density[0].upper() == 'RKS':
                        outfile.write('  DENSITY ' +
                                      eos.fluid_density[0].upper() + '\n')
                        outfile.write('    ' +
                                      eos.rks.hydrogen.upper() + '\n')
                        outfile.write('    TC ' +
                                      strD(eos.rks.tc) + '\n')
                        outfile.write('    PC ' +
                                      strD(eos.rks.pc) + '\n')
                        outfile.write('    AC ' +
                                      strD(eos.rks.ac) + '\n')
                        outfile.write('    OMEGAA ' +
                                      strD(eos.rks.a) + '\n')
                        outfile.write('    OMEGAB ' +
                                      strD(eos.rks.b) + '\n')
                        outfile.write('  /\n')
                    elif eos.fluid_density[0].upper() == 'CONSTANT' and \
                            len(eos.fluid_density) == 2:
                        outfile.write('  DENSITY ' +
                                      eos.fluid_density[0].upper() + ' '
                                      + strD(eos.fluid_density[1]) + '\n')
                    elif eos.fluid_density[0].upper() == 'EXPONENTIAL' and \
                            len(eos.fluid_density) == 4:
                        outfile.write('  DENSITY '
                                      + eos.fluid_density[0].upper() + ' '
                                      + strD(eos.fluid_density[1]) + ' '
                                      + strD(eos.fluid_density[2]) + ' '
                                      + strD(eos.fluid_density[3]) + '\n')
                    elif eos.fluid_density[0].upper() == 'LINEAR' and \
                            len(eos.fluid_density) == 4:
                        outfile.write('  DENSITY '
                                      + eos.fluid_density[0].upper() + ' '
                                      + strD(eos.fluid_density[1]) + ' '
                                      + strD(eos.fluid_density[2]) + ' '
                                      + strD(eos.fluid_density[3]) + '\n')
                    elif eos.fluid_density[0].upper() == 'DEFAULT':
                        outfile.write('  DENSITY DEFAULT\n')
                    elif eos.fluid_density[0].upper() == 'IDEAL':
                        outfile.write('  DENSITY IDEAL\n')
                    elif eos.fluid_density[0].upper() == 'BATZLE_AND_WANG':
                        outfile.write('  DENSITY BATZLE_AND_WANG\n')
                    elif eos.fluid_density[0].upper() == 'TRANGENSTEIN':
                        outfile.write('  DENSITY TRANGENSTEIN\n')
                    elif eos.fluid_density[0].upper() == 'PR_METHANE':
                        outfile.write(
                            '  DENSITY ' + eos.fluid_density[0].upper() + '\n')
                    elif eos.fluid_density[0].upper() == 'DUANMAO':
                        outfile.write(
                            '  DENSITY ' + eos.fluid_density[0].upper() + '\n')
                    elif eos.fluid_density[0].upper() == 'BRAGFLO':
                        tmp = ' '.join([strD(x) for x in eos.fluid_density[1:]])
                        outfile.write('  DENSITY BRAGFLO %s\n' % tmp)
                    elif eos.density_params is not None:
                        outfile.write('  DENSITY LINEAR\n')
                        for dkey in eos.density_params.keys():
                            outfile.write('    %s %s\n' % (dkey.upper(),
                                          strD(eos.density_params[dkey])))
                        outfile.write('  /\n')
                    else:
                        err = 'eos.fluid_density: \''+str(eos.fluid_density)+\
                                  '\' has incorrect keyword or incorrect length'
                        raise PyFLOTRAN_ERROR(err)

                if eos.fluid_viscosity:
                    if eos.fluid_viscosity[0].upper() == 'CONSTANT' and \
                            len(eos.fluid_viscosity) == 2:
                        outfile.write('  VISCOSITY ' +
                                      eos.fluid_viscosity[0].upper() + ' '
                                      + strD(eos.fluid_viscosity[1]) + '\n')
                    elif eos.fluid_viscosity[0].upper() == 'DEFAULT' and \
                            len(eos.fluid_viscosity) == 1:
                        outfile.write('  VISCOSITY ' +
                                      eos.fluid_viscosity[0].upper() + '\n')
                    elif eos.fluid_viscosity[0].upper() == 'BATZLE_AND_WANG':
                        outfile.write('  VISCOSITY BATZLE_AND_WANG\n')
                    elif eos.fluid_viscosity[0].upper() == 'QUADRATIC':
                        outfile.write('  VISCOSITY QUADRATIC\n')
                    elif eos.fluid_viscosity[0].upper() == 'GRABOWSKI':
                        outfile.write('  VISCOSITY GRABOWSKI\n')
                    else:
                        try:
                            _fl_vis = strD(eos.fluid_viscosity)
                        except TypeError:
                            _fl_vis = ' '.join(\
                                      [str(x) for x in eos.fluid_viscosity])
                        raise PyFLOTRAN_ERROR('eos.fluid_viscosity: \'' +
                                              _fl_vis +
                                              '\' has incorrect keyword or '+ \
                                              'incorrect length')

                if eos.fluid_enthalpy:
                    if eos.fluid_enthalpy[0].upper() == 'CONSTANT' and \
                            len(eos.fluid_enthalpy) == 2:
                        outfile.write('  ENTHALPY ' +
                                      eos.fluid_enthalpy[0].upper() + ' '
                                      + strD(eos.fluid_enthalpy[1]) + '\n')
                    elif eos.fluid_enthalpy[0].upper() == 'DEFAULT' and \
                            len(eos.fluid_enthalpy) == 1:
                        outfile.write('  ENTHALPY ' +
                                      eos.fluid_enthalpy[0].upper() + '\n')
                    elif eos.fluid_enthalpy[0].upper() == 'LINEAR_TEMP' and \
                            len(eos.fluid_enthalpy) == 2:
                        outfile.write('  ENTHALPY ' +
                                      eos.fluid_enthalpy[0].upper() + ' '
                                      + strD(eos.fluid_enthalpy[1]) + '\n')
                    elif eos.fluid_enthalpy[0].upper() == 'QUADRATIC_TEMP':
                        outfile.write('  ENTHALPY QUADRATIC_TEMP\n')
                    else:
                        raise PyFLOTRAN_ERROR('eos.fluid_enthalpy: \'' +
                                              strD(eos.fluid_enthalpy) +
                                              '\' has incorrect keyword or '+\
                                              'incorrect length')
                if eos.fluid_henrys_constant:
                    if eos.fluid_henrys_constant[0].upper() == 'CONSTANT' and \
                            len(eos.fluid_henrys_constant) == 2:
                        outfile.write('  HENRYS_CONSTANT ' +
                                      eos.fluid_henrys_constant[0].upper() + \
                                      ' ' + strD(eos.fluid_henrys_constant[1]) \
                                      + '\n')
                    elif eos.fluid_henrys_constant[0].upper() == 'DEFAULT' and \
                            len(eos.fluid_henrys_constant) == 1:
                        outfile.write('  HENRYS_CONSTANT ' +
                                      eos.fluid_henrys_constant[0].upper()+'\n')
                    elif eos.fluid_henrys_constant[0].upper() == 'DUANMAO' and \
                            len(eos.fluid_henrys_constant) == 1:
                        outfile.write('  HENRYS_CONSTANT ' +
                                      eos.fluid_henrys_constant[0].upper()+'\n')
                    else:
                        raise PyFLOTRAN_ERROR('eos.fluid_henrys_constant: \'' +
                                              strD(eos.fluid_henrys_constant) +
                                              '\' has incorrect keyword or ' + \
                                              'incorrect length')
                if eos.fluid_formula_weight:
                    outfile.write('  FORMULA_WEIGHT ' +
                                  strD(eos.fluid_formula_weight) + '\n')

                if eos.fluid_test:
                    for ftest in eos.fluid_test:
                        fstr = ' '.join(map(strD,ftest[:4])) + ' ' + \
                               ' '.join(map(str,map(int,ftest[4:6]))) + ' ' + \
                               ' '.join(ftest[6:])

                        outfile.write('  TEST ' + fstr + '\n')
                outfile.write('END\n\n')
            else:
                raise PyFLOTRAN_ERROR('eos.fluid_name: \'' + eos.fluid_name +
                                      '\' is invalid')

    def _add_eos(self, eos=peos(), overwrite=False):
        # check if fluid already exists
        if isinstance(eos, peos):
            if eos.fluid_name in self.eos.keys():
                if not overwrite:
                    warning = 'WARNING: Fluid property phase ' + \
                              str(fluid.phase) + '\' already exists. ' + \
                              'fluid will not be defined, ' + \
                              'use overwrite = True in add()' + \
                              ' to overwrite the old fluid.'
                    print(warning,
                    build_warnings.append(warning))
                    return
                else:
                    self.delete(self.eos[eos.fluid_name])

        if eos not in self.eoslist:
            self.eoslist.append(eos)

    def _delete_eos(self, eos=peos()):
        self.eoslist.remove(eos)

    def _read_specified_velocity(self,infile):
        spec_vel = pspecified_velocity()

        while True:
            line = get_next_line(infile)
            key = line.strip().split()[0].lower()

            if key == 'uniform?':
                _uniform = True if self.splitter(line).lower() == 'yes' else False
                spec_vel.uniform = _uniform
            elif key == 'dataset':
                if self.splitter(line).lower() == 'list':
                    _data = []
                    while True:
                        subline = get_next_line(infile).lower()
                        subkey = subline.strip().split()[0].lower()

                        if subkey == 'time_units':
                            spec_vel.time_units = subline.split()[-1]
                        elif subkey == 'data_units':
                            spec_vel.data_units = subline.split()[-1]
                        elif subkey == 'interpolation':
                            spec_vel.data_units = subline.split()[-1].lower()
                        elif subkey in ['/','end']:
                            break
                        else:
                            _raw = line.split()[1:]
                            for x in _raw:
                                try:
                                    _data.append(floatD(x))
                                except ValueError:
                                    _data.append(x)

                        spec_vel.dataset = _data
                else:
                    _raw = line.split()[1:]
                    _data = []

                    for x in _raw:
                        try:
                            _data.append(floatD(x))
                        except ValueError:
                            _data.append(x)

                    spec_vel.dataset = _data

            elif key in ['/','end']:
                break

        self.specified_velocity = spec_vel

    def _write_specified_velocity(self,outfile):
        _spec_vel = self.specified_velocity
        _uniform = 'YES' if _spec_vel.uniform == True else 'NO'
        _dataset = ' '.join([strD(x) for x in _spec_vel.dataset])

        outfile.write('SPECIFIED_VELOCITY\n')
        outfile.write('  UNIFORM? %s\n' % _uniform)

        logical = _spec_vel.time_units \
                 or _spec_vel.data_units \
                 or _spec_vel.interpolation

        if logical:
            outfile.write('  DATASET LIST\n')
            if _spec_vel.time_units:
                outfile.write('    TIME_UNITS %s\n' % _spec_vel.time_units)
            if _spec_vel.data_units:
                outfile.write('    DATA_UNITS %s\n' % _spec_vel.data_units)
            if _spec_vel.interpolation:
                outfile.write('    INTERPOLATION %s\n' % _spec_vel.interpolation)
            outfile.write('    %s\n' % _dataset)
            outfile.write('  /\n')
        else:
            outfile.write('  DATASET %s\n' % _dataset)
        outfile.write('END\n')

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
        keep_reading0 = True
        key_bank = []
        while keep_reading0:  # Read through all cards
            line = get_next_line(infile)
            if len(line.strip()) == 0:
                continue
            elif list(line)[0] in ['!', '#']:
                continue
            key0 = line.strip().split()[0].lower()  # take first key word
            if key0 == 'simulation_type':
                simulation.simulation_type = self.splitter(line)
            elif key0 == 'process_models':
                keep_reading = True
                while keep_reading:
                    line = get_next_line(infile)
                    key = line.strip().split()[0].lower()
                    if key == 'subsurface_flow':
                        simulation.subsurface_flow = self.splitter(line)
                        keep_reading_1 = True
                        while keep_reading_1:
                            line = get_next_line(infile)
                            key1 = line.strip().split()[0].lower()
                            if key1 == 'mode':
                                simulation.mode = self.splitter(line).lower()
                            elif key1 == 'options':
                                opt = psimulation.options()
                                keep_reading_2 = True
                                while keep_reading_2:
                                    line = get_next_line(infile)
                                    key2 = line.strip().split()[0].lower()
                                    if key2 == 'isothermal':
                                        opt.isothermal = True
                                    elif key2 == 'inline_surface_region':
                                        opt.inline_surface_region = \
                                            self.splitter(line)
                                    elif key2 == 'inline_surface_mannings_coeff':
                                        opt.inline_surface_mannings_coeff = \
                                            floatD(self.splitter(line))
                                    elif key2 == 'max_pressure_change':
                                        opt.max_pressure_change = \
                                            floatD(self.splitter(line))
                                    elif key2 == 'max_concentration_change':
                                        opt.max_concentration_change = \
                                            floatD(self.splitter(line))
                                    elif key2 == 'max_temperature_change':
                                        opt.max_temperature_change = \
                                            floatD(self.splitter(line))
                                    elif key2 == 'max_saturation_change':
                                        opt.max_saturation_change = \
                                            floatD(self.splitter(line))
                                    elif key2 == 'max_cfl':
                                        opt.max_cfl = \
                                            floatD(self.splitter(line))
                                    elif key2 == 'numerical_derivatives':
                                        opt.numerical_derivatives = \
                                            floatD(self.splitter(line))
                                    elif key2 == 'pressure_dampening_factor':
                                        opt.pressure_dampening_factor = \
                                            floatD(self.splitter(line))
                                    elif key2 == 'freezing':
                                        opt.freezing = True
                                    elif key2 == 'ice_model':
                                        opt.ice_model = self.splitter(line)
                                    elif key2 == 'multiple_continuum':
                                        opt.multiple_continuum = True
                                    elif key2 == 'numerical_jacobian':
                                        opt.numerical_jacobian = True
                                    elif key2 == 'analytical_jacobian':
                                        opt.analytical_jacobian = True
                                    elif key2 == 'harmonic_permeability_only':
                                        opt.harmonic_permeability_only \
                                        = True
                                    elif key2 == 'do_not_scale_jacobian':
                                        opt.do_not_scale_jacobian = True
                                    elif key2 == 'gas_component_formula_weight':
                                        opt.gas_component_formula_weight \
                                        = floatD(self.splitter(line))
                                    elif key2 == 'liquid_residual_infinity_tol':
                                        opt.liquid_residual_infinity_tol \
                                        = floatD(self.splitter(line))
                                    elif key2 == 'gas_residual_infinity_tol':
                                        opt.gas_residual_infinity_tol \
                                        = floatD(self.splitter(line))
                                    elif key2 == 'max_allow_rel_liq_pres_chang_ni':
                                        opt.max_allow_rel_liq_pres_chang_ni \
                                        = floatD(self.splitter(line))
                                    elif key2 == 'gas_sat_thresh_force_extra_ni':
                                        opt.gas_sat_thresh_force_extra_ni \
                                        = floatD(self.splitter(line))
                                    elif key2 == 'gas_sat_thresh_force_ts_cut':
                                        opt.gas_sat_thresh_force_ts_cut \
                                        = floatD(self.splitter(line))
                                    elif key2 == 'max_allow_gas_sat_change_ts':
                                        opt.max_allow_gas_sat_change_ts \
                                        = floatD(self.splitter(line))
                                    elif key2 == 'gas_sat_change_ts_governor':
                                        opt.gas_sat_change_ts_governor \
                                        = floatD(self.splitter(line))
                                    elif key2 == 'gas_sat_gov_switch_abs_to_rel':
                                        opt.gas_sat_gov_switch_abs_to_rel \
                                        = floatD(self.splitter(line))
                                    elif key2 == 'min_liq_pres_force_ts_cut':
                                        opt.min_liq_pres_force_ts_cut \
                                        = floatD(self.splitter(line))
                                    elif key2 == 'max_allow_liq_pres_change_ts':
                                        opt.max_allow_liq_pres_change_ts \
                                        = floatD(self.splitter(line))
                                    elif key2 == 'liq_pres_change_ts_governor':
                                        opt.liq_pres_change_ts_governor \
                                        = floatD(self.splitter(line))
                                    elif key2 in ['/', 'end']:
                                        keep_reading_2 = False
                                simulation.options = opt
                            elif key1 in ['/', 'end']:
                                keep_reading_1 = False
                        key_bank.append(key)
                    elif key == 'subsurface_transport':
                        simulation.subsurface_transport = self.splitter(line)
                        keep_reading_2 = True
                        while keep_reading_2:
                            line1 = get_next_line(infile)
                            key1 = line1.strip().split()[0].lower()
                            if key1 == 'global_implicit':
                                simulation.flowtran_coupling = key1
                            elif key1 == 'max_volume_fraction_change':
                                simulation.max_volume_fraction_change =\
                                    floatD(self.splitter(line1))
                            elif key1 == 'itol_relative_update':
                                simulation.itol_relative_update = \
                                    floatD(self.splitter(line1))
                            elif key1 in ['/', 'end']:
                                keep_reading_2 = False
                        # else:
                        # raise PyFLOTRAN_ERROR('coupling type missing!')
                        key_bank.append(key)
                    elif key == 'auxiliary':
                        simulation.auxiliary = psimulation.auxiliary()
                        simulation.auxiliary.aux_type = self.splitter(line)
                        while True:
                            subline = get_next_line(infile)
                            subkey = subline.lower().strip().split()[0]

                            if subkey == 'species':
                                _name = subline.split()[1]
                                _value = floatD(subline.split()[2])
                                simulation.auxiliary.species = [_name,_value]
                            elif subkey in ['/','end']:
                                break

                    elif key == 'waste_form':
                        simulation.waste_form = self.splitter(line)
                        while True:
                            subline = get_next_line(infile).strip().lower()
                            if subline in ['/','end']:
                                break

                    elif key == 'ufd_decay':
                        simulation.ufd_decay = self.splitter(line)
                        while True:
                            subline = get_next_line(infile).strip().lower()
                            if subline in ['/','end']:
                                break
                    elif key == 'ufd_biosphere':
                        simulation.ufd_biosphere = self.splitter(line)
                        while True:
                            subline = get_next_line(infile).strip().lower()
                            if subline in ['/','end']:
                                break
                    elif key == 'geomechanics_subsurface':
                        simulation.geomechanics_subsurface = self.splitter(
                            line)
                    if key in ['/', 'end']:
                        keep_reading = False
            elif key0 == 'restart':
                extract = line.strip().split()
                len_extract = len(extract)
                simulation.restart = prestart()
                if len_extract < 2:
                    PyFLOTRAN_ERROR('At least restart filename needs to be'
                                    ' specified')
                else:
                    simulation.restart.file_name = extract[1]

                if len_extract > 2:
                    simulation.restart.time_value = extract[2]
                if len_extract > 3:
                    simulation.restart.time_unit = extract[3]
            elif key0 == 'checkpoint':
                simulation.checkpoint = pcheckpoint(time_list=None)
                keep_reading = True
                while keep_reading:  # Read through all cards
                    line = get_next_line(infile)
                    # take first key word
                    key = line.strip().split()[0].lower()
                    if 'times' in key:
                        simulation.checkpoint.time_unit = \
                          line.strip().split()[1]
                        for val in line.strip().split()[2:]:
                            simulation.checkpoint.time_list.append(floatD(val))
                    elif 'periodic' in key:
                        if 'timestep' in line.strip().split()[1].lower():
                            simulation.checkpoint.periodic_timestep = \
                              line.strip().split()[2]
                        else:
                            simulation.checkpoint.periodic_time = \
                              line.strip().split()[3]
                            simulation.checkpoint.periodic_time_unit = \
                              line.strip().split()[2]
                    elif 'format' in key:
                        simulation.checkpoint.format = line.strip().split()[1]
                    elif key in ['/', 'end']:
                        keep_reading = False
            elif key0 in ['/', 'end']:
                keep_reading0 = False
        if not ('subsurface_flow' in key_bank) and \
                ('subsurface_transport' in key_bank):
            simulation.subsurface_flow = ''
            simulation.mode = ''
        self.simulation = simulation

    def _write_simulation(self, outfile):
        self._header(outfile, headers['simulation'])
        simulation = self.simulation
        # Write out simulation header

        outfile.write('SIMULATION' + '\n')
        if simulation.simulation_type.lower() in simulation_types_allowed:
            outfile.write('  SIMULATION_TYPE ' +
                          simulation.simulation_type.upper() + '\n')
        else:
            print('       valid simulation.simulation_type:', \
                simulation_types_allowed, '\n')
            raise PyFLOTRAN_ERROR(
                'simulation.simulation_type: \'' +
                simulation.simulation_type + '\' is invalid!')

        if simulation.subsurface_flow and simulation.subsurface_transport:
            outfile.write('  PROCESS_MODELS' + '\n')
            outfile.write('    SUBSURFACE_FLOW ' +
                          simulation.subsurface_flow + '\n')
            if simulation.mode in mode_names_allowed:
                outfile.write('      MODE ' + simulation.mode.upper() + '\n')

                if simulation.options:
                    simulation.options._write(outfile)

                if simulation.waste_form:
                    outfile.write('  WASTE_FORM %s\n' % simulation.waste_form)
                    outfile.write('  /\n')
                if simulation.ufd_decay:
                    outfile.write('  UFD_DECAY %s\n' % simulation.ufd_decay)
                    outfile.write('  /\n')
                if simulation.ufd_biosphere:
                    outfile.write('  UFD_BIOSPHERE %s\n' % simulation.ufd_biosphere)
                    outfile.write('  /\n')
            else:
                print('       valid simulation.mode:', mode_names_allowed, '\n')
                raise PyFLOTRAN_ERROR(
                    'simulation.mode: \'' + simulation.mode + '\' is invalid!')

            outfile.write('    / ' + '\n')
            outfile.write('    SUBSURFACE_TRANSPORT ' +
                          simulation.subsurface_transport + '\n')
            if simulation.flowtran_coupling:
                outfile.write(
                    '      ' + simulation.flowtran_coupling.upper() + '\n')
            if simulation.max_volume_fraction_change:
                outfile.write('      MAX_VOLUME_FRACTION_CHANGE ' +
                              strD(simulation.max_volume_fraction_change) +
                              '\n')
            outfile.write('    / ' + '\n')
            if simulation.auxiliary:
                simulation.auxiliary._write(outfile)
            outfile.write('  / ' + '\n')
        elif simulation.subsurface_flow:
            outfile.write('  PROCESS_MODELS' + '\n')
            outfile.write('    SUBSURFACE_FLOW ' +
                          simulation.subsurface_flow + '\n')
            if simulation.mode in mode_names_allowed:
                outfile.write('      MODE ' + simulation.mode + '\n')
                if simulation.options:
                    simulation.options._write(outfile)
            else:
                print('simulation.mode: \'' +
                      simulation.mode + '\' is invalid!')
                print('       valid simulation.mode:', mode_names_allowed, '\n')
            outfile.write('    / ' + '\n')
            if simulation.simulation_type.lower() == 'geomechanics_subsurface':
                outfile.write('    GEOMECHANICS_SUBSURFACE ' +
                              simulation.geomechanics_subsurface + '\n')
            outfile.write('  / ' + '\n')
        elif simulation.subsurface_transport:
            outfile.write('  PROCESS_MODELS' + '\n')
            outfile.write('    SUBSURFACE_TRANSPORT ' +
                          simulation.subsurface_transport + '\n')
            if simulation.itol_relative_update:
                outfile.write('      ITOL_RELATIVE_UPDATE %s\n' % \
                              strD(simulation.itol_relative_update))
            if simulation.flowtran_coupling:
                outfile.write(
                    '      ' + simulation.flowtran_coupling.upper() + '\n')
            if simulation.max_volume_fraction_change:
                outfile.write('      MAX_VOLUME_FRACTION_CHANGE ' +
                              strD(simulation.max_volume_fraction_change) +
                              '\n')
            outfile.write('    / ' + '\n')
            if simulation.ufd_decay:
                outfile.write('  UFD_DECAY %s\n' % simulation.ufd_decay)
                outfile.write('  /\n')
            if simulation.ufd_biosphere:
                outfile.write('  UFD_BIOSPHERE %s\n' % simulation.ufd_biosphere)
                outfile.write('  /\n')

            outfile.write('  / ' + '\n')
        if simulation.checkpoint:
            self._write_checkpoint(outfile)
        if simulation.restart:
            self._write_restart(outfile)

        outfile.write('END' + '\n\n')

    def _write_subsurface_simulation_begin(self, outfile):
        if self.simulation.subsurface_flow or \
                self.simulation.subsurface_transport:
            outfile.write('SUBSURFACE\n\n')

    def _write_subsurface_simulation_end(self, outfile):
        if self.simulation.subsurface_flow or \
                self.simulation.subsurface_transport:
            outfile.write('END_SUBSURFACE\n\n')

    def _read_co2_database(self, infile, line):
        self.co2_database = del_extra_slash(self.splitter(line))

    def _write_overwrite_restart_flow(self, outfile):
        outfile.write('OVERWRITE_RESTART_FLOW_PARAMS' + '\n\n')

    def _write_overwrite_restart_transport(self, outfile):
        outfile.write('OVERWRITE_RESTART_TRANSPORT' + '\n\n')

    def _write_initialize_flow_from_file(self, outfile):
        outfile.write('INITIALIZE_FLOW_FROM_FILE ' +
                      self.initialize_flow_from_file + '\n\n')

    def _write_initialize_transport_from_file(self, outfile):
        outfile.write('INITIALIZE_TRANSPORT_FROM_FILE ' +
                      self.initialize_transport_from_file + '\n\n')

    def _write_isothermal(self, outfile):
        outfile.write('ISOTHERMAL' + '\n\n')

    def _write_reference_temperature(self, outfile):
        outfile.write('REFERENCE_TEMPERATURE ' +
                      strD(self.reference_temperature) + '\n\n')

    def _write_reference_porosity(self, outfile):
        outfile.write('REFERENCE_POROSITY ' +
                      strD(self.reference_porosity) + '\n\n')

    def _write_co2_database(self, outfile):
        self._header(outfile, headers['co2_database'])
        outfile.write('CO2_DATABASE ' + self.co2_database + '\n\n')

    def _write_multiple_continuum(self, outfile):
        self._header(outfile, headers['multiple_continuum'])
        outfile.write('MULTIPLE_CONTINUUM\n\n')

    def _read_regression(self, infile, line):
        regression = pregression()
        keep_reading = True
        while keep_reading:  # Read through all cards
            line = get_next_line(infile)
            key = line.strip().split()[0].lower()  # take first key word

            if key == 'cells':
                keep_reading_2 = True
                cell_list = []
                while keep_reading_2:
                    for i in range(100):
                        line1 = get_next_line(infile)
                        if line1.strip().split()[0].lower() in ['/', 'end']:
                            keep_reading_2 = False
                            break

                        # Convert however many ints exist on this line into
                        # a list and extend
                        cell_list.extend(list(map(int, line1.split())))
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
        if regression.cells_per_process:
            outfile.write('  CELLS_PER_PROCESS' + ' ' +
                          str(regression.cells_per_process) + '\n')
        if regression.cells and regression.cells[0] != '':
            outfile.write('  CELLS' + '\n')
            for cell in regression.cells:
                outfile.write('    ' + str(cell) + '\n')
            outfile.write('  /' + '\n')
        outfile.write('END' + '\n\n')

    def _read_grid(self, infile, line):
        grid = pgrid()  # assign defaults before reading in values
        self.co2_database = ''

        keep_reading = True
        bounds_key = False
        while keep_reading:
            line = get_next_line(infile)  # get next line
            key = line.strip().split()[0].lower()  # take first keyword

            if key == 'type':
                grid.type = line.strip().split()[1].lower()
                if grid.type in ['unstructured_explicit','unstructured_implicit']:
                    grid.filename = self.splitter(line)
            elif key == 'bounds':
                keep_reading_2 = True
                while keep_reading_2:

                    line1 = get_next_line(infile)
                    grid.lower_bounds = [floatD(bnd) for bnd in line1.split()]

                    line2 = get_next_line(infile)
                    grid.upper_bounds = [floatD(bnd) for bnd in line2.split()]

                    line3 = get_next_line(infile)
                    if line3.strip().split()[0].lower() in ['/', 'end']:
                        keep_reading_2 = False

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
            elif key == 'dxyz':
                if bounds_key:
                    raise PyFLOTRAN_ERROR('specify either bounds of dxyz!')
                keep_reading_2 = True
                while keep_reading_2:
                    line = get_next_line(infile)
                    grid.dx = [floatD(val) for val in line.strip().split()]
                    line = get_next_line(infile)
                    grid.dy = [floatD(val) for val in line.strip().split()]
                    line = get_next_line(infile)
                    grid.dz = [floatD(val) for val in line.strip().split()]
                    line = get_next_line(infile)
                    if line.strip().split()[0].lower() not in ['/', 'end']:
                        raise PyFLOTRAN_ERROR(
                            'dx dy dz -- all three are not specified!')
                    else:
                        keep_reading_2 = False
            elif key in ['/', 'end']:
                keep_reading = False
        self.grid = grid

    def _write_grid(self, outfile):
        self._header(outfile, headers['grid'])
        grid = self.grid
        outfile.write('GRID\n')
        if grid.type not in grid_types_allowed:
            print('       valid grid.types:', grid_types_allowed)
            raise PyFLOTRAN_ERROR(
                'grid.type: \'' + grid.type + '\' is invalid!')
        if grid.type == 'structured':
            outfile.write('  TYPE ' + grid.type)
            if grid.symmetry_type not in grid_symmetry_types_allowed:
                print('    valid grid.symmetry_types:', \
                    grid_symmetry_types_allowed)
                raise PyFLOTRAN_ERROR('grid.symmetry_type: \''
                                      + grid.symmetry_type + '\' is invalid')
            elif grid.symmetry_type == 'cartesian' or grid.symmetry_type == '':
                outfile.write('\n')
            elif grid.symmetry_type == 'cylindrical':
                outfile.write(' ' + grid.symmetry_type + '\n')
            elif grid.symmetry_type == 'spherical':
                outfile.write(' ' + grid.symmetry_type + '\n')
            if grid.lower_bounds[0] is not None:
                if grid.symmetry_type == 'cartesian' or \
                        grid.symmetry_type == '':
                    outfile.write('  BOUNDS\n')
                    outfile.write('    ')
                    for i in range(len(grid.lower_bounds)):
                        outfile.write(strD(grid.lower_bounds[i]) + ' ')
                    outfile.write('\n    ')
                    for i in range(len(grid.upper_bounds)):
                        outfile.write(strD(grid.upper_bounds[i]) + ' ')
                    outfile.write('\n  /\n')
                elif grid.symmetry_type == 'cylindrical':
                    outfile.write('  BOUNDS\n')
                    outfile.write('    ')
                    outfile.write(strD(grid.lower_bounds[0]) + ' ')  # low x
                    outfile.write(strD(grid.lower_bounds[2]) + ' ')  # low z
                    outfile.write(strD(9999) + ' ')  # dummy for final value
                    outfile.write('\n    ')
                    outfile.write(strD(grid.upper_bounds[0]) + ' ')  # low x
                    outfile.write(strD(grid.upper_bounds[2]) + ' ')  # low z
                    # dummy value not used in PFLOTRAN
                    outfile.write(strD(9999) + ' ')
                    outfile.write('\n  /\n')  # / is end of writing out bounds
                elif grid.symmetry_type == 'spherical':
                    raise PyFLOTRAN_ERROR('grid.symmetry_type: \'' +
                                          grid.symmetry_type +
                                          '\' not currently supported')

            # DXYZ is only written if no bounds are provided
            elif len(grid.dx) != 0:  
                outfile.write('  DXYZ\n')
                if grid.symmetry_type == 'cartesian' or \
                        grid.symmetry_type == '':  # cartesian, DXYZ grid
                    for j in range(len(grid.dx)):
                        outfile.write('    ' + strD(grid.dx[j]))
                        if j % 5 == 4 and j < len(grid.dx) - 1:
                            outfile.write('   ' + '\\' + '\n')
                    outfile.write('\n')
                    for j in range(len(grid.dy)):
                        outfile.write('    ' + strD(grid.dy[j]))
                        if j % 5 == 4 and j < len(grid.dy) - 1:
                            outfile.write('   ' + '\\' + '\n')
                    outfile.write('\n')
                    for j in range(len(grid.dz)):
                        outfile.write('    ' + strD(grid.dz[j]))
                        if j % 5 == 4 and j < len(grid.dz) - 1:
                            outfile.write('   ' + '\\' + '\n')
                    outfile.write('\n')
                    outfile.write('  /\n')
                # cylindrical, DXYZ grid
                elif grid.symmetry_type == 'cylindrical':  
                    for j in range(len(grid.dx)):  # for x or r
                        outfile.write('    ' + strD(grid.dx[j]))
                        if j % 5 == 4 and j < len(grid.dx) - 1:
                            outfile.write('   ' + '\\' + '\n')
                    outfile.write('\n')

                    # for y coordinate (only 1 value allowed)
                    if len(grid.dy) == 1:  
                        outfile.write('    ' + strD(grid.dy[0]))
                        outfile.write('\n')
                    else:
                        raise PyFLOTRAN_ERROR('grid.dy must be length 1 for ' +
                                              'cylindrical grid.symmetry_type')
                    for j in range(len(grid.dz)):  # for z coordinate
                        outfile.write('    ' + strD(grid.dz[j]))
                        if j % 5 == 4 and j < len(grid.dz) - 1:
                            outfile.write('   ' + '\\' + '\n')
                    outfile.write('\n')
                    outfile.write('  /\n')
                elif grid.symmetry_type == 'spherical':
                    raise PyFLOTRAN_ERROR('grid.symmetry_type: \'' +
                                          grid.symmetry_type +
                                          '\' not supported')
            else:
                raise PyFLOTRAN_ERROR(
                    'either bounds or dx, dy, dz have to be specified!')
            outfile.write('  NXYZ' + ' ')
            if grid.lower_bounds:  # write NXYZ for BOUNDS
                for i in range(3):
                    if grid.nxyz[i]:
                        outfile.write(strI(grid.nxyz[i]) + ' ')
            else:  # write NXYZ based on length of dx, dy, dz (DXYZ)
                outfile.write(strI(len(grid.dx)) + ' ')
                outfile.write(strI(len(grid.dy)) + ' ')
                outfile.write(strI(len(grid.dz)) + ' ')
            outfile.write('\n')
        else:
            outfile.write('  TYPE ' + grid.type + ' ' + grid.filename + '\n')
        if grid.origin:
            outfile.write('  ORIGIN' + ' ')
            for i in range(3):
                outfile.write(strD(grid.origin[i]) + ' ')
            outfile.write('\n')
        if grid.gravity:
            outfile.write('  GRAVITY' + ' ')
            for i in range(3):
                outfile.write(strD(grid.gravity[i]) + ' ')
            outfile.write('\n')
        outfile.write('END\n\n')

    def _read_timestepper(self, infile, line):
        keep_reading = True
        timestepper = ptimestepper()
        if line.strip().split()[0].lower() == 'timestepper':
            timestepper.ts_mode = line.strip().split()[-1].lower()

        while keep_reading:  # read through all cards
            line = get_next_line(infile)

            key = line.strip().split()[0].lower()  # take first keyword
            if key == 'ts_acceleration':
                timestepper.ts_acceleration = int(self.splitter(line))
            elif key == 'num_steps_after_cut':
                timestepper.num_steps_after_cut = int(self.splitter(line))
            elif key == 'max_steps':
                timestepper.max_steps = int(self.splitter(line))
            elif key == 'max_ts_cuts':
                timestepper.max_ts_cuts = int(self.splitter(line))
            elif key == 'initialize_to_steady_state':
                timestepper.initialize_to_steady_state = True
            elif key == 'run_as_steady_state':
                timestepper.run_as_steady_state = True
            elif key in ['/', 'end']:
                keep_reading = False

        if timestepper.ts_mode == 'flow':
            self.timestepper_flow = timestepper
        elif timestepper.ts_mode == 'transport':
            self.timestepper_transport = timestepper
        else:
            PyFLOTRAN_WARNING('Unknown timestepping mode!')

    def _write_timestepper(self, outfile):
        self._header(outfile, headers['timestepper'])

        if self.timestepper_flow:
            outfile.write('TIMESTEPPER ' +
                          self.timestepper_flow.ts_mode.upper() + '\n')
            if self.timestepper_flow.ts_acceleration is not None:
                outfile.write('  ' + 'TS_ACCELERATION ' +
                              str(self.timestepper_flow.ts_acceleration) +
                              '\n')
            if self.timestepper_flow.num_steps_after_cut is not None:
                outfile.write('  ' + 'NUM_STEPS_AFTER_CUT ' +
                              str(self.timestepper_flow.num_steps_after_cut) +
                              '\n')
            if self.timestepper_flow.max_ts_cuts is not None:
                outfile.write('  ' + 'MAX_TS_CUTS ' +
                              str(self.timestepper_flow.max_ts_cuts) + '\n')
            if self.timestepper_flow.max_steps is not None:
                outfile.write('  ' + 'MAX_STEPS ' +
                              str(self.timestepper_flow.max_steps) + '\n')
            if self.timestepper_flow.initialize_to_steady_state:
                outfile.write('  ' + 'INITIALIZE_TO_STEADY_STATE ' + '\n')
            if self.timestepper_flow.run_as_steady_state:
                outfile.write('  ' + 'RUN_AS_STEADY_STATE ' + '\n')
            outfile.write('END\n\n')

        if self.timestepper_transport:
            outfile.write('TIMESTEPPER ' +
                          self.timestepper_transport.ts_mode.upper() + '\n')
            if self.timestepper_transport.ts_acceleration:
                outfile.write('  ' + 'TS_ACCELERATION ' +
                              str(self.timestepper_transport.ts_acceleration) +
                              '\n')
            if self.timestepper_transport.num_steps_after_cut:
                outfile.write('  ' + 'NUM_STEPS_AFTER_CUT ' +
                              str(self.timestepper_transport.
                                  num_steps_after_cut) + '\n')
            if self.timestepper_transport.max_ts_cuts:
                outfile.write('  ' + 'MAX_TS_CUTS ' +
                              str(self.timestepper_transport.max_ts_cuts) +
                              '\n')
            if self.timestepper_transport.max_steps:
                outfile.write('  ' + 'MAX_STEPS ' +
                              str(self.timestepper_transport.max_steps) +
                              '\n')
            if self.timestepper_transport.initialize_to_steady_state:
                outfile.write('  ' + 'INITIALIZE_TO_STEADY_STATE ' +
                              '\n')
            if self.timestepper_transport.run_as_steady_state:
                outfile.write('  ' + 'RUN_AS_STEADY_STATE ' +
                              '\n')
            outfile.write('END\n\n')

    def _read_prop(self, infile, line):
        np_name = self.splitter(line)  # property name
        np_id = None
        p = pmaterial(id=None, name=None, characteristic_curves=None,
                      porosity=None, tortuosity=None, permeability=None)
        np_porosity = p.porosity
        np_characteristic_curves = p.characteristic_curves
        np_tortuosity = p.tortuosity
        np_density = p.density
        np_specific_heat = p.specific_heat
        np_heat_capacity = p.heat_capacity
        np_cond_dry = p.cond_dry
        np_cond_wet = p.cond_wet
        np_saturation = p.saturation
        np_permeability = []
        np_permeability_critical_porosity = p.permeability_critical_porosity
        np_permeability_power = p.permeability_power
        np_permeability_min_scale_factor = p.permeability_min_scale_factor
        np_perm_iso = False
        np_perm_aniso = False
        np_longitudinal_dispersivity = p.longitudinal_dispersivity
        np_transverse_dispersivity_h = p.transverse_dispersivity_h
        np_transverse_dispersivity_v = p.transverse_dispersivity_v
        keep_reading = True
        np_specific_heat_unit = p.specific_heat_unit
        np_heat_capacity_unit = p.heat_capacity_unit
        np_density_unit = p.density_unit
        np_cond_dry_unit = p.cond_dry_unit
        np_cond_wet_unit = p.cond_wet_unit

        _perm_factor = None
        np_tortuosity_fnc_of_porosity = None
        _inactive = False
        _sc = None
        soil_comp_func = ''
        soil_comp = None

        while keep_reading:  # read through all cards
            line = get_next_line(infile)

            key = line.strip().split()[0].lower()  # take first keyword
            if key == 'id':
                if line.split()[1].lower() == 'dbase_value':
                    np_id = pdbase(self.splitter(line))
                else:
                    np_id = int(self.splitter(line))
            elif key == 'characteristic_curves':
                np_characteristic_curves = self.splitter(line)
            elif key == 'porosity':
                if line.split()[1].lower() == 'dataset':
                    np_porosity = self.splitter(line)
                elif line.split()[1].lower() == 'dbase_value':
                    np_porosity = pdbase(self.splitter(line))
                else:
                    np_porosity = floatD(self.splitter(line))
            elif key == 'inactive':
                _inactive = True
            elif key == 'tortuosity':
                if line.split()[1].lower() == 'dataset':
                    np_porosity = self.splitter(line)
                else:
                    np_tortuosity = floatD(self.splitter(line))
            elif key == 'tortuosity_function_of_porosity':
                np_tortuosity_fnc_of_porosity = floatD(self.splitter(line))
            elif key == 'rock_density':
                if len(line.strip().split()[1:]) > 1:
                    np_density_unit = line.strip().split()[2]
                np_density = floatD(line.strip().split()[1])
            elif key == 'specific_heat':
                if len(line.strip().split()[1:]) > 1:
                    np_specific_heat_unit = line.strip().split()[2]
                np_specific_heat = floatD(line.strip().split()[1])
            elif key == 'heat_capacity':
                if len(line.strip().split()[1:]) > 1:
                    np_heat_capacity_unit = line.strip().split()[2]
                np_heat_capacity = floatD(line.strip().split()[1])
            elif key == 'thermal_conductivity_dry':
                if len(line.strip().split()[1:]) > 1:
                    np_cond_dry_unit = line.strip().split()[2]
                np_cond_dry = floatD(line.strip().split()[1])
            elif key == 'thermal_conductivity_wet':
                if len(line.strip().split()[1:]) > 1:
                    np_cond_wet_unit = line.strip().split()[2]
                np_cond_wet = floatD(line.strip().split()[1])
            elif key == 'saturation_function':
                np_saturation = self.splitter(line)
            elif key == 'permeability_power':
                np_permeability_power = self.splitter(line)
            elif key == 'permeability_critical_porosity':
                np_permeability_critical_porosity = self.splitter(line)
            elif key == 'permeability_min_scale_factor':
                np_permeability_min_scale_factor = self.splitter(line)
            elif key == 'longitudinal_dispersivity':
                np_longitudinal_dispersivity = ' '.join(line.split()[1:])
            elif key == 'transverse_dispersivity_h':
                np_transverse_dispersivity_h = self.splitter(line)
            elif key == 'transverse_dispersivity_v':
                np_transverse_dispersivity_v = self.splitter(line)
            elif key == 'soil_compressibility_function':
                soil_comp_func = self.splitter(line)
            elif key == 'soil_compressibility':
                soil_comp = floatD(self.splitter(line))
            elif key == 'permeability':
                keep_reading_2 = True
                while keep_reading_2:
                    line = get_next_line(infile)
                    key = line.split()[0].lower()  # take first keyword

                    if key in ['perm_iso', 'perm_x', 'perm_y', 'perm_z']:
                        if 'dbase_value' in line.lower():
                            np_permeability.append(pdbase(self.splitter(line)))
                        else:
                            np_permeability.append(floatD(self.splitter(line)))
                    elif key == 'isotropic':
                        np_perm_iso = True
                    elif key == 'anisotropic':
                        np_perm_aniso = True                        
                    elif key == 'dataset':
                        np_permeability.append(self.splitter(line))
                    elif key in ['/', 'end']:
                        keep_reading_2 = False
            elif key == 'perm_factor':
                _perm_factor = pmaterial.perm_factor()
                while True:
                    line = get_next_line(infile)
                    key = line.split()[0].lower()

                    if key == 'min_pressure':
                        _perm_factor.min_pressure = floatD(self.splitter(line))
                    elif key == 'max_pressure':
                        _perm_factor.max_pressure = floatD(self.splitter(line))
                    elif key == 'max_permfactor':
                        _perm_factor.max_permfactor = floatD(self.splitter(line))
                    elif key in ['/','end']:
                        break

            elif key == 'secondary_continuum':
                _sc = pmaterial.secondary_continuum()
                while True:
                    subline = get_next_line(infile)
                    subkey = subline.split()[0].lower()

                    if subkey == 'type':
                        _sc.continuum_type = self.splitter(subline)
                    elif subkey == 'log_grid_spacing':
                        _sc.log_grid_spacing = True
                    elif subkey == 'radius':
                        _sc.radius = floatD(self.splitter(subline))
                    elif subkey == 'num_cells':
                        _sc.num_cells = floatD(self.splitter(subline))
                    elif subkey == 'outer_spacing':
                        _sc.outer_spacing = floatD(self.splitter(subline))
                    elif subkey == 'epsilon':
                        _sc.epsilon = floatD(self.splitter(subline))
                    elif subkey == 'temperature':
                        _sc.temperature = floatD(self.splitter(subline))
                    elif subkey == 'porosity':
                        _sc.porosity = floatD(self.splitter(subline))
                    elif subkey in ['/','end']:
                        break

            elif key in ['/', 'end']:
                keep_reading = False

        # create an empty material property
        new_prop = pmaterial(id=np_id, name=np_name,
                             characteristic_curves=np_characteristic_curves,
                             porosity=np_porosity, tortuosity=np_tortuosity,
                             density=np_density,
                             specific_heat=np_specific_heat,
                             cond_dry=np_cond_dry,
                             cond_wet=np_cond_wet, saturation=np_saturation,
                             permeability=np_permeability,
                             permeability_power=np_permeability_power,
                             permeability_isotropic=np_perm_iso,
                             permeability_anisotropic=np_perm_aniso,
                             permeability_critical_porosity=np_permeability_critical_porosity,
                             permeability_min_scale_factor=np_permeability_min_scale_factor,
                             longitudinal_dispersivity=np_longitudinal_dispersivity,
                             transverse_dispersivity_h=np_transverse_dispersivity_h,
                             transverse_dispersivity_v=np_transverse_dispersivity_v,
                             density_unit=np_density_unit,
                             cond_wet_unit=np_cond_wet_unit,
                             cond_dry_unit=np_cond_dry_unit,
                             specific_heat_unit=np_specific_heat_unit,
                             heat_capacity=np_heat_capacity,
                             heat_capacity_unit=np_heat_capacity_unit,
                             perm_factor=_perm_factor,
                             tortuosity_function_of_porosity=np_tortuosity_fnc_of_porosity,
                             inactive=_inactive,
                             soil_compressibility_function=soil_comp_func,
                             soil_compressibility=soil_comp,
                             secondary_continuum=_sc)

        self.add(new_prop)

    # Adds a prop object.
    def _add_prop(self, prop=pmaterial(), overwrite=False):
        # check if prop already exists
        if isinstance(prop, pmaterial):
            if prop.id in self.prop.keys():
                if not overwrite:
                    warning = 'A material property with id ' + \
                              str(prop.id) + ' already exists. Prop ' + \
                              'will not be defined, use overwrite ' + \
                              '= True in add() to overwrite the old prop.'
                    print(warning,
                    build_warnings.append(warning))
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
            if prop.inactive:
                outfile.write('  INACTIVE\n')
            if prop.characteristic_curves: #and self.simulation.subsurface_flow:
                outfile.write('  CHARACTERISTIC_CURVES ' +
                              prop.characteristic_curves + '\n')
            if prop.porosity:
                if type(prop.porosity) is str:
                    outfile.write('  POROSITY DATASET ' + prop.porosity + '\n')
                else:
                    outfile.write('  POROSITY ' + strD(prop.porosity) + '\n')
            if prop.tortuosity:
                outfile.write('  TORTUOSITY ' + strD(prop.tortuosity) + '\n')
            if prop.tortuosity_function_of_porosity:
                outfile.write('  TORTUOSITY_FUNCTION_OF_POROSITY %s\n' % \
                              prop.tortuosity_function_of_porosity)
            if prop.density:
                outfile.write('  ROCK_DENSITY ' + strD(prop.density))
                if prop.density_unit:
                    outfile.write(' ' + prop.density_unit)
                outfile.write('\n')
            if prop.specific_heat:
                outfile.write('  SPECIFIC_HEAT ' +
                              strD(prop.specific_heat))
                if prop.specific_heat_unit:
                    outfile.write(' ' + prop.specific_heat_unit)
                outfile.write('\n')
            if prop.heat_capacity:
                outfile.write('  HEAT_CAPACITY ' +
                              strD(prop.heat_capacity))
                if prop.heat_capacity_unit:
                    outfile.write(' ' + prop.heat_capacity_unit)
                outfile.write('\n')
            if prop.cond_dry:
                outfile.write('  THERMAL_CONDUCTIVITY_DRY ' +
                              strD(prop.cond_dry))
                if prop.cond_dry_unit:
                    outfile.write(' ' + prop.cond_dry_unit)
                outfile.write('\n')
            if prop.cond_wet:
                outfile.write('  THERMAL_CONDUCTIVITY_WET ' +
                              strD(prop.cond_wet))
                if prop.cond_wet_unit:
                    outfile.write(' ' + prop.cond_wet_unit)
                outfile.write('\n')
            if prop.saturation:
                outfile.write('  SATURATION_FUNCTION ' +
                              prop.saturation + '\n')
            if prop.permeability_power:
                outfile.write('  PERMEABILITY_POWER ' +
                              strD(prop.permeability_power) + '\n')
            if prop.permeability_critical_porosity:
                outfile.write('  PERMEABILITY_CRITICAL_POROSITY ' +
                              strD(prop.permeability_critical_porosity) + '\n')
            if prop.permeability_min_scale_factor:
                outfile.write('  PERMEABILITY_MIN_SCALE_FACTOR ' +
                              strD(prop.permeability_min_scale_factor) + '\n')
            if prop.longitudinal_dispersivity:
                outfile.write('  LONGITUDINAL_DISPERSIVITY ' +
                              strD(prop.longitudinal_dispersivity) + '\n')
            if prop.transverse_dispersivity_h:
                outfile.write('  TRANSVERSE_DISPERSIVITY_H ' +
                              strD(prop.transverse_dispersivity_h) + '\n')
            if prop.transverse_dispersivity_v:
                outfile.write('  TRANSVERSE_DISPERSIVITY_V ' +
                              strD(prop.transverse_dispersivity_v) + '\n')

            if prop.permeability:
                outfile.write('  PERMEABILITY\n')
                if type(prop.permeability[0]) is str:
                    outfile.write('    DATASET ' + prop.permeability[0] + '\n')
                elif len(prop.permeability) == 1:
                    outfile.write('    PERM_ISO ' +
                                  strD(prop.permeability[0]) + '\n')
                else:
                    outfile.write('    PERM_X ' +
                                  strD(prop.permeability[0]) + '\n')
                    outfile.write('    PERM_Y ' +
                                  strD(prop.permeability[1]) + '\n')
                    outfile.write('    PERM_Z ' +
                                  strD(prop.permeability[2]) + '\n')
                if prop.permeability_anisotropic:
                    outfile.write('    ANISOTROPIC\n')
                if prop.permeability_isotropic:
                    outfile.write('    ISOTROPIC\n')                    

                outfile.write('  /\n')

            if prop.perm_factor:
                outfile.write('  PERM_FACTOR\n')
                _pf = prop.perm_factor
                if _pf.min_pressure:
                    outfile.write('    MIN_PRESSURE %s\n' % floatD(_pf.min_pressure))
                if _pf.max_pressure:
                    outfile.write('    MAX_PRESSURE %s\n' % floatD(_pf.max_pressure))
                if _pf.max_permfactor:
                    outfile.write('    MAX_PERMFACTOR %s\n' % floatD(_pf.max_permfactor))
                outfile.write('  /\n')

            if prop.secondary_continuum:
                prop.secondary_continuum._write(outfile)

            if prop.soil_compressibility_function:
                if prop.soil_compressibility_function.upper() in \
                allowed_soil_compressibility_functions:
                    outfile.write('  SOIL_COMPRESSIBILITY_FUNCTION ' +
                                  prop.soil_compressibility_function + '\n')
                else:
                    raise PyFLOTRAN_ERROR(
                        'PyFLOTRAN ERROR: soil_compressibility_function ' +
                        prop.soil_compressibility_function + ' is invalid!' +
                        ' Try one of ' + \
                        str(allowed_soil_compressibility_functions))

            if prop.soil_compressibility:  # this is alpha
                outfile.write('  SOIL_COMPRESSIBILITY ' +
                              strD(prop.soil_compressibility) + '\n')

            if prop.soil_reference_pressure:  # this is alpha
                outfile.write('  SOIL_REFERENCE_PRESSURE ' +
                              strD(prop.soil_reference_pressure) + '\n')

            #if prop.secondary_continuum:
            #    self._write_sec(prop.secondary_continuum, outfile)

            if prop.compressibility_function:
                # if lsolver.name.lower() in solver_names_allowed:
                if prop.compressibility_function.lower() in \
                allowed_compressibility_functions:
                    outfile.write('  GEOMECHANICS_SUBSURFACE_PROPS\n')
                    outfile.write('    COMPRESSIBILITY_FUNCTION ' +
                                  prop.compressibility_function + '\n')
                    # check for Bandis parameters
                    if prop.bandis_A:
                        outfile.write('    BANDIS_A ' +
                                      strD(prop.bandis_A) + '\n')
                    if prop.bandis_B:
                        outfile.write('    BANDIS_B ' +
                                      strD(prop.bandis_B) + '\n')
                    if prop.maximum_aperture:
                        outfile.write('    MAXIMUM_APERTURE ' +
                                      strD(prop.maximum_aperture) + '\n')
                    if prop.normal_vector:
                        outfile.write('    NORMAL_VECTOR ' +
                                      strD(prop.normal_vector[0]) + ' ' +
                                      strD(prop.normal_vector[1]) + ' ' +
                                      strD(prop.normal_vector[2]) + '\n')
                    outfile.write('  /\n')

            if prop.B_Bandis:
                if type(prop.B_Bandis) is str:
                    outfile.write('  B_BANDIS DATASET ' + prop.B_Bandis + '\n')
                else:
                    outfile.write('  B_BANDIS ' + strD(prop.B_Bandis) + '\n')
            if prop.A_Bandis:
                if type(prop.A_Bandis) is str:
                    outfile.write('  A_BANDIS DATASET ' + prop.A_Bandis + '\n')
                else:
                    outfile.write('  A_BANDIS ' + strD(prop.A_Bandis) + '\n')
            if prop.b_min_Bandis:
                if type(prop.b_min_Bandis) is str:
                    outfile.write('  B_MIN_BANDIS DATASET ' +
                                  prop.b_min_Bandis + '\n')
                else:
                    outfile.write('  B_MIN_BANDIS ' +
                                  strD(prop.b_min_Bandis) + '\n')
            elif prop.b_min_Bandis == 0.0:
                outfile.write('  B_MIN_BANDIS ' +
                              strD(prop.b_min_Bandis) + '\n')
            if prop.b_max_Bandis:
                if type(prop.b_max_Bandis) is str:
                    outfile.write('  B_MAX_BANDIS DATASET ' +
                                  prop.b_max_Bandis + '\n')
                else:
                    outfile.write('  B_MAX_BANDIS ' +
                                  strD(prop.b_max_Bandis) + '\n')
            if prop.sigma_n_Bandis:
                if type(prop.sigma_n_Bandis) is str:
                    outfile.write('  SIGMA_N_BANDIS DATASET ' +
                                  prop.sigma_n_Bandis + '\n')
                else:
                    outfile.write('  SIGMA_N_BANDIS ' +
                                  strD(prop.sigma_n_Bandis) + '\n')

            if prop.friction_Bandis:
                if type(prop.friction_Bandis) is str:
                    outfile.write('  FRICTION_BANDIS DATASET ' +
                                  prop.friction_Bandis + '\n')
                else:
                    outfile.write('  FRICTION_BANDIS ' +
                                  strD(prop.friction_Bandis) + '\n')

            if prop.perm_multiplier_Bandis:
                if type(prop.perm_multiplier_Bandis) is str:
                    outfile.write('  PERM_MULTIPLIER_BANDIS DATASET ' +
                                  prop.perm_multiplier_Bandis + '\n')
                else:
                    outfile.write('  PERM_MULTIPLIER_BANDIS ' +
                                  strD(prop.perm_multiplier_Bandis) + '\n')

            if prop.sigma_x_Bandis:
                if type(prop.sigma_x_Bandis) is str:
                    outfile.write('  SIGMA_XX_BANDIS DATASET ' +
                                  prop.sigma_x_Bandis + '\n')
                else:
                    outfile.write('  SIGMA_XX_BANDIS ' +
                                  strD(prop.sigma_x_Bandis) + '\n')

            if prop.sigma_y_Bandis:
                if type(prop.sigma_y_Bandis) is str:
                    outfile.write('  SIGMA_YY_BANDIS DATASET ' +
                                  prop.sigma_y_Bandis + '\n')
                else:
                    outfile.write('  SIGMA_YY_BANDIS ' +
                                  strD(prop.sigma_y_Bandis) + '\n')

            if prop.sigma_z_Bandis:
                if type(prop.sigma_z_Bandis) is str:
                    outfile.write('  SIGMA_ZZ_BANDIS DATASET ' +
                                  prop.sigma_z_Bandis + '\n')
                else:
                    outfile.write('  SIGMA_ZZ_BANDIS ' +
                                  strD(prop.sigma_z_Bandis) + '\n')

            if prop.normal_x_Bandis:
                if type(prop.normal_x_Bandis) is str:
                    outfile.write('  NORMAL_X_BANDIS DATASET ' +
                                  prop.normal_x_Bandis + '\n')
                else:
                    outfile.write('  NORMAL_X_BANDIS ' +
                                  strD(prop.normal_x_Bandis) + '\n')

            if prop.normal_y_Bandis:
                if type(prop.normal_y_Bandis) is str:
                    outfile.write('  NORMAL_Y_BANDIS DATASET ' +
                                  prop.normal_y_Bandis + '\n')
                else:
                    outfile.write('  NORMAL_Y_BANDIS ' +
                                  strD(prop.normal_y_Bandis) + '\n')

            if prop.normal_z_Bandis:
                if type(prop.normal_z_Bandis) is str:
                    outfile.write('  NORMAL_Z_BANDIS DATASET ' +
                                  prop.normal_z_Bandis + '\n')
                else:
                    outfile.write('  NORMAL_Z_BANDIS ' +
                                  strD(prop.normal_z_Bandis) + '\n')

            if prop.frac_present_Bandis:
                if type(prop.frac_present_Bandis) is str:
                    outfile.write('  FRAC_PRESENT_BANDIS DATASET ' +
                                  prop.frac_present_Bandis + '\n')
                else:
                    outfile.write('  FRAC_PRESENT_BANDIS ' +
                                  strD(prop.frac_present_Bandis) + '\n')
            elif prop.frac_present_Bandis == 0.0:
                outfile.write('  FRAC_PRESENT_BANDIS ' +
                              strD(prop.frac_present_Bandis) + '\n')
            if prop.b_p_Bandis:
                if type(prop.b_p_Bandis) is str:
                    outfile.write('  B_P_BANDIS DATASET ' +
                                  prop.b_p_Bandis + '\n')
                else:
                    outfile.write('  B_P_BANDIS ' +
                                  strD(prop.b_p_Bandis) + '\n')

            outfile.write('END\n\n')

    def _write_sec(self, sec, outfile):
        self._header(outfile, headers['secondary_continuum'])
        outfile.write('  SECONDARY_CONTINUUM\n')
        if sec.continuum_type:
            outfile.write('    TYPE ' + str(sec.continuum_type).upper() + '\n')
        if sec.log_spacing:
            outfile.write('    LOG_GRID_SPACING ' + '\n')
        if sec.outer_spacing:
            outfile.write('    OUTER_SPACING ' + str(sec.outer_spacing) + '\n')
        if sec.fracture_spacing:
            outfile.write('    FRACTURE_SPACING ' +
                          str(sec.fracture_spacing) + '\n')
        if sec.num_cells:
            outfile.write('    NUM_CELLS ' + str(int(sec.num_cells)) + '\n')
        if sec.epsilon:
            outfile.write('    EPSILON ' + str(sec.epsilon) + '\n')
        if sec.temperature:
            outfile.write('    TEMPERATURE ' + str(sec.temperature) + '\n')
        if sec.diffusion_coefficient:
            outfile.write('    DIFFUSION_COEFFICIENT ' +
                          str(sec.diffusion_coefficient) + '\n')
        if sec.porosity:
            outfile.write('    POROSITY ' + str(sec.porosity) + '\n')
        outfile.write('  /\n')

    def _read_time(self, infile):
        time = ptime()
        time.dtf_list = []
        if time.tf:
            time.tf = []
        if time.dti:
            time.dti = []
        if time.dtf:
            time.dtf = []

        keep_reading = True
        while keep_reading:
            line = get_next_line(infile)
            key = line.split()[0].lower()  # take first keyword
            if key == 'final_time':
                tstring = line.split()[1:]  # temp list of strings,
                # do not include 1st sub-string
                # Do this if there is a time unit to read
                time.tf.append(floatD(tstring[0]))
                if len(tstring) > 0:
                    time.tf.append(tstring[1])
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
                        # Find index # in list (Not string)
                        at_i = tstring.index('at')
                        # Use string only after 'at'
                        tstring = line.split()[at_i + 2:]

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
                outfile.write('  FINAL_TIME ' +
                              strD(time.tf[0]))  # Write value
                if time.tf[1].lower() in time_units_allowed:
                    # Write time unit
                    outfile.write(' ' + time.tf[1].lower() + '\n')
                else:
                    print('       valid time.units', time_units_allowed, '\n')
                    raise PyFLOTRAN_ERROR(
                        'PyFLOTRAN ERROR: time.tf[1]: ' +
                        time.tf[1] + ' is invalid!')
            except:
                raise PyFLOTRAN_ERROR(
                    'time.tf (final time) input is invalid. ' +
                    'Format should be a list: [number, string]')

        # write INITIAL_TIMESTEP_SIZE statement (dti)
        if time.dti:
            try:
                outfile.write('  INITIAL_TIMESTEP_SIZE ' +
                              strD(time.dti[0]))  # Write value
                if time.dti[1].lower() in time_units_allowed:
                    outfile.write(' ' + time.dti[1] + '\n')  # Write time unit
                else:
                    print('       valid time.units', time_units_allowed, '\n')
                    raise PyFLOTRAN_ERROR(
                        'time.dti[1]: \'' + time.dti[1] + '\' is invalid.')
            except:
                raise PyFLOTRAN_ERROR(
                    'time.dti (initial timestep size) input is invalid. ' +
                    'Format should be a list: [number, string]')

        # write MAXIMUM_TIMESTEP_SIZE statement dtf
        if time.dtf:
            try:
                outfile.write('  MAXIMUM_TIMESTEP_SIZE ' + strD(time.dtf[0]))
                if time.dtf[1].lower() in time_units_allowed:
                    outfile.write(' ' + time.dtf[1] + '\n')
                else:
                    print('       valid time.units', time_units_allowed, '\n')
                    raise PyFLOTRAN_ERROR(
                        'time.dtf[1]: \'' + time.dtf[1] + '\' is invalid.')
            except:
                raise PyFLOTRAN_ERROR(
                    'time.dtf (maximum timestep size) input is invalid.' +
                    'Format should be a list: [number, string]')

        # Write more MAXIMUM_TIME_STEP_SIZE statements if applicable
        for dtf in time.dtf_list:
            outfile.write('  MAXIMUM_TIMESTEP_SIZE ')

            try:
                # Write 1st value before 'at'
                if isinstance(dtf[0], float):
                    outfile.write(strD(dtf[0]) + ' ')
                else:
                    raise PyFLOTRAN_ERROR(
                        'The 1st variable in a dtf_list is ' +
                        'not recognized as a float.')

                # Write 1st time unit before 'at'
                if isinstance(dtf[1], str):
                    outfile.write((dtf[1]) + ' ')
                else:
                    raise PyFLOTRAN_ERROR(
                        'The 2nd variable in a dtf_list is not ' +
                        'recognized as a str (string).')

                outfile.write('at ')

                # Write 2nd value after 'at'
                if isinstance(dtf[2], float):
                    outfile.write(strD(dtf[2]) + ' ')
                else:
                    raise PyFLOTRAN_ERROR(
                        'The 3rd variable in a dtf_list is not ' +
                        'recognized as a float.')

                # Write 2nd time unit after 'at'
                if isinstance(dtf[3], str):
                    outfile.write((dtf[3]))
                else:
                    raise PyFLOTRAN_ERROR('PyFLOTRAN ERROR: The 4th ' +
                                          'variable in a dtf_list ' +
                                          'is not recognized as a ' +
                                          'str (string).')
            except:
                raise PyFLOTRAN_ERROR(
                    'PyFLOTRAN ERROR: time.dtf_list ' +
                    '(maximum timestep size with \'at\') is invalid.' +
                    ' Format should be a ' +
                    'list: [float, str, float, str]')
            outfile.write('\n')
        outfile.write('END\n\n')

    def _read_lsolver(self, infile, line):
        lsolver = plsolver()  # temporary object while reading
        # solver type - tran_solver or flow_solver
        lsolver.name = self.splitter(filter_comment(line)).lower()

        keep_reading = True

        while keep_reading:  # Read through all cards
            line = get_next_line(infile)
            key = line.strip().split()[0].lower()  # take first key word

            if key == 'solver':
                lsolver.solver = self.splitter(line)  # Assign last word
            if key == 'pc_type':
                lsolver.preconditioner = self.splitter(line)
            if key == 'ksp_type':
                lsolver.ksp = self.splitter(line)
            elif key in ['/', 'end']:
                keep_reading = False

        self.add(lsolver)  # Assign object

    # Adds a Linear Solver object.
    def _add_lsolver(self, lsolver=plsolver(), overwrite=False):
        # check if lsolver already exists
        if isinstance(lsolver, plsolver):
            if lsolver.name in self.lsolver.keys():
                if not overwrite:
                    warning = 'WARNING: A linear solver with name ' + \
                              str(lsolver.name) + ' already exists. ' + \
                              'lsolver will not be defined, ' + \
                              'use overwrite = True in add() ' + \
                              'to overwrite the old lsolver.'
                    print(warning,
                    build_warnings.append(warning))
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
                print('       valid solver.names', solver_names_allowed, '\n')
                raise PyFLOTRAN_ERROR(
                    'lsolver.name: \'' + lsolver.name + '\' is invalid.')
            if lsolver.solver:
                outfile.write('  SOLVER ' + lsolver.solver.upper() + '\n')
            if lsolver.preconditioner:
                outfile.write('  PC_TYPE ' +
                              lsolver.preconditioner.upper() + '\n')
            if lsolver.ksp:
                outfile.write('  KSP_TYPE ' + lsolver.ksp.upper() + '\n')
            outfile.write('END\n\n')

    def _read_nsolver(self, infile, line):

        nsolver = pnsolver('')  # Assign Defaults

        # newton solver type - tran_solver or flow_solver
        nsolver.name = self.splitter(line).lower()

        keep_reading = True

        while keep_reading:  # Read through all cards
            line = get_next_line(infile)  # get next line
            key = line.strip().split()[0].lower()  # take first key word
            line = filter_comment(line)

            if key == 'atol':
                nsolver.atol = floatD(self.splitter(line))
            elif key == 'rtol':
                nsolver.rtol = floatD(self.splitter(line))
            elif key == 'stol':
                nsolver.stol = floatD(self.splitter(line))
            elif key == 'dtol':
                nsolver.dtol = floatD(self.splitter(line))
            elif key == 'itol':
                nsolver.itol = floatD(self.splitter(line))
            elif key == 'itol_update':
                nsolver.itol_update = floatD(self.splitter(line))
            elif key == 'maxit':
                nsolver.max_it = int(self.splitter(line))
            elif key == 'maxf':
                nsolver.max_f = int(self.splitter(line))
            elif key == 'matrix_type':
                nsolver.matrix_type = self.splitter(line)
            elif key == 'preconditioner_matrix_type':
                nsolver.preconditioner_matrix_type = self.splitter(line)
            elif key == 'no_infinity_norm':
                nsolver.no_infinity_norm = True
            elif key in ['/', 'end']:
                keep_reading = False
        self.add(nsolver,overwrite=True)  # Assign

    # Adds a Newton Solver object.
    def _add_nsolver(self, nsolver=pnsolver(), overwrite=False):
        # check if nsolver already exists
        if isinstance(nsolver, pnsolver):
            if nsolver.name in self.nsolver.keys():
                if not overwrite:
                    warning = 'WARNING: A newton solver with name ' + \
                              str(nsolver.name) + '\' already exists. ' + \
                              'nsolver will not be defined, ' + \
                              'use overwrite = True in add()' + \
                              ' to overwrite the old nsolver.'
                    print(warning,
                    build_warnings.append(warning))
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
                print('       valid solver.names', solver_names_allowed, '\n')
                raise PyFLOTRAN_ERROR(
                    'nsolver.name: \'' + nsolver.name + '\' is invalid.')
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
            if nsolver.itol_update:
                outfile.write('  ITOL_UPDATE ' +
                              strD(nsolver.itol_update) + '\n')
            if nsolver.max_it:
                outfile.write('  MAXIT ' + str(nsolver.max_it) + '\n')
            if nsolver.max_f:
                outfile.write('  MAXF ' + str(nsolver.max_f) + '\n')
            if nsolver.matrix_type:
                outfile.write('  MATRIX_TYPE ' +
                              nsolver.matrix_type.upper() + '\n')
            if nsolver.preconditioner_matrix_type:
                outfile.write('  PRECONDITIONER_MATRIX_TYPE ' +
                              nsolver.preconditioner_matrix_type.upper() + '\n')
            if nsolver.no_infinity_norm == True:
                outfile.write('  NO_INFINITY_NORM\n')
            outfile.write('END\n\n')

    def _read_output(self, infile):
        output = poutput()
        output.time_list = []
        output.format_list = []
        output.variables_list = []

        keep_reading = True

        while keep_reading:  # Read through all cards
            line = get_next_line(infile)
            key = line.strip().split()[0].lower()  # take first key word
            if key == 'times':
                tstring = line.split()[1:]  # Turn into list, exempt 1st word
                times = []
                for t in tstring:
                    try:
                        times.append(floatD(t))
                    except:
                        times.append(t)
                output.time_list.append(times)
            elif key == 'time_units':
                output.time_units = self.splitter(line).lower()
            elif key == 'screen':
                tstring = line.strip().split()[1].lower()  # Read the 2nd word
                if tstring == 'periodic':
                    output.screen_periodic = int(self.splitter(line))
            elif key == 'periodic':
                tstring = line.strip().split()[1].lower()  # Read the 2nd word
                if tstring == 'time':
                    # 2nd from last word.
                    output.periodic_time.append(floatD(line.split()[-2]))
                    output.periodic_time.append(
                        self.splitter(line))  # last word
                elif tstring == 'timestep':
                    # 2nd from last word.
                    output.periodic_timestep = int(line.split()[-1])
            elif key == 'periodic_observation':
                tstring = line.strip().split()[1].lower()  # Read the 2nd word
                if tstring == 'time':
                    # 2nd from last word.
                    output.periodic_observation_time.append(
                        floatD(line.split()[-2]))
                    output.periodic_observation_time.append(
                        self.splitter(line))  # last word
                elif tstring == 'timestep':
                    output.periodic_observation_timestep = int(
                        self.splitter(line))
            elif key == 'print_column_ids':
                output.print_column_ids = True
            elif key == 'format':
                # Do not include 1st sub-string
                tstring = (line.strip().split()[1:])
                # Convert list into a string seperated by a space
                tstring = ' '.join(tstring).lower()
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
                    line1 = get_next_line(infile)
                    key1 = line1.strip().split()[0].lower()
                    if key1 in output_variables_allowed:
                        output.variables_list.append(key1)
                    elif key1 in ['/', 'end']:
                        keep_reading_1 = False
                    else:
                        raise PyFLOTRAN_ERROR(
                            'variable ' + str(key1) +
                            ' cannot be an output variable.')
            elif key == 'snapshot_file':
                keep_reading1 = True
                while keep_reading1:
                    line1 = get_next_line(infile)
                    key1 = line1.strip().split()[0].lower()
                    if key1 == 'format':
                        if len(line1.strip().split()) == 2:
                            output.snapshot_file.format = \
                            line1.strip().split()[1].lower()
                        elif len(line1.strip().split()) == 3:
                            output.snapshot_file.format = \
                            ' '.join(line1.strip().split()[1:3]).lower()
                        elif len(line1.strip().split()) > 3 and \
                        'times_per_file' in [val.lower() for val in \
                        line1.strip().split()[1:]]:
                            output.snapshot_file.format = ' '.join(
                                line1.strip().split()[1:3]).lower()
                            output.snapshot_file.times_per_file = \
                            line1.strip().split()[4]
                    elif key1 == 'no_print_initial':
                        output.snapshot_file.print_initial = False
                    elif key1 == 'no_print_final':
                        output.snapshot_file.print_final = False
                    elif key1 == 'variables':
                        keep_reading2 = True
                        while keep_reading2:
                            line2 = get_next_line(infile)
                            key2 = line2.strip().split()[0].lower()
                            if key2 in ['/', 'end']:
                                keep_reading2 = False
                            elif key2 in output_variables_allowed:
                                output.snapshot_file.variables_list.append(key2)
                    elif key1 == 'times':
                        unit = line1.strip().split()[1].lower()
                        if unit in time_units_allowed:
                            output.snapshot_file.time_unit = unit
                        else:
                            raise PyFLOTRAN_ERROR(
                                'Unknown time unit for snapshot times!')
                        for val in line1.strip().split()[2:]:
                            output.snapshot_file.time_list.append(floatD(val))
                    elif key1 == 'periodic':
                        tstring = line1.strip().split()[
                            1].lower()  # Read the 2nd word
                        if tstring == 'time':
                            # 2nd from last word.
                            output.snapshot_file.periodic_time = floatD(
                                line1.split()[-2])
                            output.snapshot_file.periodic_time_unit = \
                              self.splitter(line1)  # last word
                        elif tstring == 'timestep':
                            # 2nd from last word.
                            output.snapshot_file.periodic_timestep = \
                              int(self.splitter(line1))
                    elif key1 == 'periodic_observation':
                        tstring = line1.strip().split()[
                            1].lower()  # Read the 2nd word
                        if tstring == 'time':
                            # 2nd from last word.
                            output.snapshot_file.periodic_observation_time = \
                              floatD(line1.split()[-2])
                            output.snapshot_file.periodic_observation_time_unit = \
                              self.splitter(line1)  # last word
                        elif tstring == 'timestep':
                            output.snapshot_file.periodic_observation_timestep = \
                              int(self.splitter(line1))
                    elif key1 == 'extend_hdf5_time_format':
                        output.snapshot_file.extend_hdf5_time_format
                    elif key1 in ['/', 'end']:
                        keep_reading1 = False
            elif key == 'observation_file':
                keep_reading1 = True
                while keep_reading1:
                    line1 = get_next_line(infile)
                    key1 = line1.strip().split()[0].lower()
                    if key1 == 'format':
                        if len(line1.strip().split()) == 2:
                            output.observation_file.format = \
                              line1.strip().split()[1].lower()
                        elif len(line1.strip().split()) == 3:
                            output.observation_file.format = \
                              line1.strip().split()[1:2].lower()
                        elif len(line1.strip().split()) > 3 and \
                        'times_per_file' in line1.strip().split()[1:]:
                            output.observation_file.format = \
                              line1.strip().split()[1:2].lower()
                            output.observation_file.times_per_file = \
                              line1.strip().split()[4]
                    elif key1 == 'no_print_initial':
                        output.observation_file.print_initial = False
                    elif key1 == 'no_print_final':
                        output.observation_file.print_final = False
                    elif key1 == 'variables':
                        keep_reading2 = True
                        while keep_reading2:
                            line2 = get_next_line(infile)
                            key2 = line2.strip().split()[0].lower()
                            if key2 in ['/', 'end']:
                                keep_reading2 = False
                            elif key2 in output_variables_allowed:
                                output.observation_file.variables_list.append(
                                    key2)
                    elif key1 == 'times':
                        unit = line1.strip().split()[1].lower()
                        if unit in time_units_allowed:
                            output.observation_file.time_unit = unit
                        else:
                            raise PyFLOTRAN_ERROR(
                                'Unknown time unit for observation times!')
                        for val in line1.strip().split()[2:]:
                            output.observation_file.time_list.append(
                                floatD(val))
                    elif key1 == 'periodic':
                        tstring = line1.strip().split()[
                            1].lower()  # Read the 2nd word
                        if tstring == 'time':
                            # 2nd from last word.
                            output.observation_file.periodic_time = floatD(
                                line1.split()[-2])
                            output.observation_file.periodic_time_unit = \
                              self.splitter(line1)  # last word
                        elif tstring == 'timestep':
                            # 2nd from last word.
                            output.observation_file.periodic_timestep = int(
                                self.splitter(line1))
                    elif key1 == 'periodic_observation':
                        tstring = line1.strip().split()[
                            1].lower()  # Read the 2nd word
                        if tstring == 'time':
                            # 2nd from last word.
                            output.observation_file.periodic_observation_time = \
                              floatD(line1.split()[-2])
                            output.observation_file.periodic_observation_time_unit = self.splitter(
                                line1)  # last word
                        elif tstring == 'timestep':
                            output.observation_file.periodic_observation_timestep = int(
                                self.splitter(line1))
                    elif key1 == 'extend_hdf5_time_format':
                        output.observation_file.extend_hdf5_time_format
                    elif key1 in ['/', 'end']:
                        keep_reading1 = False
            elif key == 'mass_balance_file':
                keep_reading1 = True
                while keep_reading1:
                    line1 = get_next_line(infile)
                    key1 = line1.strip().split()[0].lower()
                    if key1 == 'format':
                        if len(line1.strip().split()) == 2:
                            output.mass_balance_file.format = line1.strip().split()[
                                1].lower()
                        elif len(line1.strip().split()) == 3:
                            output.mass_balance_file.format = line1.strip().split()[
                                1:2].lower()
                        elif len(line1.strip().split()) > 3 and 'times_per_file' in line1.strip().split()[1:]:
                            output.mass_balance_file.format = line1.strip().split()[
                                1:2].lower()
                            output.mass_balance_file.times_per_file = line1.strip().split()[
                                4]
                    elif key1 == 'no_print_initial':
                        output.mass_balance_file.print_initial = False
                    elif key1 == 'no_print_final':
                        output.mass_balance_file.print_final = False
                    elif key1 == 'variables':
                        keep_reading2 = True
                        while keep_reading2:
                            line2 = get_next_line(infile)
                            key2 = line2.strip().split()[0].lower()
                            if key2 in ['/', 'end']:
                                keep_reading2 = False
                            elif key2 in output_variables_allowed:
                                output.mass_balance_file.variables_list.append(
                                    key2)
                    elif key1 == 'times':
                        unit = line1.strip().split()[1].lower()
                        if unit in time_units_allowed:
                            output.mass_balance_file.time_unit = unit
                        else:
                            raise PyFLOTRAN_ERROR(
                                'Unknown time unit for snapshot times!')
                        for val in line1.strip().split()[2:]:
                            output.mass_balance_file.time_list.append(
                                floatD(val))
                    elif key1 == 'periodic':
                        tstring = line1.strip().split()[
                            1].lower()  # Read the 2nd word
                        if tstring == 'time':
                            # 2nd from last word.
                            output.mass_balance_file.periodic_time = floatD(
                                line1.split()[-2])
                            output.mass_balance_file.periodic_time_unit = self.splitter(
                                line1)  # last word
                        elif tstring == 'timestep':
                            # 2nd from last word.
                            output.mass_balance_file.periodic_timestep = int(
                                self.splitter(line1))
                    elif key1 == 'periodic_observation':
                        tstring = line1.strip().split()[
                            1].lower()  # Read the 2nd word
                        if tstring == 'time':
                            # 2nd from last word.
                            output.mass_balance_file.periodic_observation_time = floatD(
                                line1.split()[-2])
                            output.mass_balance_file.periodic_observation_time_unit = self.splitter(
                                line1)  # last word
                        elif tstring == 'timestep':
                            output.mass_balance_file.periodic_observation_timestep = int(
                                self.splitter(line1))
                    elif key1 == 'total_mass_regions':
                        keep_reading2 = True
                        while keep_reading2:
                            line2 = get_next_line(infile)
                            key2 = line2.strip().split()[0].lower()
                            if key2 in ['/', 'end']:
                                keep_reading2 = False
                            else:
                                output.mass_balance_file.total_mass_regions.append(
                                    key2)
                    elif key1 == 'extend_hdf5_time_format':
                        output.mass_balance_file.extend_hdf5_time_format
                    elif key1 in ['/', 'end']:
                        keep_reading1 = False
            elif key in ['/', 'end']:
                keep_reading = False

        self.output = output

    def _write_output(self, outfile):
        self._header(outfile, headers['output'])
        output = self.output

        # Write Output - if used so null/None entries are not written
        outfile.write('OUTPUT\n')

        if output.time_list:

            for unit in output.time_list:
                if unit[0].lower() in time_units_allowed:
                    prefix = '  TIMES ' + unit[0] + ' '
                    coeffs = ' '.join([strD(ele) for ele in unit[1:]])

                    outfile.write(prefix + coeffs + '\n')
                else:
                    print('       valid time.units', time_units_allowed, '\n')
                    raise PyFLOTRAN_ERROR(
                        'output.time_list[0]: ' + output.time_list[0] +
                        ' is invalid.')
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
                raise PyFLOTRAN_ERROR('output.screen_output:' +
                                      str(output.screen_output) +
                                      ' is not bool.')
        if output.time_units:
            if output.time_units in time_units_allowed:
                outfile.write('  TIME_UNITS ' + output.time_units + '\n')
            else:
                raise PyFLOTRAN_ERROR(
                    output.time_units + ' invalid time unit\n')
        if output.screen_periodic:
            try:  # Error checking to ensure screen_periodic is int (integer).
                output.screen_periodic = int(output.screen_periodic)
                outfile.write('  ' + 'SCREEN PERIODIC ' +
                              str(output.screen_periodic) + '\n')
            except ValueError:
                raise PyFLOTRAN_ERROR(
                    'output.screen_periodic: \'' +
                    str(output.screen_periodic) + '\' is not int (integer).')
        if output.periodic_time:
            try:  # Error checking to ensure periodic_time is [float, str].
                output.periodic_time[0] = floatD(output.periodic_time[0])
                if output.periodic_time[1].lower() in time_units_allowed:
                    output.periodic_time[1] = str(
                        output.periodic_time[1].lower())
                else:
                    output.periodic_time[1] = str(
                        output.periodic_time[1].lower())
                    raise PyFLOTRAN_ERROR('time unit in ' +
                                          'output.periodic_time[1]' +
                                          ' is invalid. Valid time' +
                                          'units are:',
                                          time_units_allowed)
                outfile.write('  ' + 'PERIODIC TIME ')
                outfile.write(strD(output.periodic_time[0]) + ' ')
                outfile.write(output.periodic_time[1] + '\n')
            except:
                raise PyFLOTRAN_ERROR('output.periodic_time: \'' +
                                      str(output.periodic_time) +
                                      '\' is not [float, str].')
        if output.periodic_timestep:
            try:  # Error checking to ensure periodic_timestep is [float].
                output.periodic_timestep = int(output.periodic_timestep)
                outfile.write('  ' + 'PERIODIC TIMESTEP ')
                outfile.write(strD(output.periodic_timestep) + '\n')
            except:
                raise PyFLOTRAN_ERROR('output.periodic_timestep: \'' +
                                      str(output.periodic_timestep) +
                                      '\' is not [float].')
        if output.periodic_observation_time:
            try:
                # Error checking to ensure periodic_observation_time is [float,
                # str].
                output.periodic_observation_time[0] = floatD(
                    output.periodic_observation_time[0])
                if output.periodic_observation_time[1].lower() in \
                        time_units_allowed:
                    output.periodic_observation_time[1] = str(
                        output.periodic_observation_time[1].lower())
                else:
                    output.periodic_observation_time[1] = str(
                        output.periodic_observation_time[1].lower())
                    raise PyFLOTRAN_ERROR('time unit in ' +
                                          'output. ' +
                                          'periodic_observation_time[1] ' +
                                          ' is invalid. Valid time '
                                          'units are:', time_units_allowed)

                # Writing out results
                outfile.write('  ' + 'PERIODIC_OBSERVATION TIME ')
                outfile.write(strD(output.periodic_observation_time[0]) + ' ')
                outfile.write(output.periodic_observation_time[1] + '\n')
            except:
                raise PyFLOTRAN_ERROR(
                    'output.periodic_observation_time: \'' +
                    str(output.periodic_observation_time) +
                    '\' is not [float, str].')
        if output.periodic_observation_timestep:
            outfile.write('  PERIODIC_OBSERVATION TIMESTEP ' +
                          str(output.periodic_observation_timestep) + '\n')
        if output.print_column_ids:
            outfile.write('  ' + 'PRINT_COLUMN_IDS' + '\n')
        for out_format in output.format_list:
            if out_format.upper() in output_formats_allowed:
                outfile.write('  FORMAT ')
                outfile.write(out_format.upper() + '\n')
            else:
                print('       valid output.format:', \
                    output_formats_allowed, '\n')
                raise PyFLOTRAN_ERROR(
                    'output.format: \'' + out_format + '\' is invalid.')
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
                    print('       valid output.variable:', \
                        output_variables_allowed, '\n')
                    raise PyFLOTRAN_ERROR(
                        'output.variable: \'' + variable + '\' is invalid.')
            outfile.write('  /\n')
        if output.snapshot_file:
            outfile.write('  SNAPSHOT_FILE\n')
            if output.snapshot_file.format is not None:
                outfile.write('    FORMAT ')
                for form in output.snapshot_file.format:
                    outfile.write(form.upper())
                if output.snapshot_file.times_per_file is not None:
                    outfile.write(' TIMES_PER_FILE ')
                    outfile.write(str(output.snapshot_file.times_per_file))
                outfile.write('\n')
            if output.snapshot_file.print_final is False:
                outfile.write('    NO_PRINT_FINAL\n')
            if output.snapshot_file.print_initial is False:
                outfile.write('    NO_PRINT_INTIAL\n')
            if output.snapshot_file.variables_list:
                outfile.write('    VARIABLES\n')
                for variable in output.snapshot_file.variables_list:
                    if variable.lower() in output_variables_allowed:
                        outfile.write('      ' + variable.upper() + '\n')
                    else:
                        print('       valid output.variable:', \
                            output_variables_allowed, '\n')
                        raise PyFLOTRAN_ERROR(
                            'output.variable: \'' + variable + '\' is invalid.')
                outfile.write('    /\n')
            if output.snapshot_file.total_mass_regions:
                raise PyFLOTRAN_ERROR(
                    'SNAPSHOT_FILE cannot have TOTAL_MASS_REGIONS')
            if output.snapshot_file.time_list:
                outfile.write('    TIMES ')
                if output.snapshot_file.time_unit is not None:
                    outfile.write(output.snapshot_file.time_unit + ' ')
                for time in output.snapshot_file.time_list:
                    outfile.write(strD(time))
                    outfile.write(' ')
                outfile.write('\n')
            if output.snapshot_file.periodic_timestep is not None:
                outfile.write('    PERIODIC TIMESTEP ')
                outfile.write(str(output.snapshot_file.periodic_timestep))
                outfile.write('\n')
            if output.snapshot_file.periodic_time is not None:
                outfile.write('    PERIODIC TIME ')
                outfile.write(strD(output.snapshot_file.periodic_time))
                if output.snapshot_file.periodic_time_unit is not None:
                    outfile.write(
                        ' ' + output.snapshot_file.periodic_time_unit)
                outfile.write('\n')
            if output.snapshot_file.periodic_observation_timestep is not None:
                outfile.write('    PERIODIC_OBSERVATION TIMESTEP ')
                outfile.write(
                    str(output.snapshot_file.periodic_observation_timestep))
                outfile.write('\n')
            if output.snapshot_file.periodic_observation_time is not None:
                outfile.write('    PERIODIC_OBSERVATION TIME ')
                outfile.write(
                    strD(output.snapshot_file.periodic_observation_time))
                if output.snapshot_file.periodic_observation_time_unit is not None:
                    outfile.write(
                        ' ' + output.snapshot_file.periodic_observation_time_unit)
                outfile.write('\n')
            if output.snapshot_file.extend_hdf5_time_format:
                outfile.write('    EXTEND_HDF5_TIME_FORMAT\n')
            outfile.write('  /\n')
        if output.observation_file:
            outfile.write('  OBSERVATION_FILE\n')
            if output.observation_file.format is not None:
                raise PyFLOTRAN_ERROR(
                    'FORMAT cannot be specified with OBSERVATION_FILE')
            if output.observation_file.print_final is False:
                outfile.write('    NO_PRINT_FINAL\n')
            if output.observation_file.print_initial is False:
                outfile.write('    NO_PRINT_INTIAL\n')
            if output.observation_file.variables_list:
                outfile.write('    VARIABLES\n')
                for variable in output.observation_file.variables_list:
                    if variable.lower() in output_variables_allowed:
                        outfile.write('      ' + variable.upper() + '\n')
                    else:
                        print('       valid output.variable:', \
                            output_variables_allowed, '\n')
                        raise PyFLOTRAN_ERROR(
                            'output.variable: \'' + variable + '\' is invalid.')
                outfile.write('    /\n')
            if output.observation_file.total_mass_regions:
                raise PyFLOTRAN_ERROR(
                    'OBSERVATION_FILE cannot have TOTAL_MASS_REGIONS')
            if output.observation_file.time_list:
                outfile.write('    TIMES ')
                if output.observation_file.time_unit is not None:
                    outfile.write(output.observation_file.time_unit + ' ')
                for time in output.observation_file.time_list:
                    outfile.write(strD(time))
                    outfile.write(' ')
                outfile.write('\n')
            if output.observation_file.periodic_timestep is not None:
                outfile.write('    PERIODIC TIMESTEP ')
                outfile.write(str(output.observation_file.periodic_timestep))
                outfile.write('\n')
            if output.observation_file.periodic_time is not None:
                outfile.write('    PERIODIC TIME ')
                outfile.write(strD(output.observation_file.periodic_time))
                if output.observation_file.periodic_time_unit is not None:
                    outfile.write(
                        ' ' + output.observation_file.periodic_time_unit)
                outfile.write('\n')
            if output.observation_file.periodic_observation_timestep is not None:
                outfile.write('    PERIODIC_OBSERVATION TIMESTEP ')
                outfile.write(
                    str(output.observation_file.periodic_observation_timestep))
                outfile.write('\n')
            if output.observation_file.periodic_observation_time is not None:
                outfile.write('    PERIODIC_OBSERVATION TIME ')
                outfile.write(
                    strD(output.observation_file.periodic_observation_time))
                if output.observation_file.periodic_observation_time_unit is not None:
                    outfile.write(
                        ' ' + output.observation_file.periodic_observation_time_unit)
                outfile.write('\n')
            if output.observation_file.extend_hdf5_time_format:
                outfile.write('    EXTEND_HDF5_TIME_FORMAT\n')
            outfile.write('  /\n')
        if output.mass_balance_file:
            outfile.write('  MASS_BALANCE_FILE\n')
            if output.mass_balance_file.format is not None:
                raise PyFLOTRAN_ERROR(
                    'FORMAT cannot be specified with MASS_BALANCE_FILE')
            if output.mass_balance_file.print_final is False:
                outfile.write('    NO_PRINT_FINAL\n')
            if output.mass_balance_file.print_initial is False:
                outfile.write('    NO_PRINT_INTIAL\n')
            if output.mass_balance_file.variables_list:
                raise PyFLOTRAN_ERROR(
                    'VARIABLES cannot be used with MASS_BALANCE_FILE')
            if output.mass_balance_file.total_mass_regions:
                outfile.write('    TOTAL_MASS_REGIONS\n')
                for region in output.mass_balance_file.total_mass_regions:
                    outfile.write('      ' + region)
                    outfile.write('\n')
                outfile.write('    /\n')
            if output.mass_balance_file.time_list:
                outfile.write('    TIMES ')
                if output.mass_balance_file.time_unit is not None:
                    outfile.write(output.mass_balance_file.time_unit + ' ')
                for time in output.mass_balance_file.time_list:
                    outfile.write(strD(time))
                    outfile.write(' ')
                outfile.write('\n')
            if output.mass_balance_file.periodic_timestep is not None:
                outfile.write('    PERIODIC TIMESTEP ')
                outfile.write(str(output.mass_balance_file.periodic_timestep))
                outfile.write('\n')
            if output.mass_balance_file.periodic_time is not None:
                outfile.write('    PERIODIC TIME ')
                outfile.write(strD(output.mass_balance_file.periodic_time))
                if output.mass_balance_file.periodic_time_unit is not None:
                    outfile.write(
                        ' ' + output.mass_balance_file.periodic_time_unit)
                outfile.write('\n')
            if output.mass_balance_file.periodic_observation_timestep is not None:
                outfile.write('    PERIODIC_OBSERVATION TIMESTEP ')
                outfile.write(
                    str(output.mass_balance_file.periodic_observation_timestep))
                outfile.write('\n')
            if output.mass_balance_file.periodic_observation_time is not None:
                outfile.write('    PERIODIC_OBSERVATION TIME ')
                outfile.write(
                    strD(output.mass_balance_file.periodic_observation_time))
                if output.mass_balance_file.periodic_observation_time_unit is not None:
                    outfile.write(
                        ' ' + output.mass_balance_file.periodic_observation_time_unit)
                outfile.write('\n')
            if output.mass_balance_file.extend_hdf5_time_format:
                outfile.write('    EXTEND_HDF5_TIME_FORMAT\n')
            outfile.write('  /\n')
        outfile.write('END\n\n')

    def _read_fluid(self, infile):
        p = pfluid()

        keep_reading = True

        while keep_reading:  # Read through all cards
            line = get_next_line(infile)  # get next line
            key = line.strip().split()[0].lower()  # take first

            if key == 'diffusion_coefficient':
                p.diffusion_coefficient = floatD(
                    self.splitter(line))  # Read last entry
            if key == 'phase':
                p.phase = self.splitter(line)
            elif key in ['/', 'end']:
                keep_reading = False

        # Create new employ fluid properties object and assign read in values
        # to it
        self.add(p)

    def _add_fluid(self, fluid=pfluid(), overwrite=False):
        # check if fluid already exists
        if isinstance(fluid, pfluid):
            if fluid.phase in self.fluid.keys():
                if not overwrite:
                    warning = 'WARNING: Fluid property phase ' + \
                              str(fluid.phase) + '\' already exists. ' + \
                              'fluid will not be defined, ' + \
                              'use overwrite = True in add()' + \
                              ' to overwrite the old fluid.'
                    print(warning,
                    build_warnings.append(warning))
                    return
                else:
                    self.delete(self.fluid[fluid.phase])

        if fluid not in self.fluidlist:
            self.fluidlist.append(fluid)

    def _delete_fluid(self, fluid=pfluid()):
        self.pfluidlist.remove(fluid)

    def _write_fluid(self, outfile):
        self._header(outfile, headers['fluid_property'])
        for fluid in self.fluidlist:
            outfile.write('FLUID_PROPERTY\n')

            # Write out requested (not null) fluid properties
            if fluid.phase:
                outfile.write('  PHASE ' +
                              str(fluid.phase) +
                              '\n')  # Read last entry
            if fluid.diffusion_coefficient is not None:
                outfile.write('  DIFFUSION_COEFFICIENT ' +
                              strD(fluid.diffusion_coefficient) +
                              '\n')  # Read last entry
            outfile.write('END\n\n')

    def _read_saturation(self, infile, line):

        saturation = psaturation()  # assign defaults before reading in values
        # saturation function name, passed in.
        saturation.name = self.splitter(line).lower()

        keep_reading = True

        while keep_reading:  # Read through all cards
            line = get_next_line(infile)
            key = line.strip().split()[0].lower()  # take first  key word

            if key == 'permeability_function_type':
                saturation.permeability_function_type = self.splitter(line)
            elif key == 'saturation_function_type':
                saturation.saturation_function_type = self.splitter(line)
            elif key == 'residual_saturation_liquid':
                saturation.residual_saturation_liquid = floatD(
                    self.splitter(line))
            elif key == 'residual_saturation_gas':
                saturation.residual_saturation_gas = floatD(
                    self.splitter(line))
            elif key == 'residual_saturation':  # Alternative to check
                tstring = line.strip().split()[1].lower()  # take 2nd key word
                if tstring == 'liquid_phase':
                    saturation.residual_saturation_liquid = floatD(
                        self.splitter(line))
                elif tstring == 'gas_phase':
                    saturation.residual_saturation_gas = floatD(
                        self.splitter(line))
                else:  # if no 2nd word exists
                    saturation.residual_saturation = floatD(
                        self.splitter(line))
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
    # Adds a integral flux object

    def _add_integral_flux(self, integral_flux=pintegral_flux(),
                           overwrite=False):
        # check if integral_flux already exists
        if isinstance(integral_flux, pintegral_flux):
            if integral_flux.name in self.integral_flux.keys():
                if not overwrite:
                    warning = 'WARNING: A integra_flux with name \'' + \
                              str(
                                  integral_flux.name) + '\' already exists.' \
                        ' Integral flux will' + \
                              'not be defined, use overwrite = ' + \
                              'True in add() to overwrite the old region.'
                    print(warning,
                    build_warnings.append(warning))
                    return
                else:
                    self.delete(self.integral_flux[integral_flux.name])

        if integral_flux not in self.integral_flux_list:
            self.integral_flux_list.append(integral_flux)

    def _write_integral_flux(self, outfile):
        self._header(outfile, headers['integral_flux'])

        # Write out all valid region object entries with Region as Key word
        for integral_flux in self.integral_flux_list:
            outfile.write('INTEGRAL_FLUX\n')
            if integral_flux.name:
                outfile.write('  NAME ' + integral_flux.name + '\n')
            else:
                outfile.write('  NAME default_integral_flux\n')
            # if integral_flux.invert_direction:
            #     outfile.write('  INVERT_DIRECITON\n')

            outfile.write('  COORDINATES\n')
            outfile.write('    ')
            for i in range(3):
                outfile.write(strD(integral_flux.coordinates_lower[i]) +
                              ' ')
            outfile.write('\n    ')
            for i in range(3):
                outfile.write(strD(integral_flux.coordinates_upper[i]) +
                              ' ')
            outfile.write('\n')
            outfile.write('  /\n')
            outfile.write('END\n\n')

    def _delete_integral_flux(self, integral_flux=pintegral_flux()):
        self.integral_flux_list.remove(integral_flux)

    # Adds a saturation object.
    def _add_saturation(self, sat=psaturation(), overwrite=False):
        # check if saturation already exists
        if isinstance(sat, psaturation):
            if sat.name in self.saturation.keys():
                if not overwrite:
                    warning = 'WARNING: A saturation function with name \'' + \
                              str(sat.name) + '\' already exists. ' + \
                              ' Use overwrite = True in add() ' + \
                              ' to overwrite the old saturation function.'
                    print(warning,
                    build_warnings.append(warning))
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
                if sat.permeability_function_type in \
                        permeability_function_types_allowed:
                    outfile.write('  PERMEABILITY_FUNCTION_TYPE ' +
                                  sat.permeability_function_type + '\n')
                else:
                    print('valid saturation.permeability_function_types',
                          saturation_function_types_allowed, '\n')
                    raise PyFLOTRAN_ERROR(
                        'saturation.saturation_function_type: \'' +
                        sat.saturation_function_type + '\' is invalid.')
            if sat.saturation_function_type:
                if sat.saturation_function_type in \
                        saturation_function_types_allowed:
                    outfile.write('  SATURATION_FUNCTION_TYPE ' +
                                  sat.saturation_function_type + '\n')
            if sat.residual_saturation or sat.residual_saturation == 0:
                outfile.write('  RESIDUAL_SATURATION ' +
                              strD(sat.residual_saturation) + '\n')
            if sat.residual_saturation_liquid or \
                    sat.residual_saturation_liquid == 0:
                outfile.write('  RESIDUAL_SATURATION LIQUID_PHASE ' +
                              strD(sat.residual_saturation_liquid) + '\n')
            if sat.residual_saturation_gas or \
                    sat.residual_saturation_gas == 0:
                outfile.write('  RESIDUAL_SATURATION GAS_PHASE ' +
                              strD(sat.residual_saturation_gas) + '\n')
            if sat.a_lambda:
                outfile.write('  LAMBDA ' + strD(sat.a_lambda) + '\n')
            if sat.alpha:
                outfile.write('  ALPHA ' + strD(sat.alpha) + '\n')
            if sat.max_capillary_pressure:
                outfile.write('  MAX_CAPILLARY_PRESSURE ' +
                              strD(sat.max_capillary_pressure) + '\n')
            if sat.betac:
                outfile.write('  BETAC ' + strD(sat.betac) + '\n')
            if sat.power:
                outfile.write('  POWER ' + strD(sat.power) + '\n')
            outfile.write('END\n\n')

    def _read_characteristic_curves(self, infile, line):

        # assign defaults before reading in values
        characteristic_curves = pcharacteristic_curves()
        # Characteristic curve name, passed in.
        characteristic_curves.name = self.splitter(line).lower()
        keep_reading = True
        while keep_reading:  # Read through all cards
            line = get_next_line(infile)
            if len(line.strip()) == 0:
                continue
            elif list(line)[0] in ['!', '#']:
                continue
            key = line.strip().split()[0].lower()  # take first  key word
            word = line.strip().split()[-1].lower()
            if key == 'default':
                characteristic_curves.default = True
            if key == 'saturation_function':
                characteristic_curves.saturation_function_type = self.splitter(
                    line)
                keep_reading1 = True
                while keep_reading1:
                    line = get_next_line(infile)

                    if len(line.strip()) == 0:
                        continue
                    elif list(line)[0] in ['!', '#']:
                        continue
                    key1 = line.strip().split()[0].lower()
                    if key1 == 'alpha':
                        characteristic_curves.sf_alpha = \
                            floatD(self.splitter(line))
                    elif key1 == 'm':
                        characteristic_curves.sf_m = \
                            floatD(self.splitter(line))
                    elif key1 == 'lambda':
                        characteristic_curves.sf_lambda = \
                            floatD(self.splitter(line))
                    elif key1 == 'liquid_residual_saturation':
                        characteristic_curves.sf_liquid_residual_saturation =\
                            floatD(self.splitter(line))
                    elif key1 == 'gas_residual_saturation':
                        characteristic_curves.sf_gas_residual_saturation = \
                            floatD(self.splitter(line))
                    elif key1 == 'max_capillary_pressure':
                        characteristic_curves.max_capillary_pressure = \
                            floatD(self.splitter(line))
                    elif key1 == 'smooth':
                        characteristic_curves.smooth = True
                    elif key1 == 'power':
                        characteristic_curves.power = \
                            floatD(self.splitter(line))
                    elif key1 in ['/', 'end']:
                        keep_reading1 = False
            elif key == 'permeability_function' and 'gas' in word:
                characteristic_curves.gas_permeability_function_type = word
                keep_reading1 = True
                while keep_reading1:
                    line = get_next_line(infile)
                    key1 = line.strip().split()[0].lower()
                    if key1 == 'phase':
                        characteristic_curves.phase = \
                            self.splitter(line).lower()
                    elif key1 == 'm':
                        characteristic_curves.gpf_m = \
                            floatD(self.splitter(line))
                    elif key1 == 'liquid_residual_saturation':
                        characteristic_curves.gpf_liquid_residual_saturation = \
                            floatD(self.splitter(line))
                    elif key1 == 'gas_residual_saturation':
                        characteristic_curves.gpf_gas_residual_saturation = \
                            floatD(self.splitter(line))
                    elif key1 == 'lambda':
                        characteristic_curves.gpf_lambda = \
                            floatD(self.splitter(line))
                    elif key1 in ['/', 'end']:
                        keep_reading1 = False
            elif key == 'permeability_function' and 'gas' not in word:
                characteristic_curves.liquid_permeability_function_type = word
                keep_reading1 = True
                while keep_reading1:
                    line = get_next_line(infile)
                    key1 = line.strip().split()[0].lower()
                    if key1 == 'phase':
                        characteristic_curves.phase = \
                            self.splitter(line).lower()
                    elif key1 == 'm':
                        characteristic_curves.lpf_m = \
                            floatD(self.splitter(line))
                    elif key1 == 'liquid_residual_saturation':
                        characteristic_curves.lpf_liquid_residual_saturation = \
                            floatD(self.splitter(line))
                    elif key1 == 'gas_residual_saturation':
                        characteristic_curves.lpf_gas_residual_saturation = \
                            floatD(self.splitter(line))
                    elif key1 == 'lambda':
                        characteristic_curves.lpf_lambda = \
                            floatD(self.splitter(line))
                    elif key1 in ['/', 'end']:
                        keep_reading1 = False
            elif key in ['/', 'end']:
                keep_reading = False
        self.add(characteristic_curves)

    # Adds a char object.
    def _add_characteristic_curves(self, char=pcharacteristic_curves(),
                                   overwrite=False):
        # check if char already exists
        if isinstance(char, pcharacteristic_curves):
            if char.name in self.char.keys():
                if not overwrite:
                    warning = 'WARNING: A Characteristic Curve with name \'' + \
                              str(char.name) + '\' already exists.' + \
                              'Characteristic curve will not be defined, ' + \
                              ' use overwrite = True ' 'in add() to' + \
                              'overwrite the old characteristic curve.'
                    print(warning,
                    build_warnings.append(warning))
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
            if char.default:
                # This just prints the DEFAULT flag
                outfile.write('  DEFAULT ' + '\n')
            else:
                if char.saturation_function_type:
                    if char.saturation_function_type in \
                            characteristic_curves_saturation_function_types_allowed:
                        outfile.write('  SATURATION_FUNCTION ' +
                                      char.saturation_function_type.upper() +
                                      '\n')
                    else:
                        print('       valid  char.saturation_function_types', \
                            characteristic_curves_saturation_function_types_allowed, '\n')
                        raise PyFLOTRAN_ERROR(
                            'char.saturation_function_type: \'' +
                            char.saturation_function_type + '\' is invalid.')
                    if char.sf_alpha:
                        outfile.write('   ALPHA ' + strD(char.sf_alpha) +
                                      '\n')
                    if char.sf_m:
                        outfile.write('   M ' + strD(char.sf_m) + '\n')
                    if char.sf_lambda:
                        outfile.write('   LAMBDA ' + strD(char.sf_lambda) +
                                      '\n')
                    if char.sf_liquid_residual_saturation or \
                            char.sf_liquid_residual_saturation == 0:
                        outfile.write('   LIQUID_RESIDUAL_SATURATION ' +
                                      strD(char.sf_liquid_residual_saturation) +
                                      '\n')
                    if char.sf_gas_residual_saturation or \
                            char.sf_gas_residual_saturation == 0:
                        outfile.write('   GAS_RESIDUAL_SATURATION ' +
                                      strD(char.sf_gas_residual_saturation) +
                                      '\n')
                    if char.max_capillary_pressure:
                        outfile.write('   MAX_CAPILLARY_PRESSURE ' +
                                      strD(char.max_capillary_pressure) +
                                      '\n')
                    if char.smooth:
                        # This just prints the SMOOTH flag
                        outfile.write('   SMOOTH ' + '\n')
                    outfile.write('  / ' + '\n')

                if char.power:
                    outfile.write('  POWER ' + strD(char.power) + '\n')

                if char.liquid_permeability_function_type:
                    if char.liquid_permeability_function_type in \
                            characteristic_curves_liquid_permeability_function_types_allowed:
                        outfile.write('  PERMEABILITY_FUNCTION ' +
                                      char.liquid_permeability_function_type.upper() +
                                      '\n')
                    else:
                        print('       valid  char.liquid_permeability_function_types', \
                            characteristic_curves_liquid_permeability_function_types_allowed, '\n')
                        raise PyFLOTRAN_ERROR(
                            'char.liquid_permeability_function_type: \'' +
                            char.liquid_permeability_function_type + '\' is invalid.')
                    if char.phase:
                        outfile.write('   PHASE LIQUID' + '\n')

                    if char.lpf_m:
                        outfile.write('   M ' + strD(char.lpf_m) + '\n')
                    if char.lpf_lambda:
                        outfile.write(
                            '   LAMBDA ' + strD(char.lpf_lambda) + '\n')
                    if char.lpf_liquid_residual_saturation or \
                            char.lpf_liquid_residual_saturation == 0:
                        outfile.write('   LIQUID_RESIDUAL_SATURATION ' +
                                      strD(
                                          char.lpf_liquid_residual_saturation) +
                                      '\n')
                    outfile.write('  / ' + '\n')

                if char.gas_permeability_function_type:

                    if char.gas_permeability_function_type in \
                            characteristic_curves_gas_permeability_function_types_allowed:
                        outfile.write('  PERMEABILITY_FUNCTION ' +
                                      char.gas_permeability_function_type.upper() + '\n')

                    if char.phase:
                        outfile.write('   PHASE GAS' + '\n')
                    elif char.phase is None and 'gas' in char.gas_permeability_function_type:
                        pass
                    else:

                        print('       valid  char.gas_permeability_function_types', \
                            characteristic_curves_gas_permeability_function_types_allowed, '\n')
                        raise PyFLOTRAN_ERROR(
                            'char.gas_permeability_function_type: \'' +
                            char.gas_permeability_function_type +
                            '\' is invalid.')

                    if char.gpf_m:
                        outfile.write('   M ' + strD(char.gpf_m) + '\n')
                    if char.gpf_lambda:
                        outfile.write(
                            '   LAMBDA ' + strD(char.gpf_lambda) + '\n')
                    if char.gpf_liquid_residual_saturation or \
                            char.gpf_liquid_residual_saturation == 0:
                        outfile.write('   LIQUID_RESIDUAL_SATURATION ' +
                                      strD(
                                          char.gpf_liquid_residual_saturation) +
                                      '\n')
                    if char.gpf_gas_residual_saturation or \
                            char.gpf_gas_residual_saturation == 0:
                        outfile.write('   GAS_RESIDUAL_SATURATION ' +
                                      strD(char.gpf_gas_residual_saturation) +
                                      '\n')
                    outfile.write('  / ' + '\n')

            outfile.write('END\n\n')

    def _read_waste_form_general(self,infile):

        wfg = pwaste_form_general()

        while True:
            line = get_next_line(infile)
            key = line.strip().split()[0].lower()

            if key in ['/','end','end_waste_form_general']:
                break
            elif key == 'print_mass_balance':
                wfg.print_mass_balance = True
            elif key == 'implicit_solution':
                wfg.implicit_solution = True
            elif key == 'mechanism':
                mech = wfg.add_mechanism(mech_type=self.splitter(line))
                while True:
                    subline = get_next_line(infile)
                    subkey = subline.strip().split()[0].lower()

                    if subkey in ['/','end']:
                        break
                    elif subkey == 'name':
                        mech.name = self.splitter(subline)
                    elif subkey == 'specific_surface_area':
                        _val = floatD(subline.strip().split()[1])
                        _unit = subline.strip().split()[2]
                        mech.specific_surface_area = Coeff(_val,unit=_unit)
                    elif subkey == 'matrix_density':
                        _val = floatD(subline.strip().split()[1])
                        _unit = subline.strip().split()[2]
                        mech.matrix_density = Coeff(_val,unit=_unit)
                    elif subkey == 'kienzler_dissolution':
                        mech.kienzler_dissolution = True
                    elif subkey == 'canister_degradation_model':
                        while True:
                            _line = get_next_line(infile)
                            _key = _line.strip().split()[0].lower()

                            if _key in ['/','end']:
                                break
                            elif _key == 'vitality_log10_mean':
                                mech.vitality_log10_mean = \
                                            floatD(self.splitter(_line))
                            elif _key == 'vitality_log10_stdev':
                                mech.vitality_log10_stdev = \
                                            floatD(self.splitter(_line))
                            elif _key == 'vitality_upper_truncation':
                                mech.vitality_upper_truncation = \
                                            floatD(self.splitter(_line))
                            elif _key == 'canister_material_constant':
                                mech.canister_material_constant = \
                                            floatD(self.splitter(_line))

                    elif subkey == 'k0':
                        _val = floatD(subline.strip().split()[1])
                        _unit = subline.strip().split()[2]
                        mech.k0 = Coeff(_val,unit=_unit)
                    elif subkey == 'species':
                        while True:
                            _line = get_next_line(infile)
                            _key = _line.strip().split()[0].lower()

                            if _key in ['/','end']:
                                break
                            else:
                                spec = []
                                for (i,e) in enumerate(_line.split()):
                                    if i not in [0,5]:
                                        spec.append(floatD(e))
                                    else:
                                        spec.append(e)
                                mech.species.append(spec)
                    elif subkey == 'k_long':
                        _val = floatD(subline.strip().split()[1])
                        _unit = subline.strip().split()[2]
                        mech.k_long = Coeff(_val,unit=_unit)
                    elif subkey == 'nu':
                        mech.nu = floatD(self.splitter(subline))
                    elif subkey == 'ea':
                        _val = floatD(subline.strip().split()[1])
                        _unit = subline.strip().split()[2]
                        mech.ea = Coeff(_val,unit=_unit)
                    elif subkey == 'q':
                        _val = self.splitter(subline).lower()
                        mech.q = _val \
                               if _val == 'as_calculated' else floatD(_val)
                    elif subkey == 'k':
                        mech.k = floatD(self.splitter(subline))
                    elif subkey == 'v':
                        mech.v = floatD(self.splitter(subline))
                    elif subkey == 'ph':
                        _val = self.splitter(subline).lower()
                        mech.ph = _val \
                               if _val == 'as_calculated' else floatD(_val)
                    elif subkey == 'dissolution_rate':
                        _val = floatD(subline.strip().split()[1])
                        _unit = subline.strip().split()[2]
                        mech.dissolution_rate = Coeff(_val,unit=_unit)
                    elif subkey == 'fractional_dissolution_rate':
                        _val = floatD(subline.strip().split()[1])
                        _unit = subline.strip().split()[2]
                        mech.fractional_dissolution_rate = Coeff(_val,unit=_unit)
                    elif subkey == 'fractional_dissolution_rate_vi':
                        _val = floatD(subline.strip().split()[1])
                        _unit = subline.strip().split()[2]
                        mech.fractional_dissolution_rate_vi = Coeff(_val,unit=_unit)
                    elif subkey == 'specific_surface_area':
                        _val = floatD(subline.strip().split()[1])
                        _unit = subline.strip().split()[2]
                        mech.specific_surface_area = Coeff(_val,unit=_unit)

            elif key == 'waste_form':
                wf = wfg.add_waste_form()
                while True:
                    subline = get_next_line(infile)
                    subkey = subline.strip().split()[0].lower()

                    if subkey in ['/','end']:
                        break
                    elif subkey == 'coordinate':
                        _data = subline.strip().split()[1:]
                        _data = [floatD(x) for x in _data]
                        wf.coordinate = _data
                    elif subkey == 'exposure_factor':
                        wf.exposure_factor = floatD(self.splitter(subline))
                    elif subkey == 'volume':
                        _val = floatD(subline.strip().split()[1])
                        _unit = subline.strip().split()[2]
                        wf.volume = Coeff(_val,unit=_unit)
                    elif subkey == 'mechanism_name':
                        wf.mechanism_name = self.splitter(subline)
                    elif subkey == 'canister_vitality_rate':
                        _val = floatD(subline.strip().split()[1])
                        _unit = subline.strip().split()[2]
                        wf.canister_vitality_rate = Coeff(_val,unit=_unit)
                    elif subkey == 'cansiter_breach_time':
                        _val = floatD(subline.strip().split()[1])
                        _unit = subline.strip().split()[2]
                        wf.cansiter_breach_time = Coeff(_val,unit=_unit)
                    elif subkey == 'decay_start_time':
                        _val = floatD(subline.strip().split()[1])
                        _unit = subline.strip().split()[2]
                        wf.decay_start_time = Coeff(_val,unit=_unit)
        
        self.waste_form_general = wfg

    def _write_wasteform_general(self,outfile):
        self._header(outfile, headers['waste_form_general'])
        
        outfile.write('WASTE_FORM_GENERAL\n')
        wf = self.waste_form_general

        if wf.print_mass_balance:
            outfile.write('  PRINT_MASS_BALANCE\n')

        if wf.implicit_solution:
            outfile.write('  IMPLICIT_SOLUTION\n')

        for mech in wf.mechanism:
            outfile.write('    MECHANISM %s\n' % mech.name)
            outfile.write('      NAME %s\n' % mech.name)
            outfile.write('      SPECIFIC_SURFACE_AREA %s\n' % mech.specific_surface_area)

            if mech.fractional_dissolution_rate:
                outfile.write('      FRACTIONAL_DISSOLUTION_RATE %s\n' % mech.fractional_dissolution_rate)
            if mech.matrix_density:
                outfile.write('      MATRIX_DENSITY %s\n' % mech.matrix_density)
            if mech.kienzler_dissolution:
                outfile.write('      KIENZLER_DISSOLUTION\n')

            if mech.species:
                outfile.write('      SPECIES\n')
                for spec in mech.species:
                    _str = spec
                    _str[1:5] = [strD(x) for x in _str[1:5]]
                    _str = [x.ljust(10) for x in _str]
                    outfile.write('        '+' '.join(_str)+'\n')
                outfile.write('      /\n')

            canister_bool = mech.vitality_log10_mean is not None \
                            or mech.vitality_log10_stdev is not None \
                            or mech.canister_material_constant is not None \
                            or mech.vitality_upper_truncation is not None

            if canister_bool:
                outfile.write('      CANISTER_DEGRADATION_MODEL\n')

                if mech.vitality_log10_mean:
                    outfile.write('        VITALITY_LOG10_MEAN %s\n' % strD(mech.vitality_log10_mean))

                if mech.vitality_log10_stdev:
                    outfile.write('        VITALITY_LOG10_STDEV %s\n' % strD(mech.vitality_log10_stdev))

                if mech.vitality_upper_truncation is not None:
                    outfile.write('        VITALITY_UPPER_TRUNCATION %s\n' % strD(mech.vitality_upper_truncation))

                if mech.canister_material_constant:
                    outfile.write('        CANISTER_MATERIAL_CONSTANT %s\n' % strD(mech.canister_material_constant))

                outfile.write('      /\n')

            outfile.write('    /\n')

        for waste in wf.waste_form:
            outfile.write('    WASTE_FORM\n')

            if waste.coordinate:
                _coords = ' '.join([strD(x) for x in waste.coordinate])
                outfile.write('      COORDINATE %s\n' % _coords)

            if waste.exposure_factor:
                outfile.write('      EXPOSURE_FACTOR %s\n' % strD(waste.exposure_factor))

            if waste.volume:
                outfile.write('      VOLUME %s\n' % str(waste.volume))

            if waste.mechanism_name:
                outfile.write('      MECHANISM_NAME %s\n' % waste.mechanism_name)

            if waste.canister_vitality_rate:
                outfile.write('      CANISTER_VITALITY_RATE %s\n' % str(waste.canister_vitality_rate))

            outfile.write('    /\n')

        outfile.write('END_WASTE_FORM_GENERAL\n')


    def _read_region(self, infile, line):

        region = pregion()
        region.coordinates_lower = [None] * 3
        region.coordinates_upper = [None] * 3
        if line.strip().split()[0].lower() == 'geomechanics_region':
            region.pm = 'geomechanics'
        region.name = self.splitter(line).lower()

        keep_reading = True
        while keep_reading:  # Read through all cards
            line = get_next_line(infile)
            key = line.strip().split()[0].lower()  # take first keyword
            if key == 'coordinates':
                keep_reading_2 = True
                while keep_reading_2:
                    line1 = get_next_line(infile)
                    region.coordinates_lower[0] = floatD(line1.split()[0])
                    region.coordinates_lower[1] = floatD(line1.split()[1])
                    region.coordinates_lower[2] = floatD(line1.split()[2])

                    line2 = get_next_line(infile)
                    try:
                        region.coordinates_upper[0] = floatD(line2.split()[0])
                        region.coordinates_upper[1] = floatD(line2.split()[1])
                        region.coordinates_upper[2] = floatD(line2.split()[2])
                    except ValueError:
                        region.coordinates_upper = [None, None, None]
                        keep_reading2 = False
                        break

                    line3 = get_next_line(infile)
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
            elif key == 'file':
                region.filename = line.strip().split()[1]
            elif key == 'block':
                region.block = line.strip().split()[1:]
            elif key in ['/', 'end']:
                keep_reading = False

        self.add(region)

    # Adds a Region object.
    def _add_region(self, region=pregion(), overwrite=False):
        # check if region already exists
        if isinstance(region, pregion):
            if region.name in self.region.keys():
                if not overwrite:
                    warning = 'WARNING: A region with name \'' + \
                              str(
                                  region.name) + '\' already exists.' \
                                                 ' Region will' + \
                              'not be defined, use overwrite = ' + \
                              'True in add() to overwrite the old region.'
                    print(warning,
                    build_warnings.append(warning))
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
            if region.pm is '':
                outfile.write('REGION ')
                outfile.write(region.name.lower() + '\n')
                if region.filename:
                    outfile.write('  FILE ' + region.filename + '\n')
                elif region.block:
                    outfile.write('  BLOCK ')
                    for val in region.block:
                        outfile.write(str(int(val)) + ' ')
                    outfile.write('\n')
                else:
                    if region.face:
                        outfile.write('  FACE ' + region.face.upper() + '\n')
                        # no if statement below to ensure 0's are accepted for
                        # coordinates
                    if region.point_list:
                        for point in region.point_list:
                            outfile.write('  COORDINATE ')
                            for i in range(3):
                                outfile.write(strD(point.coordinate[i]) + ' ')
                            outfile.write('\n')
                    elif not all(x is None for x in region.coordinates_lower):
                        outfile.write('  COORDINATES\n')

                        lower = [
                            x for x in region.coordinates_lower if x is not None]
                        upper = [
                            x for x in region.coordinates_upper if x is not None]

                        lower_line = ' '.join(list(map(strD, lower)))
                        upper_line = ' '.join(list(map(strD, upper)))

                        if lower_line != '':
                            outfile.write('    ' + lower_line + '\n')

                        if upper_line != '':
                            outfile.write('    ' + upper_line + '\n')

                        outfile.write('  /\n')
                outfile.write('END\n\n')

    def _read_observation(self, infile):
        observation = pobservation()

        keep_reading = True

        while keep_reading:
            line = get_next_line(infile)
            key = line.strip().split()[0].lower()  # take first keyword
            if key == 'region':
                observation.region = self.splitter(line)
            elif key == 'velocity':
                observation.velocity = True
            elif key == 'at_cell_center':
                observation.at_cell_center = True
            elif key in ['/', 'end']:
                keep_reading = False

        self.observation_list.append(observation)

    # Adds a Observation object.
    def _add_observation(self, observation=pobservation(), overwrite=False):
        # check if observation already exists
        if isinstance(observation, pobservation):
            if observation.region in self.observation.keys():
                if not overwrite:
                    warning = 'WARNING: A observation with region \'' + \
                              str(observation.region) + '\' already ' \
                                                        'exists.' \
                                                        ' Observation will' \
                                                        ' not be defined,' + \
                              ' use overwrite = True in add() to overwrite' + \
                              'the old observation.'
                    print(warning,
                    build_warnings.append(warning))
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
            if observation.at_cell_center:
                outfile.write('  AT_CELL_CENTER\n')
            if observation.velocity:
                outfile.write('  VELOCITY\n')
            if self.multiple_continuum:
                if observation.secondary_temperature:
                    outfile.write('  SECONDARY_TEMPERATURE\n')
                if observation.secondary_concentration:
                    outfile.write('  SECONDARY_CONCENTRATION\n')
                if observation.secondary_mineral_volfrac:
                    outfile.write('  SECONDARY_MINERAL_VOLFRAC\n')
            outfile.write('END\n\n')

    def _read_flow(self, infile, line):
        flow = pflow()
        flow.datum = []
        flow.varlist = []
        flow.datum_type = ''
        # Flow Condition name passed in.
        flow.name = self.splitter(line).lower()
        if line.strip().split()[0].lower() == 'geomechanics_condition':
            flow.pm = 'geomechanics'
        keep_reading = True
        is_valid = False
        # Used so that entries outside flow conditions are ignored
        end_count = 0
        total_end_count = 1
        while keep_reading:  # Read through all cards
            line = get_next_line(infile)

            # Read key, or skip line if empty
            try:
                key = line.strip().split()[0].lower()  # take first keyword
            except IndexError:
                continue

            if key == 'type':
                total_end_count = 2  # Basically ensures that both read ifs for
                # the varlist will execute
                # This # indicates how many time a / or 'end'
                # can be read before loop terminates.

            elif key in [item.lower() for item in list(set(flow_condition_type_names_allowed).union(geomech_condition_type_names_allowed))]:
                if end_count == 0:
                    '''
                    Appending and instantiation of new flow_variables
                    occur here. Only two entries are filled,
                    the rest are assigned in the elif code block
                    where end_count == 1
                    '''
                    var = pflow_variable()
                    var.name = key
                    var.type = line.strip().split()[-1].lower()
                    if var.type in scaling_options_allowed:
                        var.subtype = var.type
                        var.type = line.strip().split()[-2].lower()
                    var.valuelist = []
                    var.list = []

                    is_valid = True  # Indicates the entries
                    # read here should be written so that entries outside
                    # flow conditions are ignored.
                    flow.varlist.append(var)

                elif end_count == 1:
                    # Assigns the 1st word on a line
                    tstring2name = line.strip().split()[0]
                    tstring2 = line.split()[1:]  # Assigns the rest of the line
                    # #2 used because this is the 2nd reading of the variables

                    # Deterine if variable is a list or stand-alone
                    # Executes only if 2nd word on line == 'list'
                    if tstring2[0].lower() == 'list':

                        # for each list in a pflow_variable object, check all
                        # pflow_variable objects by name to
                        # determine correct assignment
                        # before assigning in values from a list
                        keep_reading_list = True
                        while keep_reading_list:

                            line = get_next_line(infile)
                            # split the whole string/line
                            tstring2 = line.split()[:]
                            for var in flow.varlist:  # var represents
                                # a pflow_variable object
                                if tstring2name.lower() == var.name.lower():
                                    if line[0] == ':' or line[0] == '#' \
                                            or line[0] == '/':
                                        pass  # ignore a commented line
                                    # line[0] == '/' is a temporary fix
                                    elif tstring2[0].lower() == 'time_units':
                                        var.time_unit_type = tstring2[1]
                                    elif tstring2[0].lower() == 'data_units':
                                        var.data_unit_type = tstring2[1]
                                    elif line.split()[0].lower() in ['/', 'end']:
                                        keep_reading_list = False
                                    else:
                                        tvarlist = pflow_variable_list()
                                        tvarlist.time_unit_value = floatD(
                                            tstring2[0])
                                        tvarlist.data_unit_value_list = []
                                        tvarlist.data_unit_value_list.append(
                                            floatD(tstring2[1]))
                                        if len(tstring2) > 2:
                                            tvarlist. \
                                                data_unit_value_list.append(
                                                    floatD(tstring2[2]))
                                        var.list.append(tvarlist)
                            if line.split()[0] in ['/', 'end']:
                                keep_reading_list = False
                    else:
                        # for each single variable in a pflow_variable object,
                        # check all pflow_variable object by name to
                        # determine correct assignment
                        for substring in tstring2:
                            if substring in ['!']:
                                break
                            # Checks all values/types on this line
                            for var in flow.varlist:
                                # var represents a pflow_variable object
                                if tstring2name.lower() == var.name.lower():
                                    try:
                                        var.valuelist.append(floatD(substring))
                                    # If a string (e.g., C for temp.), assign
                                    # to unit
                                    except ValueError:
                                        var.unit = substring
            elif key == 'iphase':
                flow.iphase = int(self.splitter(line))
            elif key == 'sync_timestep_with_update':
                flow.sync_timestep_with_update = True
            elif key == 'cyclic':
                flow.cyclic = True
            elif key == 'datum':
                # Assign file_name with list of d_dx, d_dy, d_dz values.
                subkey = line.strip().split()[1].lower()
                if subkey == 'file':
                    flow.datum_type = 'file'
                    flow.datum = line.split()[-1]
                elif subkey == '':
                    flow.datum_type = 'dataset'
                    flow.datum = line.split()[1]
                # Assign d_dx, d_dy, d_dz values
                elif subkey == 'list':
                    flow.datum_type = 'list'
                    keep_reading1 = True
                    while keep_reading1:
                        line = get_next_line(infile)
                        if len(list(line)) == 0:
                            continue
                        if list(line)[0] in ['!', '#']:
                            continue
                        val = line.strip().split()[0].lower()

                        if val == 'time_units':
                            flow.datum_time_unit = line.strip().split()[1]
                        elif val in ['/', 'end']:
                            keep_reading1 = False
                        elif val == 'data_units':
                            flow.datum_data_unit = line.strip().split()[1]
                        else:
                            temp_list = [floatD(line.split()[0]), floatD(
                                line.split()[1]), floatD(line.split()[2]),
                                floatD(line.split()[3])]
                            flow.datum.append(temp_list)
                elif subkey == 'dataset':
                    flow.datum_type = 'dataset'
                    flow.datum = line.split()[1]
                else:
                    temp_list = [floatD(line.split()[1]), 
                                 floatD(line.split()[2]),
                                 floatD(line.split()[3])]
                    flow.datum = temp_list
            elif key == 'gradient':
                keep_reading1 = True
                while keep_reading1:
                    line = get_next_line(infile)
                    gtype = line.strip().split()[0].lower()

                    if gtype in gradient_types_allowed:
                        flow.gradient_type = gtype

                        if line.strip().split()[-1].lower() == 'list':
                            while True:
                                subline = get_next_line(infile)
                                subkey = subline.split()[0].lower()

                                if subkey == 'time_units':
                                    pass
                                    #print(subline.split()[-1])
                                    #TODO: add time units
                                elif subkey in ['/','end']:
                                    break
                                else:
                                    flow.gradient.extend(subline.split())                            
                        else:
                            for val in line.strip().split()[1:]:
                                flow.gradient.append(val)
                    elif gtype in ['/', 'end']:
                        keep_reading1 = False
                    else:
                        raise PyFLOTRAN_ERROR('Incorrect gradient type!')
            elif key == 'interpolation':
                flow.interpolation = line.split()[1].lower()
            elif key == 'units':
                flow.units = ''.join(line.split()[1:]).split(',')
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
                    warning = 'WARNING: A flow with name \'' + \
                              str(flow.name) + \
                              '\' already exists. Flow will not be ' + \
                              'defined, use overwrite = True in add() ' + \
                              ' to overwrite the old flow.'
                    print(warning,
                    build_warnings.append(warning))
                    return
                else:  # Executes if overwrite = True
                    self.delete(self.flow[flow.name])

        if flow not in self.flowlist:
            self.flowlist.append(flow)

    def _delete_flow(self, flow=pflow()):
        self.flowlist.remove(flow)

    '''
    Automate adding the sub-class flow_variable to a flow object.
    The flow object can be specified by name. If flow object name
    is not specified, the function will append pflow_variable to the
    last flow object added to the list. Function will provided a
    warning if a flow_variable.name already exists
    in the flow object it is trying to add it to.
    '''

    def _add_flow_variable(self, flow_variable=pflow_variable(),
                           index='', overwrite=False):
        # check if flow.name was specified
        if index:
            if isinstance(index, str):
                # Assign flow object to existing flow object with string type
                # name/index
                flow = self.flow.get(index)
                if not flow:
                    # Occurs if index/string is not found in flow object
                    print('WARNING: a flow object with flow.name', \
                        index, 'was not found. Current found entries are:', \
                        self.flow.keys(), 'pflow_variable was not added.\n')
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
                        flow_variable.name) + \
                        '\' already exists in flow with name \'' + \
                        str(flow.name) + \
                        '\'. Flow_variable will not be defined, ' + \
                        'use overwrite = True in add() to ' + \
                        'overwrite the old flow_variable. ' + \
                        'Use flow=\'name\' if you want to specify the ' + \
                        'flow object to add flow_variable to.'
                    print(warning,
                    build_warnings.append(warning))
                    return
                else:  # Executes if overwrite = True
                    self.delete(self.flow_variable(flow)[
                        flow_variable.name], flow)

        # Add flow_variable to flow (as a sub-class) if flow_variable does
        # not exist in specified flow object
        if flow_variable not in flow.varlist:
            flow.varlist.append(flow_variable)

    def _delete_flow_variable(self, flow_variable=pflow_variable(),
                              flow=pflow()):
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
                    print('       valid ' + \
                          'flow_condition pressure_types_allowed:', \
                        pressure_types_allowed, '\n')
                    raise PyFLOTRAN_ERROR(
                        'flow.varlist.type: \'' + condition_type +
                        '\' is invalid.')
                return 0  # Break out of function
            elif condition_name.upper() == 'FLUX':
                if condition_type.lower() in flux_types_allowed:
                    outfile.write(condition_type.lower())
                else:
                    print('       valid ' + \
                          'flow_condition flux_types_allowed:', \
                        flux_types_allowed, '\n')
                    raise PyFLOTRAN_ERROR(
                        'flow.varlist.type: \'' + condition_type +
                        '\' is invalid.')
                return 0  # Break out of function
            elif condition_name.upper() == 'RATE':
                if condition_type.lower() in rate_types_allowed:
                    outfile.write(condition_type.lower())
                else:
                    print('       valid flow_condition rate_types_allowed:', \
                        rate_types_allowed, '\n')
                    raise PyFLOTRAN_ERROR(
                        'flow.varlist.type: \'' + condition_type +
                        '\' is invalid.')
                return 0  # Break out of function
            elif condition_name.upper() == 'WELL':
                if condition_type.lower() in well_types_allowed:
                    outfile.write(condition_type.lower())
                else:
                    print('       valid well_conditions well_types_allowed:', \
                        well_types_allowed, '\n')
                    raise PyFLOTRAN_ERROR(
                        'flow.varlist.type: \'' + condition_type +
                        '\' is invalid.')
                return 0  # Break out of function
            elif condition_name.upper() == 'TEMPERATURE':
                if condition_type.lower() in temperature_types_allowed:
                    outfile.write(condition_type.lower())
                else:
                    print('       valid flow_condition ' + \
                          'temperature_types_allowed:', \
                        temperature_types_allowed, '\n')
                    raise PyFLOTRAN_ERROR(
                        'flow.varlist.type: \'' + condition_type +
                        '\' is invalid.')
                return 0  # Break out of function
            elif condition_name.upper() == 'CONCENTRATION':
                if condition_type.lower() in concentration_types_allowed:
                    outfile.write(condition_type.lower())
                else:
                    print('       valid flow_condition ' + \
                          'concentration_types_allowed:', \
                        concentration_types_allowed, '\n')
                    raise PyFLOTRAN_ERROR(
                        'flow.varlist.type: \'' + condition_type +
                        '\' is invalid.')
                return 0  # Break out of function
            elif condition_name.upper() == 'SATURATION':
                if condition_type.lower() in saturation_types_allowed:
                    outfile.write(condition_type.lower())
                else:
                    print('PyFLOTRAN ERROR: flow.varlist.type: \'' + \
                          condition_type + '\' is invalid.')
                    print('       valid flow_condition ' + \
                          'saturation_types_allowed:', \
                        saturation_types_allowed, '\n')
                return 0  # Break out of function
            elif condition_name.upper() == 'ENTHALPY':
                if condition_type.lower() in enthalpy_types_allowed:
                    outfile.write(condition_type.lower())
                else:
                    print('       valid flow_condition ' + \
                          'enthalpy_types_allowed:', \
                        enthalpy_types_allowed, '\n')
                    raise PyFLOTRAN_ERROR(
                        'flow.varlist.type: \'' +
                        condition_type + '\' is invalid.')
                return 0  # Break out of function
            elif condition_name.upper() == 'MOLE_FRACTION':
                if condition_type.lower() in mole_fraction_types_allowed:
                    outfile.write(condition_type.lower())
                else:
                    print('       valid ' + \
                          'flow_condition mole_fraction_types_allowed:', \
                        mole_fraction_types_allowed, '\n')
                    raise PyFLOTRAN_ERROR(
                        'flow.varlist.type: \'' + condition_type +
                        '\' is invalid.')
                return 0
            elif condition_name.upper() == 'LIQUID_PRESSURE':
                if condition_type.lower() in liquid_pressure_types_allowed:
                    outfile.write(condition_type.lower())
                else:
                    print('       valid ' + \
                          'flow_condition liquid_pressure_types_allowed:', \
                        liquid_pressure_types_allowed, '\n')
                    raise PyFLOTRAN_ERROR(
                        'flow.varlist.type: \'' + condition_type +
                        '\' is invalid.')
                return 0
            elif condition_name.upper() == 'LIQUID_FLUX':
                if condition_type.lower() in liquid_flux_types_allowed:
                    outfile.write(condition_type.lower())
                else:
                    print('       valid ' + \
                          'flow_condition liquid_flux_types_allowed:', \
                        liquid_flux_types_allowed, '\n')
                    raise PyFLOTRAN_ERROR(
                        'flow.varlist.type: \'' + condition_type +
                        '\' is invalid.')
                return 0
            elif condition_name.upper() == 'GAS_PRESSURE':
                if condition_type.lower() in gas_pressure_types_allowed:
                    outfile.write(condition_type.lower())
                else:
                    print('       valid ' + \
                          'flow_condition gas_pressure_types_allowed:', \
                        gas_pressure_types_allowed, '\n')
                    raise PyFLOTRAN_ERROR(
                        'flow.varlist.type: \'' + condition_type +
                        '\' is invalid.')
                return 0
            elif condition_name.upper() == 'GAS_FLUX':
                if condition_type.lower() in gas_flux_types_allowed:
                    outfile.write(condition_type.lower())
                else:
                    print('       valid ' + \
                          'flow_condition gas_flux_types_allowed:', \
                        gas_flux_types_allowed, '\n')
                    raise PyFLOTRAN_ERROR(
                        'flow.varlist.type: \'' + condition_type +
                        '\' is invalid.')
                return 0
            elif condition_name.upper() == 'GAS_SATURATION':
                if condition_type.lower() in gas_saturation_types_allowed:
                    outfile.write(condition_type.lower())
                else:
                    print('       valid ' + \
                          'flow_condition gas_saturation_types_allowed:', \
                        gas_saturation_types_allowed, '\n')
                    raise PyFLOTRAN_ERROR(
                        'flow.varlist.type: \'' + condition_type +
                        '\' is invalid.')
                return 0
            else:
                pass
                # Error reporting for flow_condition.name is done elsewhere
                # name should be validated before this function is called.

        # Write out all valid flow_conditions objects with FLOW_CONDITION as
        # keyword
        for flow in self.flowlist:
            if flow.pm == '':
                outfile.write('FLOW_CONDITION ' + flow.name.lower() + '\n')

                if flow.sync_timestep_with_update:
                    outfile.write('  SYNC_TIMESTEP_WITH_UPDATE\n')

                if flow.units:
                    _units = ', '.join(flow.units)
                    outfile.write('  UNITS %s\n' % _units)

                outfile.write('  TYPE\n')
                # variable name and type from lists go here
                for a_flow in flow.varlist:
                    if a_flow.name.upper() in flow_condition_type_names_allowed:
                        outfile.write('    ' + a_flow.name.upper() + ' ')
                    else:
                        print('       valid flow_condition.names:', \
                            flow_condition_type_names_allowed, '\n')
                        raise PyFLOTRAN_ERROR(
                            'flow.varlist.name: \'' +
                            a_flow.name + '\' is invalid.')

                    # Checks a_flow.type and performs write or error reporting
                    check_condition_type(a_flow.name, a_flow.type)
                    if a_flow.subtype:
                        outfile.write(' ' + a_flow.subtype)
                    outfile.write('\n')

                outfile.write('  /\n')

                if flow.interpolation:
                    outfile.write('  INTERPOLATION %s\n' %
                                  flow.interpolation.upper())

                if flow.cyclic:
                    outfile.write('  CYCLIC\n')

                if flow.datum:  # error-checking not yet added
                    outfile.write('  DATUM')

                    if isinstance(flow.datum, str):
                        if flow.datum_type == 'file':
                            outfile.write(' FILE ')
                        if flow.datum_type == 'dataset':
                            outfile.write(' DATASET ')
                        outfile.write(flow.datum + '\n')
                    elif flow.datum_type == 'list':
                        outfile.write(' LIST\n')
                        if flow.datum_time_unit is not None:
                            outfile.write('    ')
                            outfile.write('TIME_UNITS ' +
                                          flow.datum_time_unit + '\n')
                        if flow.datum_data_unit is not None:
                            outfile.write('    ')
                            outfile.write('DATA_UNITS ' +
                                          flow.datum_data_unit + '\n')
                        for val in flow.datum:
                            outfile.write('      ')
                            outfile.write(strD(val[0]) + ' ')  # time values
                            outfile.write(strD(val[1]) + ' ')
                            outfile.write(strD(val[2]) + ' ')
                            outfile.write(strD(val[3]) + '\n')
                        outfile.write('    /\n')
                    else:  # only a single list
                        outfile.write(' ')
                        outfile.write(strD(flow.datum[0]) + ' ')
                        outfile.write(strD(flow.datum[1]) + ' ')
                        outfile.write(strD(flow.datum[2]))
                        outfile.write('\n')
                # Following code is paired w/ this statement.

                if flow.iphase:
                    outfile.write('  IPHASE ' + str(flow.iphase) + '\n')
                if flow.gradient:
                    outfile.write('  GRADIENT\n')

                    if flow.gradient[0].lower() == 'file':
                        outfile.write('    ' + flow.gradient_type.upper() +
                                      ' ' + ' '.join(flow.gradient))
                    else:
                        outfile.write('    ' + flow.gradient_type.upper() +
                                      ' ' + str(flow.gradient[0]) + ' ' + str(
                            flow.gradient[1]) + ' ' +
                            str(flow.gradient[2]) + '\n')
                    outfile.write('  /\n')
                # variable name and values from lists along with units go here
                for a_flow in flow.varlist:
                    if a_flow.valuelist:
                        outfile.write('  ' + a_flow.name.upper())
                        if isinstance(a_flow.valuelist[0], str):
                            if a_flow.valuelist[0] == 'file':
                                outfile.write(' FILE ' + a_flow.valuelist[1])
                            else:
                                outfile.write(
                                    ' DATASET ' + a_flow.valuelist[0])
                        else:
                            for flow_val in a_flow.valuelist:
                                outfile.write(' ' + strD(flow_val))
                        if a_flow.unit:
                            outfile.write(
                                ' ' + a_flow.unit)
                        outfile.write('\n')
                    elif a_flow.list:
                        outfile.write(
                            '    ' + a_flow.name.upper() + ' LIST' + '\n')
                        if a_flow.time_unit_type:
                            outfile.write('      TIME_UNITS ' +
                                          a_flow.time_unit_type + '\n')
                        if a_flow.data_unit_type:
                            outfile.write('      DATA_UNITS ' +
                                          a_flow.data_unit_type + '\n')
                        for k in a_flow.list:
                            outfile.write('        ' + strD(k.time_unit_value))
                            for p in range(len(k.data_unit_value_list)):
                                outfile.write(
                                    '  ' + strD(k.data_unit_value_list[p]))
                            outfile.write('\n')
                        outfile.write('    /\n')

                outfile.write('END\n\n')

    def _read_initial_condition(self, infile, line):
        if len(line.split()) > 1:
            # Flow Condition name passed in.
            np_name = self.splitter(line).lower()
        else:
            np_name = None
        p = pinitial_condition()
        np_flow = p.flow
        np_transport = p.transport
        np_region = p.region

        keep_reading = True

        while keep_reading:  # Read through all cards
            line = get_next_line(infile)  # get next line
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
        new_initial_condition = pinitial_condition(
            np_flow, np_transport, np_region, np_name)
        self.add(new_initial_condition)

    def _add_initial_condition(self, initial_condition=pinitial_condition(),
                               overwrite=False):
        # Adds a initial_condition object.
        # check if flow already exists
        if isinstance(initial_condition, pinitial_condition):
            if initial_condition.region in self.initial_condition.keys():
                if not overwrite:
                    warning = 'WARNING: A initial_condition with region \'' + \
                              str(initial_condition.region) + \
                              '\' already exists. initial_condition will ' + \
                              'not be defined, use ' + \
                              'overwrite = True in add() to overwrite ' + \
                              'the old initial_condition.'
                    print(warning,
                    build_warnings.append(warning))
                    return
                else:
                    self.delete(self.initial_condition[
                        initial_condition.region])

        if initial_condition not in self.initial_condition_list:
            self.initial_condition_list.append(initial_condition)

    def _delete_initial_condition(self,
                                  initial_condition=pinitial_condition()):
        self.initial_condition_list.remove(initial_condition)

    def _write_initial_condition(self, outfile):
        self._header(outfile, headers['initial_condition'])
        # Write all initial conditions to file
        try:
            for b in self.initial_condition_list:  # b = initial_condition
                if b.name:
                    outfile.write('INITIAL_CONDITION ' + b.name.lower() + '\n')
                else:
                    outfile.write('INITIAL_CONDITION\n')
                if b.flow:
                    outfile.write('  FLOW_CONDITION ' + b.flow.lower() + '\n')
                if b.transport:
                    outfile.write('  TRANSPORT_CONDITION ' +
                                  b.transport.lower() + '\n')
                if b.region:
                    outfile.write('  REGION ' + b.region.lower() + '\n')
                else:
                    raise PyFLOTRAN_ERROR(
                        'initial_condition.region is required')
                outfile.write('END\n\n')
        except:
            raise PyFLOTRAN_ERROR(
                'At least one initial condition with ' +
                'valid attributes is required')

    def _read_boundary_condition(self, infile, line):
        if len(line.split()) > 1:
            # Flow Condition name passed in.
            np_name = self.splitter(line).lower()
        else:
            np_name = None
        p = pboundary_condition()
        np_flow = p.flow
        np_transport = p.transport
        np_region = p.region
        np_geomech = p.geomech

        keep_reading = True

        while keep_reading:  # Read through all cards
            line = get_next_line(infile)
            key = line.split()[0].lower()  # take first key word

            if key == 'flow_condition':
                np_flow = self.splitter(line)  # take last word
            elif key == 'transport_condition':
                np_transport = self.splitter(line)
            elif 'region' in key:
                np_region = self.splitter(line)
            elif key == 'geomechanics_condition':
                np_geomech = self.splitter(line)
            elif key in ['/', 'end']:
                keep_reading = False

        # Create an empty boundary condition and assign the values read in
        new_boundary_condition = pboundary_condition(
            np_name, np_flow, np_transport, np_region, np_geomech)
        self.add(new_boundary_condition)

    def _add_boundary_condition(self, boundary_condition=pboundary_condition(),
                                overwrite=False):
        # Adds a boundary_condition object.
        # check if flow already exists
        if isinstance(boundary_condition, pboundary_condition):
            if boundary_condition.region in self.boundary_condition.keys():
                if not overwrite:
                    warning = 'WARNING: A boundary_condition with region \'' + \
                              str(boundary_condition.region) + '\'' + \
                              ' already exists. boundary_condition will ' + \
                              ' not be defined, use overwrite = ' + \
                              'True in add()' + \
                              'to overwrite the old boundary_condition.'
                    print(warning,
                    build_warnings.append(warning))
                    return
                else:
                    self.delete(self.boundary_condition[
                        boundary_condition.region])

        if boundary_condition not in self.boundary_condition_list:
            self.boundary_condition_list.append(boundary_condition)

    def _delete_boundary_condition(self,
                                   boundary_condition=pboundary_condition()):
        self.boundary_condition_list.remove(boundary_condition)

    def _write_boundary_condition(self, outfile):
        self._header(outfile, headers['boundary_condition'])

        # Write all boundary conditions to file
        try:
            for b in self.boundary_condition_list:  # b = boundary_condition
                if b.geomech == '':
                    if b.name:
                        outfile.write('BOUNDARY_CONDITION ' +
                                      b.name.lower() + '\n')
                    else:
                        outfile.write('BOUNDARY_CONDITION ' + '\n')
                    if b.flow:
                        outfile.write('  FLOW_CONDITION ' +
                                      b.flow.lower() + '\n')
                    if b.transport:
                        outfile.write('  TRANSPORT_CONDITION ' +
                                      b.transport.lower() + '\n')
                    if b.region:
                        outfile.write('  REGION ' + b.region.lower() + '\n')
                    else:
                        raise PyFLOTRAN_ERROR(
                            'boundary_condition.region is required')
                    outfile.write('END\n\n')
        except:
            raise PyFLOTRAN_ERROR(
                'At least one boundary_condition with valid ' +
                'attributes is required')

    def _read_source_sink(self, infile, line):
        p = psource_sink()
        np_flow = p.flow
        np_transport = p.transport
        np_region = p.region

        if len(line.split()) > 1:
            # Flow Condition name passed in.
            np_name = self.splitter(line).lower()
        else:
            np_name = None

        keep_reading = True

        while keep_reading:  # Read through all cards
            line = get_next_line(infile)
            key = line.strip().split()[0].lower()  # take first key word

            if key == 'flow_condition':
                np_flow = self.splitter(line)  # take last word
            elif key == 'transport_condition':
                np_transport = self.splitter(line)  # take last word
            elif key == 'region':
                np_region = self.splitter(line)
            elif key in ['/', 'end']:
                keep_reading = False

        # Create an empty source sink and assign the values read in
        new_source_sink = psource_sink(flow=np_flow, transport=np_transport,
                                       region=np_region, name=np_name)
        self.add(new_source_sink)

    # Adds a source_sink object.
    def _add_source_sink(self, source_sink=psource_sink(), overwrite=False):
        # check if flow already exists
        if isinstance(source_sink, psource_sink):
            if source_sink.region in self.source_sink.keys():
                if not overwrite:
                    warning = 'WARNING: A source_sink with region \'' + \
                              str(source_sink.region) +\
                              '\' already exists. ' + \
                              'source_sink will not be defined,' + \
                              ' use overwrite = True ' + \
                              'in add() to overwrite the old source_sink.'
                    print(warning,
                    build_warnings.append(warning))
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
                outfile.write('  TRANSPORT_CONDITION ' +
                              b.transport.lower() + '\n')
            if b.region:
                outfile.write('  REGION ' + b.region.lower() + '\n')
            else:
                raise PyFLOTRAN_ERROR('source_sink.region is required')
            outfile.write('END\n\n')

    def _read_source_sink_sandbox(self,infile):
        ss = psource_sink_sandbox()
        while True:
            line = get_next_line(infile)
            key = line.strip().lower().split()[0]

            if key == 'mass_rate_downregulated':
                mrd = psource_sink_sandbox.mass_rate_downregulated()
                while True:
                    subline = get_next_line(infile)
                    subkey = subline.strip().lower().split()[0]

                    if subkey == 'rate':
                        mrd.rate = floatD(self.splitter(subline))
                    elif subkey == 'negative_reg_pressure':
                        mrd.negative_reg_pressure = floatD(self.splitter(subline))
                    elif subkey == 'positive_reg_pressure':
                        mrd.positive_reg_pressure = floatD(self.splitter(subline))
                    elif subkey == 'coordinate':
                        _split = subline.split()[1:]
                        _coord = [floatD(x) for x in _split]
                        mrd.coordinate = _coord
                    elif subkey in ['/','end']:
                        break

                ss.mass_rate_downregulated = mrd
            elif key in ['/','end']:
                break

        self.source_sink_sandbox_list.append(ss)

    def _add_source_sink_sandbox(self,source_sink_sandbox):
        self.source_sink_sandbox_list.append(source_sink_sandbox)

    def _delete_source_sink_sandbox(self,
                                    source_sink_sandbox=psource_sink_sandbox()):
        self.source_sink_sandbox_list.remove(source_sink_sandbox)

    def _write_source_sink_sandbox(self,outfile):
        self._header(outfile, headers['source_sink_sandbox'])
        for ss in self.source_sink_sandbox_list:
            outfile.write('SOURCE_SINK_SANDBOX\n')
            if ss.mass_rate_downregulated:
                ss.mass_rate_downregulated._write(outfile)
            outfile.write('END\n')

    def _read_wipp_source_sink(self,infile):

        wss = pwipp_source_sink()

        while True:
            line = get_next_line(infile)
            key = line.strip().split()[0].lower()

            if key in ['/','end','end_wipp_source_sink']:
                break
            elif key == 'brucitec':
                wss.brucitec = floatD(self.splitter(line))
            elif key == 'bruciteh':
                wss.bruciteh = floatD(self.splitter(line))
            elif key == 'hymagcon':
                wss.hymagcon = floatD(self.splitter(line))
            elif key == 'sat_wick':
                wss.sat_wick = floatD(self.splitter(line))
            elif key == 'salt_percent':
                wss.salt_percent = floatD(self.splitter(line))
            elif key == 'gratmici':
                wss.gratmici = floatD(self.splitter(line))
            elif key == 'gratmich':
                wss.gratmich = floatD(self.splitter(line))
            elif key == 'corrmco2':
                wss.corrmco2 = floatD(self.splitter(line))
            elif key == 'humcorr':
                wss.humcorr = floatD(self.splitter(line))
            elif key == 'asdrum':
                wss.asdrum = floatD(self.splitter(line))
            elif key == 'alpharxn':
                wss.alpharxn = floatD(self.splitter(line))
            elif key == 'socmin':
                wss.socmin = floatD(self.splitter(line))
            elif key == 'biogenfc':
                wss.biogenfc = floatD(self.splitter(line))
            elif key == 'probdeg':
                wss.probdeg = floatD(self.splitter(line))
            elif key == 'stoichiometric_matrix':
                sm = []
                while True:
                    subline = get_next_line(infile)
                    subkey = subline.strip().split()[0].lower()

                    if subkey in ['/','end']:
                        break
                    else:
                        sm.append([floatD(x) for x in subline.split()])

                wss.stoichiometric_matrix = sm

            elif key == 'inventory':
                inv = pwipp_source_sink.inventory(name=self.splitter(line))
                while True:
                    subline = get_next_line(infile)
                    subkey = subline.strip().split()[0].lower()

                    if subkey in ['/','end']:
                        break
                    elif subkey == 'vrepos':
                        _val = floatD(subline.split()[1])

                        try:
                            _unit = subline.split()[2]
                        except IndexError:
                            _unit = None

                        inv.vrepos = Coeff(_val,unit=_unit)
                    elif subkey == 'solids':
                        while True:
                            _line = get_next_line(infile)
                            _key = _line.strip().split()[0].lower()

                            if _key in ['/','end']:
                                break
                            else:

                                try:
                                    name,val,unit = _line.strip().split()
                                except ValueError:
                                    name,val = _line.strip().split()
                                    unit = None

                                val = floatD(val)
                                _solid = pwipp_source_sink.inventory.solid()
                                _solid.name = name
                                _solid.value = Coeff(val,unit=unit)
                                inv.solids.append(_solid)
                    elif subkey == 'aqueous':
                        while True:
                            _line = get_next_line(infile)
                            _key = _line.strip().split()[0].lower()

                            if _key in ['/','end']:
                                break
                            else:
                                aq = pwipp_source_sink.inventory.aqueous()
                                aq.name, aq.value = _line.strip().split()
                                aq.value = floatD(aq.value)
                                inv.aqueous.append(aq)

                wss.inventory.append(inv)

            elif key == 'waste_panel':
                wp = pwipp_source_sink.waste_panel(name=self.splitter(line))
                while True:
                    subline = get_next_line(infile)
                    subkey = subline.strip().split()[0].lower()

                    if subkey in ['/','end']:
                        break
                    elif subkey == 'region':
                        wp.region = self.splitter(subline)
                    elif subkey == 'inventory':
                        wp.inventory = self.splitter(subline)
                    elif subkey == 'scale_by_volume':
                        _bool = self.splitter(subline).lower()
                        wp.scale_by_volume = True if 'yes' in _bool else False

                wss.waste_panel.append(wp)

        self.wipp_source_sink = wss

    def _write_wipp_source_sink(self,outfile):

        wss = self.wipp_source_sink
        #self._header(outfile, headers['wipp source sink'])

        outfile.write('WIPP_SOURCE_SINK\n')
        if wss.brucitec is not None:
            outfile.write('  BRUCITEC %s\n' % strD(wss.brucitec))
        if wss.bruciteh is not None:
            outfile.write('  BRUCITEH %s\n' % strD(wss.bruciteh))
        if wss.hymagcon is not None:
            outfile.write('  HYMAGCON %s\n' % strD(wss.hymagcon))
        if wss.sat_wick is not None:
            outfile.write('  SAT_WICK %s\n' % strD(wss.sat_wick))
        if wss.salt_percent is not None:
            outfile.write('  SALT_PERCENT %s\n' % strD(wss.salt_percent))
        if wss.gratmici is not None:
            outfile.write('  GRATMICI %s\n' % strD(wss.gratmici))
        if wss.gratmich is not None:
            outfile.write('  GRATMICH %s\n' % strD(wss.gratmich))
        if wss.corrmco2 is not None:
            outfile.write('  CORRMCO2 %s\n' % strD(wss.corrmco2))
        if wss.humcorr is not None:
            outfile.write('  HUMCORR %s\n' % strD(wss.humcorr))
        if wss.asdrum is not None:
            outfile.write('  ASDRUM %s\n' % strD(wss.asdrum))
        if wss.alpharxn is not None:
            outfile.write('  ALPHARXN %s\n' % strD(wss.alpharxn))
        if wss.socmin is not None:
            outfile.write('  SOCMIN %s\n' % strD(wss.socmin))
        if wss.biogenfc is not None:
            outfile.write('  BIOGENFC %s\n' % strD(wss.biogenfc))
        if wss.probdeg is not None:
            outfile.write('  PROBDEG %s\n' % strD(wss.probdeg))
        if wss.stoichiometric_matrix:

            _text = ''
            for row in wss.stoichiometric_matrix:
                _text += '    '+' '.join([strD(x).ljust(10) for x in row]) + '\n'
            outfile.write('\n  STOICHIOMETRIC_MATRIX\n' + _text + '  END\n')

        for inv in wss.inventory:
            outfile.write('\n  INVENTORY %s\n' % inv.name)
            outfile.write('    VREPOS %s\n' % inv.vrepos)

            if inv.solids:
                outfile.write('    SOLIDS\n')
                for solid in inv.solids:
                    outfile.write('      %s %s\n' % (solid.name,solid.value))
                outfile.write('    END\n')

            if inv.aqueous:
                outfile.write('    AQUEOUS\n')
                for aq in inv.aqueous:
                    outfile.write('      %s %s\n' % (aq.name,aq.value))
                outfile.write('    END\n')

            outfile.write('  END\n')

        for wp in wss.waste_panel:
            outfile.write('\n  WASTE_PANEL %s\n' % wp.name)

            if wp.region:
                outfile.write('    REGION %s\n' % wp.region)
            if wp.inventory:
                outfile.write('    INVENTORY %s\n' % wp.inventory)
            if wp.scale_by_volume:
                _bool = 'YES' if wp.scale_by_volume == True else 'NO'
                outfile.write('    SCALE_BY_VOLUME %s\n' % _bool)

            outfile.write('  END\n')

        outfile.write('END_WIPP_SOURCE_SINK\n')

    def _delete_strata(self, strata=pstrata()):
        self.strata_list.remove(strata)

    def _read_strata(self, infile, line):
        strata = pstrata()
        keep_reading = True
        if line.strip().split()[0].lower() == 'geomechanics_strata':
            strata.pm = 'geomechanics'

        while keep_reading:  # Read through all cards
            line = get_next_line(infile)
            key = line.strip().split()[0].lower()  # take first key word

            if 'region' in key:
                strata.region = self.splitter(line)  # take last word
            elif 'material' in key:
                strata.material = self.splitter(line)  # take last word
            elif 'file' in key:
                strata.file = self.splitter(line)
            elif 'start_time' in key:
                _values = line.strip().split()[1:]
                _values[0] = floatD(_values[0])
                strata.start_time = _values
            elif 'final_time' in key:
                _values = line.strip().split()[1:]
                _values[0] = floatD(_values[0])
                strata.final_time = _values
            elif key in ['/', 'end']:
                keep_reading = False

        # Create an empty source sink and assign the values read in
        self.add(strata)

    # Adds a strata object.
    def _add_strata(self, strata=pstrata(), overwrite=False):
        # check if stratigraphy coupler already exists
        if isinstance(strata, pstrata):
            if strata.region in self.strata.keys():
                if not overwrite:
                    warning = 'WARNING: A strata with name \'' + \
                              str(strata.region) + \
                              '\' already exists. strata will' + \
                              'not be defined, use overwrite = True in' + \
                              ' add() to overwrite the old strata.'
                    print(warning,
                    build_warnings.append(warning))
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
            if strata.pm == '':
                outfile.write('STRATA\n')
                if strata.region:
                    outfile.write('  REGION ' + strata.region.lower() + '\n')
                if strata.start_time:
                    _start = strata.start_time
                    _end = strata.final_time
                    _start[0] = strD(_start[0])
                    _end[0] = strD(_end[0])
                    _start = ' '.join(_start)
                    _end = ' '.join(_end)
                    outfile.write('  START_TIME %s\n' % _start)
                    outfile.write('  FINAL_TIME %s\n' % _end)
                # else:
                # raise PyFLOTRAN_ERROR('strata.region is required')
                if strata.material:
                    outfile.write('  MATERIAL ' +
                                  strata.material + '\n')
                elif strata.file:
                    outfile.write('  FILE ' + strata.file + '\n')
                else:
                    raise PyFLOTRAN_ERROR('strata.material is required')
                outfile.write('END\n\n')

    def _read_ufd_decay(self, infile):
        ufd = pufd_decay()
        while True:
            line = get_next_line(infile)
            key = line.strip().split()[0].lower()

            if key == 'element':
                elem = pufd_decay.element()
                elem.name = self.splitter(line)
                while True:
                    subline = get_next_line(infile)
                    subkey = subline.strip().split()[0].lower()

                    if subkey == 'solubility':
                        elem.solubility = floatD(self.splitter(subline))
                    elif subkey == 'kd':
                        while True:
                            _line3 = get_next_line(infile)
                            _key3 = _line3.strip().split()[0].lower()

                            if _key3 in ['/','end']:
                                break
                            else:
                                _name = _line3.split()[0]
                                _value = floatD(_line3.split()[1])
                                elem.kd.extend([_name,_value])
                    elif subkey in ['/','end']:
                        break
                ufd.elements.append(elem)
            elif key == 'isotope':
                iso = pufd_decay.isotope()
                iso.name = self.splitter(line)
                while True:
                    subline = get_next_line(infile)
                    subkey = subline.strip().split()[0].lower()

                    if subkey == 'element':
                        iso.element = self.splitter(subline)
                    elif subkey == 'decay_rate':
                        iso.decay_rate = floatD(self.splitter(subline))
                    elif subkey == 'daughter':
                        _name = subline.split()[1]
                        _value = floatD(subline.split()[2])
                        iso.daughter = [_name,_value]
                    elif subkey in ['/','end']:
                        break
                ufd.isotopes.append(iso)
            elif key == 'implicit_solution':
                ufd.implicit_solution = True
            elif key in ['/','end']:
                break

        self.ufd_decay = ufd

    def _write_ufd_decay(self, outfile):
        self._header(outfile, headers['ufd_decay'])
        ufd = self.ufd_decay

        outfile.write('UFD_DECAY\n')
        if ufd.implicit_solution:
            outfile.write('  IMPLICIT SOLUTION\n')
        for element in ufd.elements:
            element._write(outfile)
        for isotope in ufd.isotopes:
            isotope._write(outfile)
        outfile.write('END\n')

    def _read_ufd_biosphere(self,infile):
        ufd = pufd_biosphere()
        while True:
            line = get_next_line(infile)
            key = line.strip().split()[0].lower()

            if key == 'erb_1a':
                erb = pufd_biosphere.erb_1a()
                erb.name = self.splitter(line)
                while True:
                    subline = get_next_line(infile)
                    subkey = subline.strip().split()[0].lower()

                    if subkey == 'region':
                        erb.region = self.splitter(subline)
                    elif subkey == 'individual_consumption_rate':
                        _val = floatD(subline.split()[1])
                        _un = subline.split()[2]
                        erb.individual_consumption_rate = Coeff(_val,unit=_un)
                    elif subkey == 'include_unsupported_rads':
                        erb.include_unsupported_rads = True
                    elif subkey in ['/','end']:
                        break
                ufd.erb_1a.append(erb)
            elif key == 'erb_1b':
                erb = pufd_biosphere.erb_1b()
                erb.name = self.splitter(line)
                while True:
                    subline = get_next_line(infile)
                    subkey = subline.strip().split()[0].lower()

                    if subkey == 'region':
                        erb.region = self.splitter(subline)
                    elif subkey == 'individual_consumption_rate':
                        _val = floatD(subline.split()[1])
                        _un = subline.split()[2]
                        erb.individual_consumption_rate = Coeff(_val,unit=_un)
                    elif subkey == 'include_unsupported_rads':
                        erb.include_unsupported_rads = True
                    elif subkey == 'dilution_factor':
                        erb.dilution_factor = floatD(self.splitter(subline))
                    elif subkey in ['/','end']:
                        break
                ufd.erb_1b.append(erb)

            elif key == 'supported_radionuclides' or \
                 key == 'unsupported_radionuclides':

                while True:
                    rad = pufd_biosphere.radionuclide()
                    _line = get_next_line(infile)

                    if _line.strip() in ['/','end']:
                        break

                    rad.name = self.splitter(_line)

                    while True:
                        subline = get_next_line(infile)
                        subkey = subline.strip().split()[0].lower()

                        if subkey == 'element_kd':
                            rad.element_kd = floatD(self.splitter(subline))
                        elif subkey == 'decay_rate':
                            _val = floatD(subline.split()[1])
                            _un = subline.split()[2]
                            rad.decay_rate = Coeff(_val,unit=_un)
                        elif subkey == 'supported_parent':
                            rad.supported_parent = self.splitter(subline)
                        elif subkey == 'ingestion_dose_coef':
                            _val = floatD(subline.split()[1])
                            _un = subline.split()[2]
                            rad.ingestion_dose_coef = Coeff(_val,unit=_un)
                        elif subkey == 'emanation_factor':
                            rad.emanation_factor = floatD(self.splitter(subline))
                        elif subkey in ['/','end']:
                            break

                    if key == 'supported_radionuclides':
                        ufd.supported_radionuclides.append(rad)
                    else:
                        ufd.unsupported_radionuclides.append(rad)

            elif key == 'output_start_time':
                _value = floatD(line.split()[1])
                _units = line.split()[2]
                ufd.output_start_time = Coeff(_value,unit=_units)
            elif key in ['/','end']:
                break

        self.ufd_biosphere = ufd

    def _write_ufd_biosphere(self,outfile):
        self._header(outfile, headers['ufd_biosphere'])
        ufd = self.ufd_biosphere

        outfile.write('UFD_BIOSPHERE\n')
        for erb in ufd.erb_1a:
            erb._write(outfile)
        for erb in ufd.erb_1b:
            erb._write(outfile)
        if ufd.supported_radionuclides:
            outfile.write('  SUPPORTED_RADIONUCLIDES\n')
            for rad in ufd.supported_radionuclides:
                rad._write(outfile)
            outfile.write('  /\n')
        if ufd.supported_radionuclides:
            outfile.write('  UNSUPPORTED_RADIONUCLIDES\n')
            for rad in ufd.unsupported_radionuclides:
                rad._write(outfile)
            outfile.write('  /\n')
        if ufd.output_start_time:
            outfile.write('  OUTPUT_START_TIME %s\n' % \
                          str(ufd.output_start_time))
        outfile.write('END\n')

    def _write_checkpoint(self, outfile):
        checkpoint = self.simulation.checkpoint
        # if checkpoint.time_list or checkpoint.periodic_time \
        #       or checkpoint.periodic_timestep:
        outfile.write('  CHECKPOINT\n')
        if checkpoint.time_list:
            outfile.write('    TIMES ')
            if checkpoint.time_unit is None:
                raise PyFLOTRAN_ERROR('times list unit is required!')
            else:
                outfile.write(checkpoint.time_unit + ' ')
            for val in checkpoint.time_list:
                outfile.write(strD(val) + ' ')
            outfile.write('\n')
        if checkpoint.periodic_time is not None:
            outfile.write('    PERIODIC TIME ')
            if checkpoint.periodic_time_unit is not None:
                outfile.write(str(checkpoint.periodic_time_unit) + ' ')
            outfile.write(strD(checkpoint.periodic_time))
            outfile.write('\n')
        if checkpoint.periodic_timestep:
            outfile.write('    PERIODIC TIMESTEP ')
            outfile.write(strD(checkpoint.periodic_timestep))
            outfile.write('\n')
        if checkpoint.format is not None:
            outfile.write('    FORMAT ')
            outfile.write(checkpoint.format.upper())
            outfile.write('\n')
        outfile.write('  /\n')

    def _write_restart(self, outfile):
        restart = self.simulation.restart
        # write file name
        outfile.write('  RESTART ' + str(restart.file_name) + ' ')
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

        outfile.write('\n')

    def _read_dataset(self, infile, line):
        dataset = pdataset()
        keep_reading = True
        dataset.dataset_name = self.splitter(line)
        while keep_reading:  # Read through all cards
            line = get_next_line(infile)
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
                outfile.write('DATASET MAPPED ' +
                              dataset.dataset_mapped_name + '\n')
            if dataset.dataset_name and dataset.dataset_mapped_name:
                raise PyFLOTRAN_ERROR(
                    'Cannot use both DATASET and DATASET MAPPED')
            if dataset.name:
                outfile.write('  NAME ' + dataset.name + '\n')
            if dataset.file_name:
                outfile.write('  FILENAME ' + dataset.file_name + '\n')
            if dataset.hdf5_dataset_name:
                outfile.write('  HDF5_DATASET_NAME ' +
                              dataset.hdf5_dataset_name + '\n')
            if dataset.map_hdf5_dataset_name:
                outfile.write('  MAP_HDF5_DATASET_NAME ' +
                              dataset.map_hdf5_dataset_name + '\n')
            if dataset.max_buffer_size:
                outfile.write('  MAX_BUFFER_SIZE ' +
                              strD(dataset.max_buffer_size) + '\n')
            if dataset.realization_dependent:
                outfile.write('  REALIZATION_DEPENDENT ' + '\n')
            outfile.write('END\n\n')

    # Adds a dataset object.
    def _add_dataset(self, dat=pdataset(), overwrite=False):
        # check if dataset already exists
        if isinstance(dat, pdataset):
            if dat.name in self.dataset.keys():
                if not overwrite:
                    warning = 'WARNING: A dataset with name \'' + \
                              str(dat.name) + '\' already exists. ' + \
                              'Use overwrite = True in add() to overwrite' + \
                              'the old dataset.'
                    print(warning,
                    build_warnings.append(warning))
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
            line = get_next_line(infile)

            try:
                key = line.strip().split()[0].lower()  # take first key word
            except IndexError:
                continue  # Read the next line if line is empty.
            if key == 'primary_species':
                while True:
                    line = get_next_line(infile)
                    if line.strip() in ['/', 'end']:
                        break
                    chem.primary_species_list.append(line.strip())
            elif key == 'skip':
                keep_reading_1 = True
                while keep_reading_1:
                    line1 = get_next_line(infile)
                    if line1.strip().split()[0].lower() == 'noskip':
                        keep_reading_1 = False
            elif key == 'immobile_species':
                while True:
                    subline = get_next_line(infile).strip().split()
                    subkey = subline[0]

                    if subkey in ['/', 'end']:
                        break
                    else:
                        chem.immobile_species_list.append(subkey)

            elif key == 'secondary_species':
                while True:
                    line = get_next_line(infile)
                    if line[0] in ['#', '!']:
                        continue
                    if line.strip() in ['/', 'end']:
                        break
                    chem.secondary_species_list.append(line.strip())
            elif key == 'gas_species':
                while True:
                    line = get_next_line(infile)
                    if line[0] in ['#', '!']:
                        continue
                    if line.strip() in ['/', 'end']:
                        break
                    chem.gas_species_list.append(line.strip())
            elif key == 'passive_gas_species':
                while True:
                    line = get_next_line(infile)
                    if line[0] in ['#', '!']:
                        continue
                    if line.strip() in ['/', 'end']:
                        break
                    chem.passive_gas_species_list.append(line.strip())
            elif key == 'active_gas_species':
                while True:
                    line = get_next_line(infile)
                    if line[0] in ['#', '!']:
                        continue
                    if line.strip() in ['/', 'end']:
                        break
                    chem.active_gas_species_list.append(line.strip())
            elif key == 'minerals':
                while True:
                    line = get_next_line(infile)
                    if line[0] in ['#', '!']:
                        continue
                    if line.strip() in ['/', 'end']:
                        break
                    chem.minerals_list.append(line.strip())
            elif key == 'mineral_kinetics':
                while True:
                    line = get_next_line(infile)  # get next line
                    # Check for comments

                    if line.strip() in ['/', 'end']:
                        break

                    mkinetic = pchemistry_m_kinetic()  # temporary object
                    mkinetic.rate_constant_list = []

                    # assign kinetic mineral name
                    mkinetic.name = line.strip()

                    # Write mineral attributes here
                    while True:
                        line = get_next_line(infile)

                        if line.strip().lower() in ['/', 'end']:
                            break

                        # key is a kinetic mineral attribute here
                        key = line.strip().split()[0].lower()  # take 1st

                        # Assigns the rest of the line
                        tstring = line.split()[1:]

                        # assign kinetic mineral attributes
                        if key == 'rate_constant':
                            for substring in tstring:
                                try:
                                    mkinetic.rate_constant_list.append(
                                        floatD(substring))
                                except ValueError:
                                    mkinetic.rate_constant_list.append(
                                        substring)
                        elif key == 'activation_energy':
                            try:
                                mkinetic.activation_energy = floatD(tstring[0])
                            except:
                                mkinetic.activation_energy = tstring[0]
                        elif key == 'affinity_threshold':
                            try:
                                mkinetic.affinity_threshold = floatD(
                                    tstring[0])
                            except:
                                mkinetic.affinity_threshold = tstring[0]
                        elif key == 'rate_limiter':
                            try:
                                mkinetic.rate_limiter = floatD(tstring[0])
                            except:
                                mkinetic.rate_limiter = tstring[0]
                        elif key == 'irreversible':
                            try:
                                mkinetic.irreversible = boolS(tstring[0])
                            except:
                                mkinetic.irreversible = tstring[0]
                        elif key == 'surface_area_porosity_power':
                            try:
                                mkinetic.surface_area_porosity_power = floatD(tstring[
                                                                              0])
                            except ValueError:
                                mkinetic.surface_area_porosity_power = tstring[
                                    0]
                        elif key == 'surface_area_vol_frac_power':
                            # Sample:
                            # SURFACE_AREA_VOL_FRAC_POWER 0.666667d0
                            try:
                                mkinetic.surface_area_vol_frac_power = floatD(tstring[
                                                                              0])
                            except ValueError:
                                mkinetic.surface_area_vol_frac_power = tstring[
                                    0]
                        elif key == 'prefactor':
                            # PREFACTOR is a unique card within the MINERAL_KINETICS block.
                            # PREFACTOR has its own set of cards, and own set of closing lines.
                            # In additional, PREFACTOR_SPECIES has its own set of parameters and closing lines.
                            # We're going deep into the rabbit hole.

                            # Create a new PREFACTOR
                            mk_pf = mkinetic.add_prefactor()

                            # while (PREFACTOR has not been closed)...
                            while True:

                                # Read & check for closure
                                pref_line = get_next_line(infile)

                                if pref_line.strip() in ['/', 'end']:
                                    break

                                # Capture the PREFACTOR card
                                pref_key = pref_line.strip().split()[0].lower()

                                pref_tstring = pref_line.split()[1:]

                                # Read keys...
                                if pref_key == 'rate_constant':
                                    # Example:
                                    #    RATE_CONSTANT 3.5d-8      mol/m^2-sec
                                    for pref_substring in pref_tstring:
                                        try:
                                            mk_pf.rate_constant.append(
                                                floatD(pref_substring))
                                        except ValueError:
                                            mk_pf.rate_constant.append(
                                                pref_substring)

                                if pref_key == 'activation_energy':
                                    # Example:
                                    #    ACTIVATION_ENERGY 0.31d0
                                    try:
                                        mk_pf.activation_energy = floatD(
                                            pref_tstring[0])
                                    except:
                                        mk_pf.activation_energy = \
                                                                pref_tstring[0]

                                elif pref_key == 'prefactor_species':
                                    # Example:
                                    #    PREFACTOR_SPECIES H+

                                    # Enter into the PREFACTOR_SPECIES block
                                    # Create a new PREFACTOR_SPECIES
                                    pref_spec_name = pref_line.strip().split()[
                                        1]
                                    mkpf_species = mk_pf.add_species(
                                        pref_spec_name)

                                    # while (PREFACTOR_SPECIES has not been
                                    # closed)...
                                    while True:
                                        prefspec_line = get_next_line(
                                            infile)  # get next line

                                        if prefspec_line.strip() in ['/', 'end']:
                                            break

                                        prefspec_key = prefspec_line.strip().split()[
                                            0].lower()

                                        prefspec_tstring = prefspec_line.split()[
                                            1:]

                                        if prefspec_key == 'alpha':
                                            # Example:
                                            #    ALPHA 0.37d0
                                            try:
                                                mkpf_species.alpha = floatD(
                                                    prefspec_tstring[0])
                                            except:
                                                mkpf_species.alpha = prefspec_tstring[
                                                    0]

                                        elif prefspec_key == 'beta':
                                            # Example:
                                            #    BETA 1.d0
                                            try:
                                                mkpf_species.beta = floatD(
                                                    prefspec_tstring[0])
                                            except:
                                                mkpf_species.beta = prefspec_tstring[
                                                    0]

                                        elif prefspec_key == 'attenuation_coef':
                                            # Example:
                                            #    ATTENUATION_COEF 1.d9
                                            try:
                                                mkpf_species.attenuation_coef = floatD(
                                                    prefspec_tstring[0])
                                            except:
                                                mkpf_species.attenuation_coef = prefspec_tstring[
                                                    0]

                    chem.m_kinetics_list.append(mkinetic)  # object assigned
            elif key == 'general_reaction':
                general_reaction = chem.add_general_reaction()
                while True:
                    subline = get_next_line(infile)
                    subkey = subline.split()[0].lower()

                    if subkey == 'reaction':
                        general_reaction.reaction = subline.strip().split(' ', 1)[
                            1]
                    elif subkey == 'forward_rate':
                        general_reaction.forward_rate = floatD(
                            subline.split()[-1])
                    elif subkey == 'backward_rate':
                        general_reaction.backward_rate = floatD(
                            subline.split()[-1])

                    if subline.strip() in ['/', 'end']:
                        break

            elif key == 'sorption':
                sorb = chem.add_sorption()
                while True:
                    subline = get_next_line(infile)
                    subkey = subline.split()[0].lower()

                    # Parse for ION_EXCHANGE_RXN
                    if subkey == 'ion_exchange_rxn':
                        irxn = sorb.add_ion_exchange_rxn()
                        while True:
                            subsubline = get_next_line(infile)
                            subsubkey = subsubline.split()[0].lower()
                            if subsubkey == 'cec':
                                if len(subsubline.split()[1:]) > 1:
                                    irxn.cec = subsubline.split()[1:]
                                    irxn.cec[0] = floatD(irxn.cec[0])
                                else:
                                    irxn.cec = subsubline.split()[1]
                            elif subsubkey == 'mineral':
                                irxn.mineral = subsubline.split()[1]
                            elif subsubkey == 'cations':
                                while True:
                                    catline = get_next_line(infile).split()
                                    catkey = catline[0].lower()
                                    if catkey.lower() in ['/', 'end']:
                                        break
                                    else:
                                        cat_name = catline[0]
                                        cat_val = floatD(catline[1])
                                        cat_ref = True if len(catline) >= 3 \
                                                       else False

                                        irxn.add_cation(
                                            name=cat_name, value=cat_val, reference=cat_ref)
                            if subsubkey.lower() in ['/', 'end']:
                                break
                    elif subkey == 'isotherm_reactions':

                        while True:
                            subsubline = get_next_line(infile)
                            subsubkey = subsubline.split()[0]

                            if subsubkey.lower() in ['/','end']:
                                break
                            else:
                                i_rxn = sorb.add_isotherm_reactions()
                                i_rxn.name = subsubkey

                                while True:
                                    iline = get_next_line(infile)
                                    ikey = iline.split()[0].lower()

                                    if ikey in ['/','end']:
                                        break
                                    elif ikey == 'type':
                                        i_rxn.ir_type = iline.split()[-1]
                                    elif ikey == 'kd_mineral_name':
                                        i_rxn.kd_mineral_name = iline.split()[-1]
                                    else:
                                        _split = iline.split()[1:]

                                        if len(_split) == 1:
                                            _value = floatD(_split[0])
                                            _unit = None
                                        else:
                                            _unit = _split[-1]
                                            _value = [floatD(_x) for _x in \
                                                      _split[:-1]]

                                        _coeff = Coeff(_value,unit=_unit)

                                        #_coeff = 
                                        if ikey == 'distribution_coefficient':
                                            i_rxn.distribution_coefficient = \
                                                  _coeff
                                        elif ikey == 'langmuir_b':
                                            i_rxn.langmuir_b = \
                                                  _coeff
                                        elif ikey == 'freundlich_n':
                                            i_rxn.freundlich_n = \
                                                  _coeff
                    elif subkey == 'surface_complexation_rxn':
                        scr = sorb.add_surface_complexation_rxn()
                        while True:
                            _line = get_next_line(infile)
                            _key = _line.strip().lower().split()[0]

                            if _key == 'equilibrium':
                                scr.equilibrium = True
                            elif _key == 'multirate_kinetic':
                                scr.multirate_kinetic = True
                            elif _key == 'kinetic':
                                scr.kinetic = True
                            elif _key == 'mineral':
                                scr.mineral = self.splitter(_line)
                            elif _key == 'site':
                                _name = _line.split()[1]
                                _value = floatD(_line.split()[2])
                                scr.site = [_name,_value]
                            elif _key == 'complexes':
                                while True:
                                    c_line = get_next_line(infile).strip()

                                    if c_line in ['/','end']:
                                        break
                                    else:
                                        scr.complexes.append(c_line)
                            elif _key == 'complex_kinetics':
                                pass
                            elif _key == 'site_fraction':
                                _values = get_next_line(infile)
                                _values = [floatD(x) for x in _values]
                                scr.site_fraction = _values
                            elif key == 'rates':
                                _values = get_next_line(infile)
                                _values = [floatD(x) for x in _values]
                                scr.rates = _values
                            elif _key in ['/','end']:
                                break

                    if subkey.lower() in ['/', 'end']:
                        break

            elif key == 'immobile_decay_reaction':
                id_rxn = chem.add_immobile_decay_reaction()
                while True:
                    subline = get_next_line(infile).strip().split()
                    subkey = subline[0].lower()

                    if subkey == 'species_name':
                        id_rxn.species_name = subline[-1]
                    elif subkey == 'rate_constant':
                        id_rxn.rate_constant = floatD(subline[-1])
                    elif subkey == 'half_life':
                        id_rxn.half_life = floatD(subline[-1])
                    elif subkey in ['/', 'end']:
                        break

            elif key == 'radioactive_decay_reaction':
                rad_rxn = chem.add_radioactive_decay_reaction()
                while True:
                    subline = get_next_line(infile).strip().split()
                    subkey = subline[0].lower()

                    if subkey == 'reaction':
                        rad_rxn.reaction = ' '.join(subline[1:])
                    elif subkey == 'rate_constant':
                        rad_rxn.rate_constant = floatD(subline[1]) if len(subline) <= 2 \
                            else [floatD(subline[1]), subline[2]]
                    elif subkey == 'half_life':
                        rad_rxn.half_life = floatD(subline[1]) if len(subline) <= 2 \
                            else [floatD(subline[1]), subline[2]]
                    elif subkey in ['/', 'end']:
                        break

            elif key == 'microbial_reaction':
                bio_rxn = chem.add_microbial_reaction()

                while True:
                    subline = get_next_line(infile).strip().split()
                    subkey = subline[0].lower()

                    if subkey == 'reaction':
                        bio_rxn.reaction = ' '.join(subline[1:])
                    elif subkey == 'rate_constant':
                        bio_rxn.rate_constant = floatD(subline[1])

                    elif subkey == 'monod':
                        monod = bio_rxn.add_monod()
                        while True:
                            subsubline = get_next_line(infile).strip().split()
                            subsubkey = subsubline[0].lower()

                            if subsubkey == 'species_name':
                                monod.species_name = subsubline[1]
                            elif subsubkey == 'half_saturation_constant':
                                monod.half_saturation_constant = floatD(
                                    subsubline[1])
                            elif subsubkey == 'threshold_concentration':
                                monod.threshold_concentration = floatD(
                                    subsubline[1])
                            elif subsubkey in ['/', 'end']:
                                break

                    elif subkey == 'inhibition':
                        inhibit = bio_rxn.add_inhibition()
                        while True:
                            subsubline = get_next_line(infile).strip().split()
                            subsubkey = subsubline[0].lower()

                            if subsubkey == 'species_name':
                                inhibit.species_name = subsubline[1]
                            elif subsubkey == 'type':
                                inhibit.inhibition_type = subsubline[1]
                            elif subsubkey == 'inhibition_constant':
                                inhibit.inhibition_constant = floatD(
                                    subsubline[1])
                            elif subsubkey in ['/', 'end']:
                                break

                    elif subkey == 'biomass':
                        biomass = bio_rxn.add_biomass()
                        while True:
                            subsubline = get_next_line(infile).strip().split()
                            subsubkey = subsubline[0].lower()

                            if subsubkey == 'species_name':
                                biomass.species_name = subsubline[1]
                            elif subsubkey == 'yield':
                                biomass.biomass_yield = floatD(subsubline[1])
                            elif subsubkey in ['/', 'end']:
                                break

                    elif subkey in ['/', 'end']:
                        break

            elif key == 'database':
                chem.database = self.splitter(line)  # take last word
            elif key == 'log_formulation':
                chem.log_formulation = True
            elif key == 'use_full_geochemistry':
                chem.use_full_geochemistry = True
            elif key in ['activity_water', 'activity_h2o']:
                chem.activity_water = True
            elif key == 'update_porosity':
                chem.update_porosity = True
            elif key == 'update_permeability':
                chem.update_permeability = True
            elif key == 'update_mineral_surface_area':
                chem.update_mineral_surface_area = True
            elif key == 'no_bdot':
                chem.no_bdot = True
            elif key == 'no_checkpoint_act_coefs':
                chem.no_checkpoint_act_coefs = True
            elif key == 'activity_coefficients':
                chem.activity_coefficients = self.splitter(line)
            elif key == 'truncate_concentration':
                chem.truncate_concentration = self.splitter(line)
            elif key == 'max_dlnc':
                chem.max_dlnc = self.splitter(line)
            elif key == 'max_residual_tolerance':
                chem.max_residual_tolerance = self.splitter(line)
            elif key == 'max_relative_change_tolerance':
                chem.max_relative_change_tolerance = self.splitter(line)
            elif key == 'molal':
                chem.molal = True
            elif key == 'output':
                while True:
                    line = get_next_line(infile)
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
                                 overwrite=False):
        # Adds a mineral_kinetic object

        chemistry = self.chemistry

        # check if m_kinetic already exists
        if isinstance(m_kinetic, pchemistry_m_kinetic):
            if m_kinetic.name in self.m_kinetic.keys():
                if not overwrite:
                    warning = 'WARNING: A m_kinetic with name \'' + \
                              str(m_kinetic.name) + '\' already exists in ' + \
                              'chemistry. Mineral_Kinetic ' + \
                              'will not be defined, use ' + \
                              'overwrite = True in add() ' + \
                              'to overwrite the old ' + \
                              'm_kinetic.'
                    print(warning)
                    print("")
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
        if not isinstance(c.primary_species_list, list):
            raise PyFLOTRAN_ERROR('A list needs to be passed ' +
                                  'to primary_species_list!')
        if c.primary_species_list:
            outfile.write('  PRIMARY_SPECIES\n')
            for p in c.primary_species_list:
                # p = primary_specie in primary_species_list
                outfile.write('    ' + p + '\n')
            outfile.write('  /\n')
        if not isinstance(c.secondary_species_list, list):
            raise PyFLOTRAN_ERROR('A list needs to be passed ' +
                                  'to secondary_species_list!')
        if c.secondary_species_list:
            outfile.write('  SECONDARY_SPECIES\n')
            for s in c.secondary_species_list:  # s = secondary_specie
                outfile.write('    ' + s + '\n')
            outfile.write('  /\n')

        if not isinstance(c.gas_species_list, list):
            raise PyFLOTRAN_ERROR('A list needs to be passed ' +
                                  'to gas_species_list!')
        if c.gas_species_list:
            outfile.write('  GAS_SPECIES\n')
            for g in c.gas_species_list:  # s = gas_specie
                outfile.write('    ' + g + '\n')
            outfile.write('  /\n')

        if not isinstance(c.passive_gas_species_list, list):
            raise PyFLOTRAN_ERROR('A list needs to be passed ' +
                                  'to passive_gas_species_list!')
        if c.passive_gas_species_list:
            outfile.write('  PASSIVE_GAS_SPECIES\n')
            for g in c.passive_gas_species_list:
                outfile.write('    ' + g + '\n')
            outfile.write('  /\n')

        if not isinstance(c.active_gas_species_list, list):
            raise PyFLOTRAN_ERROR('A list needs to be passed ' +
                                  'to active_gas_species_list!')
        if c.active_gas_species_list:
            outfile.write('  ACTIVE_GAS_SPECIES\n')
            for g in c.active_gas_species_list:
                outfile.write('    ' + g + '\n')
            outfile.write('  /\n')

        if not isinstance(c.minerals_list, list):
            raise PyFLOTRAN_ERROR('A list needs to be passed ' +
                                  'to mineral_list!')
        if c.minerals_list:
            outfile.write('  MINERALS\n')
            for m in c.minerals_list:  # m = mineral
                outfile.write('    ' + m + '\n')
            outfile.write('  /\n')
        if not isinstance(c.m_kinetics_list, list):
            raise PyFLOTRAN_ERROR('A list needs to be passed ' +
                                  'to m_kinetics_list!')

        if c.immobile_species_list:
            outfile.write('  IMMOBILE_SPECIES\n')
            outfile.write(
                ''.join(['    %s\n' % b for b in c.immobile_species_list]))
            outfile.write('  /\n')

        if c.immobile_decay_reaction:
            for id_rxn in c.immobile_decay_reaction:
                id_rxn._write(outfile)

        if c.radioactive_decay_reaction:
            for rd_rxn in c.radioactive_decay_reaction:
                rd_rxn._write(outfile)

        if c.microbial_reaction:
            for mb_rxn in c.microbial_reaction:
                mb_rxn._write(outfile)

        if c.m_kinetics_list:
            outfile.write('  MINERAL_KINETICS\n')
            for mk in c.m_kinetics_list:  # mk = mineral_kinetics
                outfile.write('    ' + mk.name + '\n')
                if not isinstance(mk.rate_constant_list, list):
                    raise PyFLOTRAN_ERROR('A list needs to be passed ' +
                                          'to rate_constant_list!')

                # Write out MINERAL_KINETICS RATE_CONSTANT
                if mk.rate_constant_list:
                    outfile.write('      RATE_CONSTANT ')
                    for rate in mk.rate_constant_list:
                        try:
                            outfile.write(strD(rate) + ' ')
                        except TypeError:
                            outfile.write(rate + ' ')
                    outfile.write('\n')

                if mk.activation_energy is not None:
                    outfile.write('      ACTIVATION_ENERGY ' +
                                  strD(mk.activation_energy) + '\n')
                if mk.affinity_threshold is not None:
                    outfile.write('      AFFINITY_THRESHOLD ' +
                                  strD(mk.affinity_threshold) + '\n')
                if mk.rate_limiter is not None:
                    outfile.write('      RATE_LIMITER ' +
                                  strD(mk.rate_limiter) + '\n')
                if mk.irreversible is not None:
                    outfile.write('      IRREVERSIBLE ' +
                                  strB(mk.irreversible) + '\n')
                if mk.surface_area_porosity_power is not None:
                    outfile.write('      SURFACE_AREA_POROSITY_POWER ' +
                                  strD(mk.surface_area_porosity_power) + '\n')
                if mk.surface_area_vol_frac_power is not None:
                    outfile.write('      SURFACE_AREA_VOL_FRAC_POWER ' +
                                  strD(mk.surface_area_vol_frac_power) + '\n')

                # Check if ANY prefactor-related attributes are set
                has_prefactor = (len(mk.prefactors) > 0)

                #==================================================
                # Write out PREFACTOR and PREFACTOR_SPECIES
                #==================================================

                if has_prefactor:
                    # Iterate over each PREFACTOR object
                    for pf in mk.prefactors:

                        # Does the PREFACTOR object have any PREFACTOR_SPECIES?
                        has_pfspecies = (len(pf.pf_species) > 0)

                        # Write out attributes
                        outfile.write('      PREFACTOR\n')

                        # Combine rate constant coefficient and units into
                        # string if exists
                        if pf.rate_constant is not None:
                            try:
                                rc_str = strD(pf.rate_constant[
                                              0]) + ' ' + pf.rate_constant[1]
                            except TypeError:
                                rc_str = ' '.join([str(rc)
                                                   for rc in pf.rate_constant])
                            outfile.write(
                                '        RATE_CONSTANT ' + rc_str + ' ' + '\n')

                        # Write out activation energy
                        if pf.activation_energy is not None:
                            outfile.write(
                                '        ACTIVATION_ENERGY ' + strD(pf.activation_energy) + '\n')

                        # If this PREFACTOR card has one or more
                        # PREFACTOR_SPECIES subcards...
                        if has_pfspecies:
                            # Iterate over each of them and write out
                            # attributes
                            for pf_spec in pf.pf_species:
                                outfile.write(
                                    '        PREFACTOR_SPECIES ' + pf_spec.name + '\n')
                                if pf_spec.alpha is not None:
                                    outfile.write(
                                        '          ALPHA ' + strD(pf_spec.alpha) + '\n')
                                if pf_spec.beta is not None:
                                    outfile.write(
                                        '          BETA ' + strD(pf_spec.beta) + '\n')
                                if pf_spec.attenuation_coef is not None:
                                    outfile.write(
                                        '          ATTENUATION_COEF ' + strD(pf_spec.attenuation_coef) + '\n')
                                # Close PREFACTOR_SPECIES
                                outfile.write('        /\n')

                        outfile.write('      /\n')  # Close PREFACTOR
                outfile.write('    /\n')  # Close mineral name
            outfile.write('  /\n')  # Close MINERAL_K

        # Write out the GENERAL_REACTION block if it exists
        if c.general_reaction is not None:
            outfile.write('  GENERAL_REACTION\n')
            if c.general_reaction.reaction is not None:
                outfile.write('    REACTION ' +
                              c.general_reaction.reaction + '\n')
            if c.general_reaction.forward_rate is not None:
                outfile.write('    FORWARD_RATE ' +
                              strD(c.general_reaction.forward_rate) + '\n')
            if c.general_reaction.backward_rate is not None:
                outfile.write('    BACKWARD_RATE ' +
                              strD(c.general_reaction.backward_rate) + '\n')
            outfile.write('  /\n')

        if c.sorption is not None:
            c.sorption._write(outfile)

        if c.database:
            outfile.write('  DATABASE ' + c.database + '\n')
        if c.log_formulation:
            outfile.write('  LOG_FORMULATION\n')
        if c.activity_water:
            outfile.write('  ACTIVITY_WATER\n')
        if c.use_full_geochemistry:
            outfile.write('  USE_FULL_GEOCHEMISTRY\n')
        if c.update_mineral_surface_area:
            outfile.write('  UPDATE_MINERAL_SURFACE_AREA\n')
        if c.no_bdot:
            outfile.write('  NO_BDOT\n')
        if c.no_checkpoint_act_coefs:
            outfile.write('  NO_CHECPOINT_ACT_COEFS\n')
        if c.truncate_concentration:
            outfile.write('  TRUNCATE_CONCENTRATION ' +
                          c.truncate_concentration + '\n')
        if c.max_residual_tolerance:
            outfile.write('  MAX_RESIDUAL_TOLERANCE ' +
                          c.max_residual_tolerance + '\n')
        if c.max_relative_change_tolerance:
            outfile.write('  MAX_RELATIVE_CHANGE_TOLERANCE ' +
                          c.max_relative_change_tolerance + '\n')
        if c.max_dlnc:
            outfile.write('  MAX_DLNC ' + c.max_dlnc + '\n')
        if c.update_permeability:
            outfile.write('  UPDATE_PERMEABILITY\n')
        if c.update_porosity:
            outfile.write('  UPDATE_POROSITY\n')
        if c.activity_coefficients:
            outfile.write('  ACTIVITY_COEFFICIENTS ' +
                          c.activity_coefficients.upper() + '\n')
        if c.molal:
            outfile.write('  MOLAL\n')
        if not isinstance(c.output_list, list):
            raise PyFLOTRAN_ERROR('A list needs to be passed ' +
                                  'to output_list!')
        if c.output_list:
            outfile.write('  OUTPUT\n')
            for o in c.output_list:  # o = output in in output_list
                outfile.write('    ' + o + '\n')
            outfile.write('  /\n')
        outfile.write('END\n\n')

    def _read_transport(self, infile, line):
        p = ptransport('')
        # Transport Condition name passed in.
        np_name = self.splitter(line).lower()
        np_type = p.type
        np_constraint_list_value = []
        np_constraint_list_type = []
        np_time_units = ''

        keep_reading = True

        while keep_reading:  # Read through all cards
            line = get_next_line(infile)  # get next line
            key = line.split()[0].lower()  # take first key word

            if key == 'type':
                if len(line.split()) == 2:
                    # Only Assign if 2 words are on the line
                    np_type = self.splitter(line)  # take last word
            elif key == 'time_units':
                np_time_units = self.splitter(line)
            elif key == 'constraint_list':
                keep_reading_2 = True
                line = get_next_line(infile)
                while keep_reading_2:
                    try:
                        # print
                        # np_constraint_list_value,line.split()[0].lower()
                        np_constraint_list_value.append(
                            floatD(line.split()[0].lower()))
                        # Read 1st word online
                        np_constraint_list_type.append(
                            line.split()[1].lower())  # Read 2nd word on line
                    except:
                        raise PyFLOTRAN_ERROR('constraint_list_value and ' +
                                              'constraint_list_type ' +
                                              'requires at least one' +
                                              'value. Value should = ' +
                                              'Number and type should = ' +
                                              ' String\n')

                    line = get_next_line(infile)
                    # Used to stop loop when / or end is read
                    key = line.split()[0].lower()
                    if key in ['/', 'end']:
                        keep_reading_2 = False
            elif key in ['/', 'end']:
                keep_reading = False

        # Create an empty transport condition and assign the values read in
        new_transport = ptransport(np_name, np_type, np_constraint_list_value,
                                   np_constraint_list_type, np_time_units)
        self.add(new_transport)

    # Adds a transport object.
    def _add_transport(self, transport=ptransport(), overwrite=False):
        # check if transport already exists
        if isinstance(transport, ptransport):
            if transport.name in self.transport.keys():
                if not overwrite:
                    warning = 'WARNING: A transport with name \'' + \
                              str(transport.name) + '\' already exists.' + \
                              'transport will not be defined, ' + \
                              'use overwrite = True in add() to ' + \
                              'overwrite the' + \
                              'old transport.'
                    print(warning,
                    build_warnings.append(warning))
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
                raise PyFLOTRAN_ERROR(
                    'transport_condition[' +
                    str(tl.index(t)) + '].name is required.\n')
            if t.type.lower() in transport_condition_types_allowed:
                outfile.write('  TYPE ' + t.type.lower() + '\n')
            else:
                print('       valid transport_condition.types:', \
                    transport_condition_types_allowed, '\n')
                raise PyFLOTRAN_ERROR(
                    'transport.type: \'' + t.type + '\' is invalid.')
            try:
                if t.time_units:
                    if t.time_units in time_units_allowed:
                        outfile.write('  TIME_UNITS ' + t.time_units + '\n')
                    else:
                        raise PyFLOTRAN_ERROR('Incorrect transport condition' +
                                              ' time units!\n')
                outfile.write('  CONSTRAINT_LIST\n')

                clv = t.constraint_list_value
                clt = t.constraint_list_type

                for i, a_clv in enumerate(clv):
                    if a_clv is not None:
                        outfile.write('    ' + strD(a_clv))
                    if clt[i] is not None:
                        if i == len(clv) - 1:
                            outfile.write(' ' + str(clt[i]).lower())
                        else:
                            outfile.write(' ' + str(clt[i]).lower() + '\n')
                    else:
                        raise PyFLOTRAN_ERROR('transport[' +
                                              str(tl.index(t)) +
                                              '].constraint_list_type[' +
                                              str(clt.index(i)) +
                                              '] is required to have ' +
                                              'a value when' +
                                              ' transport.' +
                                              'constraint_list_value does.')
            except:
                raise PyFLOTRAN_ERROR('transport.constraint_list_value ' +
                                      'and transport.constraint_list_type ' +
                                      'should be' +
                                      'in list format, be equal in length, ' +
                                      'and have at least one value.\n')
            outfile.write('\n  /\n')  # END FOR CONSTRAINT_LIST
            outfile.write('END\n\n')  # END FOR TRANSPORT_CONDITION

    def _read_constraint(self, infile, line):
        constraint = pconstraint()
        # constraint name passed in.
        constraint.name = self.splitter(line).lower()
        constraint.concentration_list = []
        constraint.mineral_list = []

        keep_reading = True

        while keep_reading:  # Read through all cards
            line = get_next_line(infile)
            key = line.split()[0].lower()  # take first key word

            if key == 'concentrations':
                while True:
                    line = get_next_line(infile)
                    tstring = line.split()
                    # Convert line to temporary list of strings

                    if line.strip().lower() in ['/', 'end']:
                        break  # Stop loop if line is a / or 'end'

                    # Check for comments
                    if line[0] == "#":
                        continue

                    concentrations = pconstraint_concentration()

                    # Assign concentrations - one line
                    try:
                        concentrations.pspecies = tstring[0]
                        concentrations.value = floatD(tstring[1])
                        concentrations.constraint = tstring[2]
                        concentrations.element = tstring[3]
                    except IndexError:
                        # No assigning is done if a value doesn't exist while
                        # being read in.
                        pass
                    constraint.concentration_list.append(concentrations)

            elif key == 'free_ion_guess':
                while True:
                    line = get_next_line(infile)
                    tstring = line.split()

                    if line.strip().lower() in ['/','end']:
                        break

                    assert len(tstring) == 2,'FREE_ION_GUESS missing parameters'

                    _name = tstring[0]
                    _value = floatD(tstring[1])
                    _ion = pconstraint.pfree_ion(_name,_value)

                    constraint.free_ion_guess_list.append(_ion)

            elif key == 'minerals':
                while True:
                    line = get_next_line(infile)
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
                                if len(tstring) > 5:
                                    mineral.surface_area_units = tstring[5]
                            else:
                                mineral.surface_area = floatD(tstring[3])
                                mineral.surface_area_units = tstring[4]
                        else:
                            mineral.volume_fraction = floatD(tstring[1])
                            if tstring[2].lower() == 'dataset':
                                mineral.surface_area = tstring[3]
                                if len(tstring) > 4:
                                    mineral.surface_area_units = tstring[4]
                            else:
                                mineral.surface_area = floatD(tstring[2])
                                mineral.surface_area_units = tstring[3]

                    except IndexError:
                        # No assigning is done if a value doesn't exist while
                        # being read in.
                        pass

                    constraint.mineral_list.append(mineral)

            elif key in ['/', 'end']:
                keep_reading = False

        self.add(constraint)

    # Adds a constraint object.
    def _add_constraint(self, constraint=pconstraint(), overwrite=False):
        # check if constraint already exists
        if isinstance(constraint, pconstraint):
            if constraint.name in self.constraint.keys():
                if not overwrite:
                    warning = 'WARNING: A constraint with name \'' + \
                              str(constraint.name) + '\' already exists. ' + \
                              'constraint will not be defined, ' + \
                              'use overwrite = True in add() to ' + \
                              'overwrite the old ' + \
                              'constraint.'
                    print(warning,
                    build_warnings.append(warning))
                    return
                else:
                    self.delete(self.constraint[constraint.name])

        if constraint not in self.constraint_list:
            self.constraint_list.append(constraint)

    def _delete_constraint(self, constraint=pconstraint()):
        self.constraint_list.remove(constraint)

    # Adds a constraint_concentration object
    def _add_constraint_concentration(self,
                                      constraint_concentration=pconstraint_concentration(),
                                      index='',
                                      overwrite=False):

        # check if constraint.name was specified
        if index:
            if isinstance(index, str):
                constraint = self.constraint.get(
                    index)
                # Assign constraint object to existing constraint object with
                # string type name/index
                if not constraint:
                    # Occurs if index/string is not found in constraint object
                    print('WARNING: a constraint object with ' + \
                          'constraint.name', index, 'was not found. ' + \
                                                    ' Current found' \
                                                    ' entries are:', \
                        self.constraint.keys(), \
                        'pconstraint_concentration was not added.\n')
                    return

            elif isinstance(index, pconstraint):
                constraint = index
                # Assigns if index is the constraint object reference
        else:  # Set constraint to last constraint object in list
            constraint = self.constraint_list[-1]

        # check if constraint_concentration already exists
        if isinstance(constraint_concentration, pconstraint_concentration):
            if constraint_concentration.pspecies in \
                    self.constraint_concentration(constraint).keys():
                if not overwrite:
                    warning = 'WARNING: A constraint_concentration ' + \
                              'with pspecies \'' + \
                              str(constraint_concentration.pspecies) + \
                              '\' already exists in constraint with name \'' + \
                              str(constraint.name) + \
                              '\'. constraint_concentration will not be ' + \
                              'defined, use overwrite = True in ' \
                              'add() to overwrite the old ' + \
                              'constraint_concentration. ' + \
                              'Use constraint=\'name\' ' \
                              'if you want to specify the constraint object to ' \
                              'add constraint_concentration to.'
                    print(warning,
                    build_warnings.append(warning))
                    return
                else:  # Executes if overwrite = True
                    self.delete(self.constraint_concentration(constraint)[
                        constraint_concentration.pspecies],
                        constraint)

        # Add constraint_concentration to constraint (as a sub-class)
        # if constraint_concentration does not exist in
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
            if c.secondary_continuum:
                outfile.write('SECONDARY_CONSTRAINT ')
            else:
                outfile.write('CONSTRAINT ')
            if c.name:
                outfile.write(c.name.lower() + '\n')
            else:
                raise PyFLOTRAN_ERROR(
                    'constraint_list[' +
                    str(cl.index(c)) + '].name is required.')

            outfile.write('  CONCENTRATIONS\n')

            for concn in c.concentration_list:
                # concn = concentration, c = constraint
                if concn.pspecies:
                    outfile.write('    ' + concn.pspecies.ljust(10))
                if concn.value:
                    outfile.write('  ' + strD(concn.value).ljust(10))
                else:
                    raise PyFLOTRAN_ERROR('invalid concentration value!')
                if concn.constraint:
                    outfile.write('  ' + concn.constraint.ljust(3))
                if concn.element:
                    outfile.write('  ' + concn.element.ljust(3))
                outfile.write('\n')

            outfile.write('  /\n')  # END for concentrations

            if c.free_ion_guess_list:
                outfile.write('  FREE_ION_GUESS\n')
                for ion in c.free_ion_guess_list:
                    outfile.write('    %s %s\n' % (ion.name,strD(ion.value)))
                outfile.write('  /\n')

            if c.mineral_list:
                outfile.write('  MINERALS\n')
                for mineral in c.mineral_list:
                    if mineral.name:
                        outfile.write('    ' + mineral.name.ljust(15))
                    if type(mineral.volume_fraction) is str:
                        outfile.write('  ' + 'DATASET ' +
                                      mineral.volume_fraction)
                    else:
                        outfile.write(
                            '  ' + strD(mineral.volume_fraction).ljust(5))
                    if type(mineral.surface_area) is str:
                        outfile.write('  ' + 'DATASET ' + mineral.surface_area)
                    else:
                        outfile.write(
                            '  ' + strD(mineral.surface_area).ljust(5))
                    if mineral.surface_area_units:
                        if type(mineral.surface_area_units) is str:
                            outfile.write(
                                '  ' + mineral.surface_area_units.ljust(5))
                        else:
                            raise PyFLOTRAN_ERROR(
                                'mineral surface area units have to be string!')
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
        outfile.write(ws)

    def _write_hydroquake(self, outfile):
        self._header(outfile, headers['hydroquake'])
        outfile.write('HYDROQUAKE\n')
        if self.hydroquake.mapping_file:
            outfile.write('  HYDROQUAKE_MAPPING_FILE ' +
                          self.hydroquake.mapping_file + '\n')
        if self.hydroquake.time_scaling:
            outfile.write('  TIME_SCALING ' +
                          strD(self.hydroquake.time_scaling) + '\n')
        if self.hydroquake.pressure_scaling:
            outfile.write('  PRESSURE_SCALING ' +
                          strD(self.hydroquake.pressure_scaling) + '\n')
        outfile.write('END_HYDROQUAKE')

    def _read_geomechanics_grid(self, infile, line):
        grid = pgeomech_grid()  # assign defaults before reading in values

        keep_reading = True
        while keep_reading:
            line = get_next_line(infile)
            key = line.strip().split()[0].lower()  # take first keyword
            if key in ['#']:
                pass
            elif key == 'type':
                if line.strip().split()[1].lower() != 'unstructured':
                    raise PyFLOTRAN_ERROR(
                        'Geomechanics grid can only be unstructured!')
                else:
                    grid.dirname = ''
                    grid.grid_filename = line.strip().split()[-1]
            elif key == 'gravity':
                grid.gravity = []
                grid.gravity.append(floatD(line.split()[1]))
                grid.gravity.append(floatD(line.split()[2]))
                grid.gravity.append(floatD(line.split()[3]))

            elif key in ['/', 'end']:
                keep_reading = False
        self.geomech_grid = grid

    def _write_geomechanics_grid(self, outfile):
        self._header(outfile, headers['geomechanics_grid'])
        outfile.write('GEOMECHANICS_GRID\n')
        if self.geomech_grid.dirname:
            outfile.write('  TYPE unstructured ' + self.geomech_grid.dirname +
                          '/' + self.geomech_grid.grid_filename + '\n')
        else:
            outfile.write('  TYPE unstructured ' +
                          self.geomech_grid.grid_filename + '\n')
        outfile.write('  GRAVITY ')
        for item in self.geomech_grid.gravity:
            outfile.write(strD(item) + ' ')
        outfile.write('\n')
        outfile.write('END\n\n')

    def _write_geomechanics(self, outfile):
        self._header(outfile, headers['geomechanics'])
        outfile.write('GEOMECHANICS\n\n')
        if self.geomech_grid.grid_filename:
            self._write_geomechanics_grid(outfile)
        self._write_geomechanics_subsurface_coupling(outfile)
        self._write_geomechanics_regression(outfile)
        self._write_geomechanics_time(outfile)
        self._write_geomechanics_region(outfile)
        self._write_geomechanics_condition(outfile)
        self._write_geomechanics_boundary_condition(outfile)
        self._write_geomechanics_strata(outfile)
        self._write_geomechanics_output(outfile)
        if self.geomech_proplist:
            self._write_geomechanics_prop(outfile)
        outfile.write('END_GEOMECHANICS')

    def _write_geomechanics_region(self, outfile):
        self._header(outfile, headers['geomechanics_region'])

        # Write out all valid region object entries with Region as Key word
        for region in self.regionlist:
            if region.pm in ['geomech', 'geomechanics', 'geo']:
                outfile.write('GEOMECHANICS_REGION ')
                outfile.write(region.name.lower() + '\n')
                if region.filename:
                    outfile.write('  FILE ' + region.filename + '\n')
                else:
                    raise PyFLOTRAN_ERROR('Geomechanics regions can only ' +
                                          'be defined using files ' +
                                          'with list of vertices')
                outfile.write('END\n\n')

    def _write_geomechanics_condition(self, outfile):
        self._header(outfile, headers['geomechanics_condition'])

        def check_condition_type(condition_name, condition_type):
            if condition_name.upper()[:-2] == 'DISPLACEMENT':
                if condition_type.lower() in pressure_types_allowed:
                    outfile.write(condition_type.lower())
                else:
                    raise PyFLOTRAN_ERROR(
                        'geomechanics condition type ' + condition_type +
                        '\' is invalid.')
                return 0  # Break out of function
            elif condition_name.upper()[:-2] == 'FORCE':
                if condition_type.lower() in pressure_types_allowed:
                    outfile.write(condition_type.lower())
                else:
                    raise PyFLOTRAN_ERROR(
                        'geomechanics condition type ' + condition_type +
                        '\' is invalid.')
                return 0  # Break out of function
            else:
                pass

        for flow in self.flowlist:
            if flow.pm in ['geomech', 'geomechanics', 'geo']:
                outfile.write('GEOMECHANICS_CONDITION ' +
                              flow.name.lower() + '\n')
                outfile.write('  TYPE\n')
                # variable name and type from lists go here
                for a_flow in flow.varlist:
                    if a_flow.name.upper() in geomech_condition_type_names_allowed:
                        outfile.write('    ' + a_flow.name.upper() + ' ')
                    else:
                        raise PyFLOTRAN_ERROR(
                            'geomechanics condition ' +
                            a_flow.name + ' is invalid.')

                    # Checks a_flow.type and performs write or error reporting
                    check_condition_type(a_flow.name, a_flow.type)
                    outfile.write('\n')

                outfile.write('  END\n')

                # variable name and values from lists along with units go here
                for a_flow in flow.varlist:
                    if a_flow.valuelist:
                        outfile.write('    ' + a_flow.name.upper())
                        for flow_val in a_flow.valuelist:
                            outfile.write(' ' + strD(flow_val))
                        outfile.write('\n')
                outfile.write('END\n\n')

    def _write_geomechanics_boundary_condition(self, outfile):
        self._header(outfile, headers['geomechanics_boundary_condition'])

        # Write all boundary conditions to file
        for b in self.boundary_condition_list:  # b = boundary_condition
            if b.geomech:
                if b.name:
                    outfile.write('GEOMECHANICS_BOUNDARY_CONDITION ' +
                                  b.name.lower() + '\n')
                else:
                    raise PyFLOTRAN_ERROR(
                        'Give a name for geomechanics boundary condition!')
                outfile.write('  GEOMECHANICS_CONDITION ' +
                              b.geomech.lower() + '\n')
                if b.region:
                    outfile.write('  GEOMECHANICS_REGION ' +
                                  b.region.lower() + '\n')
                else:
                    raise PyFLOTRAN_ERROR(
                        'boundary_condition.region is required')
                outfile.write('END\n\n')

    def _write_geomechanics_strata(self, outfile):
        self._header(outfile, headers['geomechanics_strata'])

        # Write out strata condition variables
        for strata in self.strata_list:
            if strata.pm in ['geomech', 'geo', 'geomechanics']:
                outfile.write('GEOMECHANICS_STRATA\n')
                if strata.region:
                    outfile.write('  GEOMECHANICS_REGION ' +
                                  strata.region.lower() + '\n')
                else:
                    raise PyFLOTRAN_ERROR('strata.region is required')
                if strata.material:
                    outfile.write('  GEOMECHANICS_MATERIAL ' +
                                  strata.material.lower() + '\n')
                else:
                    raise PyFLOTRAN_ERROR('strata.material is required')
                outfile.write('END\n\n')

    def _read_geomechanics_subsurface_coupling(self, infile, line):
        coupling = pgeomech_subsurface_coupling()
        coupling.coupling_type = line.strip().split()[-1].lower()
        keep_reading = True
        while keep_reading:
            line = get_next_line(infile)
            key = line.strip().split()[0].lower()  # take first keyword
            if key in ['#']:
                pass
            elif key == 'mapping_file':
                coupling.mapping_file = line.strip().split()[-1]
            elif key in ['/', 'end']:
                keep_reading = False
        self.geomech_subsurface_coupling = coupling

    def _write_geomechanics_subsurface_coupling(self, outfile):
        self._header(outfile, headers['geomechanics_subsurface_coupling'])
        if self.geomech_subsurface_coupling.coupling_type not in \
                geomech_subsurface_coupling_types_allowed:
            raise PyFLOTRAN_ERROR(
                'Incorrect GEOMECHANICS_SUBSURFACE_COUPLING type!')
        outfile.write('GEOMECHANICS_SUBSURFACE_COUPLING ' +
                      self.geomech_subsurface_coupling.coupling_type.upper())
        outfile.write('\n  ')
        outfile.write('MAPPING_FILE ' +
                      self.geomech_subsurface_coupling.mapping_file)
        outfile.write('\n')
        outfile.write('END\n\n')

    def _read_geomechanics_time(self, infile, line):
        time = pgeomech_time()
        keep_reading = True
        while keep_reading:
            line = get_next_line(infile)
            key = line.strip().split()[0].lower()  # take first keyword
            if key in ['#']:
                pass
            elif key == 'coupling_timestep_size':
                time.coupling_timestep = []
                time.coupling_timestep.append(line.strip().split()[1])
                if len(line.strip().split()) > 2:
                    if line.strip().split()[2] in time_units_allowed:
                        time.coupling_timestep.append(
                            line.strip().split()[2])
                    else:
                        raise PyFLOTRAN_ERROR(
                            'Unknown units used for ' +
                            'geomechanics_coupling_timestep_size!')
            elif key in ['/', 'end']:
                keep_reading = False
        self.geomech_time = time

    def _write_geomechanics_time(self, outfile):
        self._header(outfile, headers['geomechanics_time'])
        outfile.write('GEOMECHANICS_TIME\n  ')
        outfile.write('COUPLING_TIMESTEP_SIZE ' +
                      strD(self.geomech_time.coupling_timestep[0]) +
                      ' ' + strD(self.geomech_time.coupling_timestep[1]))
        outfile.write('\n')
        outfile.write('END\n\n')

    def _read_geomechanics_output(self, infile, line):
        output = pgeomech_output()
        keep_reading = True
        while keep_reading:
            line = get_next_line(infile)
            key = line.strip().split()[0].lower()  # take first keyword
            if key in ['#']:
                pass
            elif key == 'times':
                output.time_list = []
                output.time_list = line.strip().split()[1:]
            elif key == 'print_column_ids':
                output.print_column_ids = True
            elif key == 'format':
                tstring = (line.strip().split()[1:])
                tstring = ' '.join(tstring).lower()
                output.format_list.append(tstring)  # assign
            elif key in ['/', 'end']:
                keep_reading = False
        self.geomech_output = output

    def _write_geomechanics_output(self, outfile):
        self._header(outfile, headers['geomechanics_output'])
        output = self.geomech_output

        outfile.write('GEOMECHANICS_OUTPUT\n')

        if output.time_list:
            # Check if 1st variable in list a valid time unit
            if output.time_list[0].lower() in time_units_allowed:
                outfile.write('  TIMES ')
                # Write remaining number(s) after time unit is specified
                for value in output.time_list:
                    outfile.write(' ' + strD(value).lower())
            else:
                print('       valid time.units', time_units_allowed, '\n')
                raise PyFLOTRAN_ERROR(
                    'output.time_list[0]: ' + output.time_list[0] +
                    ' is invalid.')
            outfile.write('\n')

        if output.print_column_ids:
            outfile.write('  ' + 'PRINT_COLUMN_IDS' + '\n')
        for out_format in output.format_list:
            if out_format.upper() in output_formats_allowed:
                outfile.write('  FORMAT ')
                outfile.write(out_format.upper() + '\n')
            else:
                print('       valid output.format:', \
                    output_formats_allowed, '\n')
                raise PyFLOTRAN_ERROR(
                    'output.format: \'' + out_format + '\' is invalid.')

        outfile.write('END\n\n')

    def _read_geomechanics_regression(self, infile, line):
        regression = pgeomech_regression()
        keep_reading = True
        while keep_reading:  # Read through all cards
            line = get_next_line(infile)
            key = line.strip().split()[0].lower()  # take first key word

            if key == 'vertices':
                keep_reading_2 = True
                vertex_list = []
                while keep_reading_2:
                    for i in range(100000):
                        line1 = get_next_line(infile)
                        if line1.strip().split()[0].lower() in ['/', 'end']:
                            keep_reading_2 = False
                            break
                        vertex_list.append(int(line1))
                regression.vertices = vertex_list
            elif key == 'variables':
                keep_reading_2 = True
                variable_list = []
                while keep_reading_2:
                    for i in range(100):
                        line1 = get_next_line(infile)
                        if line1.strip().split()[0].lower() in ['/', 'end']:
                            keep_reading_2 = False
                            break
                        variable_list.append(self.splitter(line1))
                regression.variables = variable_list
            elif key in ['/', 'end']:
                keep_reading = False
        self.geomech_regression = regression

    def _write_geomechanics_regression(self, outfile):
        self._header(outfile, headers['regression'])
        regression = self.geomech_regression
        outfile.write('GEOMECHANICS_REGRESSION' + '\n')
        if regression.vertices and regression.vertices[0] != '':
            outfile.write('  VERTICES' + '\n')
            for vertex in regression.vertices:
                outfile.write('    ' + str(vertex) + '\n')
            outfile.write('  /' + '\n')
        if regression.vertices_per_process:
            outfile.write('  VERTICES_PER_PROCESS' + ' ' +
                          str(regression.vertices_per_process) + '\n')
        if regression.variables:
            outfile.write('  VARIABLES' + '\n')
            for variable in regression.variables:
                outfile.write('    ' + str(variable.upper()) + '\n')
            outfile.write('  /' + '\n')
        outfile.write('END' + '\n\n')

    # Adds a prop object.
    def _add_geomech_prop(self, prop=pgeomech_material(), overwrite=False):
        # check if prop already exists
        if isinstance(prop, pgeomech_material):
            if prop.id in self.geomech_prop.keys():
                if not overwrite:
                    warning = 'A geomechanics material property with id ' + \
                              str(prop.id) + ' already exists. Prop ' + \
                              'will not be defined, use overwrite ' + \
                              '= True in add() to overwrite the old prop.'
                    print(warning,
                    build_warnings.append(warning))
                    return
                else:  # Executes if overwrite = True
                    self.delete(self.prop[prop.id])

        if prop not in self.geomech_proplist:
            self.geomech_proplist.append(prop)

    def _delete_geomech_prop(self, prop=pgeomech_material()):
        self.geomech_proplist.remove(prop)

    def _read_geomechanics_prop(self, infile, line):
        np_name = self.splitter(line)  # property name
        np_id = None
        p = pgeomech_material(id=np_id)
        keep_reading = True
        while keep_reading:  # read through all cards
            line = get_next_line(infile)
            key = line.strip().split()[0].lower()  # take first keyword
            if key == 'id':
                np_id = int(self.splitter(line))
            elif key == 'rock_density':
                np_rock_density = self.splitter(line)
            elif key == 'youngs_modulus':
                np_youngs_modulus = self.splitter(line)
            elif key == 'poissons_ratio':
                np_poissons_ratio = self.splitter(line)
            elif key == 'biot_coefficient':
                np_biot_coeff = self.splitter(line)
            elif key == 'thermal_expansion_coefficient':
                np_thermal_coeff = self.splitter(line)
            elif key in ['/', 'end']:
                keep_reading = False

        # create an empty material property
        new_prop = pgeomech_material(id=np_id, name=np_name,
                                     density=np_rock_density, youngs_modulus=np_youngs_modulus,
                                     poissons_ratio=np_poissons_ratio, biot_coefficient=np_biot_coeff,
                                     thermal_expansion_coefficient=np_thermal_coeff)

        self.add(new_prop)

    def _write_geomechanics_prop(self, outfile):
        self._header(outfile, headers['geomechanics_material_property'])
        for prop in self.geomech_proplist:
            if prop.name:
                outfile.write('GEOMECHANICS_MATERIAL_PROPERTY ' +
                              prop.name + '\n')
            if not prop.id == '':
                outfile.write('  ID ' + str(prop.id) + '\n')
            if not prop.density == '':
                outfile.write('  ROCK_DENSITY ' + strD(prop.density) + '\n')
            if not prop.youngs_modulus == '':
                outfile.write('  YOUNGS_MODULUS ' +
                              strD(prop.youngs_modulus) + '\n')
            if not prop.poissons_ratio == '':
                outfile.write('  POISSONS_RATIO ' +
                              strD(prop.poissons_ratio) + '\n')
            if not prop.biot_coefficient == '':
                outfile.write('  BIOT_COEFFICIENT ' +
                              strD(prop.biot_coefficient) + '\n')
            if not prop.thermal_expansion_coefficient == '':
                outfile.write('  THERMAL_EXPANSION_COEFFICIENT ' +
                              strD(prop.thermal_expansion_coefficient) + '\n')
            outfile.write('END\n\n')

    @property
    def geomech_dirname(self):
        return self.geomech_grid.dirname

    @property
    def prop(self):
        return dict([(p.id, p) for p in self.proplist] +
                    [(p.id, p) for p in self.proplist])

    @property
    def geomech_prop(self):
        return dict([(p.id, p) for p in self.geomech_proplist] +
                    [(p.id, p) for p in self.geomech_proplist])

    @property
    def dataset(self):
        return dict([(p.dataset_name, p) for p in self.datasetlist])

    @property
    def saturation(self):
        return dict([(p.name, p) for p in self.saturationlist])

    @property
    def lsolver(self):
        return dict([lsolv.name, lsolv] for lsolv in self.lsolverlist
                    if lsolv.name)

    @property
    def nsolver(self):
        return dict([nsolv.name, nsolv] for nsolv in self.nsolverlist
                    if nsolv.name)

    @property
    def fluid(self):
        return dict([flu.phase, flu] for flu in self.fluidlist
                    if flu.phase)

    @property
    def eos(self):
        return dict([eos.fluid_name, eos] for eos in self.eoslist
                    if eos.fluid_name)

    @property
    def char(self):
        return dict(
            [(characteristic_curves.name.lower(), characteristic_curves)
             for characteristic_curves in self.charlist] +
            [(characteristic_curves.name.lower(), characteristic_curves)
             for characteristic_curves in self.charlist])

    @property
    def region(self):
        return dict([region.name.lower(), region] for region in
                    self.regionlist if region.name)

    @property
    def integral_flux(self):
        return dict([integral_flux.name.lower(), integral_flux] for integral_flux in
                    self.integral_flux_list if integral_flux.name)

    @property
    def observation(self):
        return dict(
            [observation.region.lower(), observation] for observation in
            self.observation_list if observation.region)

    @property
    def flow(self):
        return dict([flow.name.lower(), flow] for flow in self.flowlist if
                    flow.name.lower)

    def flow_variable(self, flow=pflow()):
        return dict([flow_variable.name.lower(), flow_variable] for
                    flow_variable in flow.varlist
                    if flow_variable.name.lower())

    @property
    def initial_condition(self):
        return dict([initial_condition.region, initial_condition] for
                    initial_condition in self.initial_condition_list
                    if initial_condition.region)

    @property
    def boundary_condition(self):
        return dict(
            [boundary_condition.region, boundary_condition] for
            boundary_condition in self.boundary_condition_list if
            boundary_condition.region)

    @property
    def source_sink(self):
        return dict([source_sink.region, source_sink] for
                    source_sink in self.source_sink_list if source_sink.region)

    @property
    def strata(self):
        return dict([strata.region, strata] for
                    strata in self.strata_list if strata.region)

    @property
    def m_kinetic(self):
        chemistry = self.chemistry
        return dict([m_kinetic.name, m_kinetic] for
                    m_kinetic in chemistry.m_kinetics_list if m_kinetic.name)

    @property
    def transport(self):
        return dict([transport.name, transport] for
                    transport in self.transportlist if transport.name)

    @property
    def constraint(self):
        return dict(
            [constraint.name.lower(), constraint] for
            constraint in self.constraint_list if constraint.name.lower())

    def constraint_concentration(self, constraint=pconstraint()):
        return dict([constraint_concentration.pspecies,
                     constraint_concentration] for constraint_concentration in
                    constraint.concentration_list if
                    constraint_concentration.pspecies)

    @staticmethod
    def paraview(vtk_filepath_list=None):
        if vtk_filepath_list is not None:
            imports = 'from paraview import simple'
            legacy_reader = ''
            for vtk_filepath in vtk_filepath_list:
                if not os.path.isfile(vtk_filepath):
                    raise PyFLOTRAN_ERROR(
                        vtk_filepath + ' is not a valid filepath!')
                elif vtk_filepath[-3:] != 'vtk':
                    raise PyFLOTRAN_ERROR(
                        vtk_filepath +
                        'does not have a valid extension (.vtk)!')
            legacy_reader += 'simple.LegacyVTKReader(FileNames=' + str(
                vtk_filepath_list).replace(' ', '\n') + ')\n'
            with open('paraview-script.py', 'w+') as f:
                f.write(imports + '\n')
                f.write(legacy_reader + '\n')
                f.write('simple.Show()\nsimple.Render()')
        process = subprocess.Popen('paraview --script=paraview-script.py',
                                   shell=True,
                                   stdout=subprocess.PIPE,
                                   stderr=sys.stderr)
        while True:
            out = process.stdout.read(1)
            if out == '' and process.poll() is not None:
                break
            if out != '':
                sys.stdout.write(out)
                sys.stdout.flush()

    def apply_traction(self, traction=-1e6, face='top'):
        x_verts = self.grid.nxyz[0] + 2
        y_verts = self.grid.nxyz[1] + 2
        z_verts = self.grid.nxyz[2] + 2
        x_max = self.grid.xmax
        y_max = self.grid.ymax
        z_max = self.grid.zmax
        x_min = self.grid.xmin
        y_min = self.grid.ymin
        z_min = self.grid.zmin
        delta_x = (x_max - x_min) / (x_verts - 2)
        delta_y = (y_max - y_min) / (y_verts - 2)
        delta_z = (z_max - z_min) / (z_verts - 2)
        Total_verts = x_verts * y_verts * z_verts
        if self.work_dir:
            wd = self.work_dir + os.sep
        else:
            wd = os.getcwd() + os.sep
        if face == 'top':
            # Top corner
            x = np.zeros(x_verts)
            y = np.zeros(y_verts)
            z = np.zeros(z_verts)
            x[0] = x_min
            x[1] = x_min + delta_x / 2.0
            x[x_verts - 1] = x_max
            x[x_verts - 2] = x_max - delta_x / 2.0
            if x_verts > 4:
                for i in range(2, x_verts - 2):
                    x[i] = x[i - 1] + delta_x

            y[0] = y_min
            y[1] = y_min + delta_y / 2.0
            y[y_verts - 1] = y_max
            y[y_verts - 2] = y_max - delta_y / 2.0
            if y_verts > 4:
                for i in range(2, y_verts - 2):
                    y[i] = y[i - 1] + delta_y

            z[0] = z_min
            z[1] = z_min + delta_z / 2.0
            z[z_verts - 1] = z_max
            z[z_verts - 2] = z_max - delta_z / 2.0
            if z_verts > 4:
                for i in range(2, z_verts - 2):
                    z[i] = z[i - 1] + delta_z

            xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
            Coord = np.zeros((Total_verts, 3), 'float')
            count = 0
            for k in range(z_verts):
                for j in range(y_verts):
                    for i in range(x_verts):
                        Coord[count] = [xv[i][j][k], yv[i][j][k], zv[i][j][k]]
                        count = count + 1

            count = 1
            files = []
            for j in range(y_verts):
                for i in range(x_verts):
                    if i > 0:
                        if i < x_verts - 1:
                            x_side = (xv[i + 1][j][k] - xv[i - 1][j][k]) / 2.0
                        else:
                            x_side = (xv[i][j][k] - xv[i - 1][j][k]) / 2.0
                    else:
                        x_side = (xv[i + 1][j][k] - xv[i][j][k]) / 2.0
                    if j > 0:
                        if j < y_verts - 1:
                            y_side = (yv[i][j + 1][k] - yv[i][j - 1][k]) / 2.0
                        else:
                            y_side = (yv[i][j][k] - yv[i][j - 1][k]) / 2.0
                    else:
                        y_side = (yv[i][j + 1][k] - yv[i][j][k]) / 2.0

                    area = x_side * y_side
                    node_id = x_verts * y_verts * (z_verts - 1) + count
                    fid = open(wd + self.geomech_grid.dirname +
                               '/' + str(node_id) + '.vset', 'w')
                    fid.write('%i\n' % (node_id))
                    fid.close()
                    self.add(pflow(name=str(node_id) + '_force', pm='geomech'))
                    self.add(pflow_variable(name='force_z', type='dirichlet',
                                            valuelist=[area * traction]),
                             index=str(node_id) + '_force')
                    count = count + 1
                    files.append(str(node_id) + '.vset')

            for file in files:
                self.add(pregion(name=file[:-5],
                                 filename=self.geomech_grid.dirname +
                                 '/' + file, pm='geomech'))
                self.add(pboundary_condition(name=file[:-5] + '_force',
                                             region=file[:-5],
                                             geomech=file[:-5] + '_force'))

    def apply_horizontal_critical_stress(self, rho_eff=2000.0,
                                         vertical_to_horizontal_ratio=0.7,
                                         face='east', total_depth=2500):
        x_verts = self.grid.nxyz[0] + 2
        y_verts = self.grid.nxyz[1] + 2
        z_verts = self.grid.nxyz[2] + 2
        xmax = self.grid.xmax
        ymax = self.grid.ymax
        zmax = self.grid.zmax
        xmin = self.grid.xmin
        ymin = self.grid.ymin
        zmin = self.grid.zmin
        Total_verts = x_verts * y_verts * z_verts
        N_cells = (x_verts - 1) * (y_verts - 1) * (z_verts - 1)
        delta_x = (xmax - xmin) / (x_verts - 2)
        delta_y = (ymax - ymin) / (y_verts - 2)
        delta_z = (zmax - zmin) / (z_verts - 2)

        x = np.zeros(x_verts)
        y = np.zeros(y_verts)
        z = np.zeros(z_verts)
        x[0] = xmin
        x[1] = xmin + delta_x / 2.0
        x[x_verts - 1] = xmax
        x[x_verts - 2] = xmax - delta_x / 2.0
        if x_verts > 4:
            for i in range(2, x_verts - 2):
                x[i] = x[i - 1] + delta_x

        y[0] = ymin
        y[1] = ymin + delta_y / 2.0
        y[y_verts - 1] = ymax
        y[y_verts - 2] = ymax - delta_y / 2.0
        if y_verts > 4:
            for i in range(2, y_verts - 2):
                y[i] = y[i - 1] + delta_y

        z[0] = zmin
        z[1] = zmin + delta_z / 2.0
        z[z_verts - 1] = zmax
        z[z_verts - 2] = zmax - delta_z / 2.0
        if z_verts > 4:
            for i in range(2, z_verts - 2):
                z[i] = z[i - 1] + delta_z

        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        Coord = np.zeros((Total_verts, 3), 'float')
        count = 0
        for k in range(z_verts):
            for j in range(y_verts):
                for i in range(x_verts):
                    Coord[count] = [xv[i][j][k], yv[i][j][k], zv[i][j][k]]
                    count = count + 1
        if self.work_dir:
            wd = self.work_dir + os.sep
        else:
            wd = os.getcwd() + os.sep
        if face == 'east':
            g = 9.81
            # east corner
            east_corner = []
            east_corner.append(x_verts * y_verts * (z_verts - 1) + x_verts)
            east_corner.append(x_verts * y_verts * z_verts)
            east_corner.append(x_verts)
            east_corner.append(x_verts * y_verts)
            fid = open(wd + self.geomech_grid.dirname +
                       '/east_corner.vset', 'w')
            for i in east_corner:
                fid.write('%i\n' % i)
            fid.close()

            # east boundary
            east_boundary = []
            for k in range(1, z_verts + 1):
                east_boundary.append(x_verts + (k - 1) * x_verts * y_verts)

            for k in range(1, z_verts + 1):
                east_boundary.append(x_verts + (y_verts - 1)
                                     * x_verts + (k - 1) * x_verts * y_verts)

            for j in range(1, y_verts + 1):
                east_boundary.append(x_verts + (j - 1) * x_verts)

            for j in range(1, y_verts + 1):
                east_boundary.append(
                    x_verts + (j - 1) * x_verts + (z_verts - 1) * x_verts * y_verts)

            east_boundary = list(set(east_boundary))

            # remove duplicates and corners
            east_boundary = list(set(east_boundary) - set(east_corner))
            fid = open(wd + self.geomech_grid.dirname +
                       '/east_boundary.vset', 'w')
            for i in east_boundary:
                fid.write('%i\n' % i)
            fid.close()

            # east internal
            east_internal = []
            for k in range(1, z_verts):
                for j in range(1, y_verts):
                    east_internal.append(
                        x_verts + (j - 1) * x_verts + (k - 1) * x_verts * y_verts)

            east_internal = list(set(east_internal) -
                                 set(east_boundary) - set(east_corner))
            fid = open(wd + self.geomech_grid.dirname +
                       '/east_internal.vset', 'w')
            for i in east_internal:
                fid.write('%i\n' % i)
            fid.close()

            # east corner force
            for node in east_corner:
                area = delta_y * delta_z / 16
                traction = rho_eff * g * vertical_to_horizontal_ratio * \
                    (total_depth - Coord[node - 1][2])
                force = -traction * area
                file = self.geomech_grid.dirname + \
                    '/' + 'east_corner_' + str(node)
                fid = open(wd + file, 'w')
                fid.write('%i\n' % node)
                fid.close()
                self.add(pflow(name='east_corner_force_' +
                               str(node), pm='geomech'))
                self.add(pflow_variable(name='force_x', type='dirichlet',
                                        valuelist=[force]),
                         index='east_corner_force_' + str(node))
                self.add(pregion(name='east_corner_' + str(node),
                                 filename=file, pm='geomech'))
                self.add(pboundary_condition(name='east_corner_force_' + str(node),
                                             region='east_corner_' + str(node),
                                             geomech='east_corner_force_' + str(node)))
            # east boundary force
            for node in east_boundary:
                area = delta_y * delta_z / 8
                traction = rho_eff * g * vertical_to_horizontal_ratio * \
                    (total_depth - Coord[node - 1][2])
                force = -traction * area
                file = self.geomech_grid.dirname + \
                    '/' + 'east_boundary_' + str(node)
                fid = open(wd + file, 'w')
                fid.write('%i\n' % node)
                fid.close()
                self.add(pflow(name='east_boundary_force_' +
                               str(node), pm='geomech'))
                self.add(pflow_variable(name='force_x', type='dirichlet',
                                        valuelist=[force]),
                         index='east_boundary_force_' + str(node))
                self.add(pregion(name='east_boundary_' + str(node),
                                 filename=file, pm='geomech'))
                self.add(pboundary_condition(name='east_boundary_force_' + str(node),
                                             region='east_boundary_' +
                                             str(node),
                                             geomech='east_boundary_force_' + str(node)))

            # east internal force
            for node in east_internal:
                area = delta_y * delta_z / 4
                traction = rho_eff * g * vertical_to_horizontal_ratio * \
                    (total_depth - Coord[node - 1][2])
                force = -traction * area
                file = self.geomech_grid.dirname + \
                    '/' + 'east_internal_' + str(node)
                fid = open(wd + file, 'w')
                fid.write('%i\n' % node)
                fid.close()
                self.add(pflow(name='east_internal_force_' +
                               str(node), pm='geomech'))
                self.add(pflow_variable(name='force_x', type='dirichlet',
                                        valuelist=[force]),
                         index='east_internal_force_' + str(node))
                self.add(pregion(name='east_internal_' + str(node),
                                 filename=file, pm='geomech'))
                self.add(pboundary_condition(name='east_internal_force_' + str(node),
                                             region='east_internal_' +
                                             str(node),
                                             geomech='east_internal_force_' + str(node)))

    def generate_geomech_grid(self):
        x_verts = self.grid.nxyz[0] + 2
        y_verts = self.grid.nxyz[1] + 2
        z_verts = self.grid.nxyz[2] + 2
        xmax = self.grid.xmax
        ymax = self.grid.ymax
        zmax = self.grid.zmax
        xmin = self.grid.xmin
        ymin = self.grid.ymin
        zmin = self.grid.zmin
        Total_verts = x_verts * y_verts * z_verts
        N_cells = (x_verts - 1) * (y_verts - 1) * (z_verts - 1)
        delta_x = (xmax - xmin) / (x_verts - 2)
        delta_y = (ymax - ymin) / (y_verts - 2)
        delta_z = (zmax - zmin) / (z_verts - 2)

        x = np.zeros(x_verts)
        y = np.zeros(y_verts)
        z = np.zeros(z_verts)

        x[0] = xmin
        x[1] = xmin + delta_x / 2.0
        x[x_verts - 1] = xmax
        x[x_verts - 2] = xmax - delta_x / 2.0
        if x_verts > 4:
            for i in range(2, x_verts - 2):
                x[i] = x[i - 1] + delta_x

        y[0] = ymin
        y[1] = ymin + delta_y / 2.0
        y[y_verts - 1] = ymax
        y[y_verts - 2] = ymax - delta_y / 2.0
        if y_verts > 4:
            for i in range(2, y_verts - 2):
                y[i] = y[i - 1] + delta_y

        z[0] = zmin
        z[1] = zmin + delta_z / 2.0
        z[z_verts - 1] = zmax
        z[z_verts - 2] = zmax - delta_z / 2.0
        if z_verts > 4:
            for i in range(2, z_verts - 2):
                z[i] = z[i - 1] + delta_z

        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        Coord = np.zeros((Total_verts, 3), 'float')
        Area = np.zeros((x_verts, y_verts), 'float')
        count = 0
        for k in range(z_verts):
            for j in range(y_verts):
                for i in range(x_verts):
                    Coord[count] = [xv[i][j][k], yv[i][j][k], zv[i][j][k]]
                    count = count + 1

        # Storing vertices in each element
        # Assuming all elements are hexes
        Vertices = np.zeros((N_cells, 8), 'int')

        count = 0
        for k in range(1, z_verts):
            for j in range(1, y_verts):
                for i in range(1, x_verts):
                    id = i + (j - 1) * x_verts + (k - 1) * x_verts * y_verts
                    Vertices[count, 0] = id
                    Vertices[count, 1] = id + 1
                    Vertices[count, 2] = id + x_verts + 1
                    Vertices[count, 3] = id + x_verts
                    Vertices[count, 4] = id + x_verts * y_verts
                    Vertices[count, 5] = id + x_verts * y_verts + 1
                    Vertices[count, 6] = id + x_verts * y_verts + x_verts + 1
                    Vertices[count, 7] = id + x_verts * y_verts + x_verts
                    count = count + 1

        # Writing list of all vertices
        all_nodes = []
        # print('--> Writing geomechanics mesh files')
        # print('--> Writing vertices')
        fid = open('all.vset', 'w')
        for i in range(1, Total_verts + 1):
            fid.write('%i\n' % i)
            all_nodes.append(i)
        fid.close()
        #        print('--> Finished writing all.vset')

        # Writing mesh file
        #        print('--> Writing usg file')
        fid = open('usg.mesh', 'w')
        fid.write('%i %i\n' % (N_cells, Total_verts))
        for id in range(count):
            fid.write('H %i %i %i %i %i %i %i %i\n' % (Vertices[id, 0],
                                                       Vertices[id, 1],
                                                       Vertices[id, 2],
                                                       Vertices[id, 3],
                                                       Vertices[id, 4],
                                                       Vertices[id, 5],
                                                       Vertices[id, 6],
                                                       Vertices[id, 7]))
        for id in range(Total_verts):
            fid.write('%f %f %f\n' %
                      (Coord[id, 0], Coord[id, 1], Coord[id, 2]))
        fid.close()
        #        print('--> Finished writing usg.mesh')

        # Writing vertex numbers on faces
        # Bottom (z=z_min)
        bottom = []
        #        print('--> Writing bottom vertices')
        fid = open('bottom.vset', 'w')
        for i in range(1, x_verts * y_verts + 1):
            fid.write('%i\n' % i)
            bottom.append(i)
        fid.close()
        #        print('--> Finished writing bottom.vset')

        # Top (z=z_max)
        top = []
        #        print('--> Writing top vertices')
        fid = open('top.vset', 'w')
        for i in range(x_verts * y_verts * (z_verts - 1) + 1,
                       x_verts * y_verts * z_verts + 1):
            fid.write('%i\n' % i)
            top.append(i)
        fid.close()
        #        print('--> Finished writing top.vset')

        # North (y=y_max)
        north = []
        #        print('--> Writing north vertices')
        fid = open('north.vset', 'w')
        for i in range(1, x_verts + 1):
            for k in range(1, z_verts + 1):
                j = y_verts
                id = i + (j - 1) * x_verts + (k - 1) * x_verts * y_verts
                fid.write('%i\n' % id)
                north.append(id)
        fid.close()
        #        print('--> Finished writing north.vset')

        # South (y=y_min)
        south = []
        #        print('--> Writing south vertices')
        fid = open('south.vset', 'w')
        for i in range(1, x_verts + 1):
            for k in range(1, z_verts + 1):
                j = 1
                id = i + (j - 1) * x_verts + (k - 1) * x_verts * y_verts
                fid.write('%i\n' % id)
                south.append(id)
        fid.close()
        #        print('--> Finished writing south.vset')

        # East (x=x_max)
        east = []
        #        print('--> Writing east vertices')
        fid = open('east.vset', 'w')
        for j in range(1, y_verts + 1):
            for k in range(1, z_verts + 1):
                i = x_verts
                id = i + (j - 1) * x_verts + (k - 1) * x_verts * y_verts
                fid.write('%i\n' % id)
                east.append(id)
        fid.close()
        #        print('--> Finished writing east.vset')

        # West (x=x_min)
        west = []
        #        print('--> Writing west vertices')
        fid = open('west.vset', 'w')
        for j in range(1, y_verts + 1):
            for k in range(1, z_verts + 1):
                i = 1
                id = i + (j - 1) * x_verts + (k - 1) * x_verts * y_verts
                fid.write('%i\n' % id)
                west.append(id)
        fid.close()
        #        print('--> Finished writing west.vset')

        # Create a subdirectory
        d = self.geomech_grid.dirname
        if self.work_dir:
            wd = self.work_dir + os.sep
        else:
            wd = os.getcwd() + os.sep
        d = wd + d
        if os.path.isdir(d):  # check if d exists
            shutil.rmtree(d)  # remove old directory
        os.mkdir(d)  # create new directory

        # Move *.vset *.mesh files to new subdirectory
        cmd = 'mv' + ' *.vset' + ' usg.mesh' + ' %s/.' % d
        failure = os.system(cmd)
        if failure:
            print('Unable to move *.vset, *.mesh files to subdirectory')
            sys.exit(1)
        # print('--> Finished with moving files to dat directory')

        count = 0
        if self.work_dir:
            wd = self.work_dir + os.sep
        else:
            wd = os.getcwd() + os.sep
        fid = open(wd + 'flow_geomech_mapping.dat', 'w')
        epsilon = 1.e-8
        for coord in Coord:
            i = int((coord[0] - xmin - epsilon) / delta_x)
            j = int((coord[1] - ymin - epsilon) / delta_y)
            k = int((coord[2] - zmin - epsilon) / delta_z)
            id = i + j * self.grid.nxyz[0] + k * \
                self.grid.nxyz[0] * self.grid.nxyz[1]
            fid.write('%i %i\n' % (int(id + 1), count + 1))
            count = count + 1

        fid.close()

    def terzaghi(self, x, t, dP, c, L):
        """
        Will do a terzaghi calculation with porous plate at x = 0

        :param x: distance along the column.
        :type x: float
        :param t: time since loading.
        :type name: float
        :param dP: Additional pressure that was induced by loading at t=0+
        :type dP: float
        :param c: Hydraulic diffusivity
        :type c: float
        :param L: Full length of the column
        :type L: float
        """

        import numpy as np
        inf_series = 0
        for m in range(0, 1000):
            inf_series = (1 / (2. * m + 1)) * np.exp(-(2. * m + 1)**2 * np.pi * np.pi * c * t / 4 / L / L) * \
                np.sin((2. * m + 1) * np.pi * x / 2 / L) + inf_series
        pressure = 4. * dP / np.pi * inf_series
        return pressure

    def terzaghi_flip(self, x, t, dP, c, L):
        """
        Will do a terzaghi calculation, with porous plate at x = L.

        :param x: distance along the column.
        :type x: float
        :param t: time since loading.
        :type name: float
        :param dP: Additional pressure that was induced by loading at t=0+
        :type dP: float
        :param c: Hydraulic diffusivity
        :type c: float
        :param L: Full length of the column
        :type L: float
        """
        from matplotlib import pyplot as plt
        import numpy as np

        inf_series = 0
        for m in range(0, 1000):
            inf_series = (1 / (2. * m + 1)) * np.exp(-(2. * m + 1)**2 * np.pi * np.pi * c * t / 4 / L / L) * \
                np.sin((2 * m + 1) * np.pi * (L - x) / 2 / L) + inf_series
        pressure = 4. * dP / np.pi * inf_series
        return pressure

    # def terzaghi_flip(self,x,t,P0,c,L):
    #     """
    #     Will do a terzaghi calculation where x is length, t is time, P0 is
    #     the pressure that is induced by loading at time 0+, c is the hydraulic
    #     diffusivity, and L is the length of the column.

    #     Now P = 0 at z = L instead of at z = 0.

    #     """
    #     from matplotlib import pyplot as plt
    #     import numpy as np

    #     inf_series = 0;
    #     for m in range(0,1000):
    #         inf_series = (1/(2.*m+1))*np.exp(-(2.*m+1)**2*np.pi*np.pi*c*t/4/L/L) * \
    #             np.sin((2*m+1)*np.pi*(L-x)/2/L) + inf_series
    #     pressure = 4.*P0/np.pi*inf_series
    #     return pressure

# print('--> Done writing geomechanics mesh files!')
