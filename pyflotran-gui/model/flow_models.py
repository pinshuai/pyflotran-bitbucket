def get_flow_variable_types(index):
    # 0 - PRESSURE, 1 - RATE, 2 - FLUX, 3 - TEMPERATURE, 4 - CONCENTRATION
    # 5 - SATURATION, 6 - ENTHALPY
    types_map = {0: ['Dirichlet', 'Hydrostatic', 'Zero Gradient', 'Conductance',
                     'Seepage'], 1: ['Mass Rate', 'Volumetric Rate', 'Scaled Volumetric Rate'],
                 2: ['Dirichlet', 'Neumann', 'Mass Rate', 'Hydrostatic', 'Conductance', 'Zero Gradient',
                     'Production Well', 'Seepage', 'Volumetric', 'Volumetric Rate', 'Equilibrium'],
                 3: ['Dirichlet', 'Hydrostatic', 'Zero Gradient'], 4: ['Dirichlet', 'Hydrostatic', 'Zero Gradient'],
                 5: ['Dirichlet'], 6: ['Dirichlet', 'Hydrostatic', 'Zero Gradient']}
    return types_map.get(index, 'Not a valid index')
