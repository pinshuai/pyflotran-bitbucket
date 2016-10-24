from pdata import *
import os
import socket
import argparse
import multiprocessing


__author__ =  'Satish Karra'

try:
    pflotran_dir = os.environ['PFLOTRAN_DIR']
except KeyError:
    print('PFLOTRAN_DIR must point to PFLOTRAN installation directory \
     and be defined in system environment variables.')
    sys.exit(1)
sys.path.append(pflotran_dir + '/src/python')

try:
    pyflotran_dir = os.environ['PYFLOTRAN_DIR']
except KeyError:
    print(
        'PYFLOTRAN_DIR must point to PYFLOTRAN installation directory \
        and be defined in system environment variables.')
    sys.exit(1)
sys.path.append(pyflotran_dir)

###############################################################

plt.rcParams.update({'axes.labelsize': 'large'})
plt.rcParams.update({'axes.titlesize': 'large'})


def run_sim(nx, ny, nz, num_procs):
    ''' Set's up PFLOTRAN simulation for given number of procs and
        outputs wallclock time per step
    '''


    # initialize without reading in test data
    # --------------------------------------------------------------
    dat = pdata('')
    # --------------------------------------------------------------
    # set simulation
    # --------------------------------------------------------------
    simulation = psimulation()
    simulation.simulation_type = 'subsurface'
    simulation.subsurface_flow = 'flow'
    simulation.mode = 'richards'
    dat.simulation = simulation
    # --------------------------------------------------------------

    # set grid
    # --------------------------------------------------------------
    grid = pgrid()
    grid.type = 'structured'
    (xmin, ymin, zmin) = (0.0, 0.0, 0.0)
    (xmax, ymax, zmax) = (5000.0, 2500.0, 100.0)
    (layer1_zmin, layer1_zmax) = (0.0, 30.0)
    (layer2_zmin, layer2_zmax) = (30.0, 50.0)
    (layer3_zmin, layer3_zmax) = (50.0, 70.0)
    (layer4_zmin, layer4_zmax) = (70.0, zmax)

    grid.lower_bounds = [xmin, ymin, zmin]
    grid.upper_bounds = [xmax, ymax, zmax]
    grid.nxyz = [nx, ny, nz]
    dat.grid = grid
    # --------------------------------------------------------------

    # set time stepping
    # --------------------------------------------------------------
    ts = ptimestepper()
    ts.max_steps = num_steps
    dat.timestepper = ts
    # --------------------------------------------------------------

    # set material properties
    # --------------------------------------------------------------
    material = pmaterial(id=1, name='soil1', porosity=0.15, tortuosity=0.5,
                         permeability=[1.e-10, 1.e-10, 1.e-11])
    dat.add(material)
    material = pmaterial(id=2, name='soil2', porosity=0.35, tortuosity=0.5,
                         permeability=[2.e-10, 2.e-10, 2.e-11])
    dat.add(material)
    material = pmaterial(id=3, name='soil3', porosity=0.25, tortuosity=0.5,
                         permeability=[5.e-11, 5.e-11, 5.e-12])
    dat.add(material)
    material = pmaterial(id=4, name='soil4', porosity=0.2, tortuosity=0.5,
                         permeability=[1.e-10, 1.e-10, 1.e-11])
    dat.add(material)
    # --------------------------------------------------------------

    # set time
    # --------------------------------------------------------------
    time = ptime()
    time.tf = [10.0, 'y']  # FINAL_TIME
    time.dti = [1.e-2, 'd']  # INITIAL_TIMESTEP_SIZE
    time.dtf = [0.1, 'y']  # MAXIMUM_TIMESTEP_SIZE
    dat.time = time
    # --------------------------------------------------------------

    # set newton solvers
    # --------------------------------------------------------------
    newton_solver = pnsolver('')
    newton_solver.name = 'FLOW'
    newton_solver.atol = 1e-12
    newton_solver.rtol = 1e-12
    newton_solver.stol = 1e-30
    newton_solver.dtol = 1e15
    newton_solver.itol = 1e-8
    newton_solver.max_it = 25
    newton_solver.max_f = 100
    dat.add(newton_solver)
    # --------------------------------------------------------------

    # set fluid properties
    # --------------------------------------------------------------
    fluid = pfluid()
    fluid.diffusion_coefficient = 1.e-9
    dat.add(fluid)
    # --------------------------------------------------------------

    # set regions
    # --------------------------------------------------------------
    region = pregion()
    region.name = 'all'
    region.coordinates_lower = [xmin, ymin, zmin]
    region.coordinates_upper = [xmax, ymax, zmax]
    dat.add(region)

    region = pregion()
    region.name = 'top'
    region.face = 'TOP'
    region.coordinates_lower = [xmin, ymin, zmax]
    region.coordinates_upper = [xmax, ymax, zmax]
    dat.add(region)

    region = pregion()
    region.name = 'west'
    region.face = 'WEST'
    region.coordinates_lower = [xmin, ymin, zmin]
    region.coordinates_upper = [xmin, ymax, zmax]
    dat.add(region)

    region = pregion()
    region.name = 'EAST'
    region.face = 'east'
    region.coordinates_lower = [xmax, ymin, zmin]
    region.coordinates_upper = [xmax, ymax, zmax]
    dat.add(region)

    region = pregion()
    region.name = 'injection_well'
    region.coordinates_lower = [inj_xmin, inj_ymin, inj_zmin]
    region.coordinates_upper = [inj_xmax, inj_ymax, inj_zmax]
    dat.add(region)

    region = pregion()
    region.name = 'production_well'
    region.coordinates_lower = [prod_xmin, prod_ymin, prod_zmin]
    region.coordinates_upper = [prod_xmax, prod_ymax, prod_zmax]
    dat.add(region)

    region = pregion(name='layer1', coordinates_lower=[xmin, ymin, layer1_zmin],
                     coordinates_upper=[xmax, ymax, layer1_zmax])
    dat.add(region)
    region = pregion(name='layer2', coordinates_lower=[xmin, ymin, layer2_zmin],
                     coordinates_upper=[xmax, ymax, layer2_zmax])
    dat.add(region)
    region = pregion(name='layer3', coordinates_lower=[xmin, ymin, layer3_zmin],
                     coordinates_upper=[xmax, ymax, layer3_zmax])
    dat.add(region)
    region = pregion(name='layer4', coordinates_lower=[xmin, ymin, layer4_zmin],
                     coordinates_upper=[xmax, ymax, layer4_zmax])
    dat.add(region)

    # --------------------------------------------------------------

    # set flow conditions
    # --------------------------------------------------------------
    # initial flow condition
    flow = pflow('')
    flow.name = 'initial'
    flow.iphase = 1
    dat.add(flow)
    # adding flow_variable to inital flow_condition
    variable = pflow_variable('')  # new flow var object
    variable.name = 'pressure'
    variable.type = 'hydrostatic'
    variable.valuelist = [10325]
    dat.add(variable, index='initial')

    # source flow condition
    flow = pflow('')
    flow.name = 'injection'
    flow.iphase = 1
    dat.add(flow)  # Assigning for flow condition done here
    # adding flow_variable to source flow_condition
    variable = pflow_variable('')  # new flow var object
    variable.name = 'rate'
    variable.type = 'mass_rate'
    variable.valuelist = [1.e2]
    dat.add(variable)

    # source flow condition
    flow = pflow('')
    flow.name = 'production'
    dat.add(flow)  # Assigning for flow condition done here
    # adding flow_variable to source flow_condition
    variable = pflow_variable('')  # new flow var object
    variable.name = 'rate'
    variable.type = 'mass_rate'
    variable.valuelist = [-1.e2]
    dat.add(variable)

    # --------------------------------------------------------------

    # set initial condition
    # --------------------------------------------------------------
    ic = pinitial_condition()
    ic.name = 'initial'
    ic.flow = 'INITIAL'
    ic.region = 'all'
    dat.add(ic)
    # --------------------------------------------------------------

    # set source sink
    # --------------------------------------------------------------
    ss = psource_sink(name='injection_well', flow='injection',
                      region='injection_well')
    dat.add(ss)
    ss = psource_sink(name='production_well', flow='production',
                      region='production_well')
    dat.add(ss)
    # --------------------------------------------------------------

    # set stratigraphy couplers
    # --------------------------------------------------------------
    stratigraphy_coupler = pstrata(region='layer1', material='soil1')
    dat.add(stratigraphy_coupler)
    stratigraphy_coupler = pstrata(region='layer2', material='soil2')
    dat.add(stratigraphy_coupler)
    stratigraphy_coupler = pstrata(region='layer3', material='soil3')
    dat.add(stratigraphy_coupler)
    stratigraphy_coupler = pstrata(region='layer4', material='soil4')
    dat.add(stratigraphy_coupler)

    # --------------------------------------------------------------

    pflotran_exe = pflotran_dir + '/src/pflotran/pflotran'
    # Write to file and execute that input file
    dat.run(input='flow' + str(num_procs) + '.in', exe=pflotran_exe,
            num_procs=num_procs, silent=True)
    cmd = 'grep Wall ' + 'flow' + str(num_procs) + '.out'
    process = subprocess.Popen(
        cmd.split(' '), shell=False, stdout=subprocess.PIPE,
        stderr=sys.stderr)
    out = process.stdout.read()
    wall_time = float(out.split()[3])
    wall_time_per_step = wall_time/num_steps

    return wall_time_per_step

def get_args():
    '''This function parses and return arguments passed in'''
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description='script for strong scaling of PFLOTRAN')
    # Add arguments
    parser.add_argument(
        '-nx', '--num_cells_x', type=str,
        help='Number of cells in x direction [default: 10]',
        required=False)
    parser.add_argument(
        '-ny', '--num_cells_y', type=str,
        help='Number of cells in y direction [default: 10]',
        required=False)
    parser.add_argument(
        '-nz', '--num_cells_z', type=str,
        help='Number of cells in z direction [default: 10]',
        required=False)
    parser.add_argument(
        '-mp', '--max_two_power', type=str,
        help='n, where 2^n is the maximum number of processors'
             ' [default: based on available procs. on the host]',
        required=False)
    # Array for all arguments passed to script
    args = parser.parse_args()

    # Return the args
    return args

def cleanup():
    '''Clean up the output and input files'''

    os.system('rm -f *in')
    os.system('rm -f *out')

if __name__ == "__main__":

    num_steps = 10
    (nx, ny, nz) = (10, 10, 10)
    num_cpus = multiprocessing.cpu_count()
    hostname = socket.gethostname()
    print '--> Hostname:', hostname, 'has', num_cpus, 'processors'
    mp = int(math.log(num_cpus,2))
    (inj_xmin, inj_ymin, inj_zmin) = (1250.0, 1250.0, 20.0)
    (inj_xmax, inj_ymax, inj_zmax) = (1250.0, 1250.0, 65.0)
    (prod_xmin, prod_ymin, prod_zmin) = (3750.0, 1250.0, 20.0)
    (prod_xmax, prod_ymax, prod_zmax) = (3750.0, 1250.0, 55.0)
    args = get_args()

    if args.num_cells_x: nx = args.num_cells_x; nx = int(nx)
    if args.num_cells_y: ny = args.num_cells_y; ny = int(ny)
    if args.num_cells_z: nz = args.num_cells_z; nz = int(nz)
    if args.max_two_power: mp = args.max_two_power; mp = int(mp)

    procs = [np.power(2,i) for i in range(mp+1)]
    print '--> Performing strong scaling using processors ', procs
    time_list = []
    with open('strong_scaling.txt', 'w') as f:
        f.write(('num procs').ljust(10) + ('time [s]').ljust(8) + '\n')
        print '--> Performing strong scaling'
        for proc in procs:
            print '--> Running on', str(proc), 'processors'
            time = run_sim(nx, ny, nz, proc)
            print '--> Cpu time [s] on', str(proc), 'processors is', time
            time_list.append(time)
            f.write((str(proc)).ljust(10) + (str(time)).ljust(8) + '\n')

    print '--> Completed writing to file. See strong_scaling.txt'
    xtick_labels = [str(proc) for proc in procs]
    ideal_times = [time_list[0]/proc for proc in procs]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(procs, time_list, linewidth=2, ls='-', marker='^', label='PFLOTRAN')
    ax.plot(procs, ideal_times, lw=2, ls='--', c='r', label='ideal')
    legend = ax.legend(loc='best')
    ax.set_xlabel('\# processors')
    ax.set_ylabel('wall time per step [s]')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([procs[np.asarray(procs).argmin()],
                 procs[np.asarray(procs).argmax()]])
    ax.set_title('strong scaling on ' + hostname +
                 ', flow dof=' + str(nx*ny*nz))
    plt.xticks(procs, xtick_labels)
    plt.savefig('strong_scaling.pdf')
    print '--> Completed plotting. See strong_scaling.pdf'
    cleanup()
    print '--> Finished cleaning up'
    print '--> Done!'
