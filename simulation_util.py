import subprocess

import numpy as np
import scipy.stats

lammps_path = "/Users/phankl/phd/lammps/src/lmp_g++_openmpi"
lammps_path = "lmp"


def run_lammps(
    input_file_contents,
    input_file_name,
    suppress_stdout=True
):
    # write input file and run simulation

    with open(input_file_name, "w") as input_file:
        input_file.write(input_file_contents)

    if suppress_stdout:
        subprocess.run(
            f"{lammps_path} -i {input_file_name}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )        
    else:
        subprocess.run(
            f"{lammps_path} -i {input_file_name}",
            shell=True,
        )

def equilibrate_particle(radius, temperature, species, label=0):
    if species == "Fe":
        lattice_constant = 3.571
        lattice_type = "fcc"
        nanoparticle_atom_mass = 55.847
    elif species == "Co":
        lattice_constant = 3.486
        lattice_type = "fcc"
        nanoparticle_atom_mass = 58.933

    min_coord = -radius - 1
    max_coord = -min_coord

    # general simulation parameters

    timestep = 1.0e-3
    nvt_time_constant = 1.0
    nvt_equilibration_steps = 100000
    nve_equilibration_steps = 100000

    rng = np.random.default_rng(12345)
    random_seed = rng.integers(low=0, high=100000, size=1)[0]

    # define input file

    input_file_contents = f"""
    # base simulation parameters
    
    units metal
    dimension 3
    boundary s s s
    atom_style full
    
    timestep {timestep}
    
    # generate spherical particle
    
    lattice {lattice_type} {lattice_constant}
    region box block {min_coord} {max_coord} {min_coord} {max_coord} {min_coord} {max_coord}
    create_box 2 box
    
    region particle sphere 0 0 0 {radius} side in units box
    create_atoms 1 region particle

    # colliding atom definition
    
    mass 2 12
    
    # potential definition
    
    pair_style hybrid eam/alloy lj/cut 10 zero 10
    pair_coeff * * eam/alloy FeNiCr_Bonny_2013_ptDef.eam.alloy Fe NULL
    pair_coeff 1 2 lj/cut 2.7466705 1.8530700 7.41227500
    # lj/cut, epsilon (energy unit), sigma (distance unit), LJ cutoff(distance unit)
    pair_coeff 2 2 zero
    
    # equilibrate particle
    # nvt first
    
    velocity all create {temperature} {random_seed} mom yes rot yes dist gaussian
    fix nvt all nvt temp {temperature} {temperature} {nvt_time_constant}
    fix recenter all recenter 0 0 0
    
    thermo 1000
    
    run {nvt_equilibration_steps}
    
    # then nve
    
    unfix nvt
    unfix recenter
    
    velocity all zero linear
    velocity all zero angular
    velocity all scale {temperature}
    
    fix nve all nve
    
    run {nve_equilibration_steps}
    
    # write restart file with equilibrated nanoparticle
    
    write_restart restart.nanoparticle_{label}
    """

    run_lammps(input_file_contents, f"in.nanoparticle_{label}")


def simulate_collision(
    atom_position, atom_velocity, nanoparticle_label, collision_label
):
    collision_steps = 10000

    input_file_contents = f"""
    read_restart restart.nanoparticle_{nanoparticle_label}

    change_box all x final -50 50 y final -50 50 z final -50 50 boundary f f f
    create_atoms 2 single {atom_position[0]} {atom_position[1]} {atom_position[2]}
    change_box all boundary s s s 
    group atom type 2
    velocity atom set {atom_velocity[0]} {atom_velocity[1]} {atom_velocity[2]}

    pair_style hybrid eam/alloy lj/cut 10 zero 10
    pair_coeff * * eam/alloy FeNiCr_Bonny_2013_ptDef.eam.alloy Fe NULL
    pair_coeff 1 2 lj/cut 2.7466705 1.8530700 7.41227500
    pair_coeff 2 2 zero

    fix nve all nve
    
    # dump custom all custom 100 collision_{collision_label}.lmp id type x y z

    compute pe_per_atom atom pe/atom
    compute pe_colliding_atom atom reduce sum c_pe_per_atom
    fix ave_pe all ave/time 1 1 1 c_pe_colliding_atom file results/pe_{collision_label}.dat ave one
    
    run {collision_steps}
    """

    run_lammps(input_file_contents, f"in.collision_{collision_label}")


def generate_velocities(n_samples, mb_scale, seed):
    np.random.seed(seed)

    # generate a set of random velocities according to our requirements
    magnitudes = scipy.stats.maxwell.rvs(size=n_samples, scale=mb_scale)
    directions = scipy.stats.multivariate_normal.rvs(size=np.array([n_samples, 3]))
    # ensure all velocity vectors will point towards the nanoparticle
    directions[:, 0] = -np.abs(directions[:, 0])

    # scale velocities with appropriate magnitude
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    velocities = directions * magnitudes[:, np.newaxis]

    return velocities, directions


def generate_positions(directions, nanoparticle_radius, initial_distance):
    return np.array([nanoparticle_radius, 0, 0]) - initial_distance * directions


# get maxwell-boltzmann scale in picosecond / Angstrom
# assumes temperature is in K and mass in g / mol
def calculate_mb_scale(temperature, mass):
    return 0.91183677 * np.sqrt(temperature / mass)


def generate_simulation_parameters(
    nanoparticle_radius, temperature, n_samples, seed, atom_mass=12
):
    mb_scale = calculate_mb_scale(temperature, atom_mass)
    atom_velocities, directions = generate_velocities(n_samples, mb_scale, seed)
    atom_positions = generate_positions(
        directions, nanoparticle_radius, initial_distance=20
    )

    return atom_positions, atom_velocities