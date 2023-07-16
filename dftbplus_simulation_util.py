import itertools

from pathlib import Path

import ase
import ase.build
import ase.cluster
import ase.io
import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats


def parse_output_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    output_data = []
    current_output = {}

    for line in lines:
        if line.startswith("MD step:"):
            if current_output:
                output_data.append(current_output)
            current_output = {"step": int(line.split(":")[1].strip())}
        elif line.startswith("Potential Energy:"):
            energy = line.split()[-2]
            current_output["pe"] = float(energy)
        elif line.startswith("MD Kinetic Energy:"):
            energy = line.split()[-2]
            current_output["ke"] = float(energy)
        elif line.startswith("MD Temperature:"):
            temperature = line.split()[-2]
            current_output["temperature"] = float(temperature)

    if current_output:
        output_data.append(current_output)

    df = pd.DataFrame(output_data)

    return df


def generate_collider(structure_file, molecule):
    collider = ase.build.molecule(molecule)
    ase.io.write(structure_file, collider)


def generate_nanoparticle(
    structure_file, symbol, lattice_type, lattice_constant, radius
):
    # generate a lattice cube that should contain the spherical nanoparticle
    surfaces = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    n_layers = int(np.ceil(2 * radius / lattice_constant))
    layers = 3 * [n_layers]
    if lattice_type == "fcc":
        cluster = ase.cluster.FaceCenteredCubic(
            symbol, surfaces, layers, latticeconstant=lattice_constant
        )
    elif lattice_type == "bcc":
        cluster = ase.cluster.BodyCenteredCubic(
            symbol, surfaces, layers, latticeconstant=lattice_constant
        )

    # filter for spherical nanoparticle and write to file
    cluster = cluster[
        np.sum(cluster.positions * cluster.positions, axis=1) < radius * radius
    ]
    ase.io.write(structure_file, cluster)


def generate_velocity_string(velocities):
    vel_string = "Velocities [AA/ps] { \n"
    for velocity in velocities:
        vel_string += f"        {' '.join(f'{v:16.8f}' for v in velocity)}\n"
    vel_string += "    }"
    return vel_string


def dftbplus_collide(
    structure_file,
    input_file,
    time_step,
    n_steps,
    xtb_method,
    velocities,
):
    velocity_string = generate_velocity_string(velocities)

    file_contents = f"""
    Geometry = xyzFormat {{
    <<< "{structure_file}"
    }}
    
    Driver = VelocityVerlet {{
        TimeStep [fs] = {time_step}
        Steps = {n_steps}
        MovedAtoms = 1:-1
        {velocity_string}
    }}

    Hamiltonian = xTB {{
        Method = "{xtb_method}"
    }}

    Options {{ WriteDetailedOut = No }}
    Analysis {{ CalculateForces = Yes }}
    ParserOptions {{ ParserVersion = 10 }}
    Parallel {{ UseOmpThreads = Yes }}
    """

    file_contents = file_contents.strip().replace("\n    ", "\n")

    with open(input_file, "w") as f:
        f.write(file_contents)


def dftbplus_equilibrate(
    structure_file,
    input_file,
    time_step,
    n_steps,
    temperature,
    nose_hoover_coupling,
    xtb_method,
):
    file_contents = f"""
    Geometry = xyzFormat {{
    <<< "{structure_file}"
    }}
    
    Driver = VelocityVerlet {{
        TimeStep [fs] = {time_step}
        Thermostat = NoseHoover {{
            Temperature [Kelvin] = {temperature}
            CouplingStrength = {nose_hoover_coupling}
        }}
        Steps = {n_steps}
        MovedAtoms = 1:-1
    }}

    Hamiltonian = xTB {{
        Method = "{xtb_method}"
    }}

    Options {{ WriteDetailedOut = No }}
    Analysis {{ CalculateForces = Yes }}
    ParserOptions {{ ParserVersion = 10 }}
    Parallel {{ UseOmpThreads = Yes }}
    """

    file_contents = file_contents.strip().replace("\n    ", "\n")

    with open(input_file, "w") as f:
        f.write(file_contents)


def write_cottrell_slurm_file(slurm_file, paths, n_cores):
    nl = "\n      "
    bash_path_array = f"""(
      {nl.join(paths)}
    )
    """

    file_contents = f"""
    #!/bin/bash
    
    #SBATCH --job-name=dftb_md
    #SBATCH --output=%x_%j.out
    #SBATCH --nodes=1
    #SBATCH --ntasks={n_cores}
    #SBATCH --no-requeue
    #SBATCH --partition=all
    #SBATCH --array=0-{len(paths) - 1}

    paths={bash_path_array}
    
    application="dftb+"
    options=""
    
    part=$SLURM_JOB_PARTITION
    jobname=$SLURM_JOB_NAME
    ntasks=$SLURM_NTASKS
    current_index=$SLURM_ARRAY_TASK_ID
    workdir="${{SLURM_SUBMIT_DIR}}/${{paths[current_index]}}"
    
    #load some modules and conda environments
    module purge
    module -q load intel
        
    #creating a string which has the command
    CMD="cd ${{workdir}}; ${{application}} ${{options}}"
    
    #just some info for the output file
    echo "Running command: $CMD"
    echo "# Threads: ${{ntasks}}"
    echo "Partition: ${{part}}"
    echo "Submission directory: ${{workdir}}"
    
    #now the actual commands
    eval $CMD
    """

    file_contents = file_contents.strip().replace("\n    ", "\n")

    with open(slurm_file, "w") as f:
        f.write(file_contents)


def write_nanoparticle_files(
    symbols,
    radii,
    temperatures,
    time_step=1.0,
    n_steps=10000,
    nose_hoover_coupling=0.01,
    xtb_method="GFN1-xTB",
    n_cores=4,
    structure_file="struc.xyz",
    input_file="dftb_in.hsd",
    slurm_file="slurm_submit",
):
    paths = []

    for symbol in symbols:
        if symbol == "Fe":
            lattice_constant = 3.571
            lattice_type = "fcc"
        elif symbol == "Co":
            lattice_constant = 3.486
            lattice_type = "fcc"

        for radius in radii:
            for temperature in temperatures:
                path = f"nanoparticle/{symbol}_{radius}A_{temperature}K"
                paths += [path]
                Path(path).mkdir(parents=True, exist_ok=True)

                generate_nanoparticle(
                    f"{path}/{structure_file}",
                    symbol,
                    lattice_type,
                    lattice_constant,
                    radius,
                )
                dftbplus_equilibrate(
                    structure_file,
                    f"{path}/{input_file}",
                    time_step,
                    n_steps,
                    temperature,
                    nose_hoover_coupling,
                    xtb_method,
                )

    write_cottrell_slurm_file(slurm_file, paths, n_cores)


def write_collider_files(
    molecules,
    temperatures,
    time_step=1.0,
    n_steps=10000,
    nose_hoover_coupling=0.01,
    xtb_method="GFN1-xTB",
    n_cores=4,
    structure_file="struc.xyz",
    input_file="dftb_in.hsd",
    slurm_file="slurm_submit",
):
    paths = []

    for molecule in molecules:
        for temperature in temperatures:
            path = f"collider/{molecule}_{temperature}K"
            paths += [path]
            Path(path).mkdir(parents=True, exist_ok=True)

            generate_collider(
                f"{path}/{structure_file}",
                molecule,
            )
            dftbplus_equilibrate(
                structure_file,
                f"{path}/{input_file}",
                time_step,
                n_steps,
                temperature,
                nose_hoover_coupling,
                xtb_method,
            )

    write_cottrell_slurm_file(slurm_file, paths, n_cores)


def tally_elements(elements):
    element_counts = {}

    for element in elements:
        if element in element_counts:
            element_counts[element] += 1
        else:
            element_counts[element] = 1

    element_string = ""
    for element, count in element_counts.items():
        element_string += f"{element}{count}"

    return element_string


def parse_trajectory(file_name):
    with open(file_name, "r") as file:
        lines = file.readlines()

    last_timestep_start = None

    # Find the index of the last "MD iter" line
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("MD iter"):
            last_timestep_start = i + 1
            break

    if last_timestep_start is None:
        raise ValueError('No "MD iter" line found in the file.')

    last_timestep = lines[last_timestep_start:]

    element_symbols = []
    positions = []
    velocities = []

    for line in last_timestep:
        tokens = line.split()
        element_symbols.append(tokens[0])
        positions.append([float(tokens[i]) for i in range(1, 4)])
        velocities.append([float(tokens[i]) for i in range(5, 8)])

    element_string = tally_elements(element_symbols)

    return ase.Atoms(element_string, positions=positions, velocities=velocities)


def centre_atoms(atoms):
    symbols = str(atoms.symbols)
    positions = atoms.get_positions()
    velocities = atoms.get_velocities()
    masses = atoms.get_masses()

    total_mass = np.sum(masses)
    com_position = np.sum(masses[:, np.newaxis] * positions, axis=0) / total_mass
    com_velocity = np.sum(masses[:, np.newaxis] * velocities, axis=0) / total_mass

    positions = positions - com_position
    velocities = velocities - com_velocity

    return ase.Atoms(symbols, positions=positions, velocities=velocities)


def initialise_collider(atoms, position_offset, velocity_offset):
    symbols = str(atoms.symbols)
    positions = atoms.get_positions() + position_offset
    velocities = atoms.get_velocities() + velocity_offset

    return ase.Atoms(symbols, positions=positions, velocities=velocities)


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
    nanoparticle_radius, temperature, n_samples, seed, mass=12
):
    mb_scale = calculate_mb_scale(temperature, mass)
    atom_velocities, directions = generate_velocities(n_samples, mb_scale, seed)
    atom_positions = generate_positions(
        directions, nanoparticle_radius, initial_distance=10
    )

    return atom_positions, atom_velocities


def write_collision_files(
    symbols,
    radii,
    molecules,
    temperatures,
    n_samples,
    seed,
    time_step=1.0,
    n_steps=10000,
    xtb_method="GFN1-xTB",
    n_cores=4,
    trajectory_file="geo_end.xyz",
    structure_file="struc.xyz",
    input_file="dftb_in.hsd",
    slurm_file="slurm_submit",
):
    paths = []
    parameters = itertools.product(symbols, radii, molecules, temperatures)

    for symbol, radius, molecule, temperature in parameters:
        nanoparticle_path = f"nanoparticle/{symbol}_{radius}A_{temperature}K"
        collider_path = f"collider/{molecule}_{temperature}K"

        nanoparticle = parse_trajectory(f"{nanoparticle_path}/{trajectory_file}")
        collider = parse_trajectory(f"{collider_path}/{trajectory_file}")

        nanoparticle = centre_atoms(nanoparticle)
        collider = centre_atoms(collider)

        collider_mass = np.sum(collider.get_masses())
        position_offsets, velocity_offsets = generate_simulation_parameters(
            radius, temperature, n_samples, seed, mass=collider_mass
        )

        for i in range(n_samples):
            path = f"collision/{symbol}_{radius}A_{molecule}_{temperature}K_{i}"
            paths += [path]
            Path(path).mkdir(parents=True, exist_ok=True)

            sample_collider = initialise_collider(
                collider, position_offsets[i], velocity_offsets[i]
            )
            collision_atoms = nanoparticle + sample_collider
            collision_atoms_no_velocities = ase.Atoms(
                collision_atoms.symbols, collision_atoms.get_positions()
            )
            ase.io.write(f"{path}/{structure_file}", collision_atoms_no_velocities)

            dftbplus_collide(
                structure_file,
                f"{path}/{input_file}",
                time_step,
                n_steps,
                xtb_method,
                collision_atoms.get_velocities(),
            )

    write_cottrell_slurm_file(slurm_file, paths, n_cores)
