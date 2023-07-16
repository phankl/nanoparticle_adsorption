from pathlib import Path

import ase.build
import ase.cluster
import ase.io
import numpy as np
import pandas as pd
import plotly.express as px

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

def equilibrate(
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
                equilibrate(
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
    n_cores=1,
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
            equilibrate(
                structure_file,
                f"{path}/{input_file}",
                time_step,
                n_steps,
                temperature,
                nose_hoover_coupling,
                xtb_method,
            )

    write_cottrell_slurm_file(slurm_file, paths, n_cores)