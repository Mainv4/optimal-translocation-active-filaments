#!/bin/bash
ARGS=$(getopt -o t:p:k:h --long temperature:,peclet:,kappa:,help -n 'launch_simu.sh' -- "$@")
eval set -- "$ARGS"
while true; do
    case "$1" in
        -t|--temperature) temperature="$2"; shift 2 ;;
        -p|--peclet) peclet="$2"; shift 2 ;;
        -k|--kappa) kappa="$2"; shift 2 ;;
        -h|--help)
            echo "This is a shell script to launch a simulation with LAMMPS. It will create a directory 
            with the name of the simulation, for example: Pe_0.1_T_1.0_k_0.1.
            Then, it will copy the input script in the directory, and modify it to set the temperature and the
            Péclét number and attribute a random seed to the Langevin thermostat. Finally, it will launch the 
            simulations with LAMMPS.
            Then, it will launch a python script that modifies the output of LAMMPS to make it easier to analyse afterwards."
            echo "Usage: ./launch_simu.sh -t <temperature> -p <Péclét number> -k <Kappa>"
            echo "Example: ./launch_simu.sh -t 1.0 -p 0.1 -k 0.1"
            exit 0 ;;
        --) shift; break ;;
        *) break ;;
    esac
done
if [ -z "$temperature" ] || [ -z "$peclet" ] || [ -z "$kappa" ]; then
    echo "Missing parameters..."
    exit 1
fi
dir_name="Pe_${peclet}_T_${temperature}_k_${kappa}"
mkdir $dir_name
cp *.in $dir_name
cp polymers.dat $dir_name
sed -i "s/TEMP/${temperature}/g" $dir_name/*.in
sed -i "s/PE/${peclet}/g" $dir_name/*.in
sed -i "s/KAPPA/${kappa}/g" $dir_name/*.in
sed -i "s/SEED/${RANDOM}/g" $dir_name/*.in
cd $dir_name
mpirun -np 8 -cpu-set SET ~/Softwares/lammps/build/lmp -i second_eq.in
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' 's/0 100 zlo zhi/0 10 zlo zhi/g' poly_eq_2.dat
else
    sed -i 's/0 100 zlo zhi/0 10 zlo zhi/g' poly_eq_2.dat
fi
mpirun -np 8 -cpu-set SET ~/Softwares/lammps/build/lmp -i simu.in
cd ../
python3 unwrapTrajectory.py --path $dir_name/ --prefix PolymGravity. --topology polymers.dat --L 40&
echo "Done for ${dir_name}"
