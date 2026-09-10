#!/bin/bash
KA_values=(0.4)
PE_values=(0.4)
TEMP_values=(0.05 0.1 0.2)
max_parallel_jobs=8
ncores=4
function max_jobs {
    while (( $(jobs -r | wc -l) >= $1 )); do
        sleep 1
    done
}
function inplace_sed {
    local file="$1"
    local search="$2"
    local replace="$3"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/${search}/${replace}/g" "$file"
    else
        sed -i "s/${search}/${replace}/g" "$file"
    fi
}
simulation_counter=1
num_core_sets=4
for KA in "${KA_values[@]}"
do
    for PE in "${PE_values[@]}"
    do
        for temperature in "${TEMP_values[@]}"
        do
            dir_name="Pe_${PE}_T_${temperature}_k_${KA}"
            mkdir "$dir_name"
            cp simu.in "$dir_name"
            cp "Initial_configs/initial_config_Pe_0.1_T_0.1_k_0.2__.data" "$dir_name/polymers.dat"
            cp wall.dat "$dir_name/wall.dat"
            input_file="$dir_name/simu.in"
            inplace_sed "$input_file" "TEMP" "${temperature}"
            inplace_sed "$input_file" "PE" "${PE}"
            inplace_sed "$input_file" "KAPPA" "${KA}"
            inplace_sed "$input_file" "SEED" "${RANDOM}"
            cd "$dir_name"
            core_set=$(( (simulation_counter - 1) % num_core_sets + 1 ))
            launch_script="launch_simulation_${simulation_counter}.sh"
            cat > "$launch_script" <<EOL
mpirun -np $ncores ~/Softwares/lammps/build/lmp -i simu.in
EOL
            chmod +x "$launch_script"
            bash $launch_script &
            ((simulation_counter++))
            cd -
            max_jobs "$max_parallel_jobs"
            echo "Started simulation for PE=${PE}, KA=${KA}, T=${temperature}"
        done
    done
done
wait
echo "All simulations are complete."
