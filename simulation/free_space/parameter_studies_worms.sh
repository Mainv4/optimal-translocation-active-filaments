#!/bin/bash
temperatures=(0.1 0.2)
peclets=(0.0)
kappas=(0.6 0.8 1.0 2.0)
max_concurrent_jobs=4
cpus_per_job=8
job_count=0
total_simulations=$((${#temperatures[@]} * ${#peclets[@]} * ${#kappas[@]}))
echo "Starting parameter study. Total simulations to run: ${total_simulations}"
echo "Using ${max_concurrent_jobs} concurrent jobs with ${cpus_per_job} CPUs per job, utilizing up to $((max_concurrent_jobs * cpus_per_job)) cores."
for temperature in "${temperatures[@]}"; do
  for peclet in "${peclets[@]}"; do
    for kappa in "${kappas[@]}"; do
      cpu_set_index=$((job_count % max_concurrent_jobs))
      start_core=$((cpu_set_index * cpus_per_job))
      end_core=$((start_core + cpus_per_job - 1))
      core_range="${start_core}-${end_core}"
      script_file="launch_simu_T_${temperature}_Pe_${peclet}_k_${kappa}.sh"
      echo "Preparing ${script_file} (Job $((job_count + 1))/${total_simulations}) for cores ${core_range}"
      cp launch_simu.sh "$script_file"
      sed -i "s|SET|${core_range}|g" "$script_file"
      echo "Launching ${script_file} with T=${temperature} Pe=${peclet} k=${kappa}"
      bash "$script_file" -t "$temperature" -p "$peclet" -k "$kappa" &
      job_count=$((job_count + 1))
      if ((job_count % max_concurrent_jobs == 0)); then
        echo "Waiting for batch of ${max_concurrent_jobs} jobs to finish..."
        wait
        echo "Batch finished."
      fi
    done
  done
done
if ((job_count % max_concurrent_jobs != 0)); then
  echo "Waiting for the last remaining jobs..."
  wait
  echo "Remaining jobs finished."
fi
echo "Parameter study complete. ${job_count} simulations launched."
