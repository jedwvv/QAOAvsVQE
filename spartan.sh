#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Fri Jul 30 2021 10:25:08 GMT+1000 (Australian Eastern Standard Time)

# Partition for the job:
#SBATCH --partition=physical

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="random_init_0-99_12cpus"

# The project ID which this job should run under:
#SBATCH --account="punim0147"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12



# The amount of memory in megabytes per process in the job:
#SBATCH --mem=16000

# Use this email address:
#SBATCH --mail-user=jedwvv@gmail.com

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=1-0:00:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:
module load fosscuda/2020b
module load python/3.8.6

source ~/venv1/python3.8.6/bin/activate

# The job command(s):
for i in {0..99}; do python random_init_QAOA.py -N 3 -R 3 -M 4 -S ${i} -O LN_BOBYQA -F; done | tee random_results.txt
