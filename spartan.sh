#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Fri Jul 30 2021 10:25:08 GMT+1000 (Australian Eastern Standard Time)

# Partition for the job:
#SBATCH --partition=physical

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="test_solve_qubo_qaoa"

# The project ID which this job should run under:
#SBATCH --account="punim0147"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2



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
#SBATCH --time=2:00:00

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
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 0
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 1
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 2
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 3
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 4
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 5
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 6
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 7
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 8
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 9
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 10
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 11
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 12
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 13
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 14
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 15
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 16
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 17
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 18
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 19
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 20
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 21
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 22
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 23
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 24
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 25
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 26
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 27
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 28
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 29
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 30
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 31
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 32
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 33
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 34
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 35
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 36
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 37
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 38
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 39
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 40
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 41
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 42
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 43
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 44
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 45
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 46
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 47
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 48
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 49
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 50
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 51
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 52
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 53
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 54
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 55
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 56
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 57
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 58
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 59
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 60
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 61
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 62
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 63
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 64
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 65
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 66
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 67
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 68
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 69
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 70
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 71
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 72
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 73
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 74
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 75
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 76
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 77
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 78
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 79
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 80
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 81
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 82
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 83
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 84
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 85
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 86
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 87
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 88
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 89
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 90
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 91
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 92
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 93
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 94
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 95
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 96
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 97
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 98
python solve_qubo_qaoa.py -N 4 -R 3 -P 1.5 -O SLSQP -M 5 -T 5 -S 99
