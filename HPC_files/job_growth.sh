#!/bin/sh

# qsub -q long -lnodes=1:ppn=4:msc job.sh

module load python/3.8.5

LOGFILE="jobs_growth.log"

cd $PBS_O_WORKDIR

echo "Starting Job at: "`date` > $LOGFILE 2>&1

pwd >> $LOGFILE 2>&1

which python >> $LOGFILE 2>&1

python Assembled.py -jobs 50 -growth linear >> $LOGFILE 2>&1

python Assembled.py -jobs 50 -neighbourhood True -results_file linear_results.pkl -param_file parameters.pkl -num_samples 1000 -growth linear >> $LOGFILE 2>&1

echo "Finished Job at: "`date` >> $LOGFILE 2>&1 
