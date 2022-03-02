#!/bin/sh

# qsub -q long -lnodes=1:ppn=4:msc job.sh

module load python/3.8.5

LOGFILE="jobs.log"

cd $PBS_O_WORKDIR

echo "Starting Job at: "`date` > $LOGFILE 2>&1

pwd >> $LOGFILE 2>&1

which python >> $LOGFILE 2>&1

python Assembled.py -jobs 30 >> $LOGFILE 2>&1

python Assembled.py -jobs 30 -neighbourhood True -results_file None_results.pkl -param_file parameters.pkl -num_samples 1000 >> $LOGFILE 2>&1

echo "Finished Job at: "`date` >> $LOGFILE 2>&1 
