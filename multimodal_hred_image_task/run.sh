cp params_v$3.py params.py
###########################################################
#EXAMPLE RUN: ./run.sh <DATA_DIR> <DUMP_DIR> 1 (for V1 DATASET) and ./run.sh <DATA_DIR> <DUMP_DIR> 2 (for V2 DATASET)
#jbsub -mem 200G -queue p8_7d -err $1/e.txt -out $1/o.txt -require k80 python run_model_task2.py $1 $2
python run_model_task2.py $1 $2
