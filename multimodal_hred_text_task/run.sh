########################################
# EXAMPLE RUN: ./run.sh <DATA_DIR> <DUMP_DIR> 1 (for V1 dataset) and ./run.sh <DATA_DIR> <DUMP_DIR> 2 (for V2 dataset)
########################################
cp params_v$3.py params.py
#jbsub -mem 200G -queue p8_7d -err $1/e.txt -out $1/o.txt -require k80 python run_model_task1.py $1 $2 $3
python run_model_task1.py $1 $2
