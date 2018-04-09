cp params_v$3.py params.py
###########################################################
#EXAMPLE RUN: ./run.sh <DATA_DIR> <DUMP_DIR> 1 (for V1 DATASET) and ./run.sh <DATA_DIR> <DUMP_DIR> 2 (for V2 DATASET)
###########################################################
#jbsub -mem 100g -queue p8 -err $1/e_$4_test.txt -out $1/o_$4_test.txt -require k80 python run_test_task2.py $1 $2
python run_test_task2.py $1 $2 
