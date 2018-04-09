####################################################
#EXAMPLE RUN ./run_test.sh <DATA_DIR> <DUMP_DIR> 1 (for V1 Dataset) and ./run_test.sh <DATA_DIR> <DUMP_DIR> 2 (for V2 Dataset)
####################################################
cp params_test_v$3.py params_test.py
#jbsub -mem 100g -queue p8_24h -err $1/e_test.txt -out $1/o_test.txt -require k80 python run_test_task1.py $1 $2
python run_test_task1.py $1 $2
