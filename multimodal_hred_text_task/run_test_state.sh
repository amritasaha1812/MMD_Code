cp params_test_v$3.py params_test.py
mkdir $2/e_test_state
mkdir $2/o_test_state
#jbsub -mem 200g -queue p8_12h -err $2/e_test_state/e_test_state_$4.txt -out $2/o_test_state/o_test_state_$4.txt -require k80 python run_test_task1.py $1 $2 $3 $4
python run_test_task1.py $1 $2 $4
