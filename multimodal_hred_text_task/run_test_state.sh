#rm $1/dump/*
#cp params.py $1
#python run_model_task2.py $1
mkdir $1/e_test_state
mkdir $1/o_test_state
jbsub -mem 200g -queue p8_12h -err $1/e_test_state/e_test_state_$2.txt -out $1/o_test_state/o_test_state_$2.txt -require k80 python run_test_task1.py $1 $2
