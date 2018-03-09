#rm $1/dump/*
#cp params.py $1
#python run_model_task2.py $1
jbsub -mem 100g -queue p8 -err $1/e_$3_test.txt -out $1/o_$3_test.txt -require k80 python run_test_task2.py $1 $2 $3 
