#rm $1/dump/*
cp params_test_v$2.py params_test.py
jbsub -mem 100g -queue p8_24h -err $1/e_test.txt -out $1/o_test.txt -require k80 python run_test_task1.py $1 
