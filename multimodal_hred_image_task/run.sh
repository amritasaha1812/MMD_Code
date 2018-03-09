#rm $1/dump/*
cp params_v$2.py params.py
#python run_model_task2.py $1
jbsub -mem 200G -queue p8_7d -err $1/e.txt -out $1/o.txt -require k80 python run_model_task2.py $1 
