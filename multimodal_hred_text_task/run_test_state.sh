########################################
# EXAMPLE RUN: ./run_test_state.sh <DATA_DIR> <DUMP_DIR> 1 <TEST_STATE> (for V1 dataset) and ./run.sh <DATA_DIR> <DUMP_DIR> 2 <TEST_STATE>(for V2 dataset)
# TEST_STATE can be either of the following: ask_attribute,buy,celebrity,do_not_like_earlier_show_result,do_not_like_n_show_result,do_not_like_show_result,filter_results,go_with,like_earlier_show_result,like_n_show_result,like_show_result,show_orientation,show_result,show_similar_to,sort_results,suited_for
########################################
cp params_test_v$3.py params_test.py
mkdir $2/e_test_state
mkdir $2/o_test_state
#jbsub -mem 200g -queue p8_12h -err $2/e_test_state/e_test_state_$4.txt -out $2/o_test_state/o_test_state_$4.txt -require k80 python run_test_task1.py $1 $2 $3 $4
python run_test_task1.py $1 $2 $4
