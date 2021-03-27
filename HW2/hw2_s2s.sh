
default_dir='/Users/rongwang/Desktop/MLDS_hw2_1_data/testing_data'
test_dir=${1:-$default_dir}
echo $test_dir

default_out='./att_output.txt'
output_file=${2:-$default_out}

python model_test.py --load_saver=True --test_dir=$test_dir \
                     --test_mode=True --output_filename=$output_file \
                     --batch_size=100 --save_dir=save_models
python bleu_eval.py $output_file
