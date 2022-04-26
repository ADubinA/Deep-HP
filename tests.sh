cd F://dev/pointermorpher
python test.py --algo_type hp     --exp_type same
python test.py --algo_type deephp --exp_type same
python test.py --algo_type fpfh   --exp_type same

python test.py --algo_type hp     --exp_type same --aug
python test.py --algo_type deephp --exp_type same --aug
python test.py --algo_type fpfh   --exp_type same --aug

python test.py --algo_type hp     --exp_type verse
python test.py --algo_type deephp --exp_type verse
python test.py --algo_type fpfh   --exp_type verse

python test.py --algo_type hp     --exp_type gl
python test.py --algo_type deephp --exp_type gl
python test.py --algo_type fpfh   --exp_type gl