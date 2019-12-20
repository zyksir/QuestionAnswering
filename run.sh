#python translater.py
for MARGIN in 0.3 0.5 0.75 1
do
  echo $MARGIN
  CUDA_VISIBLE_DEVICES=7 python ./code/train.py --cuda --model_name CNNRNN --name CNNRNN_$MARGIN --negative_sample_size 5 --margin $MARGIN
  # CUDA_VISIBLE_DEVICES=6 python ./code/train.py --cuda --model_name RNN --name RNN_$MARGIN --negative_sample_size 5 --margin $MARGIN
done

#CUDA_VISIBLE_DEVICES=7 python ./code/train.py --cuda -lr 0.001 --num_epochs 10 --model_name RNN --name RNN --margin 0.75
#CUDA_VISIBLE_DEVICES=7 python ./code/train.py --cuda -lr 0.001 --num_epochs 10 --model_name RNN --name RNN --margin 1
#CUDA_VISIBLE_DEVICES=7 python ./code/train.py --cuda -lr 0.001 --num_epochs 10 --model_name CNN --name CNN --negative_sample_size 10
#CUDA_VISIBLE_DEVICES=7 python ./code/train.py --cuda -lr 0.001 --num_epochs 10 --model_name CNNRNN --name CNNRNN --negative_sample_size 5 --margin 0.3
#CUDA_VISIBLE_DEVICES=7 python ./code/train.py --cuda -lr 0.001 --num_epochs 5 --model_name Coattention --name Coattention --negative_sample_size 1 --margin 0.2
