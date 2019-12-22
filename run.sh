#python translater.py
for MARGIN in 0.2 0.3 0.4 0.5
do
  echo $MARGIN
  CUDA_VISIBLE_DEVICES=7 python ./code/train.py --cuda -lr 0.0001 --num_epochs 6 --model_name Coattention --name Coattention --negative_sample_size 3 --margin 0.2
  # CUDA_VISIBLE_DEVICES=7 python ./code/train.py --cuda --model_name CNNRNN --name CNNRNN_$MARGIN --negative_sample_size 5 --margin $MARGIN
  # CUDA_VISIBLE_DEVICES=6 python ./code/train.py --cuda --model_name RNN --name RNN_$MARGIN --negative_sample_size 5 --margin $MARGIN
done

CUDA_VISIBLE_DEVICES=7 python ./code/train.py --cuda -lr 0.001 --margin 0.5 \
                                              --negative_sample_size 5 --num_epochs 10 \
                                              --model_name RNN --name RNN

CUDA_VISIBLE_DEVICES=6 python ./code/train.py --cuda -lr 0.001 --batch_size 32 --margin 0.5 \
                                              --negative_sample_size 3 --num_epochs 10 \
                                              --model_name Coattention --name Coattention

CUDA_VISIBLE_DEVICES=6 python ./code/train.py --cuda --name _ --model_name Coattention --best_model_path ./models/best_Coattention.pt
