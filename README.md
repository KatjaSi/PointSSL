# PointSSL

## Download modelnet40 set

`python download.py`

## Pretrain PointSSL model on all train data

`python trainSSL.py`

This will train for 1000 epochs and save the model point_ssl_1000.t7 to checkpoints\models

Alternatively, use the one that is already there.


## Point cloud classification task using 1 % of labeled data

Not pretrained:

`python train.py --batch_size 64  --epochs 250`

This will save the model pointSSL_without_pretraining_250_2.t7 with best test accuracy to checkpoints\models.

Alternatively, it is already there. 

Achieved Best test accuracy: 41.86%

Pretrained:

`python train.py --batch_size 64  --epochs 250 --pretrained`

pointSSL_with_pretraining_250.t7 is in checkpoints/models as well

Achieved best test accuracy: 49.64%

## Test and compare Results

Test accuracy of not pretrained model:

`python test.py --saved_model pointSSL_without_pretraining_250_2.t7`

Test accuracy of pretrained model:

`python test.py --saved_model pointSSL_with_pretraining_250_2.t7`

## Vizualization

TSNE plot of embeddings. Num_classes <= 40, batch_size is the total number of point clouds vizualization

`python vizualization.py --num_classes 10 --batch_size 512`
![scatter_plot (2)](https://github.com/KatjaSi/PointSSL/assets/69903665/433cb2fb-e5a3-4057-8569-d036552d96f9)

0 airplane
1 bathtub
2 bed
3 bench
4 bookshelf
5 bottle
6 bowl
7 car
8 chair
9 cone

## Credits

The basis for the Transformer based model is from Strawberry-Eat-Mango/PCT_Pytorch

@misc{guo2020pct,
      title={PCT: Point Cloud Transformer}, 
      author={Meng-Hao Guo and Jun-Xiong Cai and Zheng-Ning Liu and Tai-Jiang Mu and Ralph R. Martin and Shi-Min Hu},
      year={2020},
      eprint={2012.09688},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
