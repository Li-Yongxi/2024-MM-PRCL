
PyTorch implementation for Propagation based Recycling Contrastive Learning for Noisy Visible-Infrared Person Re-identification (TMM 2024).


## Introduction

### DART framework
![image](https://github.com/Li-Yongxi/2024-MM-PRCL/assets/154648436/75a4485d-a936-490b-a937-11519fa4471b)


## Requirements

- Python 3.7
- PyTorch ~1.7.1
- numpy
- scikit-learn
- apex
- faiss-gpu
## Datasets

### SYSU-MM01 and RegDB
We follow [ADP](https://github.com/mangye16/Cross-Modal-Re-ID-baseline/tree/master/ICCV21_CAJ) to obtain datasets.

## Training and Evaluation

### Training

Modify the ```data_path``` and  specify the ```noise_ratio``` to train the model.

```train
# SYSU-MM01: noise_ratio = {0, 0.2, 0.5}
python run.py --gpu 0 --dataset sysu --data-path data_path --noise-rate 0.2 --savename sysu_dart_nr20
python run.py --epoch 100 --gpu 0 --dataset sysu --batch-size 8 --lr 0.1 --data-path data_path --noise-rate 0.2 --model_path ./save_model/{} --savename sysu_nra20_contrastive --warm-epoch 2 --loss1 sid --loss2 weighted_crosscon --loss3 graph_con_feat_progressive --p-threshold 0.3

# RegDB: noise_ratio = {0, 0.2, 0.5}, trial = 1-10
python run.py --epoch 100 --gpu 0 --dataset regdb --batch-size 8 --lr 0.1 --data-path data_path --noise-rate 0.2 --model_path ./save_model/{} --savename regdb_dart_nr20_contrastive --warm-epoch 2 --loss1 sid --loss2 weighted_crosscon --loss3 graph_con_feat_progressive --p-threshold 0.3 --trial 1

```
### Evaluation

Modify the  ```data_path``` and ```model_path``` to evaluate the trained model. 

```
# SYSU-MM01: mode = {all, indoor}
python test.py --gpu 0 --dataset sysu --data-path data-path --model_path model_path --resume-net1 sysu_nra20_contrastive_graph_best_net1.t --resume-net2 sysu_nra20_contrastive_graph_best_net2.t --mode all

# RegDB: --tvsearch or not (whether thermal to visible search)
python test.py --gpu 0 --dataset regdb --data-path data-path --model_path model_path --resume-net1 regdb_dart_nr20_trial{}_net1.t --resume-net2 regdb_dart_nr20_trial{}_net2.t

```

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

## Acknowledgements
The code is based on [ADP](https://github.com/mangye16/Cross-Modal-Re-ID-baseline/tree/master/ICCV21_CAJ) and [DART](https://github.com/XLearning-SCU/2022-CVPR-DART) licensed under Apache 2.0.
