# Deep Blind Image Quality Assessment Using Dual-Order Statistics

This repository is the official PyTorch implementation of paper "[Deep Blind Image Quality Assessment Using Dual-Order Statistics](https://ieeexplore.ieee.org/abstract/document/9859608)" in ICME2022. 


### Requirements

You will need the following requirements:

- numpy >= 1.17.4
- pandas >= 0.25.3
- python >= 3.7.5
- pytorch >= 1.4.0
- torchvision >= 0.5.0
- tensorboard >= 2.0.0



### Training
Need to modify the path of dataset and code to yours:

```
python /home/Joanne/Codes/DualOrderStatistics_IQA/main.py --data /home/Joanne/IQA_datasets/KonIQ-10k -b 11 --epochs 100 --dataset koniq10k --comment koniq10k_GAP_fc_train_RTX --train-size 8000 --tensorboard 
```



### Evaluation
Need to modify the path of dataset, code and pre-trained models to yours:

```
python /home/Joanne/Codes/DualOrderStatistics_IQA/cross_test.py --data /home/Joanne/IQA_datasets/live_c/ChallengeDB_release --checkpoint 20221013-114412_koniq10k_GAP_fc_train_RTX --dataset livec
```




### Citation

```
@inproceedings{zhou2022deep,
  title={Deep Blind Image Quality Assessment Using Dual-Order Statistics},
  author={Zhou, Zihan and Xu, Yong and Quan, Yuhui and Xu, Ruotao},
  booktitle={2022 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={01--06},
  year={2022},
  organization={IEEE}
}
```



### Acknowledgments
The code is based on the implementation of "[BLIND NATURAL IMAGE QUALITY PREDICTION USING CONVOLUTIONAL NEURAL NETWORKS AND WEIGHTED SPATIAL POOLING](https://github.com/yichengsu/ICIP2020-WSP-IQA#)". See more in [GitHub - yichengsu/ICIP2020-WSP-IQA](https://github.com/yichengsu/ICIP2020-WSP-IQA).



