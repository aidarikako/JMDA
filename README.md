# [TOMM 2024] Joint Mixing Data Augmentation for Skeleton-based Action Recognition

This repo is the official implementation for [Joint Mixing Data Augmentation for Skeleton-based Action Recognition](https://dl.acm.org/doi/10.1145/3700878). The paper is accepted to ACM Transactions on Multimedia Computing, Communications and Applications 2024.



## The overall framework of Joint Mixing Data Augmentation (JMDA)
![image](https://github.com/aidarikako/JMDA/blob/main/framwork.jpg)

## Results

|  Method              |  NTU RGB+D 60  X-Sub(%) | NTU RGB+D 60  X-View(%) | NTU RGB+D 120  X-Sub(%) | NTU RGB+D 120  X-Set(%) |
|-------------------|-----------|----------|--------|--------|
| 2s-AGCN    | 88.5     | 95.1     | 82.9   | 84.9   |   
| **2s-AGCN + JMDA**           | **89.0(+0.5)**      | **95.9(+0.8)**    | **85.7(+2.8)**   | **87.4(+2.5)**   |  
|-------------------|-----------|----------|--------|--------|
|  CTR-GCN    |  92.4     |  96.8     | 88.9   |  90.6  |      
| **CTR-GCN + JMDA**           | **92.9(+0.5)**      | 96.6(-0.2)    | **89.2(+0.3)**   | **90.9(+0.3)**   | 
|-------------------|-----------|----------|--------|--------|
|   SkeletonMixFormer(3s)   |  92.6     |  96.9     | 89.8   |  91.2 |         
| **SkeletonMixFormer(3s) + JMDA**           | **93.2(+0.6)**      | **97.0(+0.1)**    | **90.1(+0.3)**   | **91.4(+0.2)**   | 
|-------------------|-----------|----------|--------|--------|
|   SkeletonMixFormer(6s)   |  93.2    |  97.2     | 90.2   |  91.5 |       
| **SkeletonMixFormer(6s) + JMDA**           | **93.7(+0.5)**      | 97.2    | **90.9(+0.7)**   | **91.9(+0.4)**   |





# Citation
If you find this repo helpful, please consider citing:

```
@article{xiang2024joint,
  title={Joint Mixing Data Augmentation for Skeleton-based Action Recognition},
  author={Xiang, Linhua and Wang, Zengfu},
  journal={ACM Transactions on Multimedia Computing, Communications and Applications},
  year={2024},
  publisher={ACM New York, NY}
}
```
