# [TOMM 2024] Joint Mixing Data Augmentation for Skeleton-based Action Recognition

This repo is the official implementation for [Joint Mixing Data Augmentation for Skeleton-based Action Recognition](https://dl.acm.org/doi/10.1145/3700878). The paper is accepted to ACM Transactions on Multimedia Computing, Communications and Applications 2024.



## The overall framework of Joint Mixing Data Augmentation (JMDA)
![image](https://github.com/aidarikako/JMDA/blob/main/framwork.jpg)

## Results

|  Method              |  NTU RGB+D 60  X-Sub(%) | NTU RGB+D 60  X-View(%) | NTU RGB+D 120  X-Sub(%) | NTU RGB+D 120  X-Set(%) |
|-------------------|-----------|----------|--------|--------|
| 2s-AGCN    | 88.5     | 95.1     | 82.9   | 84.9   |   
| **2s-AGCN + JMDA**           | **89.0(+0.5)**      | **95.9(+0.8)**    | **85.7(+2.8)**   | **87.4(+2.5)**   |  






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
