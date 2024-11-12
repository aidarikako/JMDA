# [TOMM 2024] Joint Mixing Data Augmentation for Skeleton-based Action Recognition

This repo is the official implementation for [Joint Mixing Data Augmentation for Skeleton-based Action Recognition](https://dl.acm.org/doi/10.1145/3700878). The paper is accepted to ACM Transactions on Multimedia Computing, Communications and Applications 2024.



## The overall framework of Joint Mixing Data Augmentation (JMDA)
![image](https://github.com/aidarikako/JMDA/blob/main/framwork.jpg)

## Results

|  Method              |  NTU RGB+D 60  X-Sub(%) | NTU RGB+D 60  X-View(%) | NTU RGB+D 120  X-Sub(%) | NTU RGB+D 120  X-Set(%) |
|-------------------|-----------|----------|--------|--------|
| USTC-IAT-Unite    | 0.72      | 0.73     | 0.59   | 0.68   |
| AI-lab            | 0.69      | 0.72     | 0.54   | 0.65   |
| **HFUT-LMC (Ours)**| **0.76** | **0.67** | **0.49** | **0.64** |
| Syntax            | 0.72      | 0.69     | 0.5    | 0.64   |
| ashk              | 0.72      | 0.69     | 0.42   | 0.61   |
| YKK               | 0.68      | 0.66     | 0.36   | 0.54   |
| Xpace             | 0.7       | 0.7      | 0.34   | 0.58   |
| nox               | 0.68      | 0.66     | 0.35   | 0.57   |
| SP-team           | 0.68      | 0.65     | 0.34   | 0.56   |
| YI.YJ             | 0.6       | 0.52     | 0.3    | 0.47   |
| MM24 Baseline     | 0.64      | 0.51     | 0.09   | 0.41   |





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
