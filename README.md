# [TOMM 2024] Joint Mixing Data Augmentation for Skeleton-based Action Recognition

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/joint-mixing-data-augmentation-for-skeleton/skeleton-based-action-recognition-on-ntu-rgbd-1)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-ntu-rgbd-1?p=joint-mixing-data-augmentation-for-skeleton)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/joint-mixing-data-augmentation-for-skeleton/skeleton-based-action-recognition-on-ntu-rgbd)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-ntu-rgbd?p=joint-mixing-data-augmentation-for-skeleton)

This repo is the official implementation for [Joint Mixing Data Augmentation for Skeleton-based Action Recognition](https://dl.acm.org/doi/10.1145/3700878). The paper is accepted to ACM Transactions on Multimedia Computing, Communications and Applications 2024.



## The overall framework of Joint Mixing Data Augmentation (JMDA)
![image](https://github.com/aidarikako/JMDA/blob/main/framwork.jpg)

## Data Preparation & Training & Testing

Please refer to the readme of the original project, such as [CTR-GCN](https://github.com/aidarikako/JMDA/blob/main/ctrgcn/README.md) and [Skeleton MixFormer](https://github.com/aidarikako/JMDA/blob/main/Skeleton_mixformer/README.md).

**Training & Testing for CTR-GCN**：
```
# Example: training CTRGCN on NTU RGB+D 60 cross subject
python main.py --config config/nturgbd-cross-subject/default.yaml --work-dir work_dir/ctrgcn

# Example: ensemble four modalities of CTRGCN on NTU RGB+D 60 cross subject
python ensemble.py --dataset ntu/xsub --joint-dir work_dir/ctrgcn/joint --bone-dir work_dir/ctrgcn/bone --joint-motion-dir work_dir/ctrgcn/joint_vel --bone-motion-dir work_dir/ctrgcn/bone_vel
```

**Training & Testing for Skeleton MixFormer**：
```
# Example: training SKMIXF on NTU RGB+D 60 cross subject
python main.py --config config/nturgbd-cross-subject/default.yaml --work-dir work_dir/skmixf 

# Example: ensemble six modalities of SKMIXF on NTU RGB+D 60 cross subject
python ensemble.py --dataset ntu/xsub --joint-dir work_dir/skmixf/k=K_pos --bone-dir work_dir/skmixf/k=1_pos \
--joint-k2-dir work_dir/skmixf/k=2_pos --joint-motion-dir work_dir/skmixf/k=K_vel --bone-motion-dir work_dir/skmixf/k=1_vel --joint-motion-k2-dir work_dir/skmixf/k=2_vel
```

**Note**:

* The folder names used during testing, such as "work_dir/skmixf/k=K_pos", should be modified according to your own folder naming.
  
* We added code for automatic checkpoint recovery in CTR-GCN. For example, if training is interrupted while using the command below, you can rerun the command, and the model will automatically load the checkpoint from the previous epoch to resume training.
```
# Example
python main.py --config config/nturgbd-cross-subject/default.yaml --work-dir work_dir/ctrgcn
```

* We have modified the code of Skeleton MixFormer so that you can now easily adjust the value of "k" in the configuration file. For example, you can change the value of "k" in the "model_args" section of [this file](https://github.com/aidarikako/JMDA/blob/main/Skeleton_mixformer/config/nturgbd-cross-subject/default.yaml). You can also refer to this example to make similar changes in other configuration files for Skeleton MixFormer.


## Results

|  Method              |  NTU RGB+D 60  X-Sub(%) | NTU RGB+D 60  X-View(%) | NTU RGB+D 120  X-Sub(%) | NTU RGB+D 120  X-Set(%) |
|-------------------|-----------|----------|--------|--------|
|  CTR-GCN    |  92.4     |  96.8     | 88.9   |  90.6  |      
| **CTR-GCN + JMDA**           | **92.9(+0.5)**      | 96.6(-0.2)    | **89.2(+0.3)**   | **90.9(+0.3)**   | 
|   Skeleton MixFormer(3s)   |  92.6     |  96.9     | 89.8   |  91.2 |         
| **Skeleton MixFormer(3s) + JMDA**           | **93.2(+0.6)**      | **97.0(+0.1)**    | **90.1(+0.3)**   | **91.4(+0.2)**   | 
|   Skeleton MixFormer(6s)   |  93.2    |  97.2     | 90.2   |  91.5 |       
| **Skeleton MixFormer(6s) + JMDA**           | **93.7(+0.5)**      | 97.2    | **90.9(+0.7)**   | **91.9(+0.4)**   |


## Pretrained Models

The pretrained models can be found [here](https://drive.google.com/drive/folders/1uKl08UQ0a-XG93Fw5zy6ZYMcNeuY1Gx5?usp=drive_link).

## Acknowledgements

This repo is based on [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN/tree/main) and [Skeleton MixFormer](https://github.com/ElricXin/Skeleton-MixFormer). We appreciate the contributions of the original authors.



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
