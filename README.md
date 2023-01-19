## Requirements
```
pip install gdown # Google Drive Donwloader
```

## Installation
- CUDA 11.3 is required
```
git clone git@github.com:guochengqian/PointNeXt.git
cd PointNeXt
source install.sh
```

## Data
- S3DIS Dataset
    ```bash
    $ pip install gdown # Google Drive Donwloader
    $ mkdir -p data/S3DIS/
    $ cd data/S3DIS
    $ gdown https://drive.google.com/uc?id=1MX3ZCnwqyRztG1vFRiHkKTz68ZJeHS4Y
    $ tar -xvf s3disfull.tar
    ```

## RUN
```bash
# Train
$ voxel_size=0.8 # default=0.4
$ CUDA_VISIBLE_DEVICES=0 python examples/segmentation/main.py --cfg cfgs/s3dis/pointnext-xl.yaml root_dir /home/sung/checkpoint dataset.common.voxel_size $voxel_size wandb.use_wandb=False mode=train

# Validation
$ CUDA_VISIBLE_DEVICES=0 python examples/test_s3dis_6fold.py --cfg cfgs/s3dis/pointnext-xl.yaml voxel_size $voxel_size wandb.use_wandb=False mode=test --pretrained_path pretrained/s3dis/pointnext-xl/pointnext-xl-area5/checkpoint/pointnext-xl_ckpt_best.pth
```


# Upsampling
```python
# main.py (600-606 lines)
if not nearest_neighbor:
    # average merge overlapped multi voxels logits to original point set
    idx_points = torch.from_numpy(np.hstack(idx_points)).cuda(non_blocking=True)
    all_logits = scatter(all_logits, idx_points, dim=0, reduce='mean')
else:
    # interpolate logits by nearest neighbor
    all_logits = all_logits[reverse_idx_part][voxel_idx][reverse_idx]
```


## Reference
Our code is highly referenced the codebase of [PointNeXt](https://github.com/guochengqian/PointNeXt) and [openpoints](https://github.com/guochengqian/openpoints)
