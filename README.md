# Denoising
Pipeline for denoising images in 3D.

## Git Clone repository to your host system

```
git clone https://github.com/abcucberkeley/denoising.git

# To later update the repository:
git pull
```

## Setting up the conda environment
 Conda can be installed from https://docs.anaconda.com/anaconda/install/index.html. The name of the environment can be updated inside the `environment.yml` file. The codes have been tested on **CUDA 12.4** with **PyTorch 2.5.1** and **Python 3.10**.
```
cd denoising
conda env create -f environment.yml
conda activate denoise
```

## Get Started
Set `denoising` as the present working directory. 
### Training
- First, update the `src/config.json` file containing the model hyperparameters as required. 
- The loss function, optimizer, learning_rate scheduler can be updated inside `src/train.py`.
- The `train_root` directory should contain two subdirectories `input` and `gt` representing the input to be passed to the model, and the ground truth. There should be a one-to-one correspondence of the filenames inside the individual subdirectories. 
- Currently, there is support for only TIFF files.

Run training locally:

```bash
python src/train.py --train_root /clusterfs/nvme/sayan/AI/training_denoise/ --model_root /clusterfs/nvme/sayan/AI/training_denoise/models --model_name drunet --config_file_path src/config.json
```

Or run via SLURM scheduler using the provided bash script (edit `OUTDIR`, `MODEL_NAME`, and resource allocation inside the bash file as needed):

```bash
bash src/train.sh
```

### Prediction

Run prediction locally on the `predict_root` folder containing an `input` subdirectory (`predictions` subdirectory is created inside the `predict_root` directory):

```bash
python src/predict.py --predict_root /clusterfs/nvme/sayan/AI/testing_denoise/ --model_root /clusterfs/nvme/sayan/AI/training_denoise/models --model_name drunet --config_file_path src/config.json
```

## References

[1]: “DRUNET: a dilated-residual U-Net deep learning network to segment optic nerve head tissues in optical coherence tomography images” (https://pubmed.ncbi.nlm.nih.gov/29984096/).

[2]: “Content-aware image restoration: pushing the limits of fluorescence microscopy” (https://www.nature.com/articles/s41592-018-0216-7).   
