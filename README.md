Preparation(on gpu6)
-----

To prepare dataset and environment, please run with the following command:

```
sh prepare_on_gpu6.sh
```

Preparation(public)
-----

please run with the following command:

```
conda env create -f env/env.yaml
conda activate env
```

The datasets can be downloaded from their official sites. We also provide them here:

Baidu Disk: <a href="https://pan.baidu.com/s/1yOGMBZOzlZ5UJ2EGh9y8CQ">download</a>  (code: zr4s)   

Google Drive: <a href="https://drive.google.com/drive/folders/1JprKNLCGQtaCXuVziNHz7HyOMbqzsXrM?usp=sharing">download</a>  

Note that they are saved as 'png', which are extracted from their original datasets without any further preprocessing. 

Plus, please prepare the training and the testing text files. Each file has the following format:

```
/Path/to/the/image/files /Path/to/the/label/map/files
...
...
```
We provide two example files, i.e., 'train_AtriaSeg.txt' and 'test_AtriaSeg.txt'


Training & Testing
-----
After the preperation, you can start to train your model. After training, the latest model will be saved and used for testing. please run with:

```
sh train_test.sh
```

Acknowledgement
-----------------
Our partial implementation is based on GBDL("Rethinking Bayesian Deep Learning Methods for Semi-Supervised Volumetric Medical Image Segmentation") . Thanks for these authors for their valuable work, hope our work can also contribute to related research.

