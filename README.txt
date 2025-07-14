FOLDER_CINECA/
├── configs/
│   ├── base.yaml                    # Base configuration
│   ├── 3c/
│   ├── 4c/
│   └── ssl/
│   |___ssl_ft/
│
├── classes/
│   
├── utils/
│
├── data/
│   ├── 3c_MIP/                     # 3-channel Maximum Intensity Projection images
│   │   ├── MSA-P/                  # Class 1: MSA-P patients
        |__ MSA-C/
│   │   ├── PD/                     # Class 2: Parkinson's Disease patients
│   │   ├── MSA/                # Class 3: Healthy controls (if applicable)
│   │                     # Additional classes (if applicable)
│   ├── 4c_MIP/                     # 4-channel MIP images
│   │   ├── MSA-P/                  # Class 1: MSA-P patients
        |---MSA-C/
│   │   ├── PD/                     # Class 2: Parkinson's Disease patients
│   │   ├── MSA/
│
├── slurm_files/
│   ├── pretrained/
│   ├── ssl/
│   ├── ssl_ft/
│
├── logs/                           # SLURM job logs with timestamps
│
├── mlruns/                         # MLflow tracking data
│
├── pretrained_encoders/            # SSL pretrained model weights
│   └── byol_resnet18_3c_MSA_VS_PD_patch.pth
│
├── main scripts:
│   ├── train_3c.py                 # Standard supervised training
│   ├── train_4c.py                 # Standard supervised training (4-channel)
│   ├── byol_simsiam_pretraining.py # SSL pretraining (BYOL/SimSiam)
│   └── downstream_supervised_fine_tuning.py  # SSL fine-tuning
│
└── notebooks/  (ORIGINAL CODE FROM WHICH THE .PY FILES HAVE BEEN CREATED)                     # 


NOTE:
- training is done on CINECA via slurm files.
- the slurm files correctly call the .py with the training code and pass the appropriate .yaml file (contained in the config folder) which containes the values of the hyperparameters
- the important parameters, metrics and artifacts (code,gradcam images, box plots, learning curves) are stored on mlflow and can be visualized locally using mlflow ui 
- the output of each run is stored in the folder logs/
- this project contains the code to perform binary classification:
    - supervised learning with:
        - 3 channel input
        - 4 channel input
        - using pretrained networks ( NASA imagenet-microscopy pretrained networks and torchvision imagenet pretrained networks)

    - self supervised pretraining:
        - using the data in PRETRAINING_MSA_VS_PD, PRETRAINING_MSAP_VS_PD
        - the configs in configs/ssl/
        - the slurm in slurm_files/ssl/

    - superviseed fine tuning of a self supervised pretrained encoder:
        - the slurm in slurm_files/ssl_ft/
        - the configs in configs/ssl/ssl_ft/
        

the usual training procedure is to run the slurm files which will call the .py files with the appropriate yaml file.
no need to modify the slurm files unless to change the parameters asked to cineca (name, time, gpu, cpu, etc.)
to change the parameters of the training you should modify the yaml files

NOTE: the only exception is for the supervised fine tuning of a ssl pretrained encoder in such case the parameters are still passed by the yaml but if we want to use a randomly initialized encoder
you should change the slurm and not pass the yaml


Start CINECA locally:
 $>   mlflow ui --port 5001 &

to kill eventually running mlflow processes:
 $>   pkill -f mlflow
