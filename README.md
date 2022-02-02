<h1> Pytorch Classification Training Template </h1>

A Pytroch training template to ease the processes of training.

<h2>Features </h2>

* Train with config file for reproducibility
* Config file allows for efficient hyperparameter tuning
* Easily customizable Resnet style model
* Checkpoint and log saving during training
* Genetic dataset loader for classification
* Onnx exporter with predictor

<h2> Config file </h2>

```
{
  "name": "Model",     //Model name. Used for versioning
  "n_gpu": 1,
  "arch": {
    "args": {
      "type": "resnet18"
    }
  },
  "data_loader": {
    "args": {
      "img_dir": "",    //Path to directory with images
      "train_df": "",    //Path to pandas style dataframe
      "test_df": "",
      "included_classes":     //list of labels the model is trained on
      [
      ],
      "class_mapping": null,    //Dictionary of int to label. If null sort alphabetically
      "input_size": [
        512,
        512
      ],
      "batch_size": 8,
      "num_workers": 4,
      "normalize": true,
      "mean":    //If normalize, values for mean and std. [R,G,B]
      [
      ],
      "std":
      [
      ],
      "balance": true,    //automatically balance classes if desired
      "sample_set": false,    //Train on a subset of dataset
      "sample_size": 500,    //Size of that subset
      "random_crop": false,   //Random crop augmentation
      "random_apply": true,   //Randomly apply augmentations to training set
      "transforms": {
        "ColorJitter": {
          "brightness": 0.1,
          "contrast": 0.1,
          "saturation": 0.08,
          "hue": 0.08
        },
        "Pad": {
          "padding": 20
       },
       "cutout": {
          "p":0.4,
          "scale": (0.01, 0.03)
       }
      }
    }
  },
  "model": {
    "num_classes": 14,
    "layers": [
      1,
      1,
      1,
      1
    ],
    "inplanes": 32,
    "planes_per_layer": [
      32,
      64,
      128,
      128
    ],
    "inblock_expansion": 1,
    "norm_type": "batchNorm",
    "nonlinearity": "relu",
    "downsample_type": "pool",
    "downsample_input": true,
    "export": false
  },
  "optimizer": {
    "type": "adam",
    "args": {
      "lr": 0.0003,
      "weight_decay": 0.0001
    }
  },
  "loss": "BCELoss",
  "metrics": [
    "accuracy",
    "loss"
  ],
  "lr_scheduler": {
    "type": "StepLR",
    "args": {
      "step_size": 5,
      "gamma": 0.1,
      "max_epochs": 5
    }
  },
  "trainer": {
    "epochs": 18,
    "save_dir": "./session",
    "save_freq": 1,   //save log frequency
    "tensorboard": true,
    "tensorboard_dir": "",
    "write_freq": 20,    //write to std out freq
    "save_by": "every"    //Loss, Accuracy, or every. Every will save after every epoch
  }
}
```

<h2> Code Examples </h2>

From scratch training:
```
python train.py --config ./config_files/config.json
```

Resume training from previous checkpoint:
```
python train.py --config ./config_files/config.json --resume ./sessions/model.pth
```

Training model loading pretrained weights:
(This will start training from scratch with weights initialized from trained model)
```
python train.py --config_files/config.json --load_weights ./sessions/model.pth
```
