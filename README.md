# stm-model
*StM-AI model early stages of development.*
Using Keras from Tensorflow library, this repository includes the dataset generator currently in use, a prediction file and a model training file.

# Dataset Generator
- Configured to generate 120 images of each object (total of 5 objects), with a 8-1-1 ratio, 80% being training data, 10% validation and 10% test.
- Future improvements would be generating a bigger dataset with more angle variations, and not having the object always centralized in the render.

# Model training and prediction
Currently using our dataset as training data, adding a layer of pre-trained imagenet.
The current Dataset layout needs to look like this folder architecture:
```dataset/
├── test/
│   ├── Cylinder/
│   ├── Cone/
│   ├── Cube/
    ├── Pyramid/
│   └── Sphere/
├── train/
│   ├── Cylinder/
│   ├── Cone/
│   ├── Cube/
    ├── Pyramid/
│   └── Sphere/
└── validation/
     ├── Cylinder/
     ├── Cone/
     ├── Cube/
     ├── Pyramid/
     └── Sphere/```
