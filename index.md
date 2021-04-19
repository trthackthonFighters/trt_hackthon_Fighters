## Welcome to GitHub Pages

英伟达-阿里 2021 TRT 44 组 Fighters:

### Week 1 Update 

#### Progress 
- Deployed the tensorflow model and deployed the data and built the data with pre-processing script.

- Downloaded the Inference/Evaluation data set [GLUE](https://github.com/nyu-mll/GLUE-baselines) and did the training/evaluation on the coda dataset

- Converted the model in format .meta to .pb of tensorflow.

- Run the sample in tensor plugin and profiling 

#### Blocking Issues

- Error in convert .pb to onnx format: Try to convert to trt directly
