# Knowledge Distillation Mini Project

### Dataset

Synthetic data generated randomly in a multi-domain environment.

Four pickles files and each one contains a dictionary, the keys of the dictionary are:

* x_train: train data which shape is (2400, 2)
* y_train: labels of the train data
* d_train: domains of the train data
* x_valid: validation data which shape is (300, 2)
* y_valid: labels of the validation data

![avatar](https://github.com/SLAM-CROC/KnowledgeDisstillationMiniProject/blob/main/data%20distribution.png)

The plot shows the train data, each Gaussian distribution is considered as a separate domain; each domain is plotted using a different shape (cross, square, and circle). Each Gaussian distribution has two classes in it which are plotted using different colors (blue and orange)

### Objective

Create a multi-domain feedforward neural network (FFNN).

1) Implement a basic FFNN as multi-domain model  
2) Train 3 single-domain models as teacher models  
3) 3 single-domain models to train student model by using knowledge distillation

### Environment & Dependencies

M1 Chip MacBook pro

tensorflow - 0.1a3  
numpy - 1.22.2

### Experiment Result
I performed the experiment in four Jupyter notebook: detailed process and comment can be found in there.

Experiments_On_data_d0.25.ipynb  
Experiments_On_data_d0.5.ipynb  
Experiments_On_data_d0.75.ipynb  
Experiments_On_data_d1.0.ipynb

And get the following result:

![avatar](https://github.com/SLAM-CROC/KnowledgeDisstillationMiniProject/blob/main/result.png)

By analyzing the above table, it can be found that the average accuracy of the multi-domain model trained by knowledge distillation in the test set is lower than the FFNN model obtained by direct training. There may be two reasons for this result:

1. The order of knowledge distillation is fixed:  

When performing knowledge distillation, the order in which the student model sees the data in each domain is always the same, As figure 3 shown, student is trained from domain 0, 1 to 2 according to the corresponding teacher model.  

2. The accuracy of the model is not verified by adding a test set into the training process:  

Adding test set into training process as validation, this might be a trick, but we save the model with the best performance on the test set, which does improve our accuracy on the test set
