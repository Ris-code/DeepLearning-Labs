# DeepLearning-Labs

## LAB 1

Implement an MLP for the classification of the Dry Bean dataset. You can download the dataset from the attached link.

## LAB 2

Develop a fully connected autoencoder using PyTorch, comprising three encoder and three decoder layers. Train the autoencoder on the Street View House Numbers (SVHN) dataset and conduct a comprehensive evaluation of its reconstruction accuracy. Extend the autoencoder by incorporating a sparsity constraint on the hidden layer. Systematically assess the impact of sparsity on the learned representations and the overall reconstruction performance of the model. Provide insights into how the introduction of sparsity affects the autoencoder's ability to capture meaningful features and improve reconstruction quality.

## LAB 3

Create a convolutional autoencoder that does greedy layer-wise pretraining on the STL-10 dataset. You must create an autoencoder with at most 3 convolution blocks with varying activation(ReLU, tanh) and pooling layers(Adaptive Avg pool)(i.e. 3 for encoder and 3 for decoder at most). You must add an additional linear layer after the autoencoder for classification. Appropriate plots must be made like T-SNE and train loss curves. To check for efficacy, you must also use appropriate metrics like a confusion matrix.

Bonus: Identify a known problem in this kind of architecture and apply a regularization technique as required.

## LAB 4

Depth-wise and Point-wise Convolution: Design a convolutional neural network (CNN) architecture that utilizes 4 blocks of depth-wise separable convolutions followed by point-wise convolutions. Train the CNN on the CIFAR-10 dataset and evaluate its performance in terms of classification accuracy,precison,recall,f1 score. Compare the computational efficiency and effectiveness of depth-wise and point-wise convolutions with traditional convolutional layers.

## LAB 5

In this lab exercise, your objective is to apply Long Short-Term Memory (LSTM) networks to predict shampoo sales using a dedicated dataset. Start by exploring and preprocessing the sales dataset, visualizing trends, and normalizing data. Design an LSTM model, experiment with hyperparameters, and compile the network. After model construction, undergo training, validation, and testing phases, evaluating performance with metrics like Mean Squared Error. The final task involves fine-tuning the model for optimization through adjustments, regularization techniques, and exploration of additional enhancements. This assignment encourages your practical application and understanding of LSTM networks in the context of forecasting shampoo sales based on real-world data. You can download the dataset using this link:

https://www.kaggle.com/datasets/redwankarimsony/shampoo-saled-dataset

## LAB 6

Implement a variational autoencoder architecture using PyTorch and train it on the UTKFace dataset. For enhancing the model's performance employ convolutional and dense blocks. Evaluate the quality of the generated images using both qualitative and quantitative metrics, and discuss strategies for improving the VAE's ability to generate high-fidelity and diverse samples. Compare the results of VAE with a normal autoencoder. Also prove your results by properly visualizing the outputs. 

Integration of convolutional and dense blocks into the VAE architectures.

Plotting the training , validation loss/accuracy vs epoch curve 

Comparison of the reconstruction ability of the VAE on input images of test set

Randomly sample a point from the latent space of the VAE and compare the output from the decoder with that of a normal autoencoder 

Plotting the t-SNE plot of VAE and comparison with normal autoencoder.

Smoothness of Latent Space
Interpolate between different samples to evaluate the smoothness of latent space and evaluate the performance based of ELBO score
Take the latent representation of VAE and evaluate its performance on a classification task(Age or Gender Prediction) and also compare this with a regular autoencoder.

Discuss how you can improve the VAE's ability to generate high fidelity and diverse samples.

Note
Only assignment written in Pytorch would be eligible for grading rest of the assignments would be awarded zero.

## LAB 7

In this assignment, you need to create a DCGAN. The difference between this type of GAN and
the one introduced by Goodfellow et al. is that we use deep convolutional networks in the
generator and discriminator architecture. Your task is to perform the following on the CelebA
dataset:
1) Create a DCGAN based architecture. The generator must have an architecture as
described here. The discriminator must be as described here.
2) Implement the min-max loss function for the GAN based on the GAN loss described
here. 
3) Modify the loss function in the previous step to the one implemented by the
Least-Squares GAN paper. 
4) Perform the training in both parts 2 and 3. Visualize the images created in both the
cases after the training is done. Compare their performance based on the FID score
metric. To know more of this metric, refer to this link: FID 

## LAB 8

Implementation of CycleGAN in PyTorch

In this assignment,  take a dataset of your choice, and you will delve into the world of CycleGAN, a powerful model for unsupervised image-to-image translation. Your task is to implement and analyze the performance of CycleGAN on a chosen dataset, exploring novel techniques and evaluation methods.

Part 1: Adaptive Style Transfer 

Take a dataset of your choice and Develop a CycleGAN architecture equipped with adaptive style transfer mechanisms for domain translation. Incorporate techniques such as adaptive instance normalization (AdaIN) or adaptive style attention (ASA) to enhance the model's ability to capture and transfer domain-specific styles between images. Train the adapted CycleGAN model on a dataset containing images from distinct domains, such as manga to real photos or satellite images to maps. Monitor the adversarial and cycle consistency losses during training and visualize the translated images to assess the fidelity and style preservation capabilities of the model.

Part 2: Multimodal Image Translation 

Extend the CycleGAN architecture to support multimodal image translation, enabling the generation of diverse outputs for a given input. Utilize techniques such as stochastic mapping or latent space sampling to introduce randomness into the translation process, thereby generating multiple plausible outputs for each input image. Train the multimodal CycleGAN model on a dataset featuring diverse visual styles or appearances within the same domain, such as artistic paintings to photographs or day-to-dusk scenery. Evaluate the diversity and quality of generated samples by conducting qualitative and quantitative analyses, including perceptual similarity metrics and user studies.

Part 3: Consistency Regularization 

Explore the concept of consistency regularization to enhance the stability and robustness of CycleGAN training. Implement techniques such as cycle consistency regularization or feature matching to enforce consistency between translated and original images across domains. Analyze the impact of consistency regularization on the convergence behavior and generalization performance of the CycleGAN model. Provide insights into how consistency regularization mitigates potential issues like mode collapse or overfitting during training.

## LAB 9

Download the pre-trained StyleGan(v1, v2 or v3).

1. Generate 10 realistic images using the StyleGAN.

2. Take your face image and 5 different face images of your friends (One image per friend). Perform feature disentanglement and linear interpolation between your face and your friend's face. 
   
## LAB 10

In this assignment, you are to take a pretrained convnet and apply it in the tasks
given below. Odd roll numbers must take efficientnetb0 and even roll numbers
must take efficientnetb4.
- Remove the last linear layer and replace it with appropriate linear layer(s) to
train on the dataset given in this link. Your model must only train the last linear
layer, thereby using the pretrained model. Perform the finetuning and testing
by dividing the dataset into train-test.
- Create a function to output the saliency maps corresponding to any 1 image
from each class in the following two cases:
  - Finetune only the last layer and test it.
  - Re-train the entire network on the new dataset and test it.
- Evaluate the performance of the finetuned and original network based on the
recall and accuracy metrics. Plot the training loss curve. Finally, write plausible
explanation for the difference in metric values you obtained.

## LAB-11

Code link: https://drive.google.com/file/d/1ECSqWfxP6I8HL3j-V5JmY3XjZfsej5c0/view?usp=sharing

1. Train the GNN model for the classification of the MNIST images using the paper: https://iopscience.iop.org/article/10.1088/1742-6596/1871/1/012071/pdf (Dataset: MNIST). Note: Modify the architecture and use 4 gatconv layers and 3 Fc layers
2. Tune the hyperparameters of the GNN model. (At least 2 sets of hyperparameters running logs must be shown)
3. Visualize the images in graph format (All the classes). 
4. Compare the accuracy and training time to a typical CNN of your choice. (4 cnn layers). 



