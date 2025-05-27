# MNIST_Classification-using-CNN
This project focuses on handwritten digit recognition using the MNIST dataset, which contains 70,000 grayscale images of digits (0–9), each of size 28x28 pixels. The dataset is split into 60,000 training and 10,000 test images, making it a standard benchmark for image classification tasks. A Convolutional Neural Network (CNN) was built to perform multiclass classification on this dataset.
The neural network consists of 2 convolutional layers and 2 pooling layers.
The convolutional layer has 16 and 32 filters each with a kernel size/filter dimension as 2x2 with reLU activation to avoid the vanishing gradient problem.
This layer had different strides in both layers (2x2 and 1x1).
The pooling layers had similar filter/pooling sizes and strides as 2x2 in both layers.
Then the output was flattened and a dense layer was added with 100 nodes and reLU as activation function.
Finally, at the output end, softmax activation was used with an output layer to classify the outputs .
RESULTS : Obtained over 98.35% accuracy in only 10 epochs 
Optimizer : Adam , loss : categorical_crossentropy
Thank You! Stay tuned for exciting projects.
