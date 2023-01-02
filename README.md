Overview:

This project aims to develop a deep learning-based solution for detecting deepfake technology, which is used to manipulate videos and create the illusion that someone said or did something that they didn't. The proposed solution utilizes a network with a low number of layers, making it suitable for deployment on edge devices and efficient in terms of resource utilization. In addition, the solution includes the use of support vector machines (SVMs) as a classifier to improve the binary classification of images as fake or real. The solution has been tested on both an existing dataset and a dataset created by combining images from various benchmark deepfake detection datasets.

Implementation:

The deepfake detection solution utilizes autoencoders, which are neural networks consisting of an encoder and a decoder, to process and analyze the data. The autoencoders are trained separately on datasets containing the faces of the individuals being swapped in the manipulated videos. A GAN (generative adversarial network) is also implemented to generate the face swap. The proposed architecture is an extended version of this process.

Evaluation:

The performance of the deepfake detection solution has been evaluated on both an existing dataset and a dataset created by combining images from various benchmark deepfake detection datasets. The use of SVMs as a classifier has also been shown to improve the binary classification of images as fake or real.

Conclusion:

Overall, this project presents a deep learning-based solution for detecting deepfake technology with human-level accuracy and efficient resource utilization. The use of SVMs as a classifier has been shown to improve the binary classification of images, making the solution more effective in identifying manipulated videos.
