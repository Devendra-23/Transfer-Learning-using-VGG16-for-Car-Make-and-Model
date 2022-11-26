# Transfer-Learning-using-VGG16-for-Car-Make-and-Model

The development of a computer vision application that can recognize a certain vehicle model from an image is an intriguing and difficult subject to solve. The difficulty with this issue is that different car models can often look remarkably similar, and the same vehicle can sometimes look different and difficult to identify depending on lighting, angle, and many other variables. To create a model that can recognize a specific vehicle model for this project, I choose to train a convolutional neural network (CNN) known as VGG16 using Fast ai and PyTorch.


VGG16(Medium, 2021.):https://medium.com/@mygreatlearning/everything-you-need-to-know-about-vgg16-7315defb5918

![Vgg16](https://user-images.githubusercontent.com/94075388/204093296-984dce90-6add-408e-9e93-11590ea3f69f.png)

  

# Dataset
For the Dataset: https://ai.stanford.edu/~jkrause/cars/car_dataset.html


There are 16,185 photos of 196 different kinds of cars in the Cars collection. The data has been divided into 8,144 training photos and 8,041 testing images, roughly splitting each class 50-50. Classes are usually given in the Make, Model, and Years categories.

Visualising the Dataset:
![VGG16-DATA](https://user-images.githubusercontent.com/94075388/204093225-9fbe2fe2-0833-4bd0-bc5c-252afc18cda1.png)

# Results


Classification:
<img width="1105" alt="Vgg16-Result" src="https://user-images.githubusercontent.com/94075388/204093165-b9244be7-2d0c-4cc2-b0bb-8ee5db68347b.png">




Accuracy obtained: 98% 


![vgg16_fastai_plot](https://user-images.githubusercontent.com/94075388/204093474-219075c8-fab4-456c-9205-c8bfcb20e467.png)

