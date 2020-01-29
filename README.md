<!-- # Face Detection - Facial Expression+Age+Gender Classification

### Library
`tensorflow 2.0.0`
`opencv`
`dlib`
`face_recognition`
`hog from scikit-image.feature`

### Data
The age+gender model is trained on Baidu's All-Age-Faces (AAF). [link here](https://github.com/JingchunCheng/All-Age-Faces-Dataset)

The facial expression model is trained on Kaggle dataset [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) containing labeled 3589 test images, 28709 train images. We looked at JAFFE dataset (too few images) and Cohn-Kannade dataset (unavailable) but didn't choose them.

### Model
* For **Facial Expression** model, we use functional API with 4 convolutional layers with ReLU activation and padding 'same' for image input and another separate input layer for HOG (histogram of gradients) & landmarks features. See [`FER2013.ipynb`](https://drive.google.com/file/d/1pfhGPtAEzMrAco7ChoCfrCmuiklpBN6p/view?usp=sharing) notebook for more details about the model. We don't use transfer learning because our dataset contains greyscale images and doesn't fit in 3 channel pretrained models.

<!-- ![](https://i.imgur.com/skdNLG4.png) -->
<!-- ![](https://i.imgur.com/djC5E38.png)

* For **Age and Gender** We use the two pre-trained models Inception V3 and VGG16 for gender and age detections, respectively. 
    * **Inception V3 for gender detection**
    ![](https://i.imgur.com/BiDKceY.png)

    * **VGG16 for age detection**
    ![](https://i.imgur.com/1fqjqNT.png)


### Training
#### Facial Expression training
- The data from the Kaggle dataset is already split into train and test set. The data is processed from csv file into numpy arrays and fetched into ImageDataGenerator. See jupyter notebook [`FER2013.ipynb`](https://drive.google.com/file/d/1pfhGPtAEzMrAco7ChoCfrCmuiklpBN6p/view?usp=sharing) for more details.

- Our model achieved ~99% accuracy on train set and ~75% on validation set after 21 epochs
- We use [this paper](https://arxiv.org/abs/1612.02903) as benchmark which achieves 75.2% accuracy. 
- The two tables below show training result on 5 expressions done by [amineHorseman](https://github.com/amineHorseman/facial-expression-recognition-using-cnn). It shows that face landmarks and histogram of oriented gradients (HOG) only improves the accuracy by 4.5% at best. On the other hand, batch normalization significantly improves the model performance.
![](https://i.imgur.com/F5JF0io.png)
![](https://i.imgur.com/49AbN0v.png)
-    Using example code extract face landmarks using dlib shape predictor model
-    Example code get hog features using scikit image hog
-    Using functional API https://www.tensorflow.org/guide/keras/functional
-    Model performance:
![](https://i.imgur.com/ZBhLEzL.png)
![](https://i.imgur.com/gXELMxN.png)
#### Age and Gender training
* The Baidu dataset contains 13322 face images (mostly Asian) distributed across all ages (from 2 to 80), including 7381 females and 5941 males. The orignal face images, facial landmarks and aligned face images are stored in folders `original images`, `key points`, and `aligned faces`.

* We read from the `key points` folder and turn the data into a dataframe consisting of 3 columns: `image_name`, `gender` and `age`. From there we loop through the dataframe and assign the image paths for each correponding row and store them inside sub-folders which serve as the labels. Gender has two labels Male and Female while Age has 5 labels which are the age ranges: 1-15, 16-25, 26-35, 36-45, >46.


* We both go through several pre-trained models such as MobileNet, Inception ResNet V2 and even [Levi&Hassner's model](https://talhassner.github.io/home/publication/2015_ICMI), which was specifically trained for age and gender detection, but none seems to fit with the Baidu dataset.

* See [Age](https://colab.research.google.com/drive/1AEjjg-jIbazCbG3zvH5BwSy4ogIvZEgD) and [Gender](https://colab.research.google.com/drive/1lpELZb7YmQBCekqudlNcwuQ90uepolAg) Notebooks for for information

### Flask app
-  Our Flask app connects to the webcam which user can use to scan their faces for classification with the output as the user's age, gender and emotion. 
-   Our heroku app is deployed at http://face-emotion-gender-age.herokuapp.com. However, due to size limit, we can only use the emotion model on the heroku app, the other two models are not deployed on heroku. -->
# Face Detection - Facial Expression+Age+Gender Classification

### Libraries
`tensorflow 2.0.0`
`opencv`
`hog from skimage.feature`
`dlib`

### Data
The age+gender model is trained on Baidu's All-Age-Faces (AAF). [link here](https://github.com/JingchunCheng/All-Age-Faces-Dataset)

The facial expression model is trained on Kaggle dataset [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) containing labeled 3589 test images, 28709 train images.

### Model
* For **Facial Expression** model, we use functional API with 4 convolutional layers with ReLU activation and padding 'same' for image input and another separate input layer for HOG (histogram of gradients) & landmarks features. See [`FER2013.ipynb`](https://drive.google.com/file/d/1pfhGPtAEzMrAco7ChoCfrCmuiklpBN6p/view?usp=sharing) notebook for more details about the model. We don't use transfer learning because our dataset contains greyscale images and doesn't fit in 3 channel pretrained models.

<!-- ![](https://i.imgur.com/skdNLG4.png) -->
![](https://i.imgur.com/djC5E38.png)

* For **Age and Gender** We use the two pre-trained models Inception V3 and VGG16 for gender and age detections, respectively. 
    * **Inception V3 for gender detection**
    ![](https://i.imgur.com/BiDKceY.png)

    * **VGG16 for age detection**
    ![](https://i.imgur.com/1fqjqNT.png)


### Training
#### Facial Expression training
- The data from the Kaggle dataset is already split into train and test set. The data is processed from csv file into numpy arrays and fetched into ImageDataGenerator. See jupyter notebook [`FER2013.ipynb`](https://drive.google.com/file/d/1pfhGPtAEzMrAco7ChoCfrCmuiklpBN6p/view?usp=sharing) for more details.

- Our model achieved ~99% accuracy on train set and ~75% on validation set after 21 epochs
- We use [this paper](https://arxiv.org/abs/1612.02903) as benchmark which achieves 75.2% accuracy. 
- The two tables below show training result on 5 expressions done by [amineHorseman](https://github.com/amineHorseman/facial-expression-recognition-using-cnn). It shows that face landmarks and histogram of oriented gradients (HOG) only improves the accuracy by 4.5% at best. On the other hand, batch normalization significantly improves the model performance.
![](https://i.imgur.com/F5JF0io.png)
![](https://i.imgur.com/49AbN0v.png)
-    Using example code extract face landmarks using dlib shape predictor model
-    Example code get hog features using scikit image hog
-    Model performance:
![](https://i.imgur.com/ZBhLEzL.png)
![](https://i.imgur.com/gXELMxN.png)

#### Age and Gender training
* The Baidu dataset contains 13322 face images (mostly Asian) distributed across all ages (from 2 to 80), including 7381 females and 5941 males. The orignal face images, facial landmarks and aligned face images are stored in folders `original images`, `key points`, and `aligned faces`.

* We read from the `key points` folder and turn the data into a dataframe consisting of 3 columns: `image_name`, `gender` and `age`. From there we loop through the dataframe and assign the image paths for each correponding row and store them inside sub-folders which serve as the labels. Gender has two labels Male and Female while Age has 5 labels which are the age ranges: 1-15, 16-25, 26-35, 36-45, >46.


* We both go through several pre-trained models such as MobileNet, Inception ResNet V2 and even [Levi&Hassner's model](https://talhassner.github.io/home/publication/2015_ICMI), which was specifically trained for age and gender detection, but none seems to fit with the Baidu dataset.

* See [Age](https://colab.research.google.com/drive/1AEjjg-jIbazCbG3zvH5BwSy4ogIvZEgD) and [Gender](https://colab.research.google.com/drive/1lpELZb7YmQBCekqudlNcwuQ90uepolAg) Notebooks for for information

### Flask app
-  Our Flask app connects to the webcam which user can use to scan their faces for classification with the output as the user's age, gender and emotion. 
