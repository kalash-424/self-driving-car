---------------------------------Self driving car----------------------------------
				                    -----by kalash bhagwat-----

In this project, I have developed a convolutional neural network (CNN) based model for self driving car, inspired by Nvidia's research. I have trained and tested this model using the Udacity simulator. The model has been trained on a large dataset of images and steering angles collected from the simulator. During training, the model learns to predict the steering angle from the input images. I have also used data augmentation techniques to improve the robustness of the model.

I have evaluated the performance of the model using Loss plot. The model has shown promising results in terms of its ability to navigate the car safely and accurately in different driving scenarios. This project requires a simulator to test and train the model.


Follow these steps to run the project:

1. First, Run the simulator and record the data using the record button, it will create a folder containing images and a csv file of dataset.

2. Train the model on the dataset by executing the files in specified order : data_prep.py -> train_model.py -> test_model.py

3. After executing train_model.py file, all the training logs will be updated in the train_history.npy and trained model will be saved in a Model.h5 file.

4. Before executing test_model.py, open the simulator in autonomous mode to test the trained model. Then execute the test_model.py file.




Thank you!
