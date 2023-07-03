---------------------------------Self driving car----------------------------------

This is a CNN model based on Nvidia's research for self driving car which takes only front camera images as input and generates steering angles as output.

Follow these steps to run the project:

1. First, Run the simulator and record the data using the record button, it will create a folder containing images and a csv file of dataset.

2. Train the model on the dataset by executing the files in specified order : data_prep.py -> train_model.py -> test_model.py

3. After executing train_model.py file, all the training logs will be updated in the train_history.npy and trained model will be saved in a Model.h5 file.

4. Before executing test_model.py, open the simulator in autonomous mode to test the trained model. Then execute the test_model.py file.

