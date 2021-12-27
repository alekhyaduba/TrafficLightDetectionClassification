# CSCE-643_TrafficLightDetection

Part 1-
Yolo Training on Custom Dataset from Bosch- 
Follow the Google Colab notebook for step by step guide : https://colab.research.google.com/drive/1CVyxNjKTFJ3IK6YC4L38Ix4Tr06irlCy?usp=sharing
The notebook is also available in this repo as YOLO_Train_Test.ipynb

Part 2- 
Classification Network

This folder consist of the CNN model and all the helper functions required.
Prerequisite : A .json file created in step 8 of the above mentioned google colab notebook.
Steps to Execute:
 1. Run data_handler.py to create the data set. ( Note pass the path of images in line 62). 
    df_data.csv file will be created under ../data
    Cropped images will be saved under ../data/cropped_Images
 2. Run the train.py
 3. Once the train.py is run the model will be saved under ../data/models
 4. Run getPredictedTL() once, this will save the data in the form of dataframe under outputs folder.
 5. Comment the above call from test.py and run the code to get the accuracy of Classification network.
 6. A call to getYoloAccuracy will print the accuracy of YOLO model, the input should be the result.json file obtained from the step 8 of the Google Colab notebook.

Note: For test and train, the dataset should be stored in the correct path and the paths should be updated properly.

Results:
Some results like trained weights, predicted images, cropped images samples and video can be found in the Results folder.
 
A set of training files are also made available in the forked github repo which is reference in the above google colab notebook. 
The same are also available under Yolo_Training_Files
