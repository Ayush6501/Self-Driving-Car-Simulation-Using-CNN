# Autonomous Vehicle Simulation using CNN
This project demonstrates a basic self-driving car model using Udacity's car-driving simulator. This project will build a Convolution Neural Network model to predict the steering angle for a virtual car in the 
simulator running at a constant speed. The goal is to drive the car in the simulator autonomously for a full lap without deviating from the main track/road. 
Udacity self-driving car simulator is used for testing and training our model.

## Steps involved
  1. Setting up the environment. 
  2. Setting up Udacity driving simulator.
  3. Generating training/test data from the simulator.
  4. Build and train the model with the training data.
  5. Testing the model using the Behavioral-Cloning project.
  
## 1. Setting up the environment
  Create an anaconda virtual environment 'self-driving' with the 'environment-gpu.yaml'  file for training and testing the Keras model.  
   
          conda env create -f  autopilot_project/Self-Driving-car/anaconda_env/environment-gpu.yml
    
## 2. Setting up Udacity driving simulator.    
   
   Download the Udacity pre-build term1 driving simulator with the below command. 
   
           wget https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-linux.zip
   
##  3. Creating training/test data from the simulator.
   
   1. Run the pre-build simulator executable from the extracted folder.
   2. Once it is launched, choose the training option from the main window.
   3. Now click on the record button and choose the 'data_set/train_data' folder for saving training data.
   4. Align the car on the track and then click on the record button again, then drive the car on the track for 7 or 8 laps. Click on the record button again to stop recording.
   5. Restart the simulator click on the record button and choose the 'data_set/test_data' folder to save test data.
   6. And repeat step 4 for 5-10 laps.
   7. After recording, the recorded images and driving_log.csv files can be found under respective folders.

## 4. Build and train the model with the training data.
   1. Change the current working directory to 'autopilot_project/Self-Driving-car'
   2. Activate the anaconda environment 'tensgpu_1' that was created before using the below command.

             conda activate self-driving
   3. Then run 'model_train.py' to create a model and train it with the training and test data set that was created before. 
       
             python3 model_train.py --train_csv_file 'path to training driving_log.csv file' --test_csv_file   'path to test driving_log.csv file' --batch_size 32 --epochs 50  1>train.log 2>&1
             
      The above execution will create four different models (for different learning rates) under the folder 'autopilot_project/Self-Driving-car/models'. Check 'autopilot_project/Self-Driving-car/train.log' to see the progress of training. 
      After successful training, revisit the log and check which model had minimum  'loss' and 'val_loss', and choose that as the final model for testing.
      
             
##  5. Testing the model using the Behavioral-Cloning project.

   1. Launch the simulator in autonomous mode to test the model.
   2. Run the pre-build simulator executable.
   3. Once it is launched, choose Autonomous mode from the main window (Now the simulator should be ready to accept a connection).
   4. Change the working directory to 'autopilot_project/CarND-Behavioral-Cloning-P3'
   5. Activate the anaconda environment 'tensgpu_1'.

             conda activate tensgpu_1
   6. Then run 'drive.py' with the following command.
   
             python3 drive.py 'path to the created model.h5 file'
   7. If the environment is proper and if the script can make a connection with the simulator then the car in the simulator starts running at 9kmph, and  it will try to adjust its steering angle to keep the car always on the track. 
   8. If the car can maintain on the track for a full lap, then your model is performing well :)
   10. If The car does not always stay on the track, then the model is poorly performing :( , then retrain the model with more data and with reduced batch size. and test it again and again until a good performance is achieved.
   
       The following GIF shows the output of 'model.h5' model. 
       
       ![20221112_015409](https://user-images.githubusercontent.com/78997596/201426129-31a1f8b6-6f5f-4655-a493-720745345d70.gif)

   


References:
  1. https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
  2. https://github.com/udacity/self-driving-car-sim
  
       
    
