# ML-Image_classification_dog_breed
# Image Classification using AWS SageMaker

This is an image classification project that utilises AWS Sagemaker to train and fine-tune a pretrained ResNet50 model to categorise dog breeds from dog photos by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. 
## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 
Open SageMaker Studio and connect to AWS using the gateway provided in the course.
Start by installing all the requirements in the jupyter notebook "train and deploy.ipynb."
"Python 3 (Data science - 01)"Kernel is used in this project. Also essential PyTorch libraries have been installed.
## Dataset
We'll be utilising Udacity's dataset on dog breeds.
Images of dogs from 133 different breeds from around the world are included in the dataset.
These dog photos will be used to train our image classification algorithm to distinguish between different dog breeds.
We'll upload our dataset to an S3 bucket so that we can utilise it to train our model.
### Access
To give SageMaker access to the data, upload it to an S3 bucket using the AWS Gateway.
## Hyperparameter Tuning
To offer transfer learning to the CNN, a pre-trained ResNet-50 model was employed. SageMaker has a tool for adjusting different hyperparameters of the model in order to get optimal accuracy values. The following ranges were used to fine-tune the learning rate and batch size hyperparameters:
"learning_rate": ContinuousParameter(0.001, 0.1),
"batch-size": CategoricalParameter(32, 64, 128, 256, 512)
The screenshots below show the results of completed training jobs as well as the hyperparameter values derived from the best tuning job.

 


 
 
 
## Debugging and Profiling
After finding the optimum hyperparameters, a new model was generated with SageMaker debugging and profiling facilities to track and diagnose any model faults. SMDebug hooks for PyTorch with the TRAIN and EVAL modes were added to the train and test functions, respectively, in the train model.py script. The SMDebug hook was built and registered to the model in the main method. The train and test functions both took the hook as an argument. The train and deploy notebook was used to set up debugger rules and hook settings. The SageMaker profiler may be used to track instance resources such as CPU and GPU memory use. To train the mode, a profiler and debugger settings were added to the estimator. 

### Results
During the training of the model, certain difficulties with Overtraining, and over fitting were discovered. The summary is provided in profiler report. 

## Model Deployment
Our functioning endpoint was established and deployed using the "endpoint.py" script, which was deployed to a "ml.t3.medium" instance type.
The test image from the 'DogImages/test' folder is used for testing .
The image is sent to the endpoint for inference, which is done using both methods.
Using the Boto3 client and the Predictor Object.
