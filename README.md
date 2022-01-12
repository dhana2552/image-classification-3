# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

![S3 Bucket](images_for_readme/S3.png)

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Selected Model: ResNet50
Hypermeters used: batch size, learning rate
The pretrained model resnet50 was sufficient enough to train the model for the given dataset. It was also easy enough to finetune the model and get a good prediction performance.

Remember that your README should:

- Include a screenshot of completed training jobs
![Completed Training Jobs](images_for_readme/Completed_Training_Job.png)

- Logs metrics during the training process
![Training Metrics](images_for_readme/Metrics.png)
![Logs](images_for_readme/Logs.png)

- Tune at least two hyperparameters
```
hyperparameter_ranges = {
    "learning_rate": ContinuousParameter(0.001, 0.1),
    "batch_size": CategoricalParameter([32, 64, 128, 256, 512]),
}
```

- Retrieve the best best hyperparameters from all your training jobs
```
"best_hyperparameters": {
        "learning_rate": '0.005222375403517124',
        "batch_size": "32"
    }

 ```

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

```
hook_config = DebuggerHookConfig(
    hook_parameters={
        "train.save_interval": "100",
        "eval.save_interval": "10"
    }
)

profiler_config = ProfilerConfig(
    system_monitor_interval_millis=500, framework_profile_params=FrameworkProfile(num_steps=10)
)
```

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?
As per the profiler report, the gpu was not effectively used during the training process. It could have been improved by increasing the batch size, training methods or even frameworks.

**TODO** Remember to provide the profiler html/pdf file in your submission.
Attached

## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
```
pytorch_model = PyTorchModel(
    model_data=model_data, 
    role=role, 
    entry_point='inference.py',
    py_version="py36",
    framework_version="1.8"
)
predictor = pytorch_model.deploy(initial_instance_count=1, instance_type="ml.t2.medium")
```

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

![Deployed Endpoint](images_for_readme/Endpoints.png)

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
