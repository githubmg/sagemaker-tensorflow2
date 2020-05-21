# SageMaker with Tensorflow 2

In this example I'll go trough all the necessary steps to implement a VGG16 tensorflow 2 using SageMaker.

In the first part (Classification-Train-Serve) I'm going to use SageMaker SDK to train and then deploy a Tensorflow Estimator.

In the second part (Classification-Serve) I'm going to use the resulting model and implement it using only the Tensorflow Serving container. The original model can be easily replaced with your own .pb serialized model trained locally or in SageMaker without using the training service.


