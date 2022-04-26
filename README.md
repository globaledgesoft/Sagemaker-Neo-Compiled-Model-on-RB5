# Sagemaker-Neo-Compiled-Model-on-RB5 (CPU Only)

## Introduction
This project is intended to build and deploy Object Detection model on RB5 using Sagemaker Neo Compiled model for aarch64 platform. Sagemaker is the Cloud based machine learning service provided by the Amazon’s AWS Cloud platform. Sagemaker has a service named as “Sagemaker Neo” for compiling & deployment of models for the edge devices. It provides the wide range of the ARM & x64 based devices.

## Prerequisites
1 A Linux workstation with Ubuntu 18.04.
2 Install Android Platform tools (ADB, Fastboot) 
3 Download and install the SDK Manager
4 Flash the RB5 firmware image on to the board
5 Setup the Network.
6 Amazon AWS Account Setup

## Model Compilation on Sagemaker Neo
In order to compile the model on Sagemaker Neo, there are 3 ways provided by AWS Sagemaker, 
 - AWS Console Based
 - AWS SDK Based
 - AWS CLI Based
In our project we will be following the AWS SDK Based model compilation.

### Downloading the Tiny-YoloV3 model
Please Run the commands given below inside the root of the project directory,
```sh
# mkdir tiny_yolov3_model
# cd tiny_yolov3_model
# wget https://pjreddie.com/media/files/yolov3-tiny.weights
# wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg
```

### Dependencies installation
While compiling the model from the AWS SDK, System must have the setup of AWS Python SDK Aka. Boto3. We have to setup the AWS account for the same, follow the link given below for the same.
 - https://docs.aws.amazon.com/cli/latest/userguide/getting-started-prereqs.html

### AWS-CLI & Python SDK setup
Run the commands given below for downloading & setting up the AWS-CLI,
```sh
# curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
# unzip awscliv2.zip
# ./aws/install
```

For setting up the AWS CLI Completely follow the instruction on the URL given below,
 - https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

Now install the AWS Python SDK, run the command below,
```sh
# python3 -m pip install boto3
```
### Getting Ready for Compilation
Setting up S3 bucket
Follow the link given below in order to setup the S3 bucket,
 - https://docs.aws.amazon.com/AmazonS3/latest/userguide/creating-bucket.html

### Zip & upload the artifacts (Model) on the S3
Zip the model with command given below & upload the same in S3 bucket, Run the command inside root of the project directory, 
```sh
# tar -czvf tiny_yolov3_model.tar.gz tiny_yolov3_model/   
```

### Compiling the model
Project source directory has provided with the model compilation script, please run the script for compiling the model,

NOTE: Before running the below script please do modify for the AWS Credentials & the S3 Bucket path.
```sh
# python3 model_compile.py
```
The command given above will compile the model and generate the artifacts in the provided S3 Bucket path.


