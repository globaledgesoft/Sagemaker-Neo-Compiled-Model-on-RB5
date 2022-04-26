
import boto3
import json
import time

AWS_REGION = '<SPECIFY_YOUR_REGION>'
bucket = '<YOUR_S3_BUCKET_NAME>'
sagemaker_client = boto3.client('sagemaker', region_name=AWS_REGION)
sagemaker_role_arn = "<REPLACE_WITH_SAGEMAKER_I_AM_ROLE_ARN>"

framework = 'darknet'
data_shape = '{"data":[1,3,416,416]}'

tarfile_name='<TINY_YOLOV3_ZIPPED_MODEL_PATH>'

model_directory = 'models'
s3_model_uri = 's3://{}/{}'.format(bucket,tarfile_name)

compilation_output_dir = 'output'
s3_output_location = 's3://{}/{}/'.format(bucket, compilation_output_dir)

compilation_job_name = 'darknet-framework-test-1'

compiled_model = sagemaker_client.create_compilation_job(CompilationJobName=compilation_job_name,
                                        RoleArn=sagemaker_role_arn,
                                        InputConfig={
                                            'S3Uri': s3_model_uri,
                                            'DataInputConfig': data_shape,
                                            'Framework' : framework.upper(),
                                            },
                                        OutputConfig={
                                            'S3OutputLocation': s3_output_location,
                                            'TargetPlatform':{"Os":"LINUX","Arch":'ARM64'}
                                            },
                                            StoppingCondition={'MaxRuntimeInSeconds': 900})

