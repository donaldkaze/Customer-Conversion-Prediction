import sagemaker
from sagemaker.xgboost.model import XGBoostModel
import boto3
from datetime import datetime

# Get the session with explicit bucket configuration
boto_session = boto3.Session()
sagemaker_session = sagemaker.Session(boto_session=boto_session, default_bucket='conversion-testdk1')
role = sagemaker.get_execution_role()

# Specify the S3 path to your model.tar.gz
model_data_s3_path = 's3://conversion-testdk1/model.tar.gz'

print("Creating XGBoost model object...")

# Create a SageMaker XGBoostModel object
# Using framework_version='1.0-1' which is more stable and widely compatible
# If your model was trained with a different XGBoost version, you may need to adjust this
xgboost_model = XGBoostModel(
    model_data=model_data_s3_path,
    role=role,
    framework_version='1.0-1',  # Changed from 1.7-1 to more stable 1.0-1
    sagemaker_session=sagemaker_session
)

print("Deploying model to endpoint...")
print("Note: This will take 8-10 minutes. The deployment may fail if:")
print("  - Model was trained with incompatible XGBoost version")
print("  - Model file structure is incorrect (should contain xgboost-model in tar.gz)")
print()

# Deploy the model to a SageMaker Endpoint
# Using ml.m5.large instead of ml.t2.medium for better stability
timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
endpoint_name = f'xgboost-conversion-{timestamp}'

try:
    predictor = xgboost_model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large',  # Changed to ml.m5.large for better reliability
        endpoint_name=endpoint_name,
        wait=True  # Wait for deployment to complete
    )
    print(f"\n✓ SUCCESS! Endpoint deployed: {predictor.endpoint_name}")
    print(f"You can now use this endpoint to make predictions.")

except Exception as e:
    print(f"\n✗ Deployment failed with error:")
    print(f"  {str(e)}")
    print("\nTroubleshooting suggestions:")
    print("  1. Check CloudWatch logs for detailed error messages")
    print("  2. Verify model was saved in correct XGBoost format")
    print("  3. Try different framework_version (1.2-1, 1.3-1, 1.5-1, 1.7-1)")
    print("  4. Ensure model.tar.gz contains 'xgboost-model' file at root level")
    raise
