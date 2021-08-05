import os
from io import BytesIO
import pandas as pd
import numpy as np
import boto3
import joblib
from aws_lambda_powertools.utilities.parser import event_parser, BaseModel
from aws_lambda_powertools.utilities.typing import LambdaContext
from aws_lambda_powertools.utilities.parser.models import APIGatewayProxyEventModel

from model.create_model import base_processing

s3 = boto3.resource('s3')


with BytesIO() as data:
    s3.Bucket(str(os.environ.get('bucket'))).download_fileobj(str(os.environ.get('model_name')), data)
    data.seek(0)    # move back to the beginning after writing
    MODEL = joblib.load(data)


class AmsterdamData(BaseModel):
    neighbourhood: str
    latitude: float
    longitude: float
    room_type: str
    minimum_nights: int
    availability_365: int


class InputEventModel(APIGatewayProxyEventModel):
    body: AmsterdamData


class OutputDataModel(BaseModel):
    status_code: int
    estimated_price: int
    description: str


@event_parser(model=InputEventModel)
def inference(event: InputEventModel, context: LambdaContext):
    df = pd.DataFrame([dict(event.body)])
    df = base_processing(df, train=False)
    prediction = MODEL.predict(df)
    return OutputDataModel(estimated_price=int(prediction[0]), status_code=200,
                           description="Amsterdam Apartment Price").json()
