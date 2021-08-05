# Pull the base image with python 3.8 as a runtime for your Lambda
FROM public.ecr.aws/lambda/python:3.7

WORKDIR ${LAMBDA_TASK_ROOT}
# Install OS packages for Pillow-SIMD
RUN yum -y install tar gzip zlib freetype-devel \
    gcc \
    && yum clean all


COPY requirements.txt ./
RUN python3.7 -m pip install --upgrade pip setuptools wheel bottleneck
RUN python3.7 -m pip install --trusted-host pypi.python.org -r requirements.txt

ADD model/create_model.py ./model/
ADD handler.py ./
CMD ["handler.inference"]