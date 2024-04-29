FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /code

# Install curl for healthcheck
RUN apt-get -y update; apt-get -y install curl 

# Install dependencies
COPY ./requirements/base.txt /code/requirements.txt
RUN pip install --upgrade -r /code/requirements.txt

# Copy configuration files and code
COPY ./logger.ini /code/logger.ini
COPY ./app_start.sh /code/app_start.sh
COPY ./src /code/src


