# The following line will download a Docker image that already is set up with python 3.12, so that we can run our
# simple application. 
FROM python:3.12

WORKDIR /workspace

COPY . .

# Make port 8080 available to the world outside this container.
EXPOSE 8080

# Solve cv2 error
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install opencv-python-headless

# Download and install depedencies (libraries)
RUN pip install -r api/requirements.txt
RUN pip install -r requirements-cpu.txt
RUN pip install -e .

WORKDIR /workspace/api

# Define environment variable
ENV FLASK_APP=api.py

# Run api.py when the container launches
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
