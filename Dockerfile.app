FROM python:3.12

WORKDIR /workspace

COPY api api/
COPY clipmorph clipmorph/
COPY models models/
COPY training_data/styles training_data/styles/
COPY requirements.txt .
COPY pyproject.toml .

EXPOSE 8080

RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6 libxrender-dev
RUN pip install opencv-python-headless
RUN pip install flask
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install -e .

WORKDIR /workspace/api

ENV FLASK_APP=api.py

CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
