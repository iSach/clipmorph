FROM python:3.12

WORKDIR /workspace

COPY app app/
COPY clipmorph clipmorph/
COPY models models/
COPY training_data/styles training_data/styles/
COPY requirements.txt .
COPY pyproject.toml .

EXPOSE 8080

RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6 libxrender-dev
RUN pip install opencv-python-headless
RUN pip install -r app/requirements.txt
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install -e .

WORKDIR /workspace/app

CMD ["python", "app.py"]
