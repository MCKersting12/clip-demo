FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt-get update -y && apt-get upgrade -y

RUN pip3 install --upgrade pip
RUN pip3 install torchvision transformers matplotlib glob2 pandas openpyxl sklearn tqdm streamlit

WORKDIR /workspace
COPY demo.py /workspace/

CMD ["streamlit", "run", "demo.py"]
