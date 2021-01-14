FROM python:3.8

RUN mkdir -p /foresee/rf-app

WORKDIR /foresee/rf-app

COPY requirements.txt .

RUN pip3 install -r ./requirements.txt

COPY . .

CMD ["python", "./train_traffic_intensity_RF_model.py"]