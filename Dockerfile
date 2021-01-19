FROM python:3.8.7-slim-buster

RUN mkdir -p /foresee_ethz_app/

WORKDIR /foresee_ethz_app/

COPY requirements.txt .

RUN pip install -r ./requirements.txt

COPY . .

CMD ["python3", "./train_traffic_intensity_RF_model.py"]
