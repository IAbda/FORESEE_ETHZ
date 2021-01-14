FROM python:3.8

RUN mkdir -p /FORESEE_ETHZ_APP/

WORKDIR /FORESEE_ETHZ_APP/

COPY requirements.txt .

RUN pip3 install -r ./requirements.txt

COPY . .

CMD ["python3", "./train_traffic_intensity_RF_model.py"]
