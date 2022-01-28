FROM python:latest

WORKDIR /server/
ADD . /server

RUN pip install -r requirements.txt
RUN python train.py

CMD [ "python3", "app.py" ]