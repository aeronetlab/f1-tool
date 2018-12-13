FROM aeronetlab/prod:1.2-cpu

## Additional packages etc.

RUN pip install flask
RUN pip install flask-cors

## App

ADD . /f1
WORKDIR /f1

CMD ["python3","-u","app.py"]

