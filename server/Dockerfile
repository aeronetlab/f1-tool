FROM aeronetlab/prod:1.2-cpu

## Additional packages etc.

RUN pip install flask
RUN pip install flask-cors

## App

ADD . /f1-tool
WORKDIR /f1-tool
RUN python setup.py install
WORKDIR /f1-tool/geoscore

CMD ["python3","-u","app.py"]

