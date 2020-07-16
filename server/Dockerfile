FROM aeronetlab/prod:1.2-cpu
## Additional packages etc.

RUN pip install flask && pip install flask-cors && pip install -U rasterio==1.1.4 && pip install -U aeronet==0.0.8

## App

ADD . /f1-tool
WORKDIR /f1-tool
RUN python setup.py install
WORKDIR /f1-tool/geoscore

CMD ["python3","-u","app.py"]

