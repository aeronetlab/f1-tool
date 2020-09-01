FROM aeronetlab/prod:1.2-cpu
## Additional packages etc.

RUN pip install flask && pip install flask-cors && pip install -U rasterio==1.1.4 \
 && pip install keras==2.2.4 && pip install numpy==1.15.4 && pip install shapely==1.7a1 && pip install opencv-python==4.0.0.21 && pip install tqdm==4.36.1 \
 && pip install rtree>=0.8.3 && pip install -U --no-deps aeronet==0.0.8

## App

ADD . /f1-tool
WORKDIR /f1-tool
RUN python setup.py install
WORKDIR /f1-tool/geoscore

CMD ["python3","-u","app.py"]

