FROM aeronetlab/prod:1.2-cpu

## Additional packages etc.

RUN pip install -r requirements.txt

## App

ADD . /f1
WORKDIR /f1

CMD ["python3","-u","app.py"]

