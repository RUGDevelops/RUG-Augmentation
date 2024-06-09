FROM python:3.9
ADD server.py /
ADD augmentation.py /
ADD requirements.txt /
ADD ModelCreator.py /
ADD dataset /dataset
WORKDIR .
RUN pip3 install -r requirements.txt
CMD [ "python3", "./server.py" ]