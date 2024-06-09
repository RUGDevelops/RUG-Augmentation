FROM python:3.9
RUN apt-get update && apt-get install -y libgl1-mesa-glx
ADD . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
CMD [ "python3", "./server.py" ]