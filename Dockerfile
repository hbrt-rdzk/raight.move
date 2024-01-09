FROM python:3.11.7

ARG UID
ARG GID

WORKDIR /app

RUN groupadd --gid $GID myuser && useradd --no-create-home -u $UID --gid $GID myuser && chown -R myuser:myuser /app && mkdir /home/myuser && chown myuser:myuser /home/myuser

ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN apt-get update && apt-get install -y libgl1-mesa-glx 
COPY requirements.txt .
RUN pip --no-cache-dir install -r requirements.txt

USER myuser