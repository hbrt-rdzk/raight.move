FROM python:3.10

ARG UID
ARG GID

WORKDIR /app

RUN groupadd --gid $GID myuser && useradd --no-create-home -u $UID --gid $GID myuser && chown -R myuser:myuser /app && mkdir /home/myuser && chown myuser:myuser /home/myuser

ENV PYTHONPATH "${PYTHONPATH}:/app"

COPY requirements.txt requirements_heavy.txt .
RUN pip --no-cache-dir install -r requirements_heavy.txt
RUN pip --no-cache-dir install -r requirements.txt

USER myuser