FROM python:3.9-slim


WORKDIR /app
ADD requirements.txt /app/
RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . /app/


RUN chmod +x run.sh

ENTRYPOINT ["sh", "/app/run.sh"]