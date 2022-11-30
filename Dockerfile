FROM python:3.9-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code

CMD ["uvicorn", "router:app", "--host", "0.0.0.0", "--port", "9201"]


# WORKDIR /app
# ADD requirements.txt /app/
# RUN pip install --upgrade pip

# RUN pip install -r requirements.txt

# COPY . /app/


# RUN chmod +x run.sh

# ENTRYPOINT ["sh", "/app/run.sh"]