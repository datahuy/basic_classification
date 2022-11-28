# FROM python:3.9-slim as compiler

# #Setting PYTHONUNBUFFERED to a non-empty value different from 0 ensures that the python output i.e. the stdout and stderr streams are sent straight to terminal (e.g. your container log) without being first buffered and that you can see the output of your application (e.g. django logs) in real time.
# ENV PYTHONUNBUFFERED 1

# # WORKDIR /usr/app

# # RUN python -m venv /opt/venv

# # ENV PATH="/opt/venv/bin:$PATH"

# COPY requirements.txt .

# RUN pip install -r requirements.txt

# FROM compiler as runner

# WORKDIR /usr/app

# # COPY --from=compiler /opt/venv /opt/venv

# # ENV PATH="/opt/venv/bin:$PATH"

# COPY . .

# RUN chmod +x run.sh

# ENTRYPOINT ["sh", "/app/run.sh"]


FROM continuumio/miniconda3:4.12.0


WORKDIR /app
ADD requirements.txt /app/
RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . /app/


RUN chmod +x run.sh

ENTRYPOINT ["sh", "/app/run.sh"]