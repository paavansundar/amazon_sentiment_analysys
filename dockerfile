FROM python:3.10

ADD ./sentiment_model /sentiment_model
WORKDIR /sentiment_model

COPY ./requirements/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "./sentiment_model/api/main.py"]
