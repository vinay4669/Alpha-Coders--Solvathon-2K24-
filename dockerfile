FROM python:3.9-slim-buster

WORKDIR / C:\Users\vinay\OneDrive\Desktop\my_model

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./multi_linear_regg.py" ]