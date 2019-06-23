FROM ppcelery/gargantua-base

WORKDIR /app
ADD ./requirements.txt .
RUN pip install -r requirements.txt

ADD . .
RUN python setup.py install

ENTRYPOINT python -m ramjet
