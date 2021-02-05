FROM ppcelery/gargantua-base:20200102

WORKDIR /app
ADD ./requirements.txt .
RUN pip install -r requirements.txt

ADD . .
RUN python setup.py install

ENTRYPOINT python -m ramjet
