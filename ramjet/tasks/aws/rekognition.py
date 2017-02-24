import time
from io import BytesIO

import aiohttp
import boto3

from ramjet.settings import AWS_ACCESS_KEY, AWS_SECRET_KEY, logger

logger = logger.getChild('tasks.aws.rekognition')


def bind_handle(add_route):
    logger.info('bind_handle aws.rekognition')
    add_route('/detect/', DemoHandle)


async def download_img(url):
    logger.info('download image {}'.format(url))
    with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return BytesIO(await resp.read())


def detect_img(aws_conn, im):
    logger.info('detect image...')
    return aws_conn.detect_labels(
        Image={
            'Bytes': im.read()
        },
        MaxLabels=10,
    )


description = """
    Run in command lines:
    ::
        $ pip install httpie
        $ http post app.laisky.com/image/detect/ urls:='["https://s3-us-west-1.amazonaws.com/movoto-data/demo_100x150.jpeg"]'

    Or:
    ::
        import requests

        url = 'app.laisky.com/image/detect/'
        data = {
            'urls': [
                # maximum to 5 picture urls
                xxx,
                xxx,
            ]
        }
        resp = requests.post(url, json=data)
        print(resp.json())
"""


class DemoHandle(aiohttp.web.View):

    def connect2aws(self):
        return boto3.client(
            'rekognition',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name='us-west-2'
        )

    async def get(self):
        return aiohttp.web.Response(text=description)

    async def post(self):
        try:
            data = await self.request.json()
            urls = data['urls']
            assert len(urls) < 5
        except Exception:
            return aiohttp.web.HTTPBadRequest()

        aws_conn = self.connect2aws()

        results = {
            'cost': None,
            'results': {}
        }
        start = time.time()
        for url in urls:
            im = await download_img(url)
            results['results'].update({
                url: detect_img(aws_conn, im)['Labels']
            })

        results['cost'] = '{:.2f}s'.format(time.time() - start)
        return aiohttp.web.json_response(results)
