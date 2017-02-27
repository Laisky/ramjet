import time
from io import BytesIO

import aiohttp
import boto3
import aiohttp_jinja2

from ramjet.engines import ioloop, thread_executor
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


def _load_img_labels(aws_conn, im):
    logger.info('request to deteck image...')
    return aws_conn.detect_labels(Image={'Bytes': im.read()},
                                  MaxLabels=10)


async def load_img_labels(aws_conn, im):
    logger.info('detect image...')
    return await ioloop.run_in_executor(thread_executor,
                                        _load_img_labels, aws_conn, im)


class DemoHandle(aiohttp.web.View):

    def connect2aws(self):
        return boto3.client(
            'rekognition',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name='us-west-2'
        )

    @aiohttp_jinja2.template('aws/index.tpl')
    async def get(self):
        return

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
                url: (await load_img_labels(aws_conn, im))['Labels']
            })

        results['cost'] = '{:.2f}s'.format(time.time() - start)
        return aiohttp.web.json_response(results)
