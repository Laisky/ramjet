import re
import heapq
from operator import itemgetter
from collections import defaultdict

import pymongo
import jieba.analyse

from ramjet.settings import logger
from ramjet.engines import ioloop, process_executor
from ramjet.utils import utcnow


logger = logger.getChild('tasks.heart')

N_CHINESE_KEYWORDS = 30
N_ENGLISH_KEYWORDS = 20
DB_HOST = 'localhost'
DB_PORT = 27016
DB = 'statistics'


def bind_task():
    def callback(*args, **kw):
        later = 60 * 60 * 12  # 12 hr
        logger.debug('add new task load_keywords after {}'.format(later))
        logger.info('Run task load_keywords')
        process_executor.submit(load_keywords, *args, **kw)
        ioloop.call_later(later, callback, *args, **kw)

    ioloop.call_later(1, callback, DB_HOST, DB_PORT)


def load_keywords(dbhost='localhost', dbport=27017):
    logger.debug(
        'load_keywords for dbhost {}, dbport {}'.format(dbhost, dbport)
    )

    try:
        conn = pymongo.MongoClient(dbhost, dbport)
        db = conn.blog
        coll = db[DB]

        contents = []
        for docu in db.posts.find({'post_password': ''}, {'post_content': 1}):
            contents.append(docu['post_content'])

        c = '\t'.join(contents)
        k_cn = load_chinese_keyword(c)
        logger.debug('load chinese keywords: {}'.format(k_cn))
        k_en = load_english_keyword(c)
        logger.debug('load english keywords: {}'.format(k_en))
        keywords = [] + k_cn + k_en

        coll.update(
            {'types': 'keyword'},
            {'$set': {
                'types': 'keyword',
                'keywords': keywords,
                'modified_at_gmt': utcnow()
            }}, upsert=True
        )
    except Exception as err:
        logger.exception(err)
    else:
        logger.info('keyword done.')


def load_chinese_keyword(content):
    logger.debug('load_chinese_keyword')

    return jieba.analyse.extract_tags(content, topK=N_CHINESE_KEYWORDS, allowPOS=('ns'))


def load_english_keyword(content):
    logger.debug('load_english_keyword')

    eng_kw_regex = re.compile('[a-zA-Z]{5,10}')
    ignores = set([
        'strong', 'class', 'style', 'uploads', 'align', 'height', 'image',
        'center', 'aligncenter', 'target', 'blockquote', 'blank', 'response',
        'width', 'align', 'alignleft', 'attachment', 'password', 'login',
        'valign', 'content', 'value', 'print', 'import', 'datetime', 'title', 'padding',
        'laisky', 'return', 'images', 'start', 'about', 'cinderellananako',
        'pnumber', 'subject', 'alignnone', 'ppcelery', 'color', 'config', 'shell',
        'static', 'index', 'label', 'number', 'param', 'users', 'first', 'notebook',
        'write', 'volume', 'files', 'total', 'timeit', 'latest', 'which', 'login',
        'field', 'requesthandler', 'https',
    ])
    key_map = defaultdict(int)
    kws = jieba.cut(content, cut_all=False)
    for k in kws:
        k = k.strip().lower()
        if k in ignores or not eng_kw_regex.match(k):
            continue

        key_map[k] += 1

    return [
        ks[0] for ks in
        heapq.nlargest(N_ENGLISH_KEYWORDS, key_map.items(), key=itemgetter(1))
    ]


if __name__ == '__main__':
    load_keywords()
