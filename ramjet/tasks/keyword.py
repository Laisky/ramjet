import re
import heapq
from operator import itemgetter
from collections import defaultdict

import jieba.analyse

from ramjet.settings import logger
from ramjet.engines import ioloop, process_executor
from ramjet.utils import utcnow, get_conn


logger = logger.getChild('tasks.keyword')

N_CHINESE_KEYWORDS = 30
N_ENGLISH_KEYWORDS = 20
DB_HOST = 'localhost'
DB_PORT = 27016
DB = 'statistics'
STOP_WORDS = set([
    'strong', 'class', 'style', 'uploads', 'align', 'height', 'image',
    'center', 'aligncenter', 'target', 'blockquote', 'blank', 'response',
    'width', 'align', 'alignleft', 'attachment', 'password', 'login',
    'valign', 'content', 'value', 'print', 'import', 'datetime', 'title', 'padding',
    'laisky', 'return', 'images', 'start', 'about', 'cinderellananako',
    'pnumber', 'subject', 'alignnone', 'ppcelery', 'color', 'config', 'shell',
    'static', 'index', 'label', 'number', 'param', 'users', 'first', 'notebook',
    'write', 'volume', 'files', 'total', 'timeit', 'latest', 'which', 'login',
    'field', 'requesthandler', 'https', 'tbody', 'table', 'thead', 'white', 'border',
    'cellpadding', 'wordpress', 'archives', 'nofollow', 'small', 'before', 'should',
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
    'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
    'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
    'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
    'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
    'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn',
    'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', 'http', 'br', 'li', 'div', 'tag',
    'table', 'td', 'tr', 'text', 'left', 'html', 'href', 'blog', 'sina', 'size', 'nbsp', 'span',
    'weibo', 'youku', 'douban', 'bgyx', 'tumblr', 'full', 'error', 'null', 'true', 'false', 'qiniu', 'self',
])



def bind_task():
    def callback():
        later = 60 * 60 * 12  # 12 hr
        logger.debug('add new task load_keywords after {}'.format(later))
        logger.info('Run task load_keywords')
        process_executor.submit(load_and_save_post_cnt_and_keywords)
        ioloop.call_later(later, callback)

    ioloop.call_later(1, callback)


def load_and_save_post_cnt_and_keywords():
    conn = get_conn()
    db = conn.blog
    coll = db[DB]

    all_cnt = ''
    for docu in db.posts.find():
        if docu.get('post_password'):
            continue

        cnt = docu.get('post_markdown') or docu.get('post_content')
        k_cn_each = load_chinese_keyword(cnt)
        k_en_each = load_english_keyword(cnt)
        if k_en_each and k_cn_each:
            tags = k_cn_each[:3] + k_en_each[:2]
        elif k_cn_each:
            tags = k_cn_each[:5]
        elif k_en_each:
            tags = k_en_each[:5]

        db.posts.update_one({'_id': docu['_id']},
                            {'$set': {'post_tags': tags}})
        logger.debug('update %s\'s post_tags: %s', docu.get('post_title'), tags)
        all_cnt += '\t' + cnt

    load_keywords_for_all(coll, all_cnt)


def load_keywords_for_all(coll, all_cnt):
    logger.debug('load_keywords')

    try:
        k_cn = load_chinese_keyword(all_cnt)
        logger.debug('load chinese keywords: {}'.format(k_cn))
        k_en = load_english_keyword(all_cnt)
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

    eng_kw_regex = re.compile('^[a-zA-Z]{4,15}$')
    key_map = defaultdict(int)
    kws = jieba.cut(content, cut_all=False)
    for k in kws:
        k = k.strip().lower()
        if k in STOP_WORDS or not eng_kw_regex.match(k):
            continue

        key_map[k] += 1

    return [
        ks[0] for ks in
        heapq.nlargest(N_ENGLISH_KEYWORDS, key_map.items(), key=itemgetter(1))
    ]


if __name__ == '__main__':
    load_and_save_post_cnt_and_keywords()
