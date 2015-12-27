from unittest import TestCase
from unittest.mock import patch
import datetime

import pymysql
from tornado.options import define

from ramjet.tasks.snmp.monitor import send_report_mail


def connent2db():
    return pymysql.connect(
        host='10.32.189.146',
        password='HFmWD7rDkfavyp2',
        port=3306,
        db='saltcmdb',
        user='root',
        charset='utf8mb4'
    )


class TestSnmpPortMonitor(TestCase):

    def setUp(self):
        self.connection = connent2db()
        define('debug', type=bool)

    def test_send_report_mail(self):
        """测试发送邮件
        """
        with patch('ramjet.tasks.snmp.monitor.send_mail') as mock_thing:
            arg = [{
                'address': '1.1.1.1',
                'nodename': 'Mars',
                'lastupdateat': datetime.datetime.now(),
                'basename': 'yeo',
                'error_ports': [{'portid': 1, 'portname': 'nde'}],

            }]
            send_report_mail(arg)
            mock_thing.assert_called_with(
                'messerflow@edm.chexiang.com',
                'caizhonghua@chexiang.com',
                '有交换机端口异常关闭', 'yeo机房 Mars交换机（1.1.1.1） 1号端口（nde）异常关闭。\n'
            )
