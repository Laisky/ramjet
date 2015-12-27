from unittest import TestCase
from unittest.mock import patch, call

from ramjet.tasks import snmp


class TestSnmp(TestCase):

    """测试 tasks/snmp
    """

    def setUp(self):
        self.conf = snmp.update_config()

    def test_load_mib(self):
        addr = '10.32.33.10'
        mibs = ('IF-MIB', 'ifDescr')
        ret = snmp.load_mib(addr, mibs)
        self.assertIsInstance(ret, dict)
        self.assertNotEqual(len(ret), 0)
        for oid, val in ret.items():
            self.assertIsInstance(oid, int)

    def test_update_config(self):
        self.assertIn('community', self.conf)
        self.assertIn('hosts', self.conf)
        self.assertIsInstance(self.conf['hosts'], list)

    def test_load_net_data(self):
        addr = '10.32.33.10'
        ret = snmp.load_net_data(addr)
        self.assertIsInstance(ret, dict)
        for dev, stat in ret.items():
            self.assertIsInstance(stat, dict)
            self.assertIn('in', stat)
            self.assertIn('out', stat)

    def test_save_to_db(self):
        with patch('pymongo.collection.Collection.insert_one') as mock_insert:
            argument = {'aaa': 123}
            snmp.save_to_db(argument)
            self.assertEqual(mock_insert.mock_calls, [call(argument)])
