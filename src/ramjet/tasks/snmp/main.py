import os
import time
import codecs
import pickle
import binascii
import traceback
import datetime

import yaml
import pytz
import pymysql
from pysnmp.entity.rfc3413.oneliner import cmdgen
from tornado.options import options

from ramjet.settings import logger
from .monitor import monitor_port


CWD = os.path.dirname(__file__)
SCAN_INTERVAL_SEC = 60 * 5  # 5 分钟更新一次
CLEAN_INTERVAL_SEC = 60 * 60 * 24  # 24 小时清理一次
HOSTS_CONFIG = (
    'bs_hosts.yml',  # 宝山机房
    'jq_hosts.yml',  # 金桥机房
)


logger = logger.getChild('tasks.snmp-net')
cmdGen = cmdgen.AsynCommandGenerator()
# set database
if options.debug:
    DB_HOST = '10.32.189.146'  # czh
    DB_PASSWD = 'HFmWD7rDkfavyp2'
else:
    DB_HOST = '10.32.135.33'  # cmdb
    DB_PASSWD = 'CU3pbHkPzTudWU4'

CST = pytz.timezone('Asia/Shanghai')
UTC = pytz.utc
utcnow = lambda: datetime.datetime.utcnow().replace(tzinfo=UTC)
cstnow = lambda: datetime.datetime.now(tz=CST)
ts2utc = lambda ts: datetime.datetime.fromtimestamp(ts).replace(tzinfo=CST).astimezone(UTC)
formatdt = lambda dt: datetime.datetime.strftime(dt, '%Y-%m-%d %H:%M:%S')
dumps = lambda pyobj: binascii.hexlify(pickle.dumps(pyobj, protocol=2)).decode('utf-8')  # 兼容 py2.7
loads = lambda s: pickle.loads(binascii.unhexlify(s.encode('utf-8')))


def bind_task(ioloop, thread_executor, process_executor):
    def main():
        """轮询并获取网络状态"""
        logger.info('main')
        start = time.time()
        process_executor.submit(run_fetch_snmp)
        delay = max(SCAN_INTERVAL_SEC - (time.time() - start), 0)
        ioloop.call_later(delay, main)

    def clean():
        """删除过期的 snmp 信息"""
        logger.info('delete_expired_netstats')
        start = time.time()
        process_executor.submit(clean_expired_netstats)
        delay = max(CLEAN_INTERVAL_SEC - (time.time() - start), 0)
        ioloop.call_later(delay, clean)

    clean()
    main()


def clean_expired_netstats():
    logger.info('clean_expired_netstats')
    expired_at = datetime.datetime.utcnow() - datetime.timedelta(days=7)
    sql = (
        """DELETE FROM cmdb_topology_netstat """
        """WHERE lastupdateat<%s;"""
    )
    connection = connent2db()
    with connection.cursor() as cursor:
        cursor.execute(sql, (expired_at,))

    connection.commit()
    connection.close()


def connent2db():
    return pymysql.connect(
        host=DB_HOST,
        password=DB_PASSWD,
        port=3306,
        db='saltcmdb',
        user='root',
        charset='utf8mb4'
    )


def run_fetch_snmp():
    logger.debug('run_fetch_snmp')
    try:
        hosts_conf = None
        connection = connent2db()
        result_netstat = {}
        for base in HOSTS_CONFIG:
            hosts_conf = load_hosts_config(base)
            with connection.cursor() as cursor:
                save_hostsconf2db(cursor, hosts_conf)

            result_netstat[hosts_conf['name']] = {}
            generate_snmp_tasks(hosts_conf, result_netstat)

        logger.debug('start generator!')
        result_netstat['ts'] = formatdt(utcnow())
        cmdGen.snmpEngine.transportDispatcher.runDispatcher()
        with connection.cursor() as cursor:
            save_netstat2db(cursor, result_netstat)

        connection.commit()
        monitor_port(connection)
    except Exception:
        logger.error('when hosts_conf: {}. raise: {}'
                     .format(hosts_conf, traceback.format_exc()))
    else:
        pass
    finally:
        connection.close()


def save_hostsconf2db(cursor, hosts_conf):
    """保存服务器配置在数据库"""
    logger.debug('save_hostsconf2db')
    basename = hosts_conf['name']
    sql = (
        """SELECT id, lastupdateat """
        """FROM cmdb_topology_nodeinfo """
        """WHERE basename=%s """
        """ORDER BY lastupdateat DESC;"""
    )
    logger.debug('execute sql: {}'.format(sql))
    cursor.execute(sql, (basename,))
    docu = cursor.fetchone()
    if docu and docu[1] == hosts_conf['lastupdateat']:
        logger.debug('no update for basename {}'.format(basename))
        return  # 不需要更新

    logger.debug('update basename {}'.format(basename))
    sql = (
        """DELETE FROM cmdb_topology_nodeinfo """
        """WHERE basename=%s;"""
    )
    logger.debug('execute sql: {}'.format(sql))
    cursor.execute(sql, (basename,))
    sql = (
        """INSERT IGNORE INTO cmdb_topology_nodeinfo (nodename, address, basename, community, lastupdateat) """
        """VALUES (%s, %s, %s, %s, %s);"""
    )
    args = []
    for host in hosts_conf['hosts']:
        args.append((
            host['name'],
            host['addr'],
            basename,
            host['community'],
            hosts_conf['lastupdateat']
        ))
    logger.debug('execute sql: {}'.format(sql))
    cursor.executemany(sql, args)


def filter_devs_stat(devs):
    """响应相当不完整，垃圾信息太多了"""
    result = {}
    for port_id, devstat in devs.items():
        if 'innet' in devstat and 'outnet' in devstat:
            result.update({port_id: devstat})

    return result


def save_netstat2db(cursor, result_netstat):
    """保存网络数据到数据库"""
    logger.debug('save_netstat2db for result_netstat {}'.format(result_netstat))
    args = []
    sql = (
        """INSERT IGNORE INTO cmdb_topology_netstat (nodename, address, basename, lastupdateat, devs) """
        """VALUES (%s, %s, %s, %s, %s);"""
    )
    for basename, v in result_netstat.items():
        if basename == 'ts':
            continue

        for nodename, nodestat in v.items():
            devstats = filter_devs_stat(nodestat['devs'])
            if devstats:
                args.append((
                    nodename,                        # 'nodename'
                    nodestat['addr'],                # 'addr'
                    basename,                        # 'basename'
                    result_netstat['ts'],            # 'ts'
                    dumps(devstats),                 # 'devs'
                ))

    logger.debug('execute sql: {}'.format(sql))
    logger.info('save {} new netstats.'.format(len(args)))
    cursor.executemany(sql, args)


def load_hosts_config(config_fname):
    logger.debug('load_hosts_config for config_fname {}'.format(config_fname))
    fpath = os.path.join(CWD, config_fname)
    hosts_conf = yaml.load(codecs.open(fpath, 'r', 'utf-8'))
    hosts_conf['lastupdateat'] = formatdt(ts2utc(os.stat(fpath).st_mtime))
    return hosts_conf


# http://src.gnu-darwin.org/ports/net-mgmt/py-snmp4/work/pysnmp-4.1.7a/docs/pysnmp-tutorial.html#AsynCommandGenerator
def register_mib(addr: '请求的节点地址',
                 community: '验证信息',
                 mibs: '请求的 mibs',
                 context: '上下文',
                 callback: '回调函数'
                 ) -> '注册 pysnmp 抓取任务':
    logger.debug('register for addr {}, mibs {} with community {}'
                 .format(addr, repr(mibs), community))

    cmdGen.nextCmd(
        cmdgen.CommunityData(community),
        cmdgen.UdpTransportTarget((addr, 161)),
        mibs,
        (callback, context),
        True, True
    )


def snmp_callback(sendRequestHandle, errorIndication, errorStatus, errorIndex, varBinds, cbCtx):
    if errorIndication or errorStatus or errorIndex:
        logger.error(
            'sump error with addr {}, targets {}. '
            'errorIndication {}, errorStatus {}, errorIndex {}'
            .format(cbCtx['addr'], cbCtx['targets'],
                    errorIndication, errorStatus, errorIndex)
        )

    for row in varBinds:
        for oid, val in row:

            dev_id = str(int(oid.getMibSymbol()[-1][0]))
            val = val.prettyPrint()
            mib = oid.getLabel()[-1]
            if mib not in cbCtx['targets']:
                return

            desc = cbCtx['targets'][mib]
            basename = cbCtx['basename']
            nodename = cbCtx['name']
            result_netstat = cbCtx['result_netstat']

            # 考虑到要 pickle.loads，不要使用 defaultdict
            if nodename not in result_netstat[basename]:
                result_netstat[basename][nodename] = {'addr': cbCtx['addr']}
            if 'devs' not in result_netstat[basename][nodename]:
                result_netstat[basename][nodename]['devs'] = {}
            if dev_id not in result_netstat[basename][nodename]['devs']:
                result_netstat[basename][nodename]['devs'][dev_id] = {}

            logger.debug('callback for basename {}, nodename {}, dev_id {}, desc {}'
                         .format(basename, nodename, dev_id, desc))
            result_netstat[basename][nodename]['devs'][dev_id][desc] = val

    return 1


mibs = ([
    cmdgen.MibVariable('IF-MIB', 'ifDescr'),
    cmdgen.MibVariable('IF-MIB', 'ifInOctets'),
    cmdgen.MibVariable('IF-MIB', 'ifOutOctets'),
    cmdgen.MibVariable('IF-MIB', 'ifAdminStatus'),
    cmdgen.MibVariable('IF-MIB', 'ifOperStatus'),
    cmdgen.MibVariable('IF-MIB', 'ifSpeed'),
    cmdgen.MibVariable('IF-MIB', 'ifInDiscards'),
    cmdgen.MibVariable('IF-MIB', 'ifOutDiscards'),
    cmdgen.MibVariable('IF-MIB', 'ifInErrors'),
    cmdgen.MibVariable('IF-MIB', 'ifOutErrors'),
])


def generate_snmp_tasks(hosts_conf, result_netstat):
    logger.debug('load_mib')
    for hostinfo in hosts_conf['hosts']:
        arg = {
            'addr': hostinfo['addr'],
            'community': hostinfo['community'],
            'mibs': mibs,
            'context': {
                'basename': hosts_conf['name'],
                'addr': hostinfo['addr'],
                'name': hostinfo['name'],
                'targets': {
                    'ifDescr': 'name',
                    'ifInOctets': 'innet',
                    'ifOutOctets': 'outnet',
                    'ifAdminStatus': 'admin-status',
                    'ifOperStatus': 'oper-status',
                    'ifSpeed': 'speed',
                    'ifInDiscards': 'in-discard',
                    'ifOutDiscards': 'out-discard',
                    'ifInErrors': 'in-error',
                    'ifOutErrors': 'out-error'
                },
                'result_netstat': result_netstat
            },
            'callback': snmp_callback
        }
        register_mib(**arg)
