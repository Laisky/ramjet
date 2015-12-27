"""SNMP 监控项目
当有端口状态变为关闭时发送邮件报警
"""
import pickle
import binascii

from ramjet.settings import logger
from ramjet.utils import utcnow, UTC
from ramjet.utils.mail import send_mail


monitor_targets = [
    {
        'addr': '10.33.131.254',
        'name': 'CheXiang_VS'
    }, {
        'addr': '10.33.131.251',
        'name': 'JQDCSW-H-A12'
    }, {
        'addr': '10.32.33.1',
        'name': 'VS-NeiWang'
    }, {
        'addr': '10.32.33.14',
        'name': 'C2-6800'
    }
]


i_nodename = 0
i_addr = 1
i_basename = 2
i_lastupdateat = 3
i_devs = 4

# mail settings
EMAIL_FROM = 'cmdb@chexiang.com'
EMAIL_TO = 'caizhonghua@chexiang.com,wangjinlong@chexiang.com'
EMAIL_SUBJECT = '有交换机端口异常关闭'

logger = logger.getChild('tasks.snmp-net')


def monitor_port(connection):
    logger.info('monitor_port')

    error_report = []
    for hostinfo in monitor_targets:
        netstats = fetch_netstat(connection, hostinfo['addr'])
        if len(netstats) < 2:
            logger.debug('no enough data for host {}'.format(hostinfo['addr']))
            continue

        ns_now = netstats[0]
        ns_past = netstats[1]
        if (utcnow() - ns_now[i_lastupdateat].replace(tzinfo=UTC)).seconds > 100:
            logger.debug('no current data for host {}'.format(hostinfo['addr']))
            continue

        devs_now = pickle.loads(binascii.unhexlify((ns_now[i_devs].encode('utf-8'))))
        devs_past = pickle.loads(binascii.unhexlify((ns_past[i_devs].encode('utf-8'))))

        err_ports = check_if_port_down(devs_now, devs_past)
        if not err_ports:
            logger.debug('no port down.')
            continue

        error_report.append({
            'address': ns_now[i_addr],
            'nodename': ns_now[i_nodename],
            'lastupdateat': ns_now[i_lastupdateat],
            'basename': ns_now[i_basename],
            'error_ports': err_ports
        })

    if error_report:
        send_report_mail(error_report)
    else:
        logger.debug('no error_report')


def send_report_mail(error_report):
    logger.debug('send_report_mail for error_report {}'.format(error_report))
    msg = ''
    for er in error_report:
        for ep in er['error_ports']:
            msg += (
                '{basename}机房 {nodename}交换机（{address}）{portid}({portname}) 端口异常关闭。\n'
                .format(basename=er['basename'], nodename=er['nodename'], address=er['address'],
                        portid=ep['portid'], portname=ep['portname'])
            )

    logger.debug('send mail to {}: {}'.format(EMAIL_TO, msg))
    send_mail(EMAIL_FROM, EMAIL_TO, EMAIL_SUBJECT, msg)


def fetch_netstat(connection, address):
    logger.debug('fetch_netstat for address {}'.format(address))
    sql = (
        """SELECT nodename, address, basename, lastupdateat, devs """
        """FROM cmdb_topology_netstat """
        """WHERE address=%s """
        """ORDER BY id DESC LIMIT 2;"""
    )
    with connection.cursor() as cursor:
        cursor.execute(sql, (address,))
        r = cursor.fetchall()

    return r


def check_if_port_down(devs_now, devs_past):
    logger.debug('check_if_port_down for devs_now {}, devs_past {}'
                 .format(devs_now, devs_past))
    error_port_list = []

    for portid, portstat in devs_now.items():
        if portid not in devs_past:
            continue

        if portstat['oper-status'] == 'down' \
                and devs_past[portid]['oper-status'] == 'up':
            error_port_list.append({'portid': portid, 'portname': portstat['name']})

    return error_port_list
