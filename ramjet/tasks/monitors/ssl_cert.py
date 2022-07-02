import socket
import datetime
import ssl
from textwrap import dedent
from collections import namedtuple

from ramjet.engines import ioloop, thread_executor
from ramjet.utils import send_alert
from .base import logger as monitor_logger

_SiteConf = namedtuple("_SiteConf", ["domain", "check_ssl"])


ALERT_RECEIVERS = ["ppcelery@gmail.com"]
SITES = [
    _SiteConf(domain="blog.laisky.com", check_ssl=True),
]


logger = monitor_logger.getChild("ssl_cert")


def bind_task():
    def run():
        logger.info("running...")
        thread_executor.submit(main)
        ioloop.call_later(24 * 3600, run)

    run()


def main():
    try:
        map(checker, SITES)
    except Exception as err:
        logger.exception(err)


def checker(site: _SiteConf) -> None:
    check_ssl(site)


def check_ssl(site: _SiteConf) -> None:
    if not site.check_ssl:
        return

    expired_at = load_ssl_expiry_datetime(site.domain)
    if is_ssl_cert_need_alert(expired_at):
        send_alert_email(site.domain, expired_at)


def send_alert_email(domain, expired_at):
    logger.info("send_alert_email for domain %s", domain)
    content = dedent(
        """
        These domains' cert is goging to expired!

        {domain}: will expired at {expired_at}
    """.format(
            domain=domain, expired_at=expired_at
        )
    )
    send_alert(
        to_addrs=ALERT_RECEIVERS,
        subject="HTTPS cert going to expired!",
        content=content,
    )


def load_ssl_expiry_datetime(hostname):
    logger.debug("load_ssl_expiry_datetime for hostname: %s", hostname)
    ssl_date_fmt = r"%b %d %H:%M:%S %Y %Z"

    context = ssl.create_default_context()
    conn = context.wrap_socket(
        socket.socket(socket.AF_INET),
        server_hostname=hostname,
    )
    # 3 second timeout because Lambda has runtime limitations
    conn.settimeout(3.0)

    conn.connect((hostname, 443))
    ssl_info = conn.getpeercert()
    assert ssl_info is not None
    # parse the string from the certificate into a Python datetime object
    return datetime.datetime.strptime(ssl_info["notAfter"], ssl_date_fmt)


def is_ssl_cert_need_alert(valid_to) -> bool:
    return (valid_to - datetime.datetime.utcnow()) < datetime.timedelta(days=7)


if __name__ == "__main__":
    print(load_ssl_expiry_datetime("laisky.com"))
