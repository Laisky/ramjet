import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from ramjet.settings import (MAIL_FROM_ADDR, MAIL_HOST, MAIL_PASSWD, MAIL_PORT,
                             MAIL_USERNAME)


def send_mail(*, to_addrs, subject, content, from_addr=MAIL_FROM_ADDR):
    """Send email

    Args:
        fr (string):
        to (list): receivers' addresses
    """
    smtp = smtplib.SMTP(host=MAIL_HOST, port=MAIL_PORT)
    smtp.starttls()
    smtp.login(MAIL_USERNAME, MAIL_PASSWD)

    msg = MIMEMultipart('alternative')
    msg.set_charset("utf-8")
    msg['From'] = from_addr
    msg['To'] = ', '.join(to_addrs[0].split(';'))
    msg['Subject'] = subject
    msg.attach(MIMEText(content, 'plain'))
    try:
        smtp.sendmail(from_addr, to_addrs, msg.as_string())
    except Exception:
        smtp.close()
        raise


if __name__ == '__main__':
    send_mail(to_addrs=['ppcelery@gmail.com'], subject='test', content='yooo')
