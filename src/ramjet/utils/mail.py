import smtplib
from email.mime.text import MIMEText


def send_mail(fr, to, subject, content):
    msg = MIMEText(content)

    msg['Subject'] = subject
    msg['From'] = fr
    msg['To'] = to

    # Send the message via our own SMTP server.
    s = smtplib.SMTP('10.32.135.22', port=25)
    # s.login(user='messerflow@edm.chexiang.com')
    r = s.send_message(msg)
    s.quit()
    return r
