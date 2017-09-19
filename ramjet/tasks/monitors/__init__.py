from .ssl_cert import bind_task as bind_ssl_check_task


def bind_task():
    bind_ssl_check_task()
