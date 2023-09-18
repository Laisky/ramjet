import datetime
import json
import traceback

from ramjet.settings import logger


def debug_wrapper(func):
    def wrapper(*args, **kw):
        logger.debug(f"debug_wrapper for {args=}, {kw=}")
        try:
            yield from func(*args, **kw)
        except Exception:
            self = args[0]
            err_msg = {
                "uri": self.request.uri,
                "version": self.request.version,
                "headers": self.request.headers,
                "cookies": self.request.cookies,
            }
            logger.error(
                f"{json.dumps(err_msg, indent=4, sort_keys=True)}"
                "\n-----\n"
                f"{traceback.format_exc()}"
            )
            raise

    return wrapper


class TemplateRendering:
    """
    A simple class to hold methods for rendering templates.

    Copied from
        http://bibhas.in/blog/using-jinja2-as-the-template-engine-for-tornado-web-framework/
    """

    _jinja_env = None
    _assets_env = None

    def render_template(self, template_name, **kw):
        if not self._jinja_env:
            self._jinja_env.filters.update(
                {
                    "utc2cst": lambda dt: dt + datetime.timedelta(hours=8),
                    "jstime2py": lambda ts: datetime.datetime.fromtimestamp(ts / 1000),
                    "time_format": lambda dt: datetime.datetime.strftime(
                        dt, "%Y/%m/%d %H:%M:%S"
                    ),
                }
            )

        template = self._jinja_env.get_template(template_name)
        content = template.render(kw)
        return content
