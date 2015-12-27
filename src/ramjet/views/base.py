#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import logging

import tornado

from ramjet.settings import OK, LOG_NAME
from ramjet.utils import TemplateRendering


log = logging.getLogger(LOG_NAME)
__all__ = ['BaseHandler']


class BaseHandler(tornado.web.RequestHandler, TemplateRendering):

    def get(self, url=None):
        url = url.strip(' /')
        super().get(url)

    @property
    def ip(self):
        return self.request.headers.get('X-Real-IP', self.request.remote_ip)

    def write_json(self, *, status=OK, msg='', data={}):
        j = json.dumps({'status': status, 'msg': msg, 'data': data})
        log.debug('<< {}'.format(j))
        self.write(j)

    def get_argument(self, arg_name, bool=False, *args, **kw):
        val = super().get_argument(arg_name, *args, **kw)
        if bool:
            return False if val in ('false', 'False', False, 0, None) else True
        else:
            return val

    @property
    def is_ajax(self):
        return self.request.headers.get('X-Requested-With') == "XMLHttpRequest"

    @property
    def is_https(self):
        return self.request.headers.get('X-Scheme') == "https"

    def redirect_404(self):
        self.redirect('/404.html')

    def render2(self, template_name, **kwargs):
        """
        This is for making some extra context variables available to
        the template
        """
        content = self.render_template(template_name, **kwargs)
        self.write(content)

    def render_template(self, template_name, **kwargs):
        def static_url(path):
            prefix = self.settings.get('static_url_prefix')
            return os.path.join(prefix, path)

        _kwargs = ({
            'settings': self.settings,
            'static_url': static_url,
            'reverse_url': self.reverse_url,
            'request': self.request,
            'xsrf_token': self.xsrf_token,
            'xsrf_form_html': self.xsrf_form_html,
            'max': max,
            'min': min,
            'is_ajax': self.is_ajax,
            'is_https': self.is_https,
            'current_user': self.current_user,
            'current_app': 'ramjet',
        })
        _kwargs.update(kwargs)
        return super().render_template(template_name, **_kwargs)
