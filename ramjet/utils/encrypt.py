#!/usr/bin/env python
# -*- coding: utf-8 -*-

import bcrypt
import jwt

from ramjet.settings import SECRET_KEY


def generate_passwd(passwd):
    return bcrypt.hashpw(passwd, bcrypt.gensalt())


def validate_passwd(passwd, hashed):
    return bcrypt.hashpw(passwd, hashed) == hashed


def generate_token(json_, secret=SECRET_KEY):
    """generate_token signs JSON claims with secret and returns an HS512 JWT string."""
    return jwt.encode(json_, secret, algorithm="HS512")


def validate_token(token, secret=SECRET_KEY):
    """validate_token verifies an HS512 JWT with secret and returns its JSON claims."""
    return jwt.decode(token, secret, algorithms=["HS512"])
