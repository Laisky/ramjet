import datetime
import os
import random
import re
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Generator, List

import pymongo
import re
from ramjet.settings import TWITTER_IMAGE_DIR
from ramjet.settings import logger as ramjet_logger

logger = ramjet_logger.getChild("tasks.gptchat")
