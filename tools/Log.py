import logging
from logging import handlers

logger = logging.getLogger('__name__')
logger.setLevel(logging.DEBUG)
format_str = logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')#设置日志格式

fh = handlers.TimedRotatingFileHandler(filename='/home/chase/pyinfo.log',when='MIDNIGHT',backupCount=30 ,encoding='utf-8')
fh.setFormatter(format_str)
logger.addHandler(fh)

sh = logging.StreamHandler()#往屏幕上输出
sh.setFormatter(format_str) #设置屏幕上显示的格式
logger.addHandler(sh)
