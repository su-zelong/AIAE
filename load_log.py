import logging

# 配置日志级别和日志格式
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 创建一个日志记录器
logger = logging.getLogger(__name__)
