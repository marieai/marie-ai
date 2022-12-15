class BaseConfig:
    API_PREFIX = "/api"
    TESTING = False
    DEBUG = False


class DevConfig(BaseConfig):
    FLASK_ENV = "development"
    DEBUG = True
    CELERY_BROKER = "pyamqp://rabbit_user:rabbit_password@broker-rabbitmq//"
    CELERY_RESULT_BACKEND = "rpc://rabbit_user:rabbit_password@broker-rabbitmq//"


class ProductionConfig(BaseConfig):
    FLASK_ENV = "production"
    CELERY_BROKER = "pyamqp://rabbit_user:rabbit_password@broker-rabbitmq//"
    CELERY_RESULT_BACKEND = "rpc://rabbit_user:rabbit_password@broker-rabbitmq//"


class TestConfig(BaseConfig):
    FLASK_ENV = "development"
    TESTING = True
    DEBUG = True
    # make celery execute tasks synchronously in the same process
    CELERY_ALWAYS_EAGER = True
