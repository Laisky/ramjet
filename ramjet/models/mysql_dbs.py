from .base import BaseMySQLModel


def create_mysql_model(db_name):
    class Model(BaseMySQLModel):
        _db_name = db_name

    return Model


_models = {}

def get_mysql_model(db_name):
    if db_name in _models:
        model = _models[db_name]
    else:
        model = create_mysql_model(db_name)
        _models[db_name] = model

    return model


MovotoDB = get_mysql_model('movoto')
