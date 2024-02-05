import os
from ..exceptions import PathNotFoundError


def check_path_exists(path: str, create: bool = False):
    """
        Decoratore che controlla se il path esiste.
        Se non esiste, lo crea.
    """
    def decorator(decore_class: type):
        __old_init__ = decore_class.__init__

        def __new_init__(self, *args, **kwargs):
            if not os.path.exists(path):
                if not create:
                    raise PathNotFoundError(f"Path {path} not found")
                os.makedirs(path)

            __old_init__(self, *args, **kwargs)

        decore_class.__init__ = __new_init__
        return decore_class
    return decorator
