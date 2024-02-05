import os


def validation(name, doc=None, *validate_function):
    def decorator(decore_class):

        protected_name = "_" + name
        private_name = "__" + name

        def get_param(self):
            return getattr(self, private_name)

        def set_param(self, value):
            for function in validate_function:
                value = function(protected_name, value)
            setattr(self, private_name, value)

        setattr(decore_class, protected_name, property(get_param, set_param, doc=doc))
        return decore_class

    return decorator


def ensure_valid_file_extension(extension: str, allow_none=False):
    """
        Decoratore che controlla se l'estensione del file è corretta.
        Se non lo è, la aggiunge.
    """

    def decorator(name, value):
        if value is None and allow_none:
            return None

        if not value.endswith(extension):
            value += extension
        return value

    return decorator


def ensure_file_exists(base_path, subdir, allow_none=False):
    """
        Decoratore che controlla se il file esiste.
        Se non esiste, lancia un'eccezione.
    """

    def decorator(name, value):
        if value is None and allow_none:
            return None

        full_path = os.path.join(base_path, subdir, value)
        if os.path.exists(full_path):
            return value
        raise FileNotFoundError(f"File {value} not found in {subdir} directory.")

    return decorator


def ensure_valid_type(param_type, allow_none=False):
    """
        Decoratore che controlla il tipo di un parametro.
        Se non è del tipo corretto, lancia un'eccezione.
    """

    def decorator(name, value):
        if value is None and allow_none:
            return None

        if isinstance(value, param_type):
            return value
        raise TypeError(f"Parameter {name} must be {param_type} type.")

    return decorator


def ensure_not_none(default_value=None, exception_type=TypeError):
    """
        Decoratore che controlla se il parametro è None.
        Se è None, lancia un'eccezione.
    """

    def decorator(name, value):
        if value is None:
            if default_value is None:
                raise exception_type(f"Parameter {name} must not be None.")
            return default_value
        return value

    return decorator


def ensure_into_allowed_options(options: list | tuple | set, default_value=None):
    """
        Decoratore che controlla se il parametro è nel insieme.
    """

    def decorator(name, value):
        if value not in options:
            if default_value is None:
                raise TypeError(f"Parameter {name} must be in the allowed options.\n This is the options: {options}")
            return default_value
        return value

    return decorator
