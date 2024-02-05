class ExtensionError(Exception):
    """Raised when the extension of the file is not .parquet"""
    pass


class DatasetNotFoundError(Exception):
    """Raised when the dataset is not found"""
    pass


class ModelNotFoundError(Exception):
    """Raised when the model is not found"""
    pass


class SetterNotAllowedError(Exception):
    """Raised when an attempt is made to set a class attribute when it is not allowed"""
    pass


class PathNotFoundError(Exception):
    """Raised when the path is not found"""
    pass
