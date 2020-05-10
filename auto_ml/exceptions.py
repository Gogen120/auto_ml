class UnknownMLTaskException(Exception):
    """Raised if unknown parameter task is provided. Valid choices: regression, classification"""


class InvalidBootstrapSampleSizeException(Exception):
    """Raised if sample size for bootstrap is greater than number of elements in dataset"""


class NotSupportedExtensionException(Exception):
    """Raised if file extension loading is not supported yet"""


class NotSupportedMetricException(Exception):
    """Raised if not supported metric for model provieded"""
