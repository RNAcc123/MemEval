"""Shared provider failure taxonomy."""


class ProviderError(Exception):
    """Base class for failures raised by judge providers."""


class RetryableProviderError(ProviderError):
    """A transient provider failure that may succeed on retry."""


class PermanentProviderError(ProviderError):
    """A configuration, authentication, or invalid-request failure."""


class ProviderResponseError(PermanentProviderError):
    """Provider returned a response that cannot be normalized."""
