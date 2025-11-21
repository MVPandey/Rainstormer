class RainstormerError(Exception):
    """Base exception for Rainstormer."""

    pass


class ChatModelError(RainstormerError):
    """Raised when the chat model fails."""

    pass
