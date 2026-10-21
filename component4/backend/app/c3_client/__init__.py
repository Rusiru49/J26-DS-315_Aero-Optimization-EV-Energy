from ..config import Settings
from .api_client import ApiC3Client
from .base import C3Client, C3Error
from .file_client import FileC3Client


def make_client(settings: Settings) -> C3Client:
    """The only place that decides mock/live. Change C3_MODE / C3_API_URL, nothing else."""
    if settings.c3_mode == "file":
        return FileC3Client(settings.data_file)
    return ApiC3Client(settings.c3_api_url, settings.c3_timeout_s)


__all__ = ["C3Client", "C3Error", "ApiC3Client", "FileC3Client", "make_client"]