import json
import logging.config
import traceback
from typing import Literal

from tasks_support_system_ai.core.config import settings


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        """Custom formatter that handles both dict and non-dict messages."""
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
        }

        if isinstance(record.msg, dict):
            log_data.update(record.msg)
        elif isinstance(record.msg, BaseException):
            log_data.update(
                {
                    "message": str(record.msg),
                    "error_type": record.msg.__class__.__name__,
                    "traceback": traceback.format_exc().split("\n"),
                }
            )
        else:
            log_data["message"] = str(record.msg)

        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_data, indent=4)


def get_logging_config(
    app_name: Literal["fastapi", "streamlit", "backend"],
    log_level: str = "INFO",
) -> dict:
    log_dir = settings.logs_path
    log_dir.mkdir(exist_ok=True, mode=0o777)

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {"()": JSONFormatter},
            "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
            "uvicorn": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(message)s",
                "use_colors": True,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "json",
                "filename": f"logs/{app_name}.log",
                "maxBytes": 10 * 10**20,
                "backupCount": 5,
                "encoding": "utf-8",
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "json",
                "filename": f"logs/{app_name}_error.log",
                "maxBytes": 10 * 10**20,
                "backupCount": 5,
                "encoding": "utf-8",
                "level": "ERROR",
            },
            "uvicorn_console": {
                "class": "logging.StreamHandler",
                "formatter": "uvicorn",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {},
    }

    if app_name == "fastapi":
        config["loggers"].update(
            {
                "fastapi": {
                    "handlers": ["file", "error_file"],
                    "level": log_level,
                    "propagate": False,
                },
                "uvicorn": {
                    "handlers": ["uvicorn_console"],
                    "level": log_level,
                    "propagate": False,
                },
                "uvicorn.access": {
                    "handlers": ["uvicorn_console"],
                    "level": log_level,
                    "propagate": False,
                },
                "uvicorn.error": {
                    "handlers": ["uvicorn_console", "error_file"],
                    "level": log_level,
                    "propagate": False,
                },
            }
        )
    else:
        config["loggers"].update(
            {
                app_name: {
                    "handlers": ["console", "file", "error_file"],
                    "level": log_level,
                    "propagate": False,
                },
            }
        )

    config["root"] = {
        "handlers": ["console"],
        "level": "WARNING",
    }

    return config
