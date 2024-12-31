import json
import logging.config
import traceback
from pathlib import Path
from typing import Literal


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        """Custom formatter that handles both dict and non-dict messages."""
        # Basic log record attributes
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
        }

        # Handle the message part
        if isinstance(record.msg, dict):
            log_data.update(record.msg)
        elif isinstance(record.msg, BaseException):
            log_data.update(
                {
                    "message": str(record.msg),
                    "error_type": record.msg.__class__.__name__,
                    "traceback": traceback.format_exc(),
                }
            )
        else:
            log_data["message"] = str(record.msg)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_data)


def get_logging_config(
    app_name: Literal["fastapi", "streamlit", "backend"],
    log_level: str = "INFO",
) -> dict:
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {"()": JSONFormatter},
            "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
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
                "maxBytes": 10 * 10**20,  # 10MB
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
        },
        "loggers": {
            app_name: {
                "handlers": ["console", "file", "error_file"],
                "level": log_level,
                "propagate": False,
            }
        },
        "root": {"handlers": ["console"], "level": "WARNING"},
    }
