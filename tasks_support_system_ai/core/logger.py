import logging
import logging.config

from tasks_support_system_ai.core.logging_config import get_logging_config

logging.config.dictConfig(get_logging_config("streamlit"))
logging.config.dictConfig(get_logging_config("fastapi"))
logging.config.dictConfig(get_logging_config("backend"))

streamlit_logger = logging.getLogger("streamlit")
fastapi_logger = logging.getLogger("fastapi")
backend_logger = logging.getLogger("backend")
