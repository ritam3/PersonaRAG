import json
import logging
import os


def get_model_flow_logger() -> logging.Logger:
    """Create or reuse the structured logger used across the app runtime."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger("model_flow")
    logger.setLevel(log_level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)

    return logger


def log_event(logger: logging.Logger, event: str, **payload) -> None:
    """Log JSON-serializable payloads in a consistent format."""
    safe_payload = {}
    for key, value in payload.items():
        try:
            json.dumps(value)
            safe_payload[key] = value
        except TypeError:
            safe_payload[key] = str(value)

    logger.info("%s | %s", event, json.dumps(safe_payload, ensure_ascii=False))
