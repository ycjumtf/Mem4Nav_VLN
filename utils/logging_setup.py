import logging
import sys
import os
from typing import Optional, Dict, Any

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Store a reference to the initialized logger to avoid re-configuring if called multiple times.
_INITIALIZED_LOGGERS: Dict[str, logging.Logger] = {}

def setup_logger(name: str = "Mem4NavApp",
                   config: Optional[Dict[str, Any]] = None,
                   default_level: str = "INFO",
                   log_file: Optional[str] = None,
                   log_format_str: Optional[str] = None,
                   date_format_str: Optional[str] = None,
                   force_reconfigure: bool = False) -> logging.Logger:
    """
    Sets up and configures a logger instance.

    If a logger with the given name has already been initialized, it will
    return the existing instance unless `force_reconfigure` is True.

    Args:
        name (str): The name for the logger. Typically the application name.
        config (Optional[Dict[str, Any]]): A configuration dictionary,
            which might contain logging settings under a 'logging' key.
            Expected keys in config['logging']:
            - 'level' (str): Log level (e.g., "DEBUG", "INFO", "WARNING").
            - 'log_file_path' (str): Path to the log file.
            - 'log_format' (str): Custom log message format string.
            - 'date_format' (str): Custom date format string for logs.
        default_level (str): Default log level if not specified in config.
        log_file (Optional[str]): Path to the log file (overrides config if provided).
        log_format_str (Optional[str]): Log format string (overrides config).
        date_format_str (Optional[str]): Date format string (overrides config).
        force_reconfigure (bool): If True, reconfigures the logger even if already initialized.

    Returns:
        logging.Logger: The configured logger instance.
    """
    global _INITIALIZED_LOGGERS

    if name in _INITIALIZED_LOGGERS and not force_reconfigure:
        return _INITIALIZED_LOGGERS[name]

    logger = logging.getLogger(name)

    # Determine configuration values, giving precedence to direct args, then config, then defaults
    log_conf = config.get('logging', {}) if config else {}

    level_str = log_conf.get('level', default_level).upper()
    log_level = getattr(logging, level_str, logging.INFO)

    # If log_file is passed directly, it overrides the config path
    final_log_file = log_file if log_file is not None else log_conf.get('log_file_path')
    
    final_log_format = log_format_str if log_format_str is not None else log_conf.get('log_format', DEFAULT_LOG_FORMAT)
    final_date_format = date_format_str if date_format_str is not None else log_conf.get('date_format', DEFAULT_DATE_FORMAT)

    # Reset handlers if reconfiguring to avoid duplicate messages
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(log_level)
    formatter = logging.Formatter(fmt=final_log_format, datefmt=final_date_format)

    # Console Handler (always add, unless config explicitly disables)
    if log_conf.get('enable_console_logging', True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level) # Console handler can have its own level too
        logger.addHandler(console_handler)

    # File Handler (if path is provided)
    if final_log_file:
        try:
            log_file_dir = os.path.dirname(final_log_file)
            if log_file_dir and not os.path.exists(log_file_dir):
                os.makedirs(log_file_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(final_log_file, mode='a') # Append mode
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level) # File handler can have its own level
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {final_log_file}")
        except Exception as e:
            logger.error(f"Failed to set up file logging for {final_log_file}: {e}", exc_info=True)
            # Continue with console logging if file logging fails

    # Prevent logger from propagating messages to the root logger if it has handlers
    if logger.hasHandlers():
        logger.propagate = False
    
    _INITIALIZED_LOGGERS[name] = logger
    if not force_reconfigure and name in _INITIALIZED_LOGGERS: # Check if this was the first init
        logger.info(f"Logger '{name}' configured. Level: {level_str}. Console: {'Yes' if log_conf.get('enable_console_logging', True) else 'No'}. File: {final_log_file or 'No'}.")

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Retrieves a logger instance. If name is None, returns the root logger
    (which might not be configured by setup_logger unless name was root).
    It's generally better to get a named logger that has been set up.

    Args:
        name (Optional[str]): The name of the logger.

    Returns:
        logging.Logger: The logger instance.
    """
    if name and name in _INITIALIZED_LOGGERS:
        return _INITIALIZED_LOGGERS[name]
    return logging.getLogger(name) # Returns logger, may or may not be configured by setup_logger.


if __name__ == '__main__':
    print("--- Testing Logging Setup ---")

    # 1. Basic setup (console only, default level INFO)
    print("\n1. Testing basic console logger (INFO level)...")
    logger1 = setup_logger("TestApp1", force_reconfigure=True)
    logger1.debug("This is a DEBUG message (TestApp1) - should not appear.")
    logger1.info("This is an INFO message (TestApp1).")
    logger1.warning("This is a WARNING message (TestApp1).")

    # 2. Setup with file logging and DEBUG level from direct args
    print("\n2. Testing file logger (DEBUG level) and custom format...")
    log_file_path = "./tmp_test_logs/app_debug.log"
    custom_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    custom_date_format = "%H:%M:%S"
    
    # Clean up old log file if it exists
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    if not os.path.exists("./tmp_test_logs"):
        os.makedirs("./tmp_test_logs")

    logger2 = setup_logger("TestApp2", 
                           default_level="DEBUG", 
                           log_file=log_file_path,
                           log_format_str=custom_format,
                           date_format_str=custom_date_format,
                           force_reconfigure=True)
    logger2.debug("This is a DEBUG message (TestApp2) - should appear in console and file.")
    logger2.info("This is an INFO message (TestApp2).")

    # Verify file content
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as f:
            log_content = f.read()
        print(f"  Content of '{log_file_path}':\n{log_content.strip()}")
        assert "DEBUG message (TestApp2)" in log_content
        assert "INFO message (TestApp2)" in log_content
    else:
        print(f"  ERROR: Log file '{log_file_path}' was not created.")

    # 3. Setup using a configuration dictionary
    print("\n3. Testing logger setup via config dictionary...")
    log_file_path_config = "./tmp_test_logs/app_config.log"
    if os.path.exists(log_file_path_config):
        os.remove(log_file_path_config)

    mock_config = {
        "logging": {
            "level": "WARNING",
            "log_file_path": log_file_path_config,
            "log_format": "%(levelname)-8s | %(name)s | %(module)s - %(message)s",
            "date_format": "%Y/%m/%d-%H:%M", # Different date format
            "enable_console_logging": True
        }
    }
    logger3 = setup_logger("TestApp3", config=mock_config, force_reconfigure=True)
    logger3.info("This INFO message (TestApp3) should NOT appear (level is WARNING).")
    logger3.warning("This is a WARNING message (TestApp3) - should appear.")
    logger3.error("This is an ERROR message (TestApp3) - should appear.")

    if os.path.exists(log_file_path_config):
        with open(log_file_path_config, 'r') as f:
            log_content_config = f.read()
        print(f"  Content of '{log_file_path_config}':\n{log_content_config.strip()}")
        assert "INFO message (TestApp3)" not in log_content_config
        assert "WARNING message (TestApp3)" in log_content_config
    else:
        print(f"  ERROR: Log file '{log_file_path_config}' was not created.")

    # 4. Test getting an already initialized logger
    print("\n4. Testing retrieval of initialized logger...")
    logger1_retrieved = get_logger("TestApp1")
    assert logger1 is logger1_retrieved # Should be the same instance
    logger1_retrieved.info("Another INFO message from retrieved TestApp1 logger.")

    # Clean up
    import shutil
    if os.path.exists("./tmp_test_logs"):
        shutil.rmtree("./tmp_test_logs")
    print("\nutils/logging_setup.py tests completed and cleanup done.")