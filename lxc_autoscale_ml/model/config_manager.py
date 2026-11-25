import yaml
import logging
import os


class ConfigError(Exception):
    pass


def load_config(config_path, default_config=None):
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found: {config_path}")
        raise ConfigError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path) as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise ConfigError(f"Error parsing YAML file: {e}") from e
    except OSError as e:
        logging.error(f"Unable to read configuration file {config_path}: {e}")
        raise ConfigError(f"Unable to read configuration file {config_path}: {e}") from e

    # Checked outside the try block: raising here would otherwise be swallowed
    # by the broad handler above and reported as an "unexpected" error.
    if config is None:
        logging.error(f"Configuration file is empty: {config_path}")
        raise ConfigError(f"Configuration file is empty: {config_path}")

    logging.info(f"Configuration loaded from {config_path}")

    if default_config:
        config = {**default_config, **config}
        logging.debug("Configuration merged with default values.")

    # `scaling` was absent from this list while evaluate_container and
    # determine_scaling_action both index config["scaling"] directly, so a
    # config that passed validation raised KeyError for every container, every
    # cycle -- logged as a traceback forever while the unit reported healthy.
    required_keys = ['log_file', 'interval_seconds', 'api', 'scaling']
    for key in required_keys:
        if key not in config:
            logging.error(f"Missing required configuration key: {key}")
            raise ConfigError(f"Missing required configuration key: {key}")

    if 'api_url' not in config['api']:
        logging.error("Missing required configuration key: api_url")
        raise ConfigError("Missing required configuration key: api_url")

    # Validate scaling configuration
    validate_scaling_config(config)

    logging.debug(f"Final configuration: {config}")
    return config


def validate_scaling_config(config):
    """
    Validate scaling configuration for logical consistency and required fields.

    Args:
        config (dict): Configuration dictionary to validate

    Raises:
        ConfigError: If configuration is invalid
    """
    if 'scaling' not in config:
        raise ConfigError("Missing required configuration section: scaling")

    scaling = config['scaling']

    # Validate min/max ranges
    if scaling.get('min_cpu_cores', 0) > scaling.get('max_cpu_cores', float('inf')):
        raise ConfigError("min_cpu_cores cannot be greater than max_cpu_cores")

    if scaling.get('min_ram_mb', 0) > scaling.get('max_ram_mb', float('inf')):
        raise ConfigError("min_ram_mb cannot be greater than max_ram_mb")

    # Validate thresholds are percentages (0-100) and numeric
    threshold_keys = ['cpu_scale_up_threshold', 'cpu_scale_down_threshold',
                     'ram_scale_up_threshold', 'ram_scale_down_threshold',
                     'swap_scale_up_threshold', 'swap_scale_down_block_threshold']

    for key in threshold_keys:
        if key in scaling:
            value = scaling[key]
            if not isinstance(value, (int, float)):
                raise ConfigError(f"{key} must be numeric, got {type(value).__name__}")
            if not (0 <= value <= 100):
                raise ConfigError(f"{key} must be between 0 and 100, got {value}")

    # Validate scale-down threshold is less than scale-up threshold
    if (scaling.get('cpu_scale_down_threshold', 0) >=
        scaling.get('cpu_scale_up_threshold', 100)):
        raise ConfigError("cpu_scale_down_threshold must be less than cpu_scale_up_threshold")

    if (scaling.get('ram_scale_down_threshold', 0) >=
        scaling.get('ram_scale_up_threshold', 100)):
        raise ConfigError("ram_scale_down_threshold must be less than ram_scale_up_threshold")

    # Swap above swap_scale_up_threshold already scales RAM up, so a block
    # threshold above it could never block anything and would silently do
    # nothing.
    if (scaling.get('swap_scale_down_block_threshold', 5) >
        scaling.get('swap_scale_up_threshold', 20)):
        raise ConfigError(
            "swap_scale_down_block_threshold must not be greater than swap_scale_up_threshold")

    # Swap limit management (manage_swap_limit).
    swap_ratio = scaling.get('swap_ratio', 0.5)
    if isinstance(swap_ratio, bool) or not isinstance(swap_ratio, (int, float)) or swap_ratio < 0:
        raise ConfigError(f"swap_ratio must be a number >= 0, got {swap_ratio!r}")
    for key in ('min_swap_mb', 'max_swap_mb'):
        value = scaling.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigError(f"{key} must be an integer >= 0, got {value!r}")
    if scaling.get('min_swap_mb', 0) > scaling.get('max_swap_mb', 4096):
        raise ConfigError("min_swap_mb cannot be greater than max_swap_mb")

    # Validate step sizes are positive
    if scaling.get('cpu_scale_step', 1) <= 0:
        raise ConfigError("cpu_scale_step must be positive")

    if scaling.get('ram_scale_step_mb', 1) <= 0:
        raise ConfigError("ram_scale_step_mb must be positive")

    min_confidence = scaling.get('min_confidence', 0)
    if not isinstance(min_confidence, (int, float)) or not (0 <= min_confidence <= 100):
        raise ConfigError(
            f"min_confidence must be a number between 0 and 100, got {min_confidence!r}")

    # M1: the breaker is consulted once per container per cycle. If its window
    # closes before the next cycle begins it has always expired by the time it
    # is asked, so it blocks nothing -- with the shipped 300s window and 600s
    # interval it blocked zero calls in ten consecutive failing cycles.
    interval = config.get('interval_seconds', 60)
    breaker = config.get('circuit_breaker') or {}
    if breaker.get('enabled', True):
        timeout = breaker.get('timeout_seconds', 300)
        if timeout <= interval:
            logging.warning(
                f"circuit_breaker.timeout_seconds ({timeout}s) is not longer than "
                f"interval_seconds ({interval}s). The breaker is consulted once per "
                f"container per cycle, so its window will always have expired by then "
                f"and it will never block a call. Set it above the interval."
            )

    logging.info("Scaling configuration validated successfully")
