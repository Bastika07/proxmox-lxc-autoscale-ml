import logging
import math
import time

import requests


def _swap_usage_percent(latest_metrics):
    """Swap in use as a percentage of swap available, or None without swap.

    swap_total_mb == 0 is what the monitor reports for a container without
    swap and for enable_swap: false alike; both mean "no swap signal", not
    "0% used". Missing or NaN values (older metrics files) mean the same.
    """
    try:
        total = float(latest_metrics.get("swap_total_mb", 0) or 0)
        used = float(latest_metrics.get("swap_usage_mb", 0) or 0)
    except (TypeError, ValueError):
        return None
    if math.isnan(total) or math.isnan(used) or total <= 0:
        return None
    return max(0.0, min(100.0, used / total * 100))


def _swap_usage_mb(latest_metrics):
    """Swap in use in MB; 0 when unknown, missing or NaN."""
    try:
        used = float(latest_metrics.get("swap_usage_mb", 0) or 0)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(used) else max(0.0, used)


def _optional_mb(value):
    """An allocation in MB as int, or None when missing or NaN."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else int(number)


def _optional_percent(value):
    """A percentage as float, or None when missing or NaN."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def determine_swap_limit(latest_metrics, new_ram, config):
    """
    Return the swap limit (MB) to set alongside RAM, or None to leave it alone.

    With scaling.manage_swap_limit on, the swap limit follows the RAM limit at
    a fixed ratio (swap_ratio), clamped to [min_swap_mb, max_swap_mb]. Swap is
    a buffer, not a resource to grow: pressure on it is answered by scaling RAM
    up (determine_scaling_action), and this only keeps the buffer proportional.

    It is left alone when:
      - manage_swap_limit is off (the default);
      - the container's current swap limit is unknown -- an API too old to
        report `swap_mb`, or no `swap` line in the container config. Setting it
        blind could not tell growing from shrinking;
      - the container has swap 0, which is a deliberate "no swap" choice;
      - the target equals the current limit.

    A shrink never goes below what is in swap now plus a 25% margin: lowering
    `memory.swap.max` under current usage does not bring pages back, it only
    stops further swap-out, so the next squeeze goes straight to the OOM killer.

    Args:
        latest_metrics: Metrics row with current_ram_mb, current_swap_mb and
            optionally swap_usage_mb.
        new_ram: The RAM limit determine_scaling_action chose, or None if RAM
            stays as it is.
        config: Configuration with a 'scaling' section.
    """
    scaling = config["scaling"]
    # Read every key unconditionally so each is honoured whichever branch runs.
    manage = bool(scaling.get("manage_swap_limit", False))
    ratio = float(scaling.get("swap_ratio", 0.5))
    min_swap = int(scaling.get("min_swap_mb", 0))
    max_swap = int(scaling.get("max_swap_mb", 4096))

    if not manage:
        return None

    current_swap = _optional_mb(latest_metrics.get("current_swap_mb"))
    current_ram = _optional_mb(latest_metrics.get("current_ram_mb"))
    if current_swap is None or current_ram is None:
        logging.debug(
            "Swap limit not managed for this container: its current swap limit is unknown "
            "(the API does not report swap_mb, or the container config has no swap line).")
        return None
    if current_swap == 0:
        logging.debug("Swap limit not managed for this container: it has swap 0 (no swap).")
        return None

    target_ram = new_ram if new_ram is not None else current_ram
    target = min(max(int(round(ratio * target_ram)), min_swap), max_swap)

    if target < current_swap:
        floor = math.ceil(_swap_usage_mb(latest_metrics) * 1.25)
        target = max(target, min(floor, current_swap))

    if target == current_swap:
        return None
    logging.debug(f"Swap limit {current_swap} MB -> {target} MB (RAM {target_ram} MB x {ratio}).")
    return target


def determine_scaling_action(latest_metrics, scaling_decision, confidence, config):
    """
    Determine CPU and RAM scaling actions based on usage metrics and thresholds.

    This function evaluates container resource usage against configured thresholds
    and determines whether to scale up, scale down, or maintain current resources.
    Scaling is performed incrementally using configured step sizes.

    Args:
        latest_metrics (pd.Series): Container metrics including:
            - cpu_usage_percent: Current CPU usage as percentage
            - memory_usage_mb: Current RAM usage in MB
            - current_cores: Current CPU core allocation
            - current_ram_mb: Current RAM allocation in MB
            - previous_cpu_usage_percent (optional): the container's sample
              before this one. CPU only scales up when both are above the
              threshold; without it a single sample decides, as before.
            - swap_usage_mb / swap_total_mb (optional): swap in use and swap
              available inside the container. A total of 0 means the
              container has no swap (or the monitor runs with
              enable_swap: false) and swap is ignored.
        scaling_decision (int): IsolationForest prediction (-1=anomaly, 1=normal)
        confidence (float): Prediction confidence percentage (0-100)
        config (dict): Configuration containing 'scaling' section with thresholds

    Returns:
        tuple: (cpu_action, ram_action, new_cores, new_ram)
            - cpu_action (str): "Scale Up", "Scale Down", or "No Scaling"
            - ram_action (str): "Scale Up", "Scale Down", or "No Scaling"
            - new_cores (int): Target CPU cores after scaling (or None)
            - new_ram (int): Target RAM in MB after scaling (or None)
    """
    cpu_action = "No Scaling"
    ram_action = "No Scaling"
    new_cores = None
    new_ram = None

    cpu_usage = latest_metrics["cpu_usage_percent"]
    memory_usage_mb = latest_metrics["memory_usage_mb"]
    current_ram_mb = latest_metrics.get("current_ram_mb", config["scaling"].get("min_ram_mb", 1024))

    # Calculate RAM usage as percentage for threshold comparison
    memory_usage_percent = (memory_usage_mb / current_ram_mb) * 100 if current_ram_mb > 0 else 0

    thresholds = config["scaling"]

    # Swap is the signal RAM usage alone misses: the kernel pushes pages out
    # well before MemAvailable reaches zero, so a container can sit at 60% RAM
    # while swapping hard. Read both thresholds unconditionally so they are
    # honoured (and visible as read) whether or not this container has swap.
    swap_scale_up_threshold = thresholds.get("swap_scale_up_threshold", 20)
    swap_block_threshold = thresholds.get("swap_scale_down_block_threshold", 5)
    swap_usage_percent = _swap_usage_percent(latest_metrics)

    logging.debug(
        f"CPU usage: {cpu_usage}% | Memory usage: {memory_usage_mb}MB ({memory_usage_percent:.1f}%) | "
        f"Swap usage: {'n/a' if swap_usage_percent is None else f'{swap_usage_percent:.1f}%'} | "
        f"Confidence: {confidence}%")

    # IsolationForest returns -1 for anomalies (outliers), 1 for normal points
    # Anomalies (-1) indicate unusual resource patterns that may need scaling
    if scaling_decision == -1:
        logging.debug("Anomaly detected. Evaluating for potential scaling based on thresholds.")

    # Check threshold-based scaling
    previous_cpu = _optional_percent(latest_metrics.get("previous_cpu_usage_percent"))
    if cpu_usage > thresholds["cpu_scale_up_threshold"]:
        if previous_cpu is not None and previous_cpu <= thresholds["cpu_scale_up_threshold"]:
            # A single sample above the threshold is a burst, not load. On the
            # host a job running for about a minute every half hour (container
            # 106) added a core on its one busy sample, and the next quiet
            # sample took it away again: 2 -> 3 -> 2 cores within one cycle,
            # several times a day. Two samples in a row are required instead.
            logging.debug(
                f"CPU usage {cpu_usage}% exceeds the scale-up threshold, but the sample before "
                f"was {previous_cpu:.1f}%: a single busy sample is a burst. Not scaling up.")
        else:
            cpu_action = "Scale Up"
            logging.debug(f"CPU usage {cpu_usage}% exceeds the scale-up threshold.")
    elif cpu_usage < thresholds["cpu_scale_down_threshold"]:
        cpu_action = "Scale Down"
        logging.debug(f"CPU usage {cpu_usage}% is below the scale-down threshold.")

    if memory_usage_percent > thresholds["ram_scale_up_threshold"]:
        ram_action = "Scale Up"
        logging.debug(f"Memory usage {memory_usage_percent:.1f}% exceeds the scale-up threshold.")
    elif swap_usage_percent is not None and swap_usage_percent > swap_scale_up_threshold:
        # Swap alone is not pressure. The kernel does not page swapped-out
        # memory back in just because RAM was added, so swap usage stays
        # where it is after a scale-up -- and scaling on the swap percentage
        # alone added a step every cycle until max_ram_mb (observed: +512 MB
        # every 3 minutes up to 16 GB on a container whose RAM was already
        # sufficient).
        #
        # Swap only counts while RAM could not hold what is in swap: RAM in
        # use plus swap in use must reach ram_scale_up_threshold of the
        # current limit. That is the same test the scale-down guard below
        # applies to the smaller limit, so the two cannot oscillate, and each
        # swap-driven step lowers the ratio, which ends the series once
        # everything fits -- one or two steps, not a run to the ceiling.
        held_mb = memory_usage_mb + _swap_usage_mb(latest_metrics)
        held_percent = held_mb / current_ram_mb * 100 if current_ram_mb > 0 else 0
        if held_percent >= thresholds["ram_scale_up_threshold"]:
            ram_action = "Scale Up"
            logging.debug(
                f"Swap usage {swap_usage_percent:.1f}% exceeds swap_scale_up_threshold "
                f"({swap_scale_up_threshold}%) and RAM plus swap in use ({held_mb:.0f} MB, "
                f"{held_percent:.1f}%) reaches ram_scale_up_threshold; scaling RAM up.")
        else:
            logging.debug(
                f"Swap usage {swap_usage_percent:.1f}% exceeds swap_scale_up_threshold "
                f"({swap_scale_up_threshold}%), but RAM plus swap in use ({held_mb:.0f} MB) "
                f"is only {held_percent:.1f}% of the RAM limit: the swapped pages would fit, "
                f"so more RAM does not help. Not scaling RAM up.")
    elif memory_usage_percent < thresholds["ram_scale_down_threshold"]:
        if swap_usage_percent is not None and swap_usage_percent > swap_block_threshold:
            # Low RAM usage while pages sit in swap is not idle memory: taking
            # RAM away would push more of the working set out and thrash.
            logging.debug(
                f"Memory usage {memory_usage_percent:.1f}% is below the scale-down threshold, "
                f"but swap usage {swap_usage_percent:.1f}% exceeds "
                f"swap_scale_down_block_threshold ({swap_block_threshold}%); not scaling RAM down.")
        else:
            ram_action = "Scale Down"
            logging.debug(f"Memory usage {memory_usage_percent:.1f}% is below the scale-down threshold.")

    # Get current resources from the container
    current_cores = latest_metrics.get("current_cores", thresholds.get("min_cpu_cores", 2))
    current_ram = latest_metrics.get("current_ram_mb", thresholds.get("min_ram_mb", 1024))

    # Calculate incremental scaling steps
    cpu_step = thresholds.get("cpu_scale_step", 1)  # Scale by 1 core at a time by default
    ram_step = thresholds.get("ram_scale_step_mb", 512)  # Scale by 512MB at a time by default

    # Ensure scaling stays within limits with incremental changes
    if cpu_action == "Scale Up":
        new_cores = min(current_cores + cpu_step, thresholds["max_cpu_cores"])
    elif cpu_action == "Scale Down":
        new_cores = max(current_cores - cpu_step, thresholds["min_cpu_cores"])

    if ram_action == "Scale Up":
        new_ram = min(current_ram + ram_step, thresholds["max_ram_mb"])
    elif ram_action == "Scale Down":
        new_ram = max(current_ram - ram_step, thresholds["min_ram_mb"])
        # What the container holds is RAM in use PLUS what already sits in
        # swap. A scale-down is only safe if that total fits the smaller limit
        # without immediately crossing the scale-up threshold again: otherwise
        # the kernel has to push the difference into swap (whose limit may be
        # shrinking at the same time when manage_swap_limit is on) and, if
        # swap is full, the OOM killer runs. Below the threshold also keeps the
        # next cycle from scaling straight back up.
        held_mb = memory_usage_mb + _swap_usage_mb(latest_metrics)
        if new_ram > 0 and held_mb / new_ram * 100 >= thresholds["ram_scale_up_threshold"]:
            logging.debug(
                f"Not scaling RAM down to {new_ram} MB: {held_mb:.0f} MB in RAM and swap "
                f"would be {held_mb / new_ram * 100:.1f}% of it, at or above "
                f"ram_scale_up_threshold ({thresholds['ram_scale_up_threshold']}%).")
            ram_action, new_ram = "No Scaling", None

    # The clamps above can point against the decision for a container that
    # was configured outside [min, max] by hand: a scale-down raised it to the
    # floor, a scale-up lowered it to the ceiling. A 512 MB container with
    # min_ram_mb 1024 was "scaled down" to 1024 MB every cycle. Neither is
    # what the decision means, so such a container is left as it is.
    if cpu_action == "Scale Down" and new_cores is not None and new_cores > current_cores \
            or cpu_action == "Scale Up" and new_cores is not None and new_cores < current_cores:
        logging.debug(
            f"CPU {cpu_action.lower()} would move {current_cores} cores the other way "
            f"(to {new_cores}, the configured limit); no action.")
        cpu_action, new_cores = "No Scaling", None
    if ram_action == "Scale Down" and new_ram is not None and new_ram > current_ram \
            or ram_action == "Scale Up" and new_ram is not None and new_ram < current_ram:
        logging.debug(
            f"RAM {ram_action.lower()} would move {current_ram} MB the other way "
            f"(to {new_ram} MB, the configured limit); no action.")
        ram_action, new_ram = "No Scaling", None

    # A container already at its floor or ceiling clamps to the value it
    # already has. Reporting that as an action made the loop issue a no-op
    # root `pct set` every interval and, worse, made the scaling counter and
    # the "Successfully scaled" log meaningless -- exactly the signals an
    # operator reaches for during an incident.
    if new_cores is not None and new_cores == current_cores:
        logging.debug(
            f"CPU already at the {cpu_action.split()[-1].lower()} limit "
            f"({current_cores} cores); no action.")
        cpu_action, new_cores = "No Scaling", None
    if new_ram is not None and new_ram == current_ram:
        logging.debug(
            f"RAM already at the {ram_action.split()[-1].lower()} limit "
            f"({current_ram} MB); no action.")
        ram_action, new_ram = "No Scaling", None

    logging.debug(f"Final scaling actions: CPU -> {cpu_action}, RAM -> {ram_action} | Confidence: {confidence}%")
    return cpu_action, ram_action, new_cores, new_ram




def apply_scaling(lxc_id, new_cores, new_ram, config, new_swap=None, current_ram=None):
    """
    Apply CPU and/or RAM scaling changes to a container via the API.

    Makes POST requests to the scaling API endpoints with retry logic.
    Both CPU and RAM scaling are attempted independently.

    Args:
        lxc_id (str): Container ID to scale
        new_cores (int): Target CPU cores (None to skip CPU scaling)
        new_ram (int): Target RAM in MB (None to skip RAM scaling)
        config (dict): Configuration containing API settings and retry logic
        new_swap (int): Target swap limit in MB (None to leave swap alone).
            Sent in the same /scale/ram request as RAM, which the API turns
            into one `pct set -memory -swap`.
        current_ram (int): RAM in MB to send with a swap-only change, where
            new_ram is None and the endpoint still requires `memory`.

    Returns:
        None
    """
    max_retries = config.get("retry_logic", {}).get("max_retries", 3)
    retry_delay = config.get("retry_logic", {}).get("retry_delay", 2)
    base_url = config["api"]["api_url"]
    cores_endpoint = config["api"].get("cores_endpoint", "/scale/cores")
    ram_endpoint = config["api"].get("ram_endpoint", "/scale/ram")

    # `timeout` is accepted as well for configs written against the older
    # documentation.
    request_timeout = config["api"].get("timeout_seconds", config["api"].get("timeout", 10))
    api_key = config["api"].get("api_key")
    headers = {"X-API-Key": api_key} if api_key else None

    def perform_request(url, data, resource_type):
        resource_key = "cores" if resource_type == "CPU" else "memory"
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=data, timeout=request_timeout,
                                         headers=headers)
                response.raise_for_status()
                swap_note = f" (swap {data['swap']} MB)" if "swap" in data else ""
                logging.info(f"Successfully scaled {resource_type} for LXC ID {lxc_id} to {data[resource_key]} {resource_type} units{swap_note}.")
                return True
            except requests.RequestException as e:
                # `e.response` is None for connection/timeout errors, where no
                # response was ever received. Reading it off the local variable
                # instead used to raise UnboundLocalError and mask the failure.
                status_code = e.response.status_code if e.response is not None else None
                if status_code in (401, 403):
                    # Retrying cannot help. Without this the operator saw only
                    # a generic failure while autoscaling stopped, and the
                    # usual response was to turn authentication back off.
                    logging.error(
                        f"The API rejected our credentials (HTTP {status_code}) when scaling "
                        f"{resource_type} for LXC ID {lxc_id}. It has authentication.enabled: "
                        f"true; set `api.api_key` in the model configuration to one of its "
                        f"`authentication.api_keys`."
                    )
                    return False
                if status_code == 500:
                    logging.error(f"Server error (500) encountered on attempt {attempt + 1} to scale {resource_type} for LXC ID {lxc_id}. Aborting further attempts.")
                    break  # Skip further retries for 500 errors
                logging.error(f"Attempt {attempt + 1} failed to scale {resource_type} for LXC ID {lxc_id}: {e}")
                if attempt < max_retries - 1:
                    logging.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logging.error(f"Scaling {resource_type} for LXC ID {lxc_id} failed after {max_retries} attempts.")
                    return False
        return False

    # Both spellings are sent so a newer model keeps working against an API
    # that has not been upgraded yet.
    def payload(**fields):
        return dict(lxc_id=lxc_id, vm_id=lxc_id, **fields)

    if new_cores is not None:
        cpu_url = f"{base_url}{cores_endpoint}"
        if not perform_request(cpu_url, payload(cores=new_cores), "CPU"):
            logging.error(f"Scaling operation aborted for LXC ID {lxc_id} due to CPU scaling failure.")

    memory = new_ram if new_ram is not None else (current_ram if new_swap is not None else None)
    if new_swap is not None and memory is None:
        logging.error(
            f"Not changing swap for LXC ID {lxc_id}: its RAM is unknown and /scale/ram requires it.")
        new_swap = None
    if memory is not None:
        ram_url = f"{base_url}{ram_endpoint}"
        fields = {"memory": memory}
        if new_swap is not None:
            fields["swap"] = new_swap
        if not perform_request(ram_url, payload(**fields), "RAM"):
            logging.error(f"Scaling operation aborted for LXC ID {lxc_id} due to RAM scaling failure.")
