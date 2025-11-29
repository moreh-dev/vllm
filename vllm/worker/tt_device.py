from typing import Optional, Dict
import os
from loguru import logger
# from tests.scripts.common import get_updated_device_params
import ttnn


# Device helpers to be shared with top-level conftest.py and other conftest.py files that will handle open/close of devices.


def get_updated_device_params(device_params):
    import ttnn

    new_device_params = device_params.copy()

    dispatch_core_axis = new_device_params.pop("dispatch_core_axis", None)
    dispatch_core_type = new_device_params.pop("dispatch_core_type", None)
    fabric_tensix_config = new_device_params.get("fabric_tensix_config", None)

    if ttnn.device.is_blackhole():
        # Only when both fabric_config and fabric_tensix_config are set, we can use ROW dispatch, otherwise force to use COL dispatch
        fabric_config = new_device_params.get("fabric_config", None)
        if not (fabric_config and fabric_tensix_config):
            # When not both are set, force COL dispatch
            if dispatch_core_axis == ttnn.DispatchCoreAxis.ROW:
                logger.warning(
                    "ROW dispatch requires both fabric and tensix config, using DispatchCoreAxis.COL instead."
                )
                dispatch_core_axis = ttnn.DispatchCoreAxis.COL
        elif fabric_config and fabric_tensix_config:
            logger.warning(
                f"Blackhole with fabric_config and fabric_tensix_config enabled, using fabric_tensix_config={fabric_tensix_config}"
            )

    dispatch_core_config = ttnn.DispatchCoreConfig(dispatch_core_type, dispatch_core_axis, fabric_tensix_config)
    new_device_params["dispatch_core_config"] = dispatch_core_config

    return new_device_params


_printed_device_info = False

def parse_mesh_shape_from_env() -> Optional[ttnn.MeshShape]:
    """Parse mesh shape from TT_MESHDEVICE_SHAPE environment variable.

    Format: TT_MESHDEVICE_SHAPE=rows,cols (e.g., "2,4" or "1,8")

    Returns:
        MeshShape if environment variable is set, None otherwise
    """
    env_shape = os.getenv("TT_MESHDEVICE_SHAPE")
    if not env_shape:
        return None

    try:
        rows, cols = map(int, env_shape.split(","))
        return ttnn.MeshShape(rows, cols)
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Invalid TT_MESHDEVICE_SHAPE format '{env_shape}': expected 'rows,cols' (e.g., '2,4')")


def create_mesh_device(device_params: Optional[Dict] = None):
    """Create mesh device with appropriate mesh shape based on available devices.

    Mesh shape selection:
    - TT_MESHDEVICE_SHAPE env var (e.g., "2,4" or "1,8") if set
    - Galaxy (32 devices): 4x8
    - T3000 (8+ devices): 2x4
    - Single/Few devices: 1x{num_devices}

    Args:
        device_params: Optional device parameters (trace_region_size, fabric_config, etc.)

    Returns:
        Initialized mesh device
    """
    global _printed_device_info

    params = dict(device_params or {})

    fabric_config = params.pop("fabric_config", None)
    if fabric_config:
        ttnn.set_fabric_config(fabric_config)

    updated_device_params = get_updated_device_params(params)
    device_ids = ttnn.get_device_ids()

    env_mesh_shape = parse_mesh_shape_from_env()
    if env_mesh_shape:
        default_mesh_shape = env_mesh_shape
    elif len(device_ids) == 32:  # Galaxy system
        default_mesh_shape = ttnn.MeshShape(4, 8)
    elif len(device_ids) >= 8:  # T3000 or similar
        default_mesh_shape = ttnn.MeshShape(2, 4)
    else:  # Single device or smaller configurations
        default_mesh_shape = ttnn.MeshShape(1, len(device_ids))

    updated_device_params.setdefault("mesh_shape", default_mesh_shape)
    mesh_device = ttnn.open_mesh_device(**updated_device_params)

    if not _printed_device_info:
        _printed_device_info = True
        mesh_shape = mesh_device.shape
        num_devices = mesh_device.get_num_devices()

        print(f"=" * 60)
        print(f"Mesh Device Configuration:")
        print(f"  Available Devices: {len(device_ids)}")
        print(f"  Mesh Shape: {mesh_shape[0]}x{mesh_shape[1]} ({num_devices} devices)")
        print(f"  Fabric Config: {fabric_config if fabric_config else 'DISABLED'}")
        if "trace_region_size" in params:
            trace_mb = params["trace_region_size"] // (1024 * 1024)
            print(f"  Trace Region Size: {trace_mb}MB")
        print(f"=" * 60)

    logger.debug(f"multidevice with {mesh_device.get_num_devices()} devices is created with shape {mesh_device.shape}")

    return mesh_device
