from mmengine.registry import Registry

from .base_flask_service import BaseFlaskService
from .smpl_stream_service import SMPLStreamService

SERVICES = Registry('services')

SERVICES.register_module(name='BaseFlaskService', module=BaseFlaskService)
SERVICES.register_module(name='SMPLStreamService', module=SMPLStreamService)


def build_service(cfg) -> BaseFlaskService:
    """Build a flask service."""
    return SERVICES.build(cfg)
