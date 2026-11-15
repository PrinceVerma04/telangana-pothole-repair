"""Shared API contracts. Import from here: ``from server.app.schemas import ObservationIn, ...``."""

from server.app.schemas.api import *  # noqa: F401,F403
from server.app.schemas.states import (  # noqa: F401
    INVOICE_ELIGIBLE,
    POTHOLE_TRANSITIONS,
    WORK_ORDER_TRANSITIONS,
    Role,
    WorkOrderStatus,
    pothole_can,
    work_order_can,
)
