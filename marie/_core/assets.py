from typing import NamedTuple, Optional

import marie.check as check
from marie.serdes import whitelist_for_serdes
from marie.serdes.serdes import deserialize_value


@whitelist_for_serdes
class AssetDetails(
    NamedTuple("_AssetDetails", [("last_wipe_timestamp", Optional[float])])
):
    """Set of asset fields that do not change with every materialization.  These are generally updated
    on some non-materialization action (e.g. wipe).
    """

    def __new__(cls, last_wipe_timestamp: Optional[float] = None):
        check.opt_float_param(last_wipe_timestamp, "last_wipe_timestamp")
        return super(AssetDetails, cls).__new__(cls, last_wipe_timestamp)

    @staticmethod
    def from_db_string(db_string):
        if not db_string:
            return None

        try:
            details = deserialize_value(db_string, AssetDetails)
        except Exception:
            return None

        return details
