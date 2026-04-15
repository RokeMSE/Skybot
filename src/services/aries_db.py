"""
Aries DB service — queries Oracle for unit-level test data.

Uses `oracledb` (the modern successor to cx_Oracle) in thin mode,
so no Oracle Client installation is required.

All connection parameters are read from environment / config.py.
"""
import logging
from typing import Optional

import pandas as pd

from ..config import (
    ARIES_DB_USER,
    ARIES_DB_PASSWORD,
    ARIES_DB_DSN,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL template — parameterised via bind variables (:days_back, etc.)
# ---------------------------------------------------------------------------
_UNIT_LEVEL_SQL = """\
SELECT /*+  use_nl (dt) use_nl (di) */
      v0.lot                                              AS lot
     ,v0.operation                                        AS operation
     ,To_Char(v0.test_start_date_time,
              'yyyy-mm-dd hh24:mi:ss')                    AS test_start_date
     ,To_Char(v0.test_end_date_time,
              'yyyy-mm-dd hh24:mi:ss')                    AS test_end_date
     ,v0.summary_number || v0.summary_letter              AS summary_id
     ,dt.within_session_sequence_number                   AS within_session_sequence_number
     ,v0.tester_id                                        AS tester_id
     ,dt.site_id                                          AS site_id
     ,To_Char(dt.device_start_date_time,
              'yyyy-mm-dd hh24:mi:ss')                    AS device_start_date_time
     ,To_Char(dt.device_end_date_time,
              'yyyy-mm-dd hh24:mi:ss')                    AS device_end_date_time
     ,dt.tester_interface_unit_id                         AS tester_interface_unit_id
     ,dt.thermal_head_id                                  AS thermal_head_id
     ,dt.device_tester_id                                 AS device_tester_id
     ,dt.interface_bin                                    AS interface_bin
     ,dt.functional_bin                                   AS functional_bin
     ,di.visual_id                                        AS visual_id
     ,dt.within_session_latest_flag                       AS within_session_latest_flag
     ,dt.testing_session_unit_id                          AS testing_session_unit_id
     ,CASE WHEN dt.interface_bin = 1 THEN 'G' ELSE 'B' END AS goodbad
     ,v0.devrevstep                                       AS devrevstep
     ,dt.test_time                                        AS test_time
     ,v0.program_name                                     AS program_name
     ,v0.facility                                         AS facility
     ,v0.temperature                                      AS temperature
     ,(A_PG$INTEL_WW.CALCULATE_WW(
          dt.device_end_date_time - (7/24))
      ) || '.' ||
      (TO_CHAR(dt.device_end_date_time - 7/24, 'D') - 1) ||
      (CASE WHEN TO_CHAR(dt.device_end_date_time, ' hh24') >= 7
                 AND TO_CHAR(dt.device_end_date_time, ' hh24') < 19
            THEN 'D' ELSE 'N' END)                        AS shift
     ,mp.prodgroup3                                       AS prodgroup3
     ,v0.data_source_program                              AS data_source_program
     ,(SELECT tsv99.string_value
       FROM   A_Testing_Session_Attrib_Value tsv99,
              A_Testing_Session_Attribute    tsa99
       WHERE  v0.lao_start_ww            = tsv99.lao_start_ww
         AND  v0.ts_id                   = tsv99.ts_id
         AND  v0.program_name            = tsv99.program_or_recipe_name
         AND  tsv99.program_or_recipe_name = tsa99.program_or_recipe_name
         AND  tsv99.attribute_id         = tsa99.attribute_id
         AND  tsa99.attribute_name       = 'CURRENT_PROCESS_STEP'
         AND  rownum <= 1)                                AS current_process_step
FROM
    A_Testing_Session v0
    LEFT JOIN  A_MARS_Lot     ml ON v0.lot = ml.lot
    LEFT JOIN  A_MARS_Product mp ON ml.product     = mp.product
                                 AND ml.mars_schema = mp.mars_schema
                                 AND mp.facility    = v0.facility
    INNER JOIN A_Device_Testing dt ON v0.lao_start_ww + 0 = dt.lao_start_ww
                                   AND v0.ts_id + 0       = dt.ts_id
    LEFT JOIN  A_Device_Item   di ON dt.di_id = di.di_id
WHERE 1=1
  AND (v0.operation LIKE '6%' OR v0.operation LIKE '7%')
  AND v0.tester_id LIKE :tester_filter
  AND v0.test_end_date_time >= SYSDATE - :days_back
  AND dt.interface_bin NOT IN (72)
ORDER BY device_start_date_time ASC
"""


class AriesDBService:
    """Thin wrapper around the Aries Oracle connection."""

    def __init__(self):
        self._user = ARIES_DB_USER
        self._password = ARIES_DB_PASSWORD
        self._dsn = ARIES_DB_DSN

    def _connect(self):
        import oracledb
        return oracledb.connect(
            user=self._user,
            password=self._password,
            dsn=self._dsn,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query_unit_level_data(
        self,
        days_back: float = 0.5,
        tester_filter: str = "%HXV%",
    ) -> pd.DataFrame:
        """Run the unit-level query and return a DataFrame.

        Args:
            days_back:      How far back from SYSDATE (in days). 0.5 = 12 hours.
            tester_filter:  SQL LIKE pattern for tester_id.
        """
        log.info(
            "Querying Aries — days_back=%.2f, tester_filter=%r",
            days_back, tester_filter,
        )
        conn = self._connect()
        try:
            df = pd.read_sql(
                _UNIT_LEVEL_SQL,
                conn,
                params={"days_back": days_back, "tester_filter": tester_filter},
            )
        finally:
            conn.close()

        log.info("Aries query returned %d rows", len(df))
        return df

    def summarise(
        self,
        df: pd.DataFrame,
        max_rows: int = 200,
    ) -> str:
        """Convert a DataFrame to a text summary suitable for an LLM prompt.

        Includes aggregate stats (yield, bin distribution, tester counts)
        and a truncated sample of individual rows.
        """
        if df.empty:
            return "No unit-level data returned from Aries for the given filters."

        total = len(df)
        good = (df["GOODBAD"] == "G").sum() if "GOODBAD" in df.columns else "N/A"
        bad = (df["GOODBAD"] == "B").sum() if "GOODBAD" in df.columns else "N/A"
        yield_pct = f"{good / total * 100:.2f}%" if isinstance(good, (int, float)) else "N/A"

        parts = [
            f"=== Aries Unit-Level Data Summary ===",
            f"Total units: {total}",
            f"Good: {good}  |  Bad: {bad}  |  Yield: {yield_pct}",
        ]

        if "TESTER_ID" in df.columns:
            parts.append(f"Testers: {', '.join(df['TESTER_ID'].unique()[:20])}")
        if "OPERATION" in df.columns:
            parts.append(f"Operations: {', '.join(df['OPERATION'].unique()[:20])}")
        if "INTERFACE_BIN" in df.columns:
            bin_counts = df["INTERFACE_BIN"].value_counts().head(10)
            parts.append(f"Top interface bins:\n{bin_counts.to_string()}")
        if "LOT" in df.columns:
            parts.append(f"Lots ({df['LOT'].nunique()} unique): {', '.join(df['LOT'].unique()[:15])}")
        if "SHIFT" in df.columns:
            shift_counts = df["SHIFT"].value_counts()
            parts.append(f"Shift distribution:\n{shift_counts.to_string()}")

        # Truncated sample
        sample = df.head(max_rows).to_string(index=False, max_colwidth=30)
        parts.append(f"\n--- Sample rows (first {min(max_rows, total)}) ---\n{sample}")

        return "\n".join(parts)
