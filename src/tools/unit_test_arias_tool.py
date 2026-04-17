import warnings
from typing import Any, Dict, Optional

import oracledb as cx
import pandas as pd
from pydantic import BaseModel, Field

from .base_tool import BaseTool
from ..config import ARIES_DB_USER, ARIES_DB_PASSWORD, ARIES_DB_DSN

warnings.filterwarnings("ignore")


class UnitTestAriasInput(BaseModel):
    user: str = Field(default=ARIES_DB_USER, description="Oracle username")
    password: str = Field(default=ARIES_DB_PASSWORD, description="Oracle password")
    dsn: str = Field(default=ARIES_DB_DSN, description="Oracle DSN, e.g. vn.aries")

    lot: Optional[str] = Field(default=None, description="Filter by v0.lot")
    operation: Optional[str] = Field(default=None, description="Filter by v0.operation, e.g. 6% or 7%")
    tester_id: Optional[str] = Field(default=None, description="Filter by v0.tester_id, e.g. %HXV%")
    visual_id: Optional[str] = Field(default=None, description="Filter by di.visual_id")
    interface_bin: Optional[int] = Field(default=None, description="Filter by dt.interface_bin")
    
    row_limit: Optional[int] = Field(default=5000, ge=1, le=50000, description="Max rows to return")

    test_start_date_time: Optional[str] = Field(default=None, description="Test start time in ISO format")
    device_start_date_time: Optional[str] = Field(default=None, description="Device start time in ISO format")
    device_end_date_time: Optional[str] = Field(default=None, description="Device end time in ISO format")


class UnitTestAriasTool(BaseTool):
    """
    Query unit test data from Oracle with optional filters on lot, operation, tester_id.
    """

    INPUT_MODEL = UnitTestAriasInput

    def _run(self, validated_input: BaseModel) -> Dict[str, Any]:
        data: UnitTestAriasInput = validated_input  # type: ignore[assignment]
        sql = self._build_sql(
            lot=data.lot,
            operation=data.operation,
            tester_id=data.tester_id,
            row_limit=data.row_limit,
            test_start_date_time=data.test_start_date_time,
            device_start_date_time=data.device_start_date_time,
            device_end_date_time=data.device_end_date_time,
            visual_id=data.visual_id,
            interface_bin=data.interface_bin
        )

        conn = None
        try:
            conn = cx.connect(user=data.user, password=data.password, dsn=data.dsn)
            df = pd.read_sql(sql, conn)
            return df.to_dict(orient="records")
        except Exception as ex:
            raise RuntimeError(f"UnitTestAriasTool failed: {repr(ex)}") from ex
        finally:
            if conn is not None:
                conn.close()

    def _build_sql(
        self,
        lot: Optional[str],
        operation: Optional[str],
        tester_id: Optional[str],
        row_limit: Optional[int],
        test_start_date_time: Optional[str],
        device_start_date_time: Optional[str],
        device_end_date_time: Optional[str],
        visual_id: Optional[str],
        interface_bin: Optional[int],
    ) -> str:
        where_clauses = []
        params: Dict[str, Any] = {"row_limit": row_limit}

        if lot:
            where_clauses.append(f"v0.lot = '{lot}'")
            params["lot"] = lot

        if operation:
            where_clauses.append(f"v0.operation LIKE '{operation}'")
            params["operation"] = operation

        if tester_id:
            where_clauses.append(f"v0.tester_id LIKE '{tester_id}'")
            params["tester_id"] = tester_id

        if visual_id:
            where_clauses.append(f"di.visual_id = '{visual_id}'")
            params["visual_id"] = visual_id

        if interface_bin is not None:
            where_clauses.append(f"dt.interface_bin ='{interface_bin}'")
            params["interface_bin"] = interface_bin

        if test_start_date_time:
            where_clauses.append(
                f"v0.test_start_date_time >= TO_DATE('{test_start_date_time}', 'YYYY-MM-DD HH24:MI:SS')"
            )
            params["test_start_date_time"] = test_start_date_time

        if device_start_date_time:
            where_clauses.append(
                f"dt.device_start_date_time >= TO_DATE('{device_start_date_time}', 'YYYY-MM-DD HH24:MI:SS')"
            )
            params["device_start_date_time"] = device_start_date_time

        if device_end_date_time:
            where_clauses.append(
                f"dt.device_end_date_time <= TO_DATE('{device_end_date_time}', 'YYYY-MM-DD HH24:MI:SS')"
            )
            params["device_end_date_time"] = device_end_date_time

        where_sql = "\n AND ".join(where_clauses) if where_clauses else "1=1"

        sql = f"""
        SELECT *
        FROM (
                SELECT /*+  use_nl (dt) use_nl (di) */
                v0.lot AS lot
                ,v0.operation AS operation
                ,To_Char(v0.test_start_date_time,'yyyy-mm-dd hh24:mi:ss') AS test_start_date
                ,To_Char(v0.test_end_date_time,'yyyy-mm-dd hh24:mi:ss') AS test_end_date
                ,v0.summary_number||v0.summary_letter AS summary_id
                ,dt.within_session_sequence_number AS within_session_sequence_number
                ,v0.tester_id AS tester_id
                ,dt.site_id AS site_id
                ,To_Char(dt.device_start_date_time,'yyyy-mm-dd hh24:mi:ss') AS device_start_date_time
                ,To_Char(dt.device_end_date_time,'yyyy-mm-dd hh24:mi:ss') AS device_end_date_time
                ,dt.tester_interface_unit_id AS tester_interface_unit_id
                ,dt.thermal_head_id AS thermal_head_id
                ,dt.device_tester_id AS device_tester_id
                ,dt.interface_bin AS interface_bin
                ,dt.functional_bin AS functional_bin
                ,di.visual_id AS visual_id
                ,dt.within_session_latest_flag AS within_session_latest_flag
                ,dt.testing_session_unit_id AS testing_session_unit_id
                ,CASE WHEN   dt.interface_bin = 1   THEN 'G' ELSE 'B' END AS GOODBAD
                ,v0.devrevstep AS devrevstep
                ,dt.test_time AS test_time
                ,v0.program_name AS program_name
                ,v0.facility AS facility
                ,v0.temperature AS temperature
                ,(A_PG$INTEL_WW.CALCULATE_WW(         dt.device_end_date_time -(7/24)) )    ||'.' ||     (TO_CHAR(              dt.device_end_date_time  -7/24 , 'D') -1)   ||   (CASE WHEN  TO_CHAR(             dt.device_end_date_time  ,' hh24') >=7 AND TO_CHAR(              dt.device_end_date_time  ,' hh24')  < 19 THEN 'D' ELSE 'N' END) AS Shift
                ,mp.prodgroup3 AS prodgroup3
                ,v0.data_source_program AS data_source_program
                ,(SELECT tsv99.string_value FROM A_Testing_Session_Attrib_Value tsv99, A_Testing_Session_Attribute tsa99 WHERE v0.lao_start_ww = tsv99.lao_start_ww AND v0.ts_id = tsv99.ts_id AND v0.program_name = tsv99.program_or_recipe_name AND tsv99.program_or_recipe_name = tsa99.program_or_recipe_name AND tsv99.attribute_id = tsa99.attribute_id AND tsa99.attribute_name = 'CURRENT_PROCESS_STEP' AND rownum <= 1) AS current_process_step
            FROM 
            A_Testing_Session v0
            LEFT JOIN A_MARS_Lot ml ON v0.lot=ml.lot -- AND ml.facility = v0.facility
            LEFT JOIN A_MARS_Product mp ON ml.product = mp.product AND ml.mars_schema=mp.mars_schema AND mp.facility = v0.facility
            INNER JOIN A_Device_Testing dt ON v0.lao_start_ww + 0 = dt.lao_start_ww AND v0.ts_id + 0 = dt.ts_id
            LEFT JOIN A_Device_Item di ON dt.di_id = di.di_id
            WHERE {where_sql}
            AND      (v0.operation LIKE  '6%'
            OR v0.operation LIKE '7%'
            ) 
            AND      (v0.tester_id LIKE  '%HXV%'
            ) 
            AND      v0.test_end_date_time >= SYSDATE - 0.5 
            AND      dt.interface_bin Not In (72) 
            ORDER BY
                    9 Asc
        )
        WHERE ROWNUM <= {row_limit}
        """
        return sql
    