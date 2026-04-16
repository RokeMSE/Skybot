from __future__ import annotations
import logging
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from elasticsearch import Elasticsearch
from src.tools.base_tool import BaseTool, ToolConfig

log = logging.getLogger(__name__)
from src.config import (
    KEYID_VN_LAMAS,
    KEYVAL_VN_LAMAS,
    KEYID_CD_LAMAS,
    KEYVAL_CD_LAMAS,
    KEYID_KM_LAMAS,
    KEYVAL_KM_LAMAS,
    KEYID_PG_LAMAS,
    KEYVAL_PG_LAMAS,
)


class ElasticAlarmInput(BaseModel):
    is_sc_log: bool = Field(default=False, description="Use app.log-* if True, otherwise logs-skyline.log-*")
    site_name: int = Field(default=0, description="0=vn, 1=cd, 2=km, 3=pg")
    tool_name: Optional[str] = Field(default=None, description="Tool name, e.g. 'DC53'. Omit to query all testers.")
    gte: str = Field(..., description="Start time in ISO format")
    lte: str = Field(..., description="End time in ISO format")
    zone_id: str = Field(default="+07:00", description="Timezone offset")
    sample_count: int = Field(default=10, ge=1, le=1000, description="Max records to return")
    request_timeout: int = Field(default=120, ge=1, description="Elasticsearch request timeout seconds")


class ElasticAlarmTool(BaseTool):
    """
    Query alarm/log data from Elasticsearch by tool name, time range, and site.
    """

    INPUT_MODEL = ElasticAlarmInput

    SITE_CONFIG = {
        0: ("https://vn.lamas.intel.com/rest/", KEYID_VN_LAMAS, KEYVAL_VN_LAMAS),
        1: ("https://cd.lamas.intel.com/rest/", KEYID_CD_LAMAS, KEYVAL_CD_LAMAS),
        2: ("https://km.lamas.intel.com/rest/", KEYID_KM_LAMAS, KEYVAL_KM_LAMAS),
        3: ("https://pg.lamas.intel.com/rest/", KEYID_PG_LAMAS, KEYVAL_PG_LAMAS),
    }

    def _run(self, validated_input: ElasticAlarmInput) -> Dict[str, Any]:  # type: ignore[override]
        data = validated_input

        index = "app.log-*" if data.is_sc_log else "logs-skyline.log-*"
        client = self._create_client(data.site_name, data.request_timeout)

        must_clauses = [
            {
                "range": {
                    "@timestamp": {
                        "gte": data.gte,
                        "lte": data.lte,
                        "time_zone": data.zone_id,
                    }
                }
            },
            {
                "match": {
                    "data_stream.dataset": "skyline.log"
                }
            },
        ]

        # When a specific tester is given, filter by its hostnamee <-> otherwise query across all HXV testers.
        if data.tool_name:
            must_clauses.append({
                "wildcard": {
                    "host.hostname": f"CSS01HXV01{data.tool_name}"
                }
            })

        query_body = {
            "size": data.sample_count,
            "_source": {
                "includes": [
                    "agent.name",
                    "@timestamp",
                    "message",
                    "host.hostname",
                    "data_stream.dataset",
                ]
            },
            "query": {
                "bool": {
                    "must": must_clauses
                }
            },
            "sort": [
                {
                    "@timestamp": {
                        "order": "desc"
                    }
                }
            ]
        }

        log.info(
            "ES query → index=%s, tool_name=%s, site=%d, range=[%s .. %s], size=%d",
            index, data.tool_name or "*", data.site_name, data.gte, data.lte, data.sample_count,
        )
        log.debug("ES query body: %s", query_body)

        try:
            response = client.search(index=index, body=query_body)
            hits = response.get("hits", {}).get("hits", []) or []
            total = response.get("hits", {}).get("total", {})
            total_val = total.get("value", len(hits)) if isinstance(total, dict) else total

            log.info("ES response — %d hits returned (total matched: %s)", len(hits), total_val)
            for i, hit in enumerate(hits):
                src = hit.get("_source", {})
                ts = src.get("@timestamp", "?")
                hostname = (
                    src.get("host", {}).get("hostname")
                    if isinstance(src.get("host"), dict)
                    else src.get("host.hostname", "?")
                )
                msg = (src.get("message") or "")[:150]
                log.debug("  hit[%d] %s | %s | %s", i, ts, hostname, msg)

            return dict(response)

        except Exception as ex:
            log.error("ES query failed: %s", repr(ex))
            raise RuntimeError(f"ElasticAlarmTool failed: {repr(ex)}") from ex

    def _create_client(self, site_name: int, request_timeout: int) -> Elasticsearch:
        if site_name not in self.SITE_CONFIG:
            raise ValueError(f"Unsupported site_name: {site_name}")

        uri_string, keyid, keyvalue = self.SITE_CONFIG[site_name]

        # LAMAS servers only accept Elasticsearch v7/v8 headers.
        # The elasticsearch-py v9 client sends v9 headers by default, causing
        # "media_type_header_exception".  Setting headers explicitly to v8
        # mime type fixes the compatibility issue.
        compat_headers = {
            "Accept": "application/vnd.elasticsearch+json;compatible-with=8",
            "Content-Type": "application/vnd.elasticsearch+json;compatible-with=8",
        }

        if keyid and keyvalue:
            return Elasticsearch(
                uri_string,
                api_key=(keyid, keyvalue),
                request_timeout=request_timeout,
                headers=compat_headers,
            )
        else:
            return Elasticsearch(
                uri_string,
                request_timeout=request_timeout,
                headers=compat_headers,
            )