from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from elasticsearch import Elasticsearch
from src.tools.base_tool import BaseTool, ToolConfig
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
    tool_name: str = Field(..., description="Tool name, e.g. 'DC53'")
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

    def _run(self, validated_input: BaseModel) -> Dict[str, Any]:
        data = validated_input

        index = "app.log-*" if data.is_sc_log else "logs-skyline.log-*"
        client = self._create_client(data.site_name, data.request_timeout)

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
                    "must": [
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
                        {
                            "wildcard": {
                                "host.hostname": f"CSS01HXV01{data.tool_name}"
                            }
                        },
                    ]
                }
            },
            "sort": [
                {
                    "@timestamp": {
                        "order": "asc"
                    }
                }
            ]
        }

        try:
            response = client.search(index=index, body=query_body)
            hits = response.get("hits", {}).get("hits", []) or []
            results: List[Dict[str, Any]] = []
            return response
            # for hit in hits:
            #     source = hit.get("_source", {})
            #     results.append({
            #         "timestamp": source.get("@timestamp"),
            #         "agent_name": source.get("agent", {}).get("name") if isinstance(source.get("agent"), dict) else source.get("agent.name"),
            #         "hostname": source.get("host", {}).get("hostname") if isinstance(source.get("host"), dict) else source.get("host.hostname"),
            #         "dataset": source.get("data_stream", {}).get("dataset") if isinstance(source.get("data_stream"), dict) else source.get("data_stream.dataset"),
            #         "message": source.get("message"),
            #         "_id": hit.get("_id"),
            #         "_index": hit.get("_index"),
            #         "_score": hit.get("_score"),
            #     })

            # return {
            #     "index": index,
            #     "site_name": data.site_name,
            #     "tool_name": data.tool_name,
            #     "count": len(results),
            #     "results": results,
            # }

        except Exception as ex:
            raise RuntimeError(f"ElasticAlarmTool failed: {repr(ex)}") from ex

    def _create_client(self, site_name: int, request_timeout: int) -> Elasticsearch:
        if site_name not in self.SITE_CONFIG:
            raise ValueError(f"Unsupported site_name: {site_name}")

        uri_string, keyid, keyvalue = self.SITE_CONFIG[site_name]

        if keyid and keyvalue:
            return Elasticsearch(
                uri_string,
                api_key=(keyid, keyvalue),
                request_timeout=request_timeout,
            )
        else:
            return Elasticsearch(
                uri_string,
                request_timeout=request_timeout,
            )