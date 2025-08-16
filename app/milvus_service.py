import os
from typing import Any, Dict, List, Optional

from pymilvus import MilvusClient, DataType


class MilvusService:
    def __init__(self, db_path: str, collection_name: str, dim: int) -> None:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.client = MilvusClient(uri=db_path)
        self.collection_name = collection_name
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.collection_name in [c["collection_name"] for c in self.client.list_collections()]:
            # If exists, we might ensure it's loaded
            self.client.load_collection(self.collection_name)
            return

        schema = self.client.create_schema(
            auto_id=True,
            description="RAG chunks with metadata",
            enable_dynamic_field=False,
        )
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        )
        schema.add_field(
            field_name="dataset_id",
            datatype=DataType.VARCHAR,
            max_length=128,
        )
        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.INT64,
        )
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.dim,
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            num_shards=1,
        )
        self.client.load_collection(self.collection_name)

    def upsert_chunks(
        self,
        dataset_id: str,
        chunks: List[str],
        embeddings: List[List[float]],
    ) -> int:
        assert len(chunks) == len(embeddings)
        rows: List[Dict[str, Any]] = []
        for i, (text, vec) in enumerate(zip(chunks, embeddings)):
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "chunk_id": i,
                    "text": text,
                    "vector": vec,
                }
            )
        res = self.client.insert(collection_name=self.collection_name, data=rows)
        if isinstance(res, dict) and "insert_count" in res:
            return int(res["insert_count"])  # simple client returns an int count
        return len(rows)

    def search(
        self,
        dataset_id: Optional[str],
        query_vector: List[float],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        filter_expr = None
        if dataset_id:
            # PyMilvus simple client uses `filter` for scalar filter
            filter_expr = f"dataset_id == '{dataset_id}'"

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=limit,
            search_params={"metric_type": "COSINE"},
            output_fields=["dataset_id", "chunk_id", "text"],
            filter=filter_expr,
        )
        # results is a list of hits list; take first
        if not results or not results[0]:
            return []
        # Flatten to dicts
        flat: List[Dict[str, Any]] = []
        for hit in results[0]:
            if isinstance(hit, dict):
                entity = hit.get("entity", {})
                score = hit.get("distance")
            else:
                # Hit object fallback
                entity = getattr(hit, "entity", {})
                score = getattr(hit, "distance", None)
            if isinstance(entity, dict) and entity:
                flat.append({**entity, "score": score})
        return flat