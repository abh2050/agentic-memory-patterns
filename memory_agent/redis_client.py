import redis
import json
from typing import Optional, Type, TypeVar
from pydantic import BaseModel
from memory_agent.config import REDIS_URL

T = TypeVar("T", bound=BaseModel)

class RedisClient:
    def __init__(self):
        self.client = redis.from_url(REDIS_URL, decode_responses=True)

    def set_model(self, key: str, model: BaseModel):
        self.client.set(key, model.model_dump_json())

    def get_model(self, key: str, model_type: Type[T]) -> Optional[T]:
        data = self.client.get(key)
        if data:
            return model_type.model_validate_json(data)
        return None

    def xadd_model(self, stream: str, model: BaseModel):
        self.client.xadd(stream, {"data": model.model_dump_json()})

    def xread_models(self, stream: str, model_type: Type[T], count: int = 10, block: int = 0) -> list[T]:
        streams = {stream: "0"} # Read from beginning for demo simplicity, or use $ for new
        messages = self.client.xread(streams, count=count, block=block)
        results = []
        if messages:
            for _, msgs in messages:
                for msg_id, data in msgs:
                    results.append(model_type.model_validate_json(data["data"]))
        return results

    def xread_latest(self, stream: str, model_type: Type[T], last_id: str = "0") -> tuple[list[T], str]:
        messages = self.client.xread({stream: last_id}, block=0)
        results = []
        new_last_id = last_id
        if messages:
            for _, msgs in messages:
                for msg_id, data in msgs:
                    results.append(model_type.model_validate_json(data["data"]))
                    new_last_id = msg_id
        return results, new_last_id

    def get_stream_length(self, stream: str) -> int:
        return self.client.xlen(stream)

    def delete_stream(self, stream: str):
        self.client.delete(stream)

    def set_raw(self, key: str, value: str):
        self.client.set(key, value)

    def get_raw(self, key: str) -> Optional[str]:
        return self.client.get(key)

redis_client = RedisClient()
