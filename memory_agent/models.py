from uuid import UUID, uuid4
from datetime import datetime
from typing import Literal, List, Optional
from pydantic import BaseModel, Field

class UserMessage(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class MessagePayload(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class FactCandidate(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    category: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class MemoryContext(BaseModel):
    work: str = ""
    personal: str = ""
    top_of_mind: str = ""

class MemoryHistory(BaseModel):
    recent_months: str = ""
    earlier_context: str = ""
    long_term_background: str = ""

class MemoryStore(BaseModel):
    context: MemoryContext = Field(default_factory=MemoryContext)
    history: MemoryHistory = Field(default_factory=MemoryHistory)
    facts: List[FactCandidate] = Field(default_factory=list)

class AgentState(BaseModel):
    current_message: Optional[UserMessage] = None
    current_prompt: str = ""
    last_response: str = ""
    memory_version: int = 0

class BuildReport(BaseModel):
    context_tokens: int
    history_tokens: int
    facts_included: int
    facts_excluded: int
    total_tokens: int

class MemoryUpdatedEvent(BaseModel):
    version: int
    facts_added: int
    facts_discarded: int
    facts_evicted: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class FactExtractionResult(BaseModel):
    facts: List[FactCandidate]
