from typing import List, Literal, Optional

from langchain_core.pydantic_v1 import BaseModel, Field


class HeaderCatalogEntry(BaseModel):
    header: str = Field(...)
    section_label: str = Field(default="")
    section_type: str = Field(default="")
    source: str = Field(default="")
    summary: str = Field(default="")


class RetrievalPlanStep(BaseModel):
    step_id: str = Field(...)
    action: Literal["retrieve", "extract"] = Field(...)
    purpose: str = Field(...)
    query: str = Field(default="")
    target_headers: List[str] = Field(default_factory=list)
    depends_on: Optional[str] = Field(default=None)
    extract_fields: List[Literal["project_names", "company_names", "role_titles", "keywords"]] = Field(default_factory=list)


class RetrievalPlan(BaseModel):
    strategy: Literal["single_section", "multi_section", "dependent_lookup"] = Field(...)
    reasoning: str = Field(...)
    steps: List[RetrievalPlanStep] = Field(default_factory=list)


class ExtractedEntities(BaseModel):
    project_names: List[str] = Field(default_factory=list)
    company_names: List[str] = Field(default_factory=list)
    role_titles: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
