"""结构化输出模型"""

from typing import Literal

from pydantic import BaseModel, Field


class TaskClassification(BaseModel):
    """PPT 任务分类结果 — 使用 Pydantic BaseModel"""

    is_complex: bool = Field(description="是否为复杂任务（需要规划多步骤）")
    reason: str = Field(default="", description="分类理由，一句话说明")
    suggested_tools: list[str] = Field(
        default_factory=list,
        description="建议使用的工具列表，如 ['refine_content', 'generate_ppt', 'enhance_ppt']",
    )


class PlanStep(BaseModel):
    """单个执行步骤 — 使用 Pydantic 进行字段和类型校验。"""

    description: str = Field(description="步骤描述")
    expected_tools: list[str] = Field(
        default_factory=list,
        description="预期使用的工具列表",
    )


class ExecutionPlan(BaseModel):
    """PPT 执行计划 — 使用 Pydantic 校验完整计划结构。"""

    task_summary: str = Field(description="任务摘要")
    steps: list[PlanStep] = Field(
        min_length=1,
        description="非空执行步骤列表",
    )
    estimated_difficulty: Literal["easy", "medium", "hard"] = Field(
        description="预估难度",
    )


PPT_SLIDES_SCHEMA: dict = {
    "title": "PPTSlides",
    "description": "PPT 幻灯片结构化数据，由多张幻灯片组成",
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "PPT 整体标题",
        },
        "style": {
            "type": "string",
            "enum": ["business", "creative", "academic", "minimalist"],
            "description": "PPT 主题风格",
        },
        "slides": {
            "type": "array",
            "items": {},
            "description": "幻灯片列表",
        },
    },
    "required": ["title", "style", "slides"],
}
