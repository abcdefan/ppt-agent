"""将公共 IntentRouterService 适配成 Workflow Router Node。"""

from app.agents.router import IntentRouterService, RouteContext
from app.agents.workflow.state import WorkflowState

INTENT_ROUTER_NODE = "intent_router_node"


def build_intent_router_node(router: IntentRouterService):
    async def route_intent(state: WorkflowState) -> dict:
        decision = await router.route(
            RouteContext(
                user_message=state["user_message"],
                requested_action=state.get("requested_action"),
                active_ppt_id=state.get("active_ppt_id"),
                style=state.get("style") or "business",
                recent_messages=state.get("messages", []),
            )
        )
        return {
            "intent": decision.intent,
            "route_source": decision.source,
            "route_confidence": decision.confidence,
            "route_reason": decision.reason,
            # Create/Edit 的真实 PPT 上下文分别由后续初始化/目标解析节点加载。
            **(
                {
                    "ppt_id": None,
                    "ppt_context_error": None,
                    "workflow_error": None,
                    "outline": None,
                    "research_report": None,
                    "filename": None,
                    "slides_manifest": None,
                    "asset_tasks": [],
                    "required_stages": [],
                    "requirements_initialized": False,
                    "next": None,
                }
                if decision.intent in {"create", "edit"}
                else {}
            ),
        }

    return route_intent
