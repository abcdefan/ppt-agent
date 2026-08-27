"""将公共 IntentRouterService 适配成 Workflow Router Node。"""

from app.agents.router import IntentRouterService, RouteContext
from app.agents.workflow.state import WorkflowState

INTENT_ROUTER = "intent_router"


def build_intent_router_node(router: IntentRouterService):
    async def route_intent(state: WorkflowState) -> dict:
        decision = await router.route(
            RouteContext(
                user_message=state["user_message"],
                requested_action=state.get("requested_action"),
                active_ppt_filename=state.get("filename"),
                style=state.get("style") or "business",
                recent_messages=state.get("messages", []),
            )
        )
        return {
            "intent": decision.intent,
            "route_source": decision.source,
            "route_confidence": decision.confidence,
            "route_reason": decision.reason,
            # V1 的 create 始终表示新建文件，不能让上一轮持久化的
            # outline/filename 使 Supervisor 误以为任务已经完成。
            **(
                {"outline": None, "research_report": None, "filename": None}
                if decision.intent == "create"
                else {}
            ),
        }

    return route_intent
