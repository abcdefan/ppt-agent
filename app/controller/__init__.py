"""控制层包"""

from app.controller.assistant import router as assistant_router
from app.controller.chat import router as chat_router
from app.controller.user import router as user_router

__all__ = ["assistant_router", "chat_router", "user_router"]
