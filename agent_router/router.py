from agent_roles.coder import send_to_coder
from agent_roles.executor import send_to_executor
from agent_roles.planner import send_to_planner
from agent_roles.reviewer import send_to_reviewer


def agent_router(text, agent, route, context, memories):
    #Insert routing logic to decide what agent, may be based on current state

    if agent == "coder":
        response = send_to_coder(
                text,
                context=context,
                memories=memories,
            )

    else: None
    return response 
    
    