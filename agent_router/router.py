from agent_roles.coder import send_to_coder
from agent_roles.reviewer import run_reviewer_agent


def agent_router(text, agent, route=None, context=None, memories=None):
    handlers = {'coder': send_to_coder, 'reviewer': run_reviewer_agent}
    if agent not in handlers:
        raise ValueError(f'Agent role is not implemented: {agent}')
    return handlers[agent](text, context=context, memories=memories)
