Potentially should consider having seperate models/reasoning levevls for each section of the agent pipeline,

eg if coder is only writing code according to a plan rather than starting from scratch, it may not need a large model as it is a heavily restricted section of the task rather than the whole thing

Context management between systems will be a core part of the challenge headers


1. Propose Blackboard and shared state as the architecture to adopy
- Managed passing between agents
- shared state/blackboard, but different agent roles 

Router use the blackbboard to decide where to pass or could have a planner to call the best next agent (this may be a bit of a supervisor/blackboard if was adopted)
-> Plan -> set of tasks -> task status -> deligate task (via contract) -> router to agent based on contract status


Subagent -> Works on task, updates status of their stage -> passes to executor to text -> 

How to set up agent communication -> If we need to split via multiple vm machines 
