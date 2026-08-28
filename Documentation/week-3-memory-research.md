# Agent Memory Research and Initial Design

**Unit:** COS30018 Intelligent Systems  
**Project:** LLM-Powered Multi-Agent Coding Harness  
**Student:** Le Nhu Ngoc Ho  
**Area of responsibility:** Memory  
**Sprint stage:** Research and initial design  
**Week:** 3  

## 1. Purpose of This Research

The purpose of this work is to understand and design the memory component for our multi-agent coding system. The memory component will allow agents to retain useful information, retrieve relevant information when completing a task, and share selected state with other agents.

This document records the research and design work completed before implementation. It does not claim that the complete memory system has already been implemented.

## 2. Why the Project Needs Memory

A language model does not automatically remember previous sessions. OpenRouter provides access to language models, but our application is responsible for storing conversation history, task state and long-term information.

Without a memory system, the coding agents may:

- Forget requirements and previous decisions.
- Ask the user the same questions repeatedly.
- Repeat actions that have already failed.
- Lose all context when the program closes.
- Produce code that conflicts with earlier architectural decisions.
- Send the complete conversation history on every request, increasing token usage.

With memory, the system should be able to:

- Maintain the current task and plan.
- Remember important project requirements.
- Retrieve previous errors and successful fixes.
- Share verified results between agents.
- Continue relevant work across multiple sessions.
- Reduce repeated questions and unnecessary tool calls.

## 3. Context Is Not Persistent Memory

The context window is the information included in the request sent to the language model. It is temporary, limited in size and must be sent again with each request.

Persistent memory is stored outside the language model. It can survive after the application is closed, can be searched, and can be updated or deleted over time.

The context window is where selected memories are **used**. It is not where long-term memories permanently **live**.

## 4. Types of Agent Memory

### 4.1 Working Memory

Working memory contains the information required for the agent's current step.

Examples:

- Current user request.
- Current goal.
- Active plan step.
- Latest tool result.
- Current file being edited.
- Retry count.
- Temporary variables.

Working memory normally lasts for one task or agent run. It should be represented as structured task state rather than a collection of permanent memories.

### 4.2 Short-Term Memory

Short-term memory stores information from the current session.

Examples:

- Recent user and assistant messages.
- Recent agent actions.
- Temporary requirements.
- Current conversation summary.
- Recent tool observations.

The system should not continue adding every message to the prompt forever. A bounded recent history and a rolling summary can control token usage.

### 4.3 Long-Term Semantic Memory

Semantic memory stores facts that may remain useful across sessions.

Examples:

- The project uses Python.
- The team uses OpenRouter.
- Project architecture decisions.
- Coding conventions.
- Stable user preferences.
- Important assignment requirements.

These memories should include their source, creation time and reliability so the system can decide whether they are still trustworthy.

### 4.4 Long-Term Episodic Memory

Episodic memory stores useful experiences from earlier tasks.

An episode can be represented as:

```text
Situation -> Action -> Observation -> Outcome -> Lesson
```

Examples:

- A command failed because a dependency was missing.
- A test failed after a particular code change.
- A reviewer rejected an implementation because it did not handle timeouts.
- A previous bug was fixed by changing a request header.

The lesson or outcome is usually more useful than saving the complete raw transcript.

### 4.5 Procedural Memory

Procedural memory stores information about how an agent should perform tasks.

Examples:

- Coding standards.
- Tool-use instructions.
- Testing procedures.
- Error-recovery rules.
- Prompt templates.
- Reviewer acceptance rules.

Procedural memory should normally be controlled by developers and stored in version-controlled code, prompts or configuration files. Agents should not silently modify their own important safety rules.

## 5. Current Repository Findings

The following files were reviewed:

- `main.py`
- `memory/memory.py`
- `memory/__init__.py`
- `agent_roles/coder.py`
- `agent_router/router.py`

### 5.1 Current Memory Stub

The current function in `memory/memory.py` is:

```python
def get_relevant_memories(text: str, memories: list[str]) -> list[str]:
    """Stub to build out"""
    return []
```

This means that memory retrieval is not currently implemented. The function always returns an empty list.

### 5.2 Current Conversation Context

`main.py` creates the following values when the program starts:

```python
memories = []
context = []
```

The current `context` list provides basic short-term conversation history, but it has several limitations:

- It disappears when the application closes.
- It grows without a defined token limit.
- It does not have project, user, task or agent identifiers.
- It does not distinguish messages from decisions, errors or tool results.
- It is not searchable.

The `memories` list is never populated, so the current application has no long-term memory.

### 5.3 Current Prompt Injection Method

The coder currently adds memories directly to the user's text. A safer design should provide clear boundaries between:

- System instructions.
- Current task state.
- Retrieved memory.
- Latest user message.
- Latest tool observation.

Retrieved memory should be treated as supporting information, not as trusted system instructions.

## 6. Proposed Memory Architecture

The recommended design is **scoped shared memory with private working state**.

```text
User Interface
      |
Session and Task Manager
      |
Shared Task State
      |
      +------ Planner
      +------ Coder
      +------ Executor
      +------ Reviewer
      +------ Router
                 |
            Memory Service
                 |
      +----------+-----------+
      |          |           |
Short-Term   Long-Term    Episodic
Memory       Semantic     Memory
             Memory
```

Agents should access memory through a shared memory service. They should not directly read and write arbitrary database rows.

## 7. Memory Scopes

Each memory should have a defined scope.

| Scope | Purpose |
|---|---|
| Global | General rules that are safe for every project |
| Project | Project requirements, architecture and conventions |
| User | Stable preferences belonging to one user |
| Session | Information needed during one conversation |
| Task | State and observations for one task or run |
| Agent | Private information only needed by a particular agent role |

Scope filtering should happen before relevance ranking. This helps prevent unrelated or private memories from being sent to the wrong agent.

## 8. Memory Lifecycle

The proposed memory lifecycle is:

```text
Observe -> Store -> Index -> Retrieve -> Use -> Update -> Forget
```

### Observe

Possible observations include user messages, plans, tool calls, tool results, test results, reviewer decisions and human corrections.

### Store

The memory policy decides whether the observation is useful enough to keep. Not every observation should become long-term memory.

### Index

A stored memory should be indexed using its text, metadata, keywords and optionally an embedding vector.

### Retrieve

The system searches for memories related to the current goal, plan step or error.

### Use

Selected memories are added to the model context with clear source and reliability information.

### Update

A memory can be corrected, marked as disputed, or replaced by newer information.

### Forget

A memory can expire, be archived, have its retrieval priority reduced, or be deleted.

## 9. What Should Be Remembered

Suitable long-term memories include:

- Stable project requirements.
- Explicit user preferences.
- Architectural decisions and their reasons.
- Coding conventions.
- Verified errors and fixes.
- Reviewer-confirmed lessons.
- Important failure patterns.
- Repeatedly useful commands or procedures.

Information that should normally not be stored permanently includes:

- Greetings and small talk.
- Temporary calculations.
- Raw chain-of-thought.
- Duplicate facts.
- Very large terminal outputs.
- API keys, passwords and authentication tokens.
- Unverified model-generated claims.
- Complete copies of files that already exist in the repository.

A useful memory policy question is:

> Will this information probably be useful in another session, and is it trustworthy enough to keep?

## 10. Proposed Data Structures

### 10.1 Memory Record

```text
MemoryRecord
- id
- project_id
- user_id
- session_id
- task_id
- agent_id
- memory_type
- content
- summary
- source_type
- source_reference
- created_at
- updated_at
- valid_until
- importance
- reliability
- status
- supersedes_id
- metadata
- embedding_model
- embedding
```

Suggested memory types include:

- `requirement`
- `preference`
- `project_fact`
- `decision`
- `constraint`
- `tool_observation`
- `error`
- `bug_fix`
- `review_feedback`
- `episode`
- `lesson`
- `session_summary`
- `procedure`

### 10.2 Retrieval Result

A retrieval function should return structured results rather than only returning a list of strings.

```text
MemoryHit
- memory_id
- content
- memory_type
- relevance_score
- source
- reliability
- created_at
- reason_retrieved
```

This structure will make retrieval easier to inspect, test and explain in the user interface.

### 10.3 Shared Task State

```text
TaskState
- task_id
- project_id
- session_id
- user_goal
- constraints
- current_plan
- active_step
- assigned_agent
- latest_observation
- artifacts
- retry_count
- status
- error
```

The current task state should remain separate from long-term memory.

## 11. Inputs and Outputs

### Inputs Required by Memory

The memory component will require:

- Project, user, session and task identifiers.
- Requesting agent role.
- Current user goal.
- Current plan step.
- Recent conversation messages.
- Tool commands and results.
- Test results.
- Reviewer decisions.
- Source references and timestamps.
- Memory type, importance and reliability.

### Outputs Provided by Memory

The memory component should provide:

- Ranked memories relevant to the current task.
- Short-term session history or summary.
- Previous errors and lessons.
- Project requirements and constraints.
- Source and reliability metadata.
- Retrieval scores and reasons.
- Persistent task checkpoints.
- Memory events for logging and evaluation.

## 12. Interaction With Other Agents

### Planner Agent

The planner can retrieve project requirements, user constraints, previous plans and previous failure lessons. It can write important plan decisions and plan revisions.

### Coder Agent

The coder can retrieve coding conventions, relevant project decisions, previous bugs and reviewer-approved lessons. Generated source code should be stored in repository files rather than copied into memory.

### Executor Agent

The executor can store command names, exit codes, bounded output summaries, error messages, durations and artifact paths. Executor observations should have high reliability because they come from the actual environment.

### Reviewer Agent

The reviewer can retrieve requirements, implementation decisions and test evidence. It can store verified defects, approval decisions and lessons that may help future tasks.

### Router or Coordinator

The router should use structured task state to decide which agent acts next. It can store routing events, task ownership, retry counts, recovery decisions and stopping conditions.

## 13. Retrieval Design

Retrieval should use more than one signal.

```text
retrieval score =
    keyword relevance
  + semantic similarity
  + recency
  + importance
  + reliability
  + scope relevance
```

Keyword matching is important for file paths, function names, package names, task IDs and error codes. Semantic similarity is useful when the current wording is different from the stored memory.

The retrieval process should:

1. Filter by project, user, task and agent scope.
2. Search by keyword and, later, semantic similarity.
3. Rank by relevance, recency, importance and reliability.
4. Remove duplicate or nearly identical results.
5. Select the top results within a token budget.
6. Return the selected memories with their source information.

## 14. Storage Decision

The recommended initial storage system is SQLite with metadata and SQLite FTS5 keyword search.

Reasons for selecting SQLite:

- It is local and free.
- It does not require a database server.
- It survives program restarts.
- It supports transactions.
- It is easy to inspect and test.
- It helps make the project reproducible.
- It is suitable for the expected project data size.

Semantic embeddings can be added later through a separate interface. The interface could support a local embedding model or the OpenRouter embeddings endpoint without changing the rest of the memory service.

## 15. Safety and Data Quality Risks

### Hallucinated Memory

The model may produce an incorrect statement that is later stored as a fact. Model-generated claims should remain unverified until confirmed by a tool, reviewer or user.

### Conflicting Memory

Two memories may contain different information. The system should store timestamps, sources and reliability levels and allow newer information to supersede older information.

### Outdated Memory

A fact may have been correct when stored but become incorrect later. Mutable facts should be verified using current tools.

### Duplicate Memory

Repeated information can reduce retrieval quality. The system should check for exact or similar existing memories before inserting a new record.

### Memory Overload

Saving everything creates noise and high token usage. The system needs a selective write policy, top-k retrieval and a prompt token budget.

### Prompt Injection

Retrieved memory may contain malicious or instruction-like text. Retrieved memory must be clearly marked as untrusted context and kept separate from system instructions.

### Privacy

The memory system must not store API keys, passwords, access tokens or unnecessary personal information. It should support deletion and expiration.

## 16. Initial Implementation Plan

### Stage 1: Define Contracts

- Create `MemoryRecord`, `MemoryHit` and `TaskState` models.
- Define memory types, scopes, reliability levels and statuses.
- Define the public memory-service interface.

### Stage 2: Add SQLite Persistence

- Create the database schema.
- Implement insert, retrieve, update, supersede, expire and delete operations.
- Add filtering by project, user, session, task and agent.

### Stage 3: Add Basic Retrieval

- Implement SQLite FTS5 keyword search.
- Add metadata, recency, importance and reliability ranking.
- Add top-k and token-budget limits.
- Remove duplicate results.

### Stage 4: Add Short-Term Memory

- Store bounded recent messages.
- Add session summaries.
- Restore a session after restarting the application.

### Stage 5: Integrate With Agents

- Add memory retrieval before an agent call.
- Update state and create memory candidates after an agent or tool result.
- Add safe delimiters and source information to prompts.

### Stage 6: Add Automated Tests

- Test persistence and retrieval.
- Test scope isolation.
- Test duplicate and conflicting memories.
- Test expiration and deletion.
- Test prompt token limits.
- Test behaviour when retrieval fails.

### Stage 7: Experiment With Embeddings

- Define a replaceable embedding-provider interface.
- Compare keyword-only and hybrid retrieval.
- Record retrieval quality, latency, token usage and cost.

## 17. Evaluation Plan

The memory feature should be compared against a no-memory baseline.

Possible configurations are:

1. Agent with no persistent memory.
2. Agent with complete raw conversation history.
3. Agent with selective structured memory.

Useful metrics include:

- Task success rate.
- Retrieval precision and recall.
- Number of repeated questions.
- Number of repeated failed actions.
- Tool-use success rate.
- Recovery rate.
- Number of agent iterations.
- Retrieval latency.
- Total latency.
- Prompt token usage.
- Approximate model cost.
- Cross-session consistency.

Memory-specific test cases should include:

- Retrieving a previous architectural decision.
- Remembering a user constraint in a later session.
- Finding an exact error code.
- Ignoring a memory belonging to another project.
- Preferring a verified memory over an unverified claim.
- Handling two conflicting memories.
- Expiring temporary information.
- Deleting a memory on request.
- Ignoring malicious instructions inside retrieved text.
- Continuing safely when the embedding service is unavailable.

## 18. Sprint Outcome

The research produced the following design decisions:

- Separate task state, short-term memory and long-term memory.
- Use scoped shared memory rather than full-context sharing.
- Treat current filesystem and tool results as more authoritative than old memory.
- Store provenance, time, importance and reliability with every long-term memory.
- Use SQLite and keyword retrieval as the first reproducible baseline.
- Add semantic embeddings only after the baseline works.
- Test memory using both retrieval metrics and end-to-end task outcomes.
- Avoid storing secrets, raw chain-of-thought and unverified claims as trusted facts.

## 19. Sources Consulted

- COS30018 Lecture 1: Intelligent Systems Introduction.
- COS30018 Lecture 2: Agentic Architecture and Design Patterns.
- COS30018 Lecture 3: Engineering Agentic AI Systems.
- COS30018 Lecture 4: Multi-Agent Systems.
- COS30018 Project Assignment - Option A.
- OpenRouter API documentation for chat completions, embeddings and privacy.
- Current COS30018 Project A repository source code.

