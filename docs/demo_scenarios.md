# Opti-Oignon — Demo Scenarios

Step-by-step walkthroughs for the main features of Opti-Oignon v2.2.0. Each scenario assumes the backend is running on `http://localhost:8001` and the frontend on `http://localhost:5173`.


## Scenario 1: Basic Chat with Model Selection

This scenario demonstrates a simple chat interaction with smart model routing.

### Steps

1. **Open the app** — Navigate to `http://localhost:5173`. The app opens on the chats index; the sidebar lists the six most recent chats, and an open conversation's bar holds the model selector.

2. **Check available models** — Click the model selector dropdown. It shows all Ollama models currently pulled on your system, with capability badges (code, creative, analysis, etc.) from the model profiles.

3. **Select a model** — Choose `qwen3-coder:30b` (or whichever model you have available). The routing indicator below the selector updates to show the selected model.

4. **Send a message** — Type "Write a Python function that calculates the Fibonacci sequence using memoization" and press Enter (or click Send).

5. **Observe the pipeline** — The agentic executor automatically classifies this as a code task and routes it through the `code_verify` pipeline. You will see:
   - A streaming response appearing token by token
   - The routing indicator showing which pipeline was selected
   - Token count and generation speed in the message footer

6. **Try thinking mode** — In the Chat Controls bar, toggle "Think" on. Send "Compare the trade-offs between recursion and iteration for tree traversals." The response now includes a collapsible thinking section showing the model's internal reasoning before the final answer.

7. **View conversation** — The conversation is saved automatically. It appears under Recent in the sidebar and in the chats index, where each row's menu renames it.

### What to verify

- Model selector shows installed models with profile info
- Pipeline auto-selection works (code queries → `code_verify`, complex queries → `think`)
- Streaming works smoothly with token count display
- Conversation persists in the chats index


## Scenario 2: Running a Benchmark and Viewing Results

This scenario walks through evaluating your models and comparing their performance.

### Steps

1. **Navigate to Benchmarks** — Switch to the Workshop (the switch at the foot of the sidebar) and choose "Benchmarks" (bar-chart icon). The page is one row of tabs over the quality evaluation: Run, Leaderboard, Head-to-head, Trends, Compare, History, Profiles. The tab you are on stays in the address (`?tab=`), so a reload or a shared link lands on it.

2. **Configure a run** — On the Run tab:
   - Choose an evaluation profile (a built-in one, or one you made on the Profiles tab)
   - Select one or more models
   - Optionally turn on the LLM judge

3. **Start the run** — Start it from the Run tab. Its progress shows while each model answers the profile's tasks, and the results appear once the run completes: accuracy, code, structure and speed per model, the judge's scores when it ran, and a radar chart comparing the models.

4. **Rank the models** — The Leaderboard tab ranks every model the evaluation has seen, and suggests which model suits each role.

5. **Compare** — Head-to-head puts two models side by side; Compare aggregates several; Trends follows one model's scores over its runs.

6. **View history** — The History tab lists the fifty latest runs. Open one to see its detail in a drawer (every model's accuracy, code, structure and speed); closing the drawer brings you back to History.

7. **Export** — A completed run's results export as JSON or CSV from the Run tab.

8. **Configure model roles** — Roles are assigned in **Workshop > Models and inference > Model assignment**. Assign models to roles:
   - **Primary** — Default model for general use
   - **Fast** — Quick model for simple queries
   - **Quality** — Best model for complex tasks

   A save the server refuses is shown under its role, with the editor still open on your choices.

### What to verify

- The run completes and its results appear on the Run tab
- The leaderboard ranks the models that ran
- History persists across page reloads, and a run opened from it closes back onto History
- The tab survives a reload and Back


## Scenario 3: Creating a Project with File Context

This scenario demonstrates the RAG-powered project system for contextual conversations.

### Steps

1. **Navigate to Projects** — Click "Projects" in the sidebar. The project list loads (empty on first use).

2. **Create a project** — Click "New Project" and fill in:
   - **Name**: "Bioacoustics Analysis"
   - **Description**: "BCI field research data and analysis scripts"
   - **System Instructions**: "You are a bioinformatics assistant helping analyze acoustic biodiversity data from Barro Colorado Island. Use technical terminology appropriate for an ecology M2 researcher."

3. **Upload files** — In the project detail view, click "Upload Files" and add:
   - A Python script (e.g., `diversity_analysis.R`)
   - A data description document (e.g., `methods.md`)
   - A CSV data file (e.g., `species_counts.csv`)

   Files are validated against allowed extensions and size limits (configurable in `projects.yaml`). Each file is automatically indexed into a per-project ChromaDB collection.

4. **Link a conversation** — Go back to the chat page. The Project Context Badge appears in the chat header. Click it and select "Bioacoustics Analysis" to link the current conversation to the project.

5. **Chat with context** — Send a message that references your project files: "How should I modify the diversity analysis script to use Shannon-Wiener index instead of Simpson's?"

   The 3-level trigger detection activates:
   - **Level 1 (regex)**: Detects direct file references
   - **Level 2 (term matching)**: Matches domain terms from indexed files
   - **Level 3 (LLM classification)**: Determines relevance if levels 1–2 are inconclusive

   Relevant file chunks are injected into the context via RAG, and the model responds with project-aware information.

6. **Verify context injection** — The Project Context Badge shows a green indicator when context was injected. The Context Panel (accessible from the side panel) shows which chunks were retrieved and their relevance scores.

### What to verify

- Project creation and file upload work
- Files are indexed (check project stats endpoint)
- Trigger detection fires on relevant messages
- RAG-injected context improves response relevance


## Scenario 4: Comparing Benchmark Runs for Model Selection

This scenario demonstrates using benchmarks to make an informed model selection decision.

### Steps

1. **Run a baseline** — Go to **Workshop > Benchmarks**, Run tab. Select every installed model and an evaluation profile. Run it and wait for completion.

2. **Record the run** — The run is saved to history automatically; note its id on the History tab.

3. **Change model parameters** — Open **Workshop > Models and inference** and adjust the temperature of one model's profile (e.g., lower temperature for code tasks). Alternatively, pull a new model variant: `ollama pull qwen3-coder:30b-q4_0`.

4. **Run it again** — Run the same profile with the updated configuration.

5. **Compare the models** — The Compare tab aggregates the models' scores over their runs, and Head-to-head sets two of them side by side.

6. **Check model trends** — The Trends tab shows a model's composite score over its runs.

7. **Update model roles** — Based on the comparison, go to **Workshop > Models and inference > Model assignment** and update:
   - Assign the highest-scoring model to the "Quality" role
   - Assign the fastest model to the "Fast" role
   - Choose the best all-rounder for "Primary"

8. **Verify routing** — Go back to chat. The smart router now uses your updated role assignments. Send a simple question (routed to Fast model) and a complex analysis question (routed to Quality model). Verify via the routing indicator.

### What to verify

- Multiple runs of the same profile produce consistent, comparable results
- The trends and the comparison show the change between the two configurations
- Model role changes propagate to the smart router
- Routing indicator reflects the updated model assignments


## Scenario 5: Using the Pipeline Editor

This scenario demonstrates creating and running custom pipelines.

### Steps

1. **Navigate to Pipeline Editor** — Open a conversation, then the Pipelines side panel from the conversation's panel toggle. The editor shows builtin pipelines (Code Expert, Creative Writer, Research Assistant, Thorough Analyst) as read-only cards.

2. **Create a custom pipeline** — Click "New Pipeline" and configure:
   - **Name**: "Thorough Code Review"
   - **Description**: "Multi-step code analysis with reasoning and self-correction"
   - Add steps:
     1. **Step 1** — Type: `think`, prompt: "Analyze the code structure, identify potential bugs, and consider edge cases."
     2. **Step 2** — Type: `self_correct`, prompt: "Review your analysis for accuracy. Check if you missed any security vulnerabilities or performance issues."
     3. **Step 3** — Type: `direct`, prompt: "Provide a final structured code review with sections: Summary, Bugs Found, Suggestions, Security Notes."

3. **Preview model assignments** — Each step shows which model will be used based on the current smart routing configuration. The per-step preview updates when you change step types.

4. **Save the pipeline** — Click "Save". The pipeline appears in the custom pipelines section.

5. **Run the pipeline** — Go to chat. In the Chat Controls bar, select your "Thorough Code Review" pipeline from the pipeline dropdown. Paste a code snippet and send.

6. **Observe multi-step execution** — The execution panel shows progress through each step:
   - Step 1 thinking output (collapsible)
   - Step 2 self-correction iterations
   - Step 3 final formatted response

7. **Duplicate and modify** — Back in the editor, click "Duplicate" on your pipeline. Modify the duplicate to add a consensus step (type: `consensus`) that queries multiple models for the code review, then merges their findings.

### What to verify

- Pipeline CRUD operations work (create, read, update, delete, duplicate)
- Step configuration supports all 9 pipeline types
- Per-step model preview reflects smart routing
- Pipeline execution shows progress through each step
- Custom pipelines appear in the chat pipeline selector


## Scenario 6: Coding Agent — Create a Utility Script

This scenario demonstrates the autonomous coding agent creating a new file from scratch.

### Steps

1. **Open Coding Agent** — Navigate to the Coding Agent panel (accessible from the sidebar or the tools menu). The panel shows the agent status as idle.

2. **Submit a task** — Enter: "Create a Python script called `csv_stats.py` that reads a CSV file from a command-line argument, calculates mean/median/std for each numeric column, and prints a formatted summary table."

3. **Watch the plan** — The agent generates a JSON plan with steps:
   - Step 1: `create_file` — Create `csv_stats.py` with the implementation
   - Step 2: `create_file` — Create `test_csv_stats.py` with test cases
   - Step 3: `bash` — Run the tests

4. **Monitor execution** — The WebSocket progress stream shows:
   - Current phase (planning, implementing, testing)
   - Working memory updates (decisions, modified files)
   - Test results in real time

5. **Review diffs** — Once all steps complete, the agent presents unified diffs showing every file created or modified. Each diff includes the SHA-256 integrity hash.

6. **Apply or reject** — Click "Apply" to copy files from sandbox to workspace, or "Reject" to discard. The sandbox is destroyed after the decision.

### What to verify

- Plan generation produces valid JSON with correct step types
- Sandbox isolation (files created inside sandbox, not on host)
- Test auto-execution detects pass/fail
- Diff presentation includes integrity hashes
- Apply requires explicit human confirmation


## Scenario 7: Coding Agent — Fix a Bug with Auto-Retry

This scenario tests the agent's fix loop and working memory on a broken file.

### Steps

1. **Prepare a broken file** — Upload or create a file with an intentional bug, e.g., a Python function with an off-by-one error in a list slicing operation.

2. **Submit a fix task** — Enter: "Fix the bug in `broken_sort.py` — the function returns incorrect results for lists with duplicate elements. Add tests to verify the fix."

3. **Observe the fix loop** — The agent:
   - Reads the file (step type: `bash` with `cat`)
   - Creates test cases first (TDD approach)
   - Runs tests, sees failures
   - Edits the file to fix the bug
   - Re-runs tests, sees them pass

4. **Check working memory** — The working memory panel shows:
   - `decisions`: "Identified off-by-one in partition logic"
   - `errors_encountered`: original test failure with traceback
   - `modified_files`: `broken_sort.py` with description of change

5. **Verify cascading** — If the first model fails to fix the bug within 2 attempts, the agent auto-escalates to a more capable model (visible in the progress stream as an `escalated` event).

6. **Apply the fix** — Review the diff and apply.

### What to verify

- Fix loop retries up to max_fix_retries before giving up
- Working memory tracks context across steps
- Cascading escalation triggers after `escalate_after_failures` consecutive failures
- Diff shows only the relevant changes


## Scenario 8: First Run — Presets and Onboarding

This scenario walks through the first-run experience with system presets.

### Steps

1. **Fresh start** — If you have used the app before, go to Workshop > Backup > Configuration and click "Reset onboarding" to simulate a first run. Reload the page.

2. **Onboarding dialog** — The "Welcome to Opti-Oignon" dialog appears with:
   - The Opti-Oignon logo, on a flat tone, and Stop all in the dialog's head
   - "Scanning installed models" while the models are read
   - After the scan: the number of models detected, and their list, each with its size
   - Three presets, one choice: Minimal, Balanced, Power, the dialog opening on the recommended one
   - The word "Recommended" on the preset matching your hardware

3. **Apply a preset** — Choose a preset (the recommended one is chosen already), then click "Apply <name> preset". The dialog shows:
   - "Applying configuration" while the configs are being updated (the dialog cannot be closed meanwhile)
   - "<name> preset applied", the default model it chose, and any warnings under the word "Warnings" (e.g., "Some models referenced by Power preset are not installed")
   - "Get started" button: it closes the dialog, and the application takes the preset without a reload (the chat's models and default model, the feature list and the chat's switches are read again); closing the dialog by Escape or its close button does the same

4. **Verify configuration** — After applying:
   - Go to Workshop > Models and inference > System preset: the active preset reads "Applied"
   - Feature toggles match the preset (e.g., Power enables cascading and speculative)
   - Default model is set to the largest/smallest model per strategy

5. **Try different presets** — In Workshop > Models and inference > System preset, click "Apply" on another preset. Configs update live. Use the smoke test to verify: `bash scripts/smoke_test.sh`

### What to verify

- Onboarding overlay appears on first run
- Model detection correctly identifies installed models
- Preset apply updates all relevant YAML config files
- Workshop > Models and inference reflects the applied preset
- Skip button works (closes overlay without applying)

