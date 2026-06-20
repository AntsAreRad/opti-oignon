# Opti-Oignon — Demo Scenarios

Step-by-step walkthroughs for the main features of Opti-Oignon v1.9.0. Each scenario assumes the backend is running on `http://localhost:8000` and the frontend on `http://localhost:5173`.


## Scenario 1: Basic Chat with Model Selection

This scenario demonstrates a simple chat interaction with smart model routing.

### Steps

1. **Open the app** — Navigate to `http://localhost:5173`. The chat interface loads with a model selector in the top bar and a conversation list in the sidebar.

2. **Check available models** — Click the model selector dropdown. It shows all Ollama models currently pulled on your system, with capability badges (code, creative, analysis, etc.) from the model profiles.

3. **Select a model** — Choose `qwen3-coder:30b` (or whichever model you have available). The routing indicator below the selector updates to show the selected model.

4. **Send a message** — Type "Write a Python function that calculates the Fibonacci sequence using memoization" and press Enter (or click Send).

5. **Observe the pipeline** — The agentic executor automatically classifies this as a code task and routes it through the `code_verify` pipeline. You will see:
   - A streaming response appearing token by token
   - The routing indicator showing which pipeline was selected
   - Token count and generation speed in the message footer

6. **Try thinking mode** — In the Chat Controls bar, toggle "Think" on. Send "Compare the trade-offs between recursion and iteration for tree traversals." The response now includes a collapsible thinking section showing the model's internal reasoning before the final answer.

7. **View conversation** — The conversation is saved automatically. It appears in the sidebar. Click the conversation title to rename it.

### What to verify

- Model selector shows installed models with profile info
- Pipeline auto-selection works (code queries → `code_verify`, complex queries → `think`)
- Streaming works smoothly with token count display
- Conversation persists in the sidebar


## Scenario 2: Running a Benchmark and Viewing Results

This scenario walks through benchmarking your models and comparing their performance.

### Steps

1. **Navigate to Benchmark** — Click "Benchmark" in the sidebar (bar-chart icon). The Benchmark page loads with three tabs: Run, History, and Model Assignment.

2. **Configure a run** — On the Run tab:
   - Select one or more models using the model chips (e.g., `qwen3-coder:30b` and `qwen3:32b`)
   - Select one or more benchmark suites (e.g., "General Knowledge", "Code Generation")
   - Optionally expand "Advanced Options" to adjust temperature and timeout

3. **Start the benchmark** — Click "Run Benchmark". A progress bar appears with:
   - Current task name and model being tested
   - Completion percentage and ETA
   - Individual results appearing in a table as they complete

4. **Review results** — Once complete, the results table shows each model/task combination with:
   - Auto-score (keyword-based scoring, 0–10)
   - Time taken per task
   - Status badges (success, refusal, error, timeout)
   - Color-coded scores (green > 7, yellow 4–7, red < 4)

5. **Submit user scores** — Click on any result row to provide your own score (1–10). The final score is computed as a weighted average of auto and user scores.

6. **View history** — Switch to the History tab. Previous runs appear as cards with summary stats. Click any card to see:
   - Full results table
   - Best-by-category breakdown
   - Model ranking

7. **Compare runs** — Select two or more runs using checkboxes, then click "Compare". The comparison view shows:
   - A model × task score matrix
   - Score deltas between runs
   - Regression warnings (highlighted in red if a model dropped > 1.5 points)

8. **Configure model roles** — Switch to the Model Assignment tab. Assign models to roles:
   - **Primary** — Default model for general use
   - **Fast** — Quick model for simple queries
   - **Quality** — Best model for complex tasks

### What to verify

- WebSocket progress updates in real time
- Scoring and result display work correctly
- History persists across page reloads
- Comparison detects regressions


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

1. **Run baseline benchmark** — Go to Benchmark > Run tab. Select all installed models, choose the "General Knowledge" and "Code Generation" suites. Run the benchmark and wait for completion.

2. **Record the run** — Note the run ID. The run is automatically saved to history.

3. **Change model parameters** — Go to Settings and adjust the temperature for one model (e.g., lower temperature for code tasks). Alternatively, pull a new model variant: `ollama pull qwen3-coder:30b-q4_0`.

4. **Run second benchmark** — Run the same suites again with the updated configuration.

5. **Compare both runs** — Go to History tab, select both runs, click "Compare". The comparison matrix shows:
   - Per-task score changes for each model
   - Overall average score delta
   - Regressions highlighted (any drop > 1.5 points)

6. **Check model trends** — For any model that appeared in multiple runs, the trends view (via API at `GET /api/benchmark/trends/{model}`) shows score and speed progression over time.

7. **Update model roles** — Based on the comparison, go to Model Assignment tab and update:
   - Assign the highest-scoring model to the "Quality" role
   - Assign the fastest model to the "Fast" role
   - Choose the best all-rounder for "Primary"

8. **Verify routing** — Go back to chat. The smart router now uses your updated role assignments. Send a simple question (routed to Fast model) and a complex analysis question (routed to Quality model). Verify via the routing indicator.

### What to verify

- Multiple benchmark runs produce consistent, comparable results
- Comparison correctly identifies improvements and regressions
- Model role changes propagate to the smart router
- Routing indicator reflects the updated model assignments


## Scenario 5: Using the Pipeline Editor

This scenario demonstrates creating and running custom pipelines.

### Steps

1. **Navigate to Pipeline Editor** — Go to Settings, then the Pipelines tab. The editor shows builtin pipelines (Code Expert, Creative Writer, Research Assistant, Thorough Analyst) as read-only cards.

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

1. **Fresh start** — If you have used the app before, go to Settings > Advanced > Onboarding and click "Reset Onboarding" to simulate a first run. Reload the page.

2. **Onboarding overlay** — A full-screen overlay appears with:
   - Opti-Oignon logo with copper glow
   - "Scanning models..." loading state
   - After scan: list of detected Ollama models as chips (with size badges)
   - Three preset cards: Minimal, Balanced, Power
   - A "Recommended" badge on the preset matching your hardware

3. **Apply a preset** — Click the recommended preset card. The overlay shows:
   - Spinner while configs are being updated
   - Confirmation with any warnings (e.g., "Some models referenced by Power preset are not installed")
   - "Get Started" button

4. **Verify configuration** — After applying:
   - Go to Settings > Quick tab: the preset selector shows the active preset
   - Feature toggles match the preset (e.g., Power enables cascading and speculative)
   - Default model is set to the largest/smallest model per strategy

5. **Try different presets** — In Settings > Quick tab, click another preset card and "Apply". Configs update live. Use the smoke test to verify: `bash scripts/smoke_test.sh`

### What to verify

- Onboarding overlay appears on first run
- Model detection correctly identifies installed models
- Preset apply updates all relevant YAML config files
- Settings page reflects the applied preset
- Skip button works (closes overlay without applying)

