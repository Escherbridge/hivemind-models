# %% [markdown]
# # HiveMind Pipeline: Generate Tool-Calling Training Datasets
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Escherbridge/laxame-hivemind/blob/main/hivemind-models/notebooks/03_generate_datasets.py)
#
# This notebook generates per-expert training datasets using a teacher model (Claude):
# 1. Load tool category definitions from notebook 02
# 2. Configure the teacher model (Claude API) or use dry-run mode
# 3. Generate N training samples per category
# 4. Validate dataset quality (JSON parsing, diversity metrics)
# 5. Save datasets as JSONL (one per category)
# 6. Print statistics and cost estimates

# %% [markdown]
# ## 1. Environment Setup

# %%
import sys
import json
import yaml
import time
import hashlib
import random
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any

COLAB = "google.colab" in str(get_ipython()) if hasattr(__builtins__, "__IPYTHON__") else False

if COLAB:
    get_ipython().system('pip install -q anthropic pyyaml tqdm tabulate')
    REPO_ROOT = Path("/content/laxame-hivemind")
    DRIVE_ROOT = Path("/content/drive/MyDrive/hivemind-pipeline")
    TOOL_CONFIG_DIR = DRIVE_ROOT / "tool_configs"
    DATASET_DIR = DRIVE_ROOT / "datasets"
else:
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent if "__file__" in dir() else Path.cwd().parent.parent
    TOOL_CONFIG_DIR = REPO_ROOT / "hivemind-models" / "output" / "tool_configs"
    DATASET_DIR = REPO_ROOT / "hivemind-models" / "output" / "datasets"

DATASET_DIR.mkdir(parents=True, exist_ok=True)

print(f"Colab: {COLAB}")
print(f"Tool config dir: {TOOL_CONFIG_DIR}")
print(f"Dataset output dir: {DATASET_DIR}")

# %% [markdown]
# ## 2. Load Tool Categories

# %%
# Load from notebook 02 output
config_path = TOOL_CONFIG_DIR / "tool_categories.yaml"

if not config_path.exists():
    print(f"WARNING: {config_path} not found. Run notebook 02 first.")
    print("Using inline defaults for demonstration.\n")

    # Inline minimal defaults so this notebook is runnable standalone
    tool_config = {
        "num_experts": 8,
        "categories": [
            {"id": i, "name": name, "description": desc, "tools": tools, "example_queries": queries}
            for i, (name, desc, tools, queries) in enumerate([
                ("web_search", "Web search and retrieval", [
                    {"type": "function", "function": {"name": "web_search", "description": "Search the web", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]}}}
                ], ["What is the latest news on AI?"]),
                ("code_generation", "Code writing and analysis", [
                    {"type": "function", "function": {"name": "generate_code", "description": "Generate code", "parameters": {"type": "object", "properties": {"task": {"type": "string", "description": "Task"}, "language": {"type": "string", "description": "Language"}}, "required": ["task", "language"]}}}
                ], ["Write a Python quicksort"]),
                ("math_computation", "Math and statistics", [
                    {"type": "function", "function": {"name": "calculate", "description": "Calculate expression", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "Math expression"}}, "required": ["expression"]}}}
                ], ["What is the integral of x^2?"]),
                ("file_data_ops", "File and data operations", [
                    {"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File path"}}, "required": ["path"]}}}
                ], ["Read the config file"]),
                ("system_shell", "System and shell ops", [
                    {"type": "function", "function": {"name": "run_command", "description": "Run shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "Command"}}, "required": ["command"]}}}
                ], ["List running processes"]),
                ("database_api", "Database and API calls", [
                    {"type": "function", "function": {"name": "sql_query", "description": "Run SQL", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "SQL query"}, "database": {"type": "string", "description": "DB name"}}, "required": ["query", "database"]}}}
                ], ["Query users table"]),
                ("image_media", "Image and media processing", [
                    {"type": "function", "function": {"name": "generate_image", "description": "Generate image", "parameters": {"type": "object", "properties": {"prompt": {"type": "string", "description": "Image prompt"}}, "required": ["prompt"]}}}
                ], ["Generate a landscape photo"]),
                ("reasoning_planning", "Reasoning and task planning", [
                    {"type": "function", "function": {"name": "create_plan", "description": "Create a plan", "parameters": {"type": "object", "properties": {"goal": {"type": "string", "description": "Goal"}}, "required": ["goal"]}}}
                ], ["Plan a product launch"]),
            ])
        ],
    }
else:
    with open(config_path) as f:
        tool_config = yaml.safe_load(f)
    print(f"Loaded tool config from {config_path}")

categories = tool_config["categories"]
print(f"\nLoaded {len(categories)} categories:")
for cat in categories:
    print(f"  Expert {cat['id']}: {cat['name']} ({len(cat['tools'])} tools)")

# %% [markdown]
# ## 3. Configure Teacher Model

# %%
# Configuration
SAMPLES_PER_CATEGORY = 500  # Adjust based on budget; 50 for testing, 500 for production
DRY_RUN = True  # Set to False to actually call the Claude API
TEACHER_MODEL = "claude-sonnet-4-20250514"  # Teacher model for generation
MAX_RETRIES = 3
BATCH_SIZE = 10  # Generate this many per API call for efficiency

# API key setup
api_key = None
if not DRY_RUN:
    if COLAB:
        try:
            from google.colab import userdata
            api_key = userdata.get("ANTHROPIC_API_KEY")
            print("API key loaded from Colab secrets.")
        except Exception as e:
            print(f"Could not load API key from Colab secrets: {e}")
    if api_key is None:
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            print("API key loaded from environment.")
        else:
            print("WARNING: No API key found. Set ANTHROPIC_API_KEY or use DRY_RUN=True.")
            DRY_RUN = True

if DRY_RUN:
    print("\n*** DRY RUN MODE: Generating synthetic samples without API calls ***")
    print(f"Samples per category: {SAMPLES_PER_CATEGORY}")
else:
    print(f"\nUsing teacher model: {TEACHER_MODEL}")
    print(f"Samples per category: {SAMPLES_PER_CATEGORY}")
    print(f"Estimated API calls: ~{len(categories) * (SAMPLES_PER_CATEGORY // BATCH_SIZE)}")

# %% [markdown]
# ## 4. Dataset Generator

# %%
class ToolCallingDatasetGenerator:
    """Generate tool-calling training data using a teacher model or synthetic generation."""

    # Diverse query templates per category for synthetic generation
    QUERY_TEMPLATES = {
        "web_search": [
            "What are the latest developments in {topic}?",
            "Find information about {topic}",
            "Search for {topic} news from this week",
            "What does the internet say about {topic}?",
            "Look up {topic} and give me a summary",
            "I need to research {topic} for a report",
            "Can you find recent articles about {topic}?",
            "What are people saying about {topic} online?",
        ],
        "code_generation": [
            "Write a {language} function that {task}",
            "Help me debug this {language} code: {code_snippet}",
            "Create a {language} class for {task}",
            "Implement {algorithm} in {language}",
            "Refactor this code to be more efficient: {code_snippet}",
            "Write unit tests for a {language} function that {task}",
            "Convert this Python code to {language}: {code_snippet}",
            "Optimize this {language} code for performance: {code_snippet}",
        ],
        "math_computation": [
            "Calculate {expression}",
            "Solve the equation {equation}",
            "What is the {measure} of {data}?",
            "Compute the derivative of {expression}",
            "Find the integral of {expression} from {a} to {b}",
            "What is {number1} {operation} {number2}?",
            "Simplify the expression {expression}",
            "Convert {value} {from_unit} to {to_unit}",
        ],
        "file_data_ops": [
            "Read the file at {path}",
            "Convert this CSV data to JSON: {data}",
            "Write {content} to {path}",
            "Parse this YAML configuration: {data}",
            "Filter this JSON data where {condition}",
            "Merge these two CSV files based on {key}",
            "Extract all email addresses from {path}",
            "Create a {format} file with {description}",
        ],
        "system_shell": [
            "Run {command} and show me the output",
            "Check the disk usage on this system",
            "List all {process_type} processes",
            "What is the value of {env_var}?",
            "Show system memory usage",
            "Find all {file_type} files in {directory}",
            "Check if port {port} is in use",
            "Show the system uptime",
        ],
        "database_api": [
            "Query the {table} table for {condition}",
            "Make a {method} request to {url}",
            "Insert a new record into {table}: {data}",
            "Call the {api_name} API to {action}",
            "Run this SQL query: {query}",
            "Fetch data from {endpoint} with parameters {params}",
            "Update records in {table} where {condition}",
            "Delete expired sessions from the {table} table",
        ],
        "image_media": [
            "Generate an image of {description}",
            "Analyze this image and tell me what you see: {image_url}",
            "Create a {style} illustration of {subject}",
            "Edit this image to {instruction}",
            "Resize this image to {width}x{height}",
            "Generate a thumbnail for {description}",
            "What objects are in this photo: {image_url}?",
            "Create a {style} version of {description}",
        ],
        "reasoning_planning": [
            "Create a plan for {goal}",
            "Break down {task} into subtasks",
            "Evaluate these options: {options}",
            "What is the best approach to {problem}?",
            "Help me prioritize these tasks: {tasks}",
            "Create a decision matrix for {decision}",
            "Outline a strategy for {goal}",
            "What are the pros and cons of {options}?",
        ],
    }

    # Diverse fill-in values
    FILL_VALUES = {
        "topic": ["quantum computing", "renewable energy", "machine learning", "climate change",
                   "cryptocurrency regulation", "space exploration", "CRISPR gene editing",
                   "autonomous vehicles", "cybersecurity threats", "nuclear fusion"],
        "language": ["Python", "JavaScript", "Rust", "Go", "TypeScript", "Java"],
        "task": ["sorts a list of integers", "validates email addresses",
                 "implements a binary search tree", "parses JSON from a string",
                 "computes fibonacci numbers", "handles HTTP requests"],
        "algorithm": ["quicksort", "Dijkstra's algorithm", "binary search",
                      "depth-first search", "dynamic programming knapsack"],
        "expression": ["x^2 + 3x - 7", "sin(x) * cos(x)", "e^(2x) + ln(x)",
                       "sqrt(x^2 + y^2)", "1/(1 + e^(-x))"],
        "equation": ["2x^2 - 5x + 3 = 0", "ln(x) = 5", "e^x = 100",
                     "sin(x) = 0.5", "x^3 - 6x^2 + 11x - 6 = 0"],
        "path": ["/data/config.yaml", "/home/user/report.csv", "/var/log/app.log",
                 "./output/results.json", "/tmp/data.xml"],
        "command": ["df -h", "ps aux", "top -bn1", "free -h", "uname -a"],
        "table": ["users", "orders", "products", "sessions", "events"],
        "description": ["a serene mountain landscape at sunset",
                        "a futuristic city skyline", "a cozy cabin in the woods"],
        "goal": ["launching a new SaaS product", "migrating to microservices",
                 "building a recommendation engine", "scaling to 1M users"],
        "style": ["photorealistic", "digital_art", "sketch", "anime", "oil_painting"],
        "code_snippet": [
            "def fib(n):\n  if n <= 1: return n\n  return fib(n-1) + fib(n-2)",
            "const data = fetch(url).then(r => r.json())",
            "for i in range(len(arr)):\n  for j in range(len(arr)):\n    if arr[i] > arr[j]: swap()",
        ],
    }

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model
        self.client = None
        if api_key:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)

    def _fill_template(self, template: str) -> str:
        """Fill a template with random values."""
        result = template
        for key, values in self.FILL_VALUES.items():
            placeholder = "{" + key + "}"
            while placeholder in result:
                result = result.replace(placeholder, random.choice(values), 1)
        # Clean up any remaining unfilled placeholders
        result = re.sub(r'\{[a-z_]+\}', 'example_value', result)
        return result

    def _generate_synthetic_sample(self, category: dict) -> dict | None:
        """Generate a single synthetic training sample without API calls."""
        cat_name = category["name"]
        tools = category["tools"]

        if not tools:
            return None

        # Pick a random tool
        tool_schema = random.choice(tools)
        func = tool_schema["function"]
        func_name = func["name"]
        params_schema = func["parameters"]
        required = params_schema.get("required", [])
        properties = params_schema.get("properties", {})

        # Generate a query from templates
        templates = self.QUERY_TEMPLATES.get(cat_name, ["Please help me with {topic}"])
        query = self._fill_template(random.choice(templates))

        # Generate plausible arguments
        arguments = {}
        for param_name, param_schema in properties.items():
            ptype = param_schema.get("type", "string")
            enum_vals = param_schema.get("enum")
            default = param_schema.get("default")

            if param_name not in required and random.random() < 0.3:
                continue  # Skip some optional params

            if enum_vals:
                arguments[param_name] = random.choice(enum_vals)
            elif default is not None and random.random() < 0.5:
                arguments[param_name] = default
            elif ptype == "string":
                # Use query content for the first required string param
                if param_name in required and not arguments:
                    arguments[param_name] = query
                else:
                    fill_key = param_name if param_name in self.FILL_VALUES else "topic"
                    arguments[param_name] = random.choice(self.FILL_VALUES.get(fill_key, ["example"]))
            elif ptype == "integer":
                arguments[param_name] = random.choice([5, 10, 20, 30, 60, 100])
            elif ptype == "number":
                arguments[param_name] = round(random.uniform(0.1, 100.0), 2)
            elif ptype == "boolean":
                arguments[param_name] = random.choice([True, False])
            elif ptype == "array":
                arguments[param_name] = random.choice([
                    ["item1", "item2", "item3"],
                    [1, 2, 3, 4, 5],
                    ["mean", "median", "std"],
                ])
            elif ptype == "object":
                arguments[param_name] = {"key": "value"}

        sample = {
            "messages": [
                {"role": "user", "content": query},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{hashlib.md5(query.encode()).hexdigest()[:12]}",
                            "type": "function",
                            "function": {
                                "name": func_name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
            ],
            "tools": [tool_schema],
            "expert_id": category["id"],
            "category": cat_name,
        }

        return sample

    def _generate_with_teacher(self, category: dict, batch_size: int = 10) -> list[dict]:
        """Generate training samples using the Claude teacher model."""
        if self.client is None:
            raise RuntimeError("No API client configured")

        cat_name = category["name"]
        tools = category["tools"]

        system_prompt = f"""You are a training data generator for a tool-calling AI model.
Your task is to generate {batch_size} diverse, realistic user queries that would require
using tools from the "{cat_name}" category.

For each query, you must also generate the correct tool call with appropriate arguments.

The available tools are:
{json.dumps(tools, indent=2)}

Output a JSON array of {batch_size} objects, each with:
- "query": a natural user query
- "tool_name": which tool to call
- "arguments": a dict of argument values

Make the queries diverse: vary complexity, phrasing, and which tool is used.
Output ONLY valid JSON, no other text."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": system_prompt}],
            )

            text = response.content[0].text
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                raw_samples = json.loads(json_match.group())
            else:
                raw_samples = json.loads(text)

            samples = []
            for raw in raw_samples:
                tool_schema = next(
                    (t for t in tools if t["function"]["name"] == raw["tool_name"]),
                    tools[0],
                )
                sample = {
                    "messages": [
                        {"role": "user", "content": raw["query"]},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call_{hashlib.md5(raw['query'].encode()).hexdigest()[:12]}",
                                    "type": "function",
                                    "function": {
                                        "name": raw["tool_name"],
                                        "arguments": json.dumps(raw["arguments"]),
                                    },
                                }
                            ],
                        },
                    ],
                    "tools": [tool_schema],
                    "expert_id": category["id"],
                    "category": cat_name,
                }
                samples.append(sample)

            return samples

        except Exception as e:
            print(f"  Teacher API error: {e}")
            return []

    def generate_dataset(
        self,
        category: dict,
        num_samples: int,
        dry_run: bool = True,
        batch_size: int = 10,
    ) -> list[dict]:
        """Generate a complete dataset for a single category."""
        samples = []
        cat_name = category["name"]

        if dry_run:
            for i in range(num_samples):
                sample = self._generate_synthetic_sample(category)
                if sample:
                    samples.append(sample)
        else:
            num_batches = (num_samples + batch_size - 1) // batch_size
            for batch_idx in range(num_batches):
                remaining = num_samples - len(samples)
                current_batch = min(batch_size, remaining)
                if current_batch <= 0:
                    break

                batch_samples = self._generate_with_teacher(category, current_batch)
                samples.extend(batch_samples)
                print(f"    Batch {batch_idx + 1}/{num_batches}: generated {len(batch_samples)} samples")

                # Rate limiting
                if batch_idx < num_batches - 1:
                    time.sleep(1.0)

        return samples[:num_samples]


print("ToolCallingDatasetGenerator ready.")

# %% [markdown]
# ## 5. Generate Datasets

# %%
from tqdm import tqdm

generator = ToolCallingDatasetGenerator(api_key=api_key, model=TEACHER_MODEL)

all_datasets = {}
generation_stats = []

print(f"Generating {SAMPLES_PER_CATEGORY} samples per category ({'dry run' if DRY_RUN else 'teacher model'})...\n")

for cat in tqdm(categories, desc="Categories"):
    cat_name = cat["name"]
    cat_id = cat["id"]

    start_time = time.time()

    dataset = generator.generate_dataset(
        category=cat,
        num_samples=SAMPLES_PER_CATEGORY,
        dry_run=DRY_RUN,
        batch_size=BATCH_SIZE,
    )

    elapsed = time.time() - start_time
    all_datasets[cat_name] = dataset

    generation_stats.append({
        "category": cat_name,
        "expert_id": cat_id,
        "samples": len(dataset),
        "time_s": elapsed,
    })

    print(f"  Expert {cat_id} ({cat_name}): {len(dataset)} samples in {elapsed:.1f}s")

total_samples = sum(len(d) for d in all_datasets.values())
print(f"\nTotal samples generated: {total_samples}")

# %% [markdown]
# ## 6. Show Sample Data

# %%
print("Sample generated data:")
print("=" * 70)

for cat_name, dataset in all_datasets.items():
    if not dataset:
        continue

    sample = dataset[0]
    user_msg = sample["messages"][0]["content"]
    tool_call = sample["messages"][1]["tool_calls"][0]
    func = tool_call["function"]

    print(f"\n--- {cat_name} (Expert {sample['expert_id']}) ---")
    print(f"  User: {user_msg[:100]}{'...' if len(user_msg) > 100 else ''}")
    print(f"  Tool: {func['name']}")
    args = json.loads(func["arguments"])
    args_str = json.dumps(args, indent=4)
    if len(args_str) > 200:
        args_str = args_str[:200] + "..."
    print(f"  Args: {args_str}")

# %% [markdown]
# ## 7. Validate Dataset Quality

# %%
from tabulate import tabulate

print("Dataset Quality Validation")
print("=" * 70)

validation_results = []

for cat_name, dataset in all_datasets.items():
    if not dataset:
        validation_results.append([cat_name, 0, 0, 0, 0, "EMPTY"])
        continue

    total = len(dataset)
    valid_json = 0
    unique_queries = set()
    unique_tools = set()
    parse_errors = 0

    for sample in dataset:
        # Check JSON parsing of tool call arguments
        try:
            tool_call = sample["messages"][1]["tool_calls"][0]
            args = json.loads(tool_call["function"]["arguments"])
            valid_json += 1
            unique_tools.add(tool_call["function"]["name"])
        except (json.JSONDecodeError, KeyError, IndexError):
            parse_errors += 1

        # Track query diversity
        query = sample["messages"][0]["content"]
        unique_queries.add(query)

    diversity = len(unique_queries) / total * 100 if total > 0 else 0
    status = "OK" if valid_json == total and diversity > 50 else "WARN"

    validation_results.append([
        cat_name,
        total,
        valid_json,
        parse_errors,
        f"{diversity:.1f}%",
        len(unique_tools),
        status,
    ])

print(tabulate(
    validation_results,
    headers=["Category", "Samples", "Valid JSON", "Parse Errors", "Query Diversity", "Unique Tools", "Status"],
    tablefmt="grid",
))

# %% [markdown]
# ## 8. Save Datasets as JSONL

# %%
print("Saving datasets...")
saved_files = []

for cat_name, dataset in all_datasets.items():
    if not dataset:
        continue

    output_path = DATASET_DIR / f"expert_{dataset[0]['expert_id']}_{cat_name}.jsonl"

    with open(output_path, "w") as f:
        for sample in dataset:
            f.write(json.dumps(sample) + "\n")

    file_size = output_path.stat().st_size
    saved_files.append({
        "path": output_path,
        "category": cat_name,
        "samples": len(dataset),
        "size_bytes": file_size,
    })
    print(f"  Saved: {output_path.name} ({len(dataset)} samples, {file_size / 1024:.1f} KB)")

# Also save a combined dataset
combined_path = DATASET_DIR / "combined_all_experts.jsonl"
with open(combined_path, "w") as f:
    for cat_name, dataset in all_datasets.items():
        for sample in dataset:
            f.write(json.dumps(sample) + "\n")

combined_size = combined_path.stat().st_size
print(f"\n  Combined: {combined_path.name} ({total_samples} samples, {combined_size / 1024:.1f} KB)")

# %% [markdown]
# ## 9. Statistics & Cost Estimate

# %%
import statistics

# Token estimation (rough: ~4 chars per token)
CHARS_PER_TOKEN = 4
INPUT_COST_PER_MTOK = 3.0   # Claude Sonnet input cost $/Mtok (approx)
OUTPUT_COST_PER_MTOK = 15.0  # Claude Sonnet output cost $/Mtok (approx)

print("Dataset Statistics")
print("=" * 70)

stats_table = []
total_tokens = 0
total_input_tokens = 0
total_output_tokens = 0

for cat_name, dataset in all_datasets.items():
    if not dataset:
        continue

    sample_lengths = []
    input_tokens = 0
    output_tokens = 0

    for sample in dataset:
        user_content = sample["messages"][0]["content"]
        tool_call = sample["messages"][1]["tool_calls"][0]
        args_str = tool_call["function"]["arguments"]

        input_len = len(user_content) + len(json.dumps(sample.get("tools", [])))
        output_len = len(tool_call["function"]["name"]) + len(args_str)

        input_tokens += input_len // CHARS_PER_TOKEN
        output_tokens += output_len // CHARS_PER_TOKEN
        sample_lengths.append(input_len + output_len)

    avg_len = statistics.mean(sample_lengths) if sample_lengths else 0
    total_input_tokens += input_tokens
    total_output_tokens += output_tokens
    total_tokens += input_tokens + output_tokens

    stats_table.append([
        cat_name,
        len(dataset),
        f"{input_tokens:,}",
        f"{output_tokens:,}",
        f"{avg_len:.0f}",
    ])

print(tabulate(
    stats_table,
    headers=["Category", "Samples", "Input Tokens (est)", "Output Tokens (est)", "Avg Length (chars)"],
    tablefmt="grid",
))

print(f"\nTotal estimated tokens: {total_tokens:,}")
print(f"  Input: {total_input_tokens:,}")
print(f"  Output: {total_output_tokens:,}")

# Cost estimate for teacher model generation (not the synthetic data itself)
if not DRY_RUN:
    input_cost = (total_input_tokens / 1_000_000) * INPUT_COST_PER_MTOK
    output_cost = (total_output_tokens / 1_000_000) * OUTPUT_COST_PER_MTOK
    total_cost = input_cost + output_cost
    print(f"\nEstimated API cost for generation:")
    print(f"  Input:  ${input_cost:.4f}")
    print(f"  Output: ${output_cost:.4f}")
    print(f"  Total:  ${total_cost:.4f}")
else:
    # Estimate what it WOULD cost with the teacher model
    est_input_tokens = SAMPLES_PER_CATEGORY * len(categories) * 500  # ~500 tokens per prompt
    est_output_tokens = SAMPLES_PER_CATEGORY * len(categories) * 200  # ~200 tokens per response
    est_input_cost = (est_input_tokens / 1_000_000) * INPUT_COST_PER_MTOK
    est_output_cost = (est_output_tokens / 1_000_000) * OUTPUT_COST_PER_MTOK
    est_total = est_input_cost + est_output_cost
    print(f"\nEstimated cost if using teacher model (Claude Sonnet):")
    print(f"  Est. input tokens:  {est_input_tokens:,}")
    print(f"  Est. output tokens: {est_output_tokens:,}")
    print(f"  Est. input cost:  ${est_input_cost:.4f}")
    print(f"  Est. output cost: ${est_output_cost:.4f}")
    print(f"  Est. total cost:  ${est_total:.4f}")

# %% [markdown]
# ## Summary
#
# This notebook has:
# 1. Loaded 8 tool categories from notebook 02
# 2. Generated **{total_samples}** training samples ({SAMPLES_PER_CATEGORY} per category)
# 3. Validated all samples (JSON parsing, diversity)
# 4. Saved per-expert JSONL files and a combined dataset
# 5. Estimated token counts and API costs
#
# **Next step**: Run `04_moe_upcycling.py` to convert the dense model to MoE.

# %%
print("\n" + "=" * 60)
print("NOTEBOOK 03 COMPLETE: Generate Datasets")
print("=" * 60)
print(f"\nTotal samples: {total_samples}")
print(f"Categories: {len(all_datasets)}")
print(f"Dataset dir: {DATASET_DIR}")
print(f"Mode: {'DRY RUN (synthetic)' if DRY_RUN else 'Teacher model (' + TEACHER_MODEL + ')'}")
print(f"\nSaved files:")
for f in saved_files:
    print(f"  {f['path'].name}: {f['samples']} samples")
print(f"\nReady for notebook 04: MoE Upcycling")
