# %% [markdown]
# # HiveMind Pipeline: Define Tool Categories & Schemas
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Escherbridge/laxame-hivemind/blob/main/hivemind-models/notebooks/02_define_tool_categories.py)
#
# This notebook defines the 8 tool categories that map to MoE experts:
# 1. Load the tool schema system
# 2. Define the 8 default tool categories with JSON schemas
# 3. Allow customization: add/remove tools, modify schemas
# 4. Visualize category-to-tool mapping
# 5. Export the final configuration as YAML
# 6. Show example prompts for each category

# %% [markdown]
# ## 1. Environment Setup

# %%
import sys
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any

COLAB = "google.colab" in str(get_ipython()) if hasattr(__builtins__, "__IPYTHON__") else False

if COLAB:
    get_ipython().system('pip install -q pyyaml tabulate')
    REPO_ROOT = Path("/content/laxame-hivemind")
    DRIVE_ROOT = Path("/content/drive/MyDrive/hivemind-pipeline")
    CHECKPOINT_DIR = DRIVE_ROOT / "checkpoints"
    OUTPUT_DIR = DRIVE_ROOT / "tool_configs"
else:
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent if "__file__" in dir() else Path.cwd().parent.parent
    CHECKPOINT_DIR = REPO_ROOT / "hivemind-models" / "output" / "checkpoints"
    OUTPUT_DIR = REPO_ROOT / "hivemind-models" / "output" / "tool_configs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS_SRC = REPO_ROOT / "hivemind-models"
if str(MODELS_SRC) not in sys.path:
    sys.path.insert(0, str(MODELS_SRC))

print(f"Colab: {COLAB}")
print(f"Output dir: {OUTPUT_DIR}")

# %% [markdown]
# ## 2. Tool Schema System

# %%
@dataclass
class ToolParameter:
    """A single parameter in a tool's input schema."""
    name: str
    type: str  # "string", "number", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: list[str] | None = None
    default: Any = None

    def to_json_schema(self) -> dict:
        schema = {"type": self.type, "description": self.description}
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class ToolDefinition:
    """A single tool that the model can call."""
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)

    def to_openai_schema(self) -> dict:
        """Convert to OpenAI function-calling format."""
        properties = {}
        required = []
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


@dataclass
class ToolCategory:
    """A category of related tools, mapped to one MoE expert."""
    id: int
    name: str
    description: str
    tools: list[ToolDefinition] = field(default_factory=list)
    example_queries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tools": [t.to_openai_schema() for t in self.tools],
            "example_queries": self.example_queries,
        }

print("Tool schema system defined.")

# %% [markdown]
# ## 3. Define the 8 Default Tool Categories
#
# Each category corresponds to one MoE expert. The categories cover a broad
# range of agentic capabilities.

# %%
# Category 0: Web Search & Retrieval
cat_web_search = ToolCategory(
    id=0,
    name="web_search",
    description="Web search, URL fetching, and information retrieval from the internet",
    tools=[
        ToolDefinition(
            name="web_search",
            description="Search the web for information on a topic",
            parameters=[
                ToolParameter("query", "string", "The search query"),
                ToolParameter("num_results", "integer", "Number of results to return", required=False, default=5),
            ],
        ),
        ToolDefinition(
            name="fetch_url",
            description="Fetch and extract text content from a URL",
            parameters=[
                ToolParameter("url", "string", "The URL to fetch"),
                ToolParameter("extract_mode", "string", "Extraction mode", required=False, enum=["text", "html", "markdown"], default="text"),
            ],
        ),
        ToolDefinition(
            name="search_news",
            description="Search recent news articles on a topic",
            parameters=[
                ToolParameter("query", "string", "News search query"),
                ToolParameter("days_back", "integer", "How many days back to search", required=False, default=7),
            ],
        ),
    ],
    example_queries=[
        "What are the latest developments in quantum computing?",
        "Find the current weather in Tokyo",
        "Search for recent papers on mixture of experts",
    ],
)

# Category 1: Code Generation & Analysis
cat_code = ToolCategory(
    id=1,
    name="code_generation",
    description="Write, analyze, debug, and execute code in various programming languages",
    tools=[
        ToolDefinition(
            name="generate_code",
            description="Generate code to solve a programming task",
            parameters=[
                ToolParameter("task", "string", "Description of what the code should do"),
                ToolParameter("language", "string", "Programming language", enum=["python", "javascript", "typescript", "rust", "go", "java"]),
                ToolParameter("style", "string", "Code style preference", required=False, enum=["concise", "verbose", "production"], default="production"),
            ],
        ),
        ToolDefinition(
            name="analyze_code",
            description="Analyze code for bugs, performance issues, or improvements",
            parameters=[
                ToolParameter("code", "string", "The code to analyze"),
                ToolParameter("language", "string", "Programming language"),
                ToolParameter("focus", "string", "Analysis focus", required=False, enum=["bugs", "performance", "security", "style", "all"], default="all"),
            ],
        ),
        ToolDefinition(
            name="execute_code",
            description="Execute code in a sandboxed environment and return output",
            parameters=[
                ToolParameter("code", "string", "Code to execute"),
                ToolParameter("language", "string", "Programming language", enum=["python", "javascript"]),
                ToolParameter("timeout", "integer", "Execution timeout in seconds", required=False, default=30),
            ],
        ),
    ],
    example_queries=[
        "Write a Python function to merge two sorted linked lists",
        "Debug this JavaScript code that fails on edge cases",
        "Analyze this Rust code for potential memory safety issues",
    ],
)

# Category 2: Math & Computation
cat_math = ToolCategory(
    id=2,
    name="math_computation",
    description="Mathematical calculations, symbolic math, statistics, and data analysis",
    tools=[
        ToolDefinition(
            name="calculate",
            description="Perform a mathematical calculation",
            parameters=[
                ToolParameter("expression", "string", "Mathematical expression to evaluate"),
                ToolParameter("precision", "integer", "Decimal precision", required=False, default=10),
            ],
        ),
        ToolDefinition(
            name="solve_equation",
            description="Solve a mathematical equation symbolically",
            parameters=[
                ToolParameter("equation", "string", "Equation to solve (use 'x' as variable)"),
                ToolParameter("variable", "string", "Variable to solve for", required=False, default="x"),
            ],
        ),
        ToolDefinition(
            name="statistics",
            description="Compute statistical measures on a dataset",
            parameters=[
                ToolParameter("data", "array", "Array of numerical values"),
                ToolParameter("measures", "array", "Which statistics to compute", required=False, default=["mean", "median", "std"]),
            ],
        ),
    ],
    example_queries=[
        "Calculate the integral of x^2 * sin(x) from 0 to pi",
        "Solve the equation 3x^2 - 7x + 2 = 0",
        "What is the standard deviation of [23, 45, 12, 67, 34, 89]?",
    ],
)

# Category 3: File & Data Operations
cat_file_data = ToolCategory(
    id=3,
    name="file_data_ops",
    description="File manipulation, data format conversion, CSV/JSON processing",
    tools=[
        ToolDefinition(
            name="read_file",
            description="Read contents of a file",
            parameters=[
                ToolParameter("path", "string", "File path to read"),
                ToolParameter("encoding", "string", "File encoding", required=False, default="utf-8"),
            ],
        ),
        ToolDefinition(
            name="write_file",
            description="Write content to a file",
            parameters=[
                ToolParameter("path", "string", "File path to write"),
                ToolParameter("content", "string", "Content to write"),
                ToolParameter("mode", "string", "Write mode", required=False, enum=["overwrite", "append"], default="overwrite"),
            ],
        ),
        ToolDefinition(
            name="convert_data",
            description="Convert data between formats (CSV, JSON, YAML, XML)",
            parameters=[
                ToolParameter("data", "string", "Input data as string"),
                ToolParameter("from_format", "string", "Source format", enum=["csv", "json", "yaml", "xml"]),
                ToolParameter("to_format", "string", "Target format", enum=["csv", "json", "yaml", "xml"]),
            ],
        ),
        ToolDefinition(
            name="query_data",
            description="Query structured data using SQL-like syntax",
            parameters=[
                ToolParameter("data", "string", "Data as JSON array of objects"),
                ToolParameter("query", "string", "SQL-like query (SELECT, WHERE, ORDER BY, LIMIT)"),
            ],
        ),
    ],
    example_queries=[
        "Read the contents of config.yaml and convert it to JSON",
        "Write a CSV file with user records",
        "Filter this JSON data to find users older than 30",
    ],
)

# Category 4: System & Shell Operations
cat_system = ToolCategory(
    id=4,
    name="system_shell",
    description="System commands, process management, environment variables, shell operations",
    tools=[
        ToolDefinition(
            name="run_command",
            description="Execute a shell command and return the output",
            parameters=[
                ToolParameter("command", "string", "Shell command to execute"),
                ToolParameter("working_dir", "string", "Working directory", required=False),
                ToolParameter("timeout", "integer", "Timeout in seconds", required=False, default=60),
            ],
        ),
        ToolDefinition(
            name="get_env",
            description="Get the value of an environment variable",
            parameters=[
                ToolParameter("name", "string", "Environment variable name"),
            ],
        ),
        ToolDefinition(
            name="list_processes",
            description="List running processes with optional filtering",
            parameters=[
                ToolParameter("filter", "string", "Process name filter", required=False),
                ToolParameter("sort_by", "string", "Sort criteria", required=False, enum=["cpu", "memory", "name", "pid"], default="memory"),
            ],
        ),
    ],
    example_queries=[
        "List all Python processes running on this system",
        "Run 'df -h' to check disk usage",
        "What is the value of the PATH environment variable?",
    ],
)

# Category 5: Database & API
cat_db_api = ToolCategory(
    id=5,
    name="database_api",
    description="Database queries, REST API calls, GraphQL, authentication",
    tools=[
        ToolDefinition(
            name="sql_query",
            description="Execute a SQL query against a connected database",
            parameters=[
                ToolParameter("query", "string", "SQL query to execute"),
                ToolParameter("database", "string", "Database connection name"),
                ToolParameter("params", "object", "Query parameters for parameterized queries", required=False),
            ],
        ),
        ToolDefinition(
            name="http_request",
            description="Make an HTTP request to an API endpoint",
            parameters=[
                ToolParameter("method", "string", "HTTP method", enum=["GET", "POST", "PUT", "PATCH", "DELETE"]),
                ToolParameter("url", "string", "Request URL"),
                ToolParameter("headers", "object", "Request headers", required=False),
                ToolParameter("body", "string", "Request body (for POST/PUT/PATCH)", required=False),
            ],
        ),
        ToolDefinition(
            name="graphql_query",
            description="Execute a GraphQL query",
            parameters=[
                ToolParameter("endpoint", "string", "GraphQL endpoint URL"),
                ToolParameter("query", "string", "GraphQL query string"),
                ToolParameter("variables", "object", "Query variables", required=False),
            ],
        ),
    ],
    example_queries=[
        "Query the users table for all admins created in the last 30 days",
        "Make a GET request to the GitHub API to list my repositories",
        "Execute a GraphQL query to fetch the latest orders",
    ],
)

# Category 6: Image & Media
cat_media = ToolCategory(
    id=6,
    name="image_media",
    description="Image generation, editing, analysis, and media processing",
    tools=[
        ToolDefinition(
            name="generate_image",
            description="Generate an image from a text description",
            parameters=[
                ToolParameter("prompt", "string", "Image description prompt"),
                ToolParameter("width", "integer", "Image width in pixels", required=False, default=1024),
                ToolParameter("height", "integer", "Image height in pixels", required=False, default=1024),
                ToolParameter("style", "string", "Art style", required=False, enum=["photorealistic", "digital_art", "sketch", "anime", "oil_painting"]),
            ],
        ),
        ToolDefinition(
            name="analyze_image",
            description="Analyze an image and describe its contents",
            parameters=[
                ToolParameter("image_url", "string", "URL or path to the image"),
                ToolParameter("questions", "array", "Specific questions about the image", required=False),
            ],
        ),
        ToolDefinition(
            name="edit_image",
            description="Edit an existing image with instructions",
            parameters=[
                ToolParameter("image_url", "string", "URL or path to the source image"),
                ToolParameter("instruction", "string", "Editing instruction"),
                ToolParameter("mask_url", "string", "URL or path to mask image (for inpainting)", required=False),
            ],
        ),
    ],
    example_queries=[
        "Generate a photorealistic image of a sunset over mountains",
        "What objects are in this image?",
        "Remove the background from this product photo",
    ],
)

# Category 7: Reasoning & Planning
cat_reasoning = ToolCategory(
    id=7,
    name="reasoning_planning",
    description="Complex reasoning, task decomposition, scheduling, and decision-making",
    tools=[
        ToolDefinition(
            name="create_plan",
            description="Create a step-by-step plan to accomplish a complex task",
            parameters=[
                ToolParameter("goal", "string", "The goal to plan for"),
                ToolParameter("constraints", "array", "Constraints to consider", required=False),
                ToolParameter("max_steps", "integer", "Maximum number of steps", required=False, default=10),
            ],
        ),
        ToolDefinition(
            name="evaluate_options",
            description="Evaluate multiple options and provide a recommendation",
            parameters=[
                ToolParameter("options", "array", "List of options to evaluate"),
                ToolParameter("criteria", "array", "Evaluation criteria"),
                ToolParameter("weights", "array", "Weight for each criterion (must sum to 1)", required=False),
            ],
        ),
        ToolDefinition(
            name="decompose_task",
            description="Break a complex task into smaller subtasks with dependencies",
            parameters=[
                ToolParameter("task", "string", "The complex task to decompose"),
                ToolParameter("detail_level", "string", "Level of decomposition detail", required=False, enum=["high", "medium", "low"], default="medium"),
            ],
        ),
    ],
    example_queries=[
        "Create a plan to migrate a monolith to microservices",
        "Evaluate these 3 cloud providers for our use case",
        "Break down building a recommendation system into subtasks",
    ],
)

# Collect all categories
ALL_CATEGORIES = [
    cat_web_search,
    cat_code,
    cat_math,
    cat_file_data,
    cat_system,
    cat_db_api,
    cat_media,
    cat_reasoning,
]

print(f"Defined {len(ALL_CATEGORIES)} tool categories:")
for cat in ALL_CATEGORIES:
    print(f"  Expert {cat.id}: {cat.name} ({len(cat.tools)} tools)")

# %% [markdown]
# ## 4. Visualize Category-to-Tool Mapping

# %%
from tabulate import tabulate

# Overview table
overview_table = []
for cat in ALL_CATEGORIES:
    tool_names = ", ".join(t.name for t in cat.tools)
    total_params = sum(len(t.parameters) for t in cat.tools)
    overview_table.append([
        f"Expert {cat.id}",
        cat.name,
        len(cat.tools),
        total_params,
        tool_names,
    ])

print(tabulate(
    overview_table,
    headers=["Expert", "Category", "Tools", "Params", "Tool Names"],
    tablefmt="grid",
    maxcolwidths=[10, 25, 8, 8, 50],
))

# %%
# Detailed tool schemas
for cat in ALL_CATEGORIES:
    print(f"\n{'=' * 70}")
    print(f"Expert {cat.id}: {cat.name}")
    print(f"Description: {cat.description}")
    print(f"{'=' * 70}")

    for tool in cat.tools:
        schema = tool.to_openai_schema()
        print(f"\n  Tool: {tool.name}")
        print(f"  Description: {tool.description}")
        print(f"  Parameters:")
        for param in tool.parameters:
            req = "required" if param.required else "optional"
            extra = ""
            if param.enum:
                extra = f", enum={param.enum}"
            if param.default is not None:
                extra += f", default={param.default}"
            print(f"    - {param.name} ({param.type}, {req}{extra})")

# %% [markdown]
# ## 5. Customize Categories (Interactive)

# %%
# Customization section -- modify categories here before exporting.
# Examples of common customizations:

def add_tool_to_category(category: ToolCategory, tool: ToolDefinition):
    """Add a tool to an existing category."""
    category.tools.append(tool)
    print(f"Added tool '{tool.name}' to category '{category.name}'")

def remove_tool_from_category(category: ToolCategory, tool_name: str):
    """Remove a tool from a category by name."""
    original_count = len(category.tools)
    category.tools = [t for t in category.tools if t.name != tool_name]
    if len(category.tools) < original_count:
        print(f"Removed tool '{tool_name}' from category '{category.name}'")
    else:
        print(f"Tool '{tool_name}' not found in category '{category.name}'")

def create_custom_category(
    cat_id: int,
    name: str,
    description: str,
    tools: list[ToolDefinition],
    example_queries: list[str],
) -> ToolCategory:
    """Create a new custom category."""
    cat = ToolCategory(
        id=cat_id,
        name=name,
        description=description,
        tools=tools,
        example_queries=example_queries,
    )
    print(f"Created custom category: Expert {cat_id} - {name}")
    return cat


# Example: Uncomment to add a custom tool
# add_tool_to_category(cat_code, ToolDefinition(
#     name="lint_code",
#     description="Run a linter on the provided code and return issues",
#     parameters=[
#         ToolParameter("code", "string", "Code to lint"),
#         ToolParameter("language", "string", "Programming language"),
#     ],
# ))

# Example: Uncomment to replace category 7 with a custom one
# ALL_CATEGORIES[7] = create_custom_category(
#     cat_id=7,
#     name="custom_tools",
#     description="Your custom tool category",
#     tools=[...],
#     example_queries=[...],
# )

print("Customization section ready. Modify the code above to customize categories.")
print(f"Current category count: {len(ALL_CATEGORIES)}")
for cat in ALL_CATEGORIES:
    print(f"  Expert {cat.id}: {cat.name} -> {len(cat.tools)} tools")

# %% [markdown]
# ## 6. Show Example Prompts

# %%
# For each category, show what a query + tool_call pair looks like in training data format
print("Example Training Data Format")
print("=" * 70)

for cat in ALL_CATEGORIES:
    print(f"\n--- Expert {cat.id}: {cat.name} ---")

    if not cat.example_queries or not cat.tools:
        print("  (no examples)")
        continue

    query = cat.example_queries[0]
    tool = cat.tools[0]

    # Build an example tool call with plausible arguments
    example_args = {}
    for param in tool.parameters:
        if param.default is not None:
            example_args[param.name] = param.default
        elif param.enum:
            example_args[param.name] = param.enum[0]
        elif param.type == "string":
            example_args[param.name] = f"<{param.name}>"
        elif param.type == "integer":
            example_args[param.name] = 5
        elif param.type == "number":
            example_args[param.name] = 1.0
        elif param.type == "boolean":
            example_args[param.name] = True
        elif param.type == "array":
            example_args[param.name] = ["item1", "item2"]
        elif param.type == "object":
            example_args[param.name] = {}

    # Fill in the first required param with the query content
    for param in tool.parameters:
        if param.required and param.type == "string":
            example_args[param.name] = query
            break

    example = {
        "messages": [
            {"role": "user", "content": query},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{cat.name}_001",
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "arguments": json.dumps(example_args),
                        },
                    }
                ],
            },
        ],
        "tools": [tool.to_openai_schema()],
        "expert_id": cat.id,
    }

    print(f"  Query: {query}")
    print(f"  Tool call: {tool.name}({json.dumps(example_args, indent=2)[:200]}...)")
    print()

# %% [markdown]
# ## 7. Export Configuration

# %%
# Export as YAML for use by downstream notebooks
config_output = {
    "version": "1.0",
    "num_experts": len(ALL_CATEGORIES),
    "categories": [cat.to_dict() for cat in ALL_CATEGORIES],
}

yaml_path = OUTPUT_DIR / "tool_categories.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(config_output, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print(f"Tool categories exported to: {yaml_path}")
print(f"File size: {yaml_path.stat().st_size / 1024:.1f} KB")

# Also save as JSON for programmatic access
json_path = OUTPUT_DIR / "tool_categories.json"
with open(json_path, "w") as f:
    json.dump(config_output, f, indent=2)

print(f"Also saved as JSON: {json_path}")

# %%
# Verify the exported config
with open(yaml_path) as f:
    loaded = yaml.safe_load(f)

print(f"\nVerification - loaded {loaded['num_experts']} categories:")
for cat_data in loaded["categories"]:
    n_tools = len(cat_data["tools"])
    print(f"  Expert {cat_data['id']}: {cat_data['name']} ({n_tools} tools)")

# %% [markdown]
# ## Summary
#
# This notebook has:
# 1. Defined the tool schema system (ToolParameter, ToolDefinition, ToolCategory)
# 2. Created **8 tool categories** mapping to 8 MoE experts:
#    - 0: Web Search & Retrieval
#    - 1: Code Generation & Analysis
#    - 2: Math & Computation
#    - 3: File & Data Operations
#    - 4: System & Shell Operations
#    - 5: Database & API
#    - 6: Image & Media
#    - 7: Reasoning & Planning
# 3. Exported configuration to YAML and JSON
#
# **Next step**: Run `03_generate_datasets.py` to generate training data for each category.

# %%
total_tools = sum(len(cat.tools) for cat in ALL_CATEGORIES)
total_params = sum(len(p) for cat in ALL_CATEGORIES for t in cat.tools for p in t.parameters)

print("\n" + "=" * 60)
print("NOTEBOOK 02 COMPLETE: Define Tool Categories")
print("=" * 60)
print(f"\nCategories: {len(ALL_CATEGORIES)}")
print(f"Total tools: {total_tools}")
print(f"Config path: {yaml_path}")
print(f"\nReady for notebook 03: Generate Datasets")
