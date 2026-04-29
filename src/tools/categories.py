"""
Default tool categories with realistic tool schemas.

Provides 8 pre-built categories that cover common LLM tool-calling domains.
Import `DEFAULT_CATEGORIES` for direct use or call `build_default_registry()`
to get a populated ToolRegistry.
"""

from __future__ import annotations

from src.tools.schema import ToolCategory, ToolRegistry, ToolSchema

# ---------------------------------------------------------------------------
# Helper to cut boilerplate when defining JSON-Schema parameter blocks
# ---------------------------------------------------------------------------

def _obj(props: dict, required: list[str] | None = None) -> dict:
    """Shorthand for a JSON-Schema 'object' with given properties."""
    schema: dict = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


# ===================================================================
# 1. web_search
# ===================================================================

_web_search_tools = [
    ToolSchema(
        name="web_search",
        description="Search the web for information matching a query.",
        parameters=_obj(
            {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "description": "Max results to return", "default": 10},
                "language": {"type": "string", "description": "ISO 639-1 language code", "default": "en"},
            },
            required=["query"],
        ),
    ),
    ToolSchema(
        name="fetch_url",
        description="Fetch the raw content of a URL.",
        parameters=_obj(
            {
                "url": {"type": "string", "description": "URL to fetch"},
                "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
            },
            required=["url"],
        ),
    ),
    ToolSchema(
        name="scrape_page",
        description="Extract structured text content from a web page.",
        parameters=_obj(
            {
                "url": {"type": "string", "description": "Page URL"},
                "selector": {"type": "string", "description": "Optional CSS selector to narrow extraction"},
                "include_links": {"type": "boolean", "description": "Include hyperlinks", "default": False},
            },
            required=["url"],
        ),
    ),
]

WEB_SEARCH = ToolCategory(
    name="web_search",
    description="Tools for searching the web, fetching pages, and scraping content.",
    tools=_web_search_tools,
)

# ===================================================================
# 2. code_execution
# ===================================================================

_code_execution_tools = [
    ToolSchema(
        name="run_python",
        description="Execute a Python code snippet and return stdout/stderr.",
        parameters=_obj(
            {
                "code": {"type": "string", "description": "Python source code to execute"},
                "timeout": {"type": "integer", "description": "Execution timeout in seconds", "default": 60},
            },
            required=["code"],
        ),
    ),
    ToolSchema(
        name="run_bash",
        description="Execute a bash command and return stdout/stderr.",
        parameters=_obj(
            {
                "command": {"type": "string", "description": "Bash command to run"},
                "timeout": {"type": "integer", "description": "Execution timeout in seconds", "default": 60},
            },
            required=["command"],
        ),
    ),
    ToolSchema(
        name="write_file",
        description="Write content to a file at the given path.",
        parameters=_obj(
            {
                "path": {"type": "string", "description": "File path"},
                "content": {"type": "string", "description": "Content to write"},
                "overwrite": {"type": "boolean", "description": "Overwrite if exists", "default": False},
            },
            required=["path", "content"],
        ),
    ),
    ToolSchema(
        name="read_file",
        description="Read the content of a file.",
        parameters=_obj(
            {
                "path": {"type": "string", "description": "File path"},
                "max_lines": {"type": "integer", "description": "Maximum lines to read"},
            },
            required=["path"],
        ),
    ),
]

CODE_EXECUTION = ToolCategory(
    name="code_execution",
    description="Tools for running code, reading and writing files.",
    tools=_code_execution_tools,
)

# ===================================================================
# 3. database
# ===================================================================

_database_tools = [
    ToolSchema(
        name="sql_query",
        description="Execute a read-only SQL query and return rows.",
        parameters=_obj(
            {
                "query": {"type": "string", "description": "SQL SELECT statement"},
                "database": {"type": "string", "description": "Database name or connection alias"},
                "limit": {"type": "integer", "description": "Max rows to return", "default": 100},
            },
            required=["query", "database"],
        ),
    ),
    ToolSchema(
        name="create_table",
        description="Create a new table in the database.",
        parameters=_obj(
            {
                "database": {"type": "string", "description": "Database name"},
                "table_name": {"type": "string", "description": "Name for the new table"},
                "columns": {
                    "type": "array",
                    "description": "Column definitions",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string"},
                            "nullable": {"type": "boolean", "default": True},
                        },
                        "required": ["name", "type"],
                    },
                },
            },
            required=["database", "table_name", "columns"],
        ),
    ),
    ToolSchema(
        name="insert_row",
        description="Insert a single row into a table.",
        parameters=_obj(
            {
                "database": {"type": "string", "description": "Database name"},
                "table_name": {"type": "string", "description": "Table name"},
                "row": {"type": "object", "description": "Column-value mapping for the row"},
            },
            required=["database", "table_name", "row"],
        ),
    ),
    ToolSchema(
        name="list_tables",
        description="List all tables in a database.",
        parameters=_obj(
            {
                "database": {"type": "string", "description": "Database name"},
            },
            required=["database"],
        ),
    ),
]

DATABASE = ToolCategory(
    name="database",
    description="Tools for querying and managing relational databases.",
    tools=_database_tools,
)

# ===================================================================
# 4. api_calls
# ===================================================================

_api_calls_tools = [
    ToolSchema(
        name="http_get",
        description="Perform an HTTP GET request.",
        parameters=_obj(
            {
                "url": {"type": "string", "description": "Request URL"},
                "headers": {"type": "object", "description": "Optional headers"},
                "params": {"type": "object", "description": "Query string parameters"},
            },
            required=["url"],
        ),
    ),
    ToolSchema(
        name="http_post",
        description="Perform an HTTP POST request with a JSON body.",
        parameters=_obj(
            {
                "url": {"type": "string", "description": "Request URL"},
                "body": {"type": "object", "description": "JSON body payload"},
                "headers": {"type": "object", "description": "Optional headers"},
            },
            required=["url", "body"],
        ),
    ),
    ToolSchema(
        name="parse_json",
        description="Parse a JSON string and return the structured data.",
        parameters=_obj(
            {
                "json_string": {"type": "string", "description": "Raw JSON text to parse"},
                "jq_filter": {"type": "string", "description": "Optional jq-style filter expression"},
            },
            required=["json_string"],
        ),
    ),
    ToolSchema(
        name="authenticate",
        description="Authenticate with an API using credentials and return an access token.",
        parameters=_obj(
            {
                "provider": {"type": "string", "description": "Auth provider name (e.g. oauth2, api_key)"},
                "credentials": {"type": "object", "description": "Provider-specific credential fields"},
                "scopes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Requested permission scopes",
                },
            },
            required=["provider", "credentials"],
        ),
    ),
]

API_CALLS = ToolCategory(
    name="api_calls",
    description="Tools for making HTTP requests, parsing responses, and managing authentication.",
    tools=_api_calls_tools,
)

# ===================================================================
# 5. math_compute
# ===================================================================

_math_compute_tools = [
    ToolSchema(
        name="calculate",
        description="Evaluate a mathematical expression and return the result.",
        parameters=_obj(
            {
                "expression": {"type": "string", "description": "Math expression (e.g. '2*pi*r')"},
                "variables": {"type": "object", "description": "Variable-value mapping"},
                "precision": {"type": "integer", "description": "Decimal places in result", "default": 10},
            },
            required=["expression"],
        ),
    ),
    ToolSchema(
        name="solve_equation",
        description="Solve a symbolic equation for a given variable.",
        parameters=_obj(
            {
                "equation": {"type": "string", "description": "Equation string (e.g. 'x**2 - 4 = 0')"},
                "variable": {"type": "string", "description": "Variable to solve for", "default": "x"},
            },
            required=["equation"],
        ),
    ),
    ToolSchema(
        name="plot_function",
        description="Plot a mathematical function over a range and return the image.",
        parameters=_obj(
            {
                "expression": {"type": "string", "description": "Function of x to plot"},
                "x_min": {"type": "number", "description": "Lower bound of x", "default": -10},
                "x_max": {"type": "number", "description": "Upper bound of x", "default": 10},
                "title": {"type": "string", "description": "Plot title"},
            },
            required=["expression"],
        ),
    ),
    ToolSchema(
        name="statistics",
        description="Compute descriptive statistics for a dataset.",
        parameters=_obj(
            {
                "data": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Numeric data points",
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Metrics to compute (mean, median, std, min, max, etc.)",
                    "default": ["mean", "median", "std"],
                },
            },
            required=["data"],
        ),
    ),
]

MATH_COMPUTE = ToolCategory(
    name="math_compute",
    description="Tools for mathematical calculation, symbolic solving, plotting, and statistics.",
    tools=_math_compute_tools,
)

# ===================================================================
# 6. file_io
# ===================================================================

_file_io_tools = [
    ToolSchema(
        name="file_read",
        description="Read the content of a file on the local filesystem.",
        parameters=_obj(
            {
                "path": {"type": "string", "description": "Absolute or relative file path"},
                "encoding": {"type": "string", "description": "Text encoding", "default": "utf-8"},
            },
            required=["path"],
        ),
    ),
    ToolSchema(
        name="file_write",
        description="Write content to a file, creating directories as needed.",
        parameters=_obj(
            {
                "path": {"type": "string", "description": "Destination file path"},
                "content": {"type": "string", "description": "Content to write"},
                "append": {"type": "boolean", "description": "Append instead of overwrite", "default": False},
            },
            required=["path", "content"],
        ),
    ),
    ToolSchema(
        name="list_directory",
        description="List files and directories at the given path.",
        parameters=_obj(
            {
                "path": {"type": "string", "description": "Directory path"},
                "recursive": {"type": "boolean", "description": "List recursively", "default": False},
                "pattern": {"type": "string", "description": "Glob pattern filter"},
            },
            required=["path"],
        ),
    ),
    ToolSchema(
        name="search_files",
        description="Search for files whose name or content matches a pattern.",
        parameters=_obj(
            {
                "directory": {"type": "string", "description": "Root directory to search"},
                "pattern": {"type": "string", "description": "Regex pattern to match"},
                "search_content": {"type": "boolean", "description": "Search file contents too", "default": False},
            },
            required=["directory", "pattern"],
        ),
    ),
]

FILE_IO = ToolCategory(
    name="file_io",
    description="Tools for reading, writing, listing, and searching local files.",
    tools=_file_io_tools,
)

# ===================================================================
# 7. communication
# ===================================================================

_communication_tools = [
    ToolSchema(
        name="send_email",
        description="Send an email to one or more recipients.",
        parameters=_obj(
            {
                "to": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Recipient email addresses",
                },
                "subject": {"type": "string", "description": "Email subject line"},
                "body": {"type": "string", "description": "Email body (plain text or HTML)"},
                "cc": {"type": "array", "items": {"type": "string"}, "description": "CC addresses"},
            },
            required=["to", "subject", "body"],
        ),
    ),
    ToolSchema(
        name="send_message",
        description="Send an instant message to a user or channel.",
        parameters=_obj(
            {
                "channel": {"type": "string", "description": "Channel or user ID"},
                "text": {"type": "string", "description": "Message text"},
                "platform": {"type": "string", "description": "Messaging platform (slack, teams, discord)", "default": "slack"},
            },
            required=["channel", "text"],
        ),
    ),
    ToolSchema(
        name="schedule_meeting",
        description="Create a calendar event / meeting.",
        parameters=_obj(
            {
                "title": {"type": "string", "description": "Meeting title"},
                "start_time": {"type": "string", "description": "ISO 8601 start datetime"},
                "duration_minutes": {"type": "integer", "description": "Duration in minutes", "default": 30},
                "attendees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Attendee email addresses",
                },
                "description": {"type": "string", "description": "Meeting description"},
            },
            required=["title", "start_time"],
        ),
    ),
    ToolSchema(
        name="create_reminder",
        description="Set a reminder for the user at a specified time.",
        parameters=_obj(
            {
                "message": {"type": "string", "description": "Reminder text"},
                "remind_at": {"type": "string", "description": "ISO 8601 datetime for the reminder"},
                "recurring": {"type": "string", "description": "Recurrence rule (daily, weekly, etc.)"},
            },
            required=["message", "remind_at"],
        ),
    ),
]

COMMUNICATION = ToolCategory(
    name="communication",
    description="Tools for email, messaging, calendar, and reminders.",
    tools=_communication_tools,
)

# ===================================================================
# 8. knowledge
# ===================================================================

_knowledge_tools = [
    ToolSchema(
        name="search_docs",
        description="Search a document corpus or knowledge base.",
        parameters=_obj(
            {
                "query": {"type": "string", "description": "Semantic search query"},
                "collection": {"type": "string", "description": "Document collection name"},
                "top_k": {"type": "integer", "description": "Number of results", "default": 5},
            },
            required=["query"],
        ),
    ),
    ToolSchema(
        name="summarize",
        description="Generate a concise summary of a given text.",
        parameters=_obj(
            {
                "text": {"type": "string", "description": "Text to summarize"},
                "max_length": {"type": "integer", "description": "Max summary length in words", "default": 150},
                "style": {"type": "string", "description": "Summary style (bullets, paragraph, tldr)", "default": "paragraph"},
            },
            required=["text"],
        ),
    ),
    ToolSchema(
        name="translate",
        description="Translate text between languages.",
        parameters=_obj(
            {
                "text": {"type": "string", "description": "Text to translate"},
                "source_language": {"type": "string", "description": "Source language code (e.g. en)"},
                "target_language": {"type": "string", "description": "Target language code (e.g. fr)"},
            },
            required=["text", "target_language"],
        ),
    ),
    ToolSchema(
        name="extract_entities",
        description="Extract named entities (people, places, organisations, dates) from text.",
        parameters=_obj(
            {
                "text": {"type": "string", "description": "Text to analyse"},
                "entity_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Entity types to extract",
                    "default": ["PERSON", "ORG", "GPE", "DATE"],
                },
            },
            required=["text"],
        ),
    ),
]

KNOWLEDGE = ToolCategory(
    name="knowledge",
    description="Tools for document search, summarisation, translation, and entity extraction.",
    tools=_knowledge_tools,
)


# ===================================================================
# Convenience aggregates
# ===================================================================

DEFAULT_CATEGORIES: list[ToolCategory] = [
    WEB_SEARCH,
    CODE_EXECUTION,
    DATABASE,
    API_CALLS,
    MATH_COMPUTE,
    FILE_IO,
    COMMUNICATION,
    KNOWLEDGE,
]
"""All eight default tool categories."""


def build_default_registry() -> ToolRegistry:
    """Return a ToolRegistry populated with every default category."""
    registry = ToolRegistry()
    for cat in DEFAULT_CATEGORIES:
        registry.register_category(cat)
    return registry
