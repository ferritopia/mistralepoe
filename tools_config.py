"""
Tool configurations for Mistral AI function calling integration.
This file contains all the tool definitions and wrapper functions for connectors.
"""

import json
import os
from typing import Dict, Any, Callable, Optional
from datetime import datetime, timedelta

# ============================================================================
# TOOL DEFINITIONS FOR MISTRAL
# ============================================================================

TOOLS_FOR_MISTRAL = [
    # Dataroom / Metabase tools
    {
        "type": "function",
        "function": {
            "name": "dataroom_list_databases",
            "description": "List available Metabase databases to find database IDs for SQL queries",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dataroom_run_sql_query",
            "description": "Execute SQL query against Metabase databases (ULI Datalake or Unilever DB). Returns up to 2000 rows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "database_id": {
                        "type": "number",
                        "description": "Numeric Metabase database ID. Use dataroom_list_databases first to find the correct ID.",
                    },
                    "sql_query": {
                        "type": "string",
                        "description": "The full SQL query to execute",
                    },
                },
                "required": ["database_id", "sql_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dataroom_get_question_results",
            "description": "Get results from a specific Metabase question by its ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "question_id": {
                        "type": "string",
                        "description": "The Metabase question ID",
                    }
                },
                "required": ["question_id"],
            },
        },
    },

    # Freshdesk tools
    {
        "type": "function",
        "function": {
            "name": "freshdesk_get_ticket",
            "description": "Get details of a specific Freshdesk ticket by ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "The Freshdesk ticket ID",
                    }
                },
                "required": ["ticket_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "freshdesk_list_tickets",
            "description": "List Freshdesk tickets with optional filters",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of tickets to return (default: 10, max: 100)",
                    },
                    "status": {
                        "type": "string",
                        "description": "Filter by status",
                        "enum": ["Open", "Pending", "Resolved", "Closed"],
                    },
                    "priority": {
                        "type": "string",
                        "description": "Filter by priority",
                        "enum": ["Low", "Medium", "High", "Urgent"],
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "freshdesk_create_ticket",
            "description": "Create a new Freshdesk ticket",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "Ticket subject",
                    },
                    "description": {
                        "type": "string",
                        "description": "Ticket description",
                    },
                    "priority": {
                        "type": "string",
                        "description": "Ticket priority",
                        "enum": ["Low", "Medium", "High", "Urgent"],
                        "default": "Medium",
                    },
                    "status": {
                        "type": "string",
                        "description": "Initial ticket status",
                        "enum": ["Open", "Pending"],
                        "default": "Open",
                    },
                },
                "required": ["subject", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "freshdesk_update_ticket",
            "description": "Update an existing Freshdesk ticket",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "The ticket ID to update",
                    },
                    "status": {
                        "type": "string",
                        "description": "New status",
                        "enum": ["Open", "Pending", "Resolved", "Closed"],
                    },
                    "priority": {
                        "type": "string",
                        "description": "New priority",
                        "enum": ["Low", "Medium", "High", "Urgent"],
                    },
                    "response": {
                        "type": "string",
                        "description": "Response to add to the ticket",
                    },
                },
                "required": ["ticket_id"],
            },
        },
    },

    # Gmail tools
    {
        "type": "function",
        "function": {
            "name": "gmail_search_threads",
            "description": "Search Gmail threads by query. Returns thread summaries with message snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Gmail search query (e.g., 'from:john@example.com')",
                    },
                    "pageSize": {
                        "type": "number",
                        "description": "Maximum threads to return (1-50, default: 20)",
                        "minimum": 1,
                        "maximum": 50,
                    },
                    "includeTrash": {
                        "type": "boolean",
                        "description": "Include threads from TRASH (default: false)",
                    },
                    "pageToken": {
                        "type": "string",
                        "description": "Token for pagination",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gmail_get_thread",
            "description": "Get full details of a Gmail thread including all messages and attachments",
            "parameters": {
                "type": "object",
                "properties": {
                    "thread_id": {
                        "type": "string",
                        "description": "The Gmail thread ID",
                    }
                },
                "required": ["thread_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gmail_create_draft",
            "description": "Create a new email draft in Gmail",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Comma-separated list of recipient email addresses",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body (plain text)",
                    },
                    "cc": {
                        "type": "string",
                        "description": "Comma-separated list of CC recipients",
                    },
                    "bcc": {
                        "type": "string",
                        "description": "Comma-separated list of BCC recipients",
                    },
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gmail_list_labels",
            "description": "List all Gmail labels with their IDs",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },

    # Google Calendar tools
    {
        "type": "function",
        "function": {
            "name": "google_calendar_list_events",
            "description": "List calendar events. Can filter by date range, calendar ID, and search query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "calendarId": {
                        "type": "string",
                        "description": "Calendar ID to list events from (default: primary calendar)",
                    },
                    "startTime": {
                        "type": "string",
                        "description": "Lower bound for event end time (ISO 8601). Only events ending after this time are returned.",
                    },
                    "endTime": {
                        "type": "string",
                        "description": "Upper bound for event start time (ISO 8601). Only events starting before this time are returned.",
                    },
                    "timeZone": {
                        "type": "string",
                        "description": "Time zone for response (IANA format, e.g., 'Asia/Jakarta')",
                    },
                    "pageSize": {
                        "type": "number",
                        "description": "Maximum events to return (1-250, default: 10)",
                    },
                    "fullText": {
                        "type": "string",
                        "description": "Free-form search query across title, description, location, and attendees",
                    },
                    "orderBy": {
                        "type": "string",
                        "description": "Order of results",
                        "enum": ["default", "startTime", "startTimeDesc", "lastModified"],
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "google_calendar_get_event",
            "description": "Get details of a specific calendar event by ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {
                        "type": "string",
                        "description": "The calendar event ID",
                    },
                    "calendarId": {
                        "type": "string",
                        "description": "Calendar ID (default: primary)",
                    },
                },
                "required": ["event_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "google_calendar_create_event",
            "description": "Create a new calendar event",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Event title",
                    },
                    "description": {
                        "type": "string",
                        "description": "Event description",
                    },
                    "startTime": {
                        "type": "string",
                        "description": "Event start time (ISO 8601, e.g., '2026-07-22T14:00:00+07:00')",
                    },
                    "endTime": {
                        "type": "string",
                        "description": "Event end time (ISO 8601)",
                    },
                    "timeZone": {
                        "type": "string",
                        "description": "Time zone (IANA format)",
                        "default": "Asia/Jakarta",
                    },
                    "location": {
                        "type": "string",
                        "description": "Event location",
                    },
                    "attendees": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of attendee email addresses",
                    },
                },
                "required": ["summary", "startTime", "endTime"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "google_calendar_update_event",
            "description": "Update an existing calendar event",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {
                        "type": "string",
                        "description": "The event ID to update",
                    },
                    "calendarId": {
                        "type": "string",
                        "description": "Calendar ID (default: primary)",
                    },
                    "summary": {
                        "type": "string",
                        "description": "New event title",
                    },
                    "description": {
                        "type": "string",
                        "description": "New event description",
                    },
                    "startTime": {
                        "type": "string",
                        "description": "New start time (ISO 8601)",
                    },
                    "endTime": {
                        "type": "string",
                        "description": "New end time (ISO 8601)",
                    },
                },
                "required": ["event_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "google_calendar_delete_event",
            "description": "Delete a calendar event",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {
                        "type": "string",
                        "description": "The event ID to delete",
                    },
                    "calendarId": {
                        "type": "string",
                        "description": "Calendar ID (default: primary)",
                    },
                },
                "required": ["event_id"],
            },
        },
    },

    # MileApp Unilever tools
    {
        "type": "function",
        "function": {
            "name": "mileapp_get_tasks",
            "description": "Get list of tasks from MileApp Unilever",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of tasks to return",
                    },
                    "status": {
                        "type": "string",
                        "description": "Filter by task status",
                    },
                    "assignee": {
                        "type": "string",
                        "description": "Filter by assignee user ID or email",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mileapp_get_task_detail",
            "description": "Get detailed information about a specific task",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task ID",
                    }
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mileapp_get_users",
            "description": "Get list of users from MileApp Unilever",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of users to return",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mileapp_get_flows",
            "description": "Get list of flows/workflows from MileApp Unilever",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of flows to return",
                    },
                },
                "required": [],
            },
        },
    },

    # Sentry tools
    {
        "type": "function",
        "function": {
            "name": "sentry_search_issues",
            "description": "Search Sentry issues by query, project, status, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for issues",
                    },
                    "project": {
                        "type": "string",
                        "description": "Filter by project slug",
                    },
                    "status": {
                        "type": "string",
                        "description": "Filter by status",
                        "enum": ["unresolved", "resolved", "ignored"],
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of issues to return (default: 20, max: 100)",
                    },
                    "sort": {
                        "type": "string",
                        "description": "Sort by field",
                        "enum": ["date", "freq", "priority", "last_seen"],
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sentry_search_events",
            "description": "Search Sentry error events",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_id": {
                        "type": "string",
                        "description": "Filter by issue ID",
                    },
                    "project": {
                        "type": "string",
                        "description": "Filter by project slug",
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum events to return (default: 20, max: 100)",
                    },
                    "start_time": {
                        "type": "string",
                        "description": "Start time for events (ISO 8601)",
                    },
                    "end_time": {
                        "type": "string",
                        "description": "End time for events (ISO 8601)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sentry_update_issue",
            "description": "Update a Sentry issue (assign, change status, add comment)",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_id": {
                        "type": "string",
                        "description": "The Sentry issue ID",
                    },
                    "status": {
                        "type": "string",
                        "description": "New status",
                        "enum": ["unresolved", "resolved", "ignored"],
                    },
                    "assignee": {
                        "type": "string",
                        "description": "User ID or email to assign to",
                    },
                    "comment": {
                        "type": "string",
                        "description": "Comment to add",
                    },
                },
                "required": ["issue_id"],
            },
        },
    },

    # WhatsApp tools
    {
        "type": "function",
        "function": {
            "name": "whatsapp_archive_chat",
            "description": "Archive a WhatsApp chat or group",
            "parameters": {
                "type": "object",
                "properties": {
                    "chat_id": {
                        "type": "string",
                        "description": "The WhatsApp chat or group ID",
                    }
                },
                "required": ["chat_id"],
            },
        },
    },
]

# ============================================================================
# FUNCTION WRAPPERS FOR CONNECTOR TOOLS
# ============================================================================

async def call_dataroom_list_databases() -> str:
    try:
        from tools.dataroom import Get_many_databases_in_Metabase
        result = await Get_many_databases_in_Metabase()
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

async def call_dataroom_run_sql_query(database_id: int, sql_query: str) -> str:
    try:
        from tools.dataroom import Run_SQL_Query
        result = await Run_SQL_Query({
            "database_id": database_id,
            "sql_query": sql_query,
            "parameters1_Value": ""
        })
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

async def call_dataroom_get_question_results(question_id: str) -> str:
    try:
        from tools.dataroom import Get_the_results_from_a_question_in_Metabase
        result = await Get_the_results_from_a_question_in_Metabase({"question_id": question_id})
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

async def call_freshdesk_get_ticket(ticket_id: str) -> str:
    try:
        from tools.freshdesk import Get_a_ticket
        result = await Get_a_ticket({"Ticket_ID": ticket_id})
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

async def call_freshdesk_list_tickets(
    limit: Optional[int] = 10,
    status: Optional[str] = None,
    priority: Optional[str] = None
) -> str:
    try:
        from tools.freshdesk import Get_many_tickets
        params = {"limit": limit}
        if status: params["status"] = status
        if priority: params["priority"] = priority
        result = await Get_many_tickets(params)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

async def call_freshdesk_create_ticket(
    subject: str,
    description: str,
    priority: str = "Medium",
    status: str = "Open"
) -> str:
    try:
        from tools.freshdesk import Create_a_ticket
        result = await Create_a_ticket({
            "subject": subject,
            "description": description,
            "priority": priority,
            "status": status
        })
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

async def call_freshdesk_update_ticket(
    ticket_id: str,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    response: Optional[str] = None
) -> str:
    try:
        from tools.freshdesk import Update_a_ticket
        params = {"ticket_id": ticket_id}
        if status: params["status"] = status
        if priority: params["priority"] = priority
        if response: params["response"] = response
        result = await Update_a_ticket(params)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

async def call_gmail_search_threads(
    query: Optional[str] = None,
    pageSize: Optional[int] = 20,
    includeTrash: Optional[bool] = False,
    pageToken: Optional[str] = None
) -> str:
    try:
        from tools.gmail import search_threads
        params = {}
        if query: params["query"] = query
        if pageSize: params["pageSize"] = pageSize
        if includeTrash: params["includeTrash"] = includeTrash
        if pageToken: params["pageToken"] = pageToken
        result = await search_threads(params)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

async def call_gmail_get_thread(thread_id: str) -> str:
    try:
        from tools.gmail import get_thread
        result = await get_thread({"thread_id": thread_id, "format": "FULL_CONTENT"})
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

async def call_gmail_create_draft(
    to: str,
    subject: str,
    body: str,
    cc: Optional[str] = None,
    bcc: Optional[str] = None
) -> str:
    try:
        from tools.gmail import create_draft
        params = {"to": to, "subject": subject, "body": body}
        if cc: params["cc"] = cc
        if bcc: params["bcc"] = bcc
        result = await create_draft(params)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

async def call_gmail_list_labels() -> str:
    try:
        from tools.gmail import list_labels
        result = await list_labels()
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

async def call_google_calendar_list_events(
    calendarId: Optional[str] = None,
    startTime: Optional[str] = None,
    endTime: Optional[str] = None,
    timeZone: Optional[str] = "Asia/Jakarta",
    pageSize: Optional[int] = 10,
    fullText: Optional[str] = None,
    orderBy: Optional[str] = "startTime"
) -> str:
    try:
        from tools.google_calendar import list_events
        params = {"timeZone": timeZone, "pageSize": pageSize, "orderBy": orderBy}
        if calendarId: params["calendarId"] = calendarId
        if startTime: params["startTime"] = startTime
        if endTime: params["endTime"] = endTime
        if fullText: params["fullText"] = fullText
        result = await list_events(params)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

async def call_google_calendar_get_event(
    event_id: str,
    calendarId: Optional[str] = None
) -> str:
    try:
        from tools.google_calendar import get_event
        params = {"event_id": event_id}
        if calendarId: params["calendarId"] = calendarId
        result = await get_event(params)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

async def call_google_calendar_create_event(
    summary: str,
    startTime: str,
    endTime: str,
    description: Optional[str] = None,
    location: Optional[str] = None,
    timeZone: str = "Asia/Jakarta",
    attendees: Optional[list] = None,
    calendarId: Optional[str] = None
) -> str:
    try:
        from tools.google_calendar import create_event
        event_data = {
            "summary": summary,
            "start": {"dateTime": startTime, "timeZone": timeZone},
            "end": {"dateTime": endTime, "timeZone": timeZone},
        }
        if description: event_data["description"] = description
        if location: event_data["location"] = location
        if attendees: event_data["attendees"] = [{"email": a} for a in attendees]

        params = {"event": event_data}
        if calendarId: params["calendarId"] = calendarId
        result = await create_event(params)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

async def call_google_calendar_update_event(
    event_id: str,
    calendarId: Optional[str] = None,
    summary: Optional[str] = None,
    description: Optional[str] = None,
    startTime: Optional[str] = None,
    endTime: Optional[str] = None
) -> str:
    try:
        from tools.google_calendar import update_event
        update_data = {}
        if summary: update_data["summary"] = summary
        if description: update_data["description"] = description
        if startTime: update_data["start"] = {"dateTime": startTime}
        if endTime: update_data["end"] = {"dateTime": endTime}

        params = {"event_id": event_id, "event": update_data}
        if calendarId: params["calendarId"] = calendarId
        result = await update_event(params)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

async def call_google_calendar_delete_event(
    event_id: str,
    calendarId: Optional[str] = None
) -> str:
    try:
        from tools.google_calendar import delete_event
        params = {"event_id": event_id}
        if calendarId: params["calendarId"] = calendarId
        result = await delete_event(params)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

async def call_mileapp_get_tasks(
    limit: Optional[int] = None,
    status: Optional[str] = None,
    assignee: Optional[str] = None
) -> str:
    try:
        from tools.mileapp_unilever import Get_Tasks
        params = {}
        if limit: params["limit"] = limit
        if status: params["status"] = status
        if assignee: params["assignee"] = assignee
        result = await Get_Tasks(params)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

async def call_mileapp_get_task_detail(task_id: str) -> str:
    try:
        from tools.mileapp_unilever import Get_Task_Detail
        result = await Get_Task_Detail({"task_id": task_id})
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

async def call_mileapp_get_users(limit: Optional[int] = None) -> str:
    try:
        from tools.mileapp_unilever import Get_Users
        params = {}
        if limit: params["limit"] = limit
        result = await Get_Users(params)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

async def call_mileapp_get_flows(limit: Optional[int] = None) -> str:
    try:
        from tools.mileapp_unilever import Get_Flows
        params = {}
        if limit: params["limit"] = limit
        result = await Get_Flows(params)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

async def call_sentry_search_issues(
    query: Optional[str] = None,
    project: Optional[str] = None,
    status: Optional[str] = None,
    limit: Optional[int] = 20,
    sort: Optional[str] = "date"
) -> str:
    try:
        from tools.sentry import search_issues
        params = {"limit": limit, "sort": sort}
        if query: params["query"] = query
        if project: params["project"] = project
        if status: params["status"] = status
        result = await search_issues(params)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

async def call_sentry_search_events(
    issue_id: Optional[str] = None,
    project: Optional[str] = None,
    query: Optional[str] = None,
    limit: Optional[int] = 20,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
) -> str:
    try:
        from tools.sentry import search_events
        params = {"limit": limit}
        if issue_id: params["issue_id"] = issue_id
        if project: params["project"] = project
        if query: params["query"] = query
        if start_time: params["start_time"] = start_time
        if end_time: params["end_time"] = end_time
        result = await search_events(params)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

async def call_sentry_update_issue(
    issue_id: str,
    status: Optional[str] = None,
    assignee: Optional[str] = None,
    comment: Optional[str] = None
) -> str:
    try:
        from tools.sentry import update_issue
        params = {"issue_id": issue_id}
        if status: params["status"] = status
        if assignee: params["assignee"] = assignee
        if comment: params["comment"] = comment
        result = await update_issue(params)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

async def call_whatsapp_archive_chat(chat_id: str) -> str:
    try:
        from tools.whatsapp import WhatsApp_Archive
        result = await WhatsApp_Archive({"chat_id": chat_id})
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

# ============================================================================
# FUNCTION NAME TO CALLABLE MAPPING
# ============================================================================

FUNCTION_MAP = {
    "dataroom_list_databases": call_dataroom_list_databases,
    "dataroom_run_sql_query": call_dataroom_run_sql_query,
    "dataroom_get_question_results": call_dataroom_get_question_results,
    "freshdesk_get_ticket": call_freshdesk_get_ticket,
    "freshdesk_list_tickets": call_freshdesk_list_tickets,
    "freshdesk_create_ticket": call_freshdesk_create_ticket,
    "freshdesk_update_ticket": call_freshdesk_update_ticket,
    "gmail_search_threads": call_gmail_search_threads,
    "gmail_get_thread": call_gmail_get_thread,
    "gmail_create_draft": call_gmail_create_draft,
    "gmail_list_labels": call_gmail_list_labels,
    "google_calendar_list_events": call_google_calendar_list_events,
    "google_calendar_get_event": call_google_calendar_get_event,
    "google_calendar_create_event": call_google_calendar_create_event,
    "google_calendar_update_event": call_google_calendar_update_event,
    "google_calendar_delete_event": call_google_calendar_delete_event,
    "mileapp_get_tasks": call_mileapp_get_tasks,
    "mileapp_get_task_detail": call_mileapp_get_task_detail,
    "mileapp_get_users": call_mileapp_get_users,
    "mileapp_get_flows": call_mileapp_get_flows,
    "sentry_search_issues": call_sentry_search_issues,
    "sentry_search_events": call_sentry_search_events,
    "sentry_update_issue": call_sentry_update_issue,
    "whatsapp_archive_chat": call_whatsapp_archive_chat,
}

def get_tools_for_mistral():
    return TOOLS_FOR_MISTRAL

def get_function_map():
    return FUNCTION_MAP
