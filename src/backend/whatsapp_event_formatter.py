"""
WhatsApp Event Formatter - Dict-based orchestration event processing.

Converts orchestration events into OpenAI/LiteLLM message dicts for WhatsApp
agent conversations.  No LangChain dependency — all output is plain dicts.

Ported from the LangChain-based WhatsAppEventFormatter in chask_lambdas.

Key Features:
- Registry-based event handling (easy to add/remove event types)
- Configurable event filtering via enabled_events parameter
- Proper tool call/response pairing for LLM compatibility
- Batch tool execution support
- Clean separation of concerns with individual handler methods
"""

import json
import logging
import re
from typing import Dict, List, Tuple, Set, Optional, Any, Callable

logger = logging.getLogger(__name__)

# Type aliases
EventDict = Dict[str, Any]
MessageDict = Dict[str, Any]
ChannelMap = Dict[str, Tuple[int, str]]
HandlerFunc = Callable[[EventDict, ChannelMap, Dict], Optional[List[MessageDict]]]


# =============================================================================
# DEFAULT EVENT CONFIGURATION
# =============================================================================

WHATSAPP_DEFAULT_EVENTS: Set[str] = {
    "received_whatsapp_message",
    "response_to_whatsapp_message",
    "message_to_whatsapp_agent",
    "function_call",
    "function_call_response",
    "function_call_async_error",
    "analyst_request",
    "analyst_response",
    "context",
    "execute_plan",
    "batch_tool_execution",
}

EVENT_PREFIXES: Dict[str, Tuple[str, str]] = {
    # (prefix, role)
    "received_whatsapp_message": ("[Usuario WhatsApp]", "user"),
    "response_to_whatsapp_message": ("[Agente WhatsApp]", "assistant"),
    "message_to_whatsapp_agent": ("[Operador]", "system"),
    "received_email": ("[Email recibido]", "user"),
    "email_to_user": ("[Email enviado]", "assistant"),
    "function_call": ("[Llamada a herramienta]", "assistant"),
    "function_call_response": ("[Respuesta de herramienta]", "assistant"),
    "function_call_async_error": ("[Error de herramienta]", "system"),
    "analyst_request": ("[Solicitud a analista]", "assistant"),
    "analyst_response": ("[Respuesta de analista]", "assistant"),
    "execution_step": ("[Paso de ejecución]", "assistant"),
    "csv_analyst_response": ("[Respuesta CSV Analyst]", "assistant"),
    "internal_reasoning_response": ("[Razonamiento interno]", "assistant"),
    "context": ("[Contexto]", "system"),
    "execute_plan": ("[Información del pipeline]", "assistant"),
}

# Generic orchestrator prompts that add no conversation value — skip them
_SKIP_PROMPTS = frozenset({
    "Continuar la ejecución del pipeline",
})

EVENT_SPEAKER_NAMES: Dict[str, str] = {
    "received_whatsapp_message": "whatsapp_user",
    "received_email": "email_user",
}


_TOOL_EVENT_TYPES = {
    "function_call", "function_call_response", "function_call_async_error",
    "batch_tool_execution", "analyst_request", "analyst_response",
}


class WhatsAppEventFormatter:
    """Dict-based event formatter with registry-based event handlers.

    Processes orchestration events into OpenAI/LiteLLM message dicts, with
    special handling for tool call/response pairing.
    """

    EVENT_HANDLERS: Dict[str, str] = {
        "batch_tool_execution": "_handle_batch_tool_execution",
        "function_call": "_handle_function_call",
        "analyst_request": "_handle_analyst_request",
        "function_call_response": "_handle_tool_response",
        "analyst_response": "_handle_tool_response",
        "function_call_async_error": "_handle_tool_error",
        "received_whatsapp_message": "_handle_regular_message",
        "response_to_whatsapp_message": "_handle_regular_message",
        "message_to_whatsapp_agent": "_handle_regular_message",
        "received_email": "_handle_email_message",
        "email_to_user": "_handle_regular_message",
        "execution_step": "_handle_regular_message",
        "csv_analyst_response": "_handle_regular_message",
        "internal_reasoning_response": "_handle_regular_message",
        "context": "_handle_context_event",
        "execute_plan": "_handle_regular_message",
    }

    _REPLY_SPLITTERS = (
        re.compile(r"\r?\nOn .+ wrote:", re.I),
        re.compile(r"\r?\nEl .+ escribió:", re.I),
    )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    @classmethod
    def format_events(
        cls,
        events: List[EventDict],
        channel_map: Optional[ChannelMap] = None,
        enabled_events: Optional[Set[str]] = None,
    ) -> List[MessageDict]:
        """Format orchestration events into OpenAI/LiteLLM message dicts."""
        if enabled_events is None:
            enabled_events = WHATSAPP_DEFAULT_EVENTS

        sorted_events = sorted(events, key=lambda x: x.get("created_at", ""))
        logger.info(f"Processing {len(sorted_events)} events with {len(enabled_events)} enabled types")

        state: Dict[str, Any] = {
            "look_ahead": {},
            "buffered": {},
            "pending_batch_ids": [],
            "processed_event_uuids": set(),
            "processed_content_keys": set(),
        }

        output: List[MessageDict] = []

        for evt in sorted_events:
            event_type = evt.get("event_type", "")
            event_uuid = evt.get("event_id", evt.get("uuid"))

            if event_uuid and event_uuid in state["processed_event_uuids"]:
                logger.warning(f"Skipping duplicate event UUID: {event_uuid}")
                continue
            if event_uuid:
                state["processed_event_uuids"].add(event_uuid)

            if event_type not in enabled_events:
                continue

            # Skip execute_plan events with generic orchestrator prompts
            if event_type == "execute_plan" and evt.get("prompt", "") in _SKIP_PROMPTS:
                continue

            # Content-based dedup: skip events with same type+prompt.
            # Tool events are exempt — they differentiate via extra_params
            # (call IDs, tool names, args), not prompt text.
            if event_type not in _TOOL_EVENT_TYPES:
                content_key = (event_type, evt.get("prompt", ""))
                if content_key in state["processed_content_keys"]:
                    logger.warning(f"Skipping duplicate content for {event_type}: {event_uuid}")
                    continue
                state["processed_content_keys"].add(content_key)

            handler_name = cls.EVENT_HANDLERS.get(event_type)
            if not handler_name:
                logger.warning(f"No handler for event type: {event_type}")
                continue

            handler = getattr(cls, handler_name)
            result = handler(evt, channel_map or {}, state)
            if result:
                output.extend(result)

        output.extend(cls._handle_unmatched_calls(state, channel_map or {}))

        logger.info(f"Formatted {len(output)} messages from {len(sorted_events)} events")
        return output

    # =========================================================================
    # TOOL CALL HANDLERS
    # =========================================================================

    @classmethod
    def _handle_batch_tool_execution(
        cls, evt: EventDict, channel_map: ChannelMap, state: Dict,
    ) -> List[MessageDict]:
        """Handle batch tool execution events (multiple parallel tool calls)."""
        source = evt.get("source", "")
        if source != "agent":
            return []

        extra = evt.get("extra_params") or {}
        tool_calls = extra.get("tool_calls", [])
        if not tool_calls:
            return []

        formatted_calls = []
        for tc in tool_calls:
            call_id = tc.get("id")
            if not call_id:
                continue

            formatted_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": json.dumps(tc.get("args", {})),
                },
            })

            if call_id not in state["look_ahead"]:
                state["look_ahead"][call_id] = []
                state["buffered"][call_id] = []

            state["look_ahead"][call_id].append({
                "event": None,
                "tool_call": tc,
                "from_batch": True,
            })
            state["pending_batch_ids"].append(call_id)

        if formatted_calls:
            return [{"role": "assistant", "content": None, "tool_calls": formatted_calls}]
        return []

    @classmethod
    def _handle_function_call(
        cls, evt: EventDict, channel_map: ChannelMap, state: Dict,
    ) -> List[MessageDict]:
        """Handle single function_call events (source=agent only)."""
        source = evt.get("source", "")
        if source != "agent":
            return []

        extra = evt.get("extra_params") or {}
        if extra.get("batch_id"):
            return []

        tool_calls = extra.get("tool_calls", [])
        if not tool_calls:
            return []

        tool_call = tool_calls[0]
        call_id = tool_call.get("id")
        if not call_id:
            return []

        if call_id not in state["look_ahead"]:
            state["look_ahead"][call_id] = []
            state["buffered"][call_id] = []
        elif state["look_ahead"][call_id]:
            return []  # duplicate

        state["look_ahead"][call_id].append({
            "event": evt,
            "tool_call": tool_call,
            "from_batch": False,
        })
        return []

    @classmethod
    def _handle_analyst_request(
        cls, evt: EventDict, channel_map: ChannelMap, state: Dict,
    ) -> List[MessageDict]:
        """Handle analyst_request events (synthetic tool call)."""
        extra = evt.get("extra_params") or {}
        call_id = extra.get("tool_call_id")
        analyst_uuid = extra.get("analyst_uuid")
        if not call_id or not analyst_uuid:
            return []

        tool_name = extra.get("tool_name", f"analyst_{analyst_uuid[:8]}")
        tool_call = {
            "id": call_id,
            "name": tool_name,
            "args": {
                "prompt": evt.get("prompt", ""),
                "analyst_uuid": analyst_uuid,
                "node_id": extra.get("node_id"),
            },
        }

        if call_id not in state["look_ahead"]:
            state["look_ahead"][call_id] = []
            state["buffered"][call_id] = []

        state["look_ahead"][call_id].append({
            "event": evt,
            "tool_call": tool_call,
            "from_batch": False,
        })
        return []

    @classmethod
    def _handle_tool_response(
        cls, evt: EventDict, channel_map: ChannelMap, state: Dict,
    ) -> List[MessageDict]:
        """Handle function_call_response / analyst_response events."""
        extra = evt.get("extra_params") or {}
        call_id = extra.get("tool_call_id") or extra.get("id")
        tool_name = extra.get("tool_name", "unknown")
        response_content = evt.get("prompt", "Tool execution completed")

        matched_id = cls._match_tool_call(call_id, state)
        if not matched_id:
            logger.warning(f"Skipping tool response (no matching agent call): {tool_name}")
            return []

        call_data = state["look_ahead"][matched_id].pop(0)
        tool_call = call_data.get("tool_call")
        from_batch = call_data.get("from_batch", False)

        if not state["look_ahead"][matched_id]:
            del state["look_ahead"][matched_id]
        if matched_id in state["pending_batch_ids"]:
            state["pending_batch_ids"].remove(matched_id)

        output: List[MessageDict] = []

        # Emit AIMessage if not from batch (batch already emitted)
        if not from_batch and tool_call:
            output.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_call.get("id"),
                    "type": "function",
                    "function": {
                        "name": tool_call.get("name", ""),
                        "arguments": json.dumps(tool_call.get("args", {})),
                    },
                }],
            })

        # Emit ToolMessage
        if tool_call:
            effective_name = extra.get("tool_name") or tool_call.get("name", "unknown")
            output.append({
                "role": "tool",
                "tool_call_id": matched_id,
                "content": response_content,
            })

        output.extend(cls._flush_buffered_events(matched_id, state, channel_map))
        return output

    @classmethod
    def _handle_tool_error(
        cls, evt: EventDict, channel_map: ChannelMap, state: Dict,
    ) -> List[MessageDict]:
        """Handle function_call_async_error events."""
        extra = evt.get("extra_params") or {}
        call_id = extra.get("tool_call_id")
        tool_name = extra.get("tool_name", "Unknown tool")
        error_content = evt.get("prompt", "Unknown error occurred")

        matched_id = cls._match_tool_call(call_id, state)

        if matched_id:
            call_data = state["look_ahead"][matched_id].pop(0)
            tool_call = call_data.get("tool_call")
            from_batch = call_data.get("from_batch", False)

            if not state["look_ahead"][matched_id]:
                del state["look_ahead"][matched_id]
            if matched_id in state["pending_batch_ids"]:
                state["pending_batch_ids"].remove(matched_id)

            output: List[MessageDict] = []

            if not from_batch and tool_call:
                output.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call.get("id"),
                        "type": "function",
                        "function": {
                            "name": tool_call.get("name", ""),
                            "arguments": json.dumps(tool_call.get("args", {})),
                        },
                    }],
                })

            output.append({
                "role": "tool",
                "tool_call_id": matched_id,
                "content": f"ERROR: {error_content}",
            })

            output.extend(cls._flush_buffered_events(matched_id, state, channel_map))
            return output

        return [{"role": "system", "content": f"ERROR EN HERRAMIENTA: {tool_name}\n\n{error_content}\n---"}]

    # =========================================================================
    # REGULAR MESSAGE HANDLERS
    # =========================================================================

    @classmethod
    def _handle_regular_message(
        cls, evt: EventDict, channel_map: ChannelMap, state: Dict,
    ) -> List[MessageDict]:
        """Handle regular (non-tool) message events."""
        if not state["look_ahead"]:
            return [cls._format_regular_message(evt, channel_map)]

        latest_id = list(state["look_ahead"].keys())[-1]
        state["buffered"][latest_id].append(evt)
        return []

    @classmethod
    def _handle_email_message(
        cls, evt: EventDict, channel_map: ChannelMap, state: Dict,
    ) -> List[MessageDict]:
        """Handle email events with special formatting."""
        if state["look_ahead"]:
            latest_id = list(state["look_ahead"].keys())[-1]
            state["buffered"][latest_id].append(evt)
            return []

        return [cls._format_email_message(evt, channel_map)]

    @classmethod
    def _format_email_message(cls, evt: EventDict, channel_map: ChannelMap) -> MessageDict:
        """Format an email event into a message dict."""
        extra = evt.get("extra_params") or {}
        event_type = evt.get("event_type", "")
        prefix, role = EVENT_PREFIXES.get(event_type, ("[Email]", "user"))

        sender_name = extra.get("sender_name", "")
        sender_email = extra.get("sender_email", "")
        if sender_name or sender_email:
            contact_info = f"{sender_name} <{sender_email}>" if sender_email else sender_name
            prefix = f"{prefix} {contact_info}"

        ch_id = evt.get("channel_id")
        if channel_map and ch_id in channel_map:
            idx, ch_type = channel_map[ch_id]
            prefix = f"{prefix} [{idx}: {ch_type}]"

        body = extra.get("body", "")
        if extra.get("attachments"):
            body += f" [incluye {len(extra['attachments'])} archivos adjuntos]"
        content = cls._extract_latest_email_content(body)

        return {"role": role, "content": f"{prefix} {content}\n---"}

    @classmethod
    def _handle_context_event(
        cls, evt: EventDict, channel_map: ChannelMap, state: Dict,
    ) -> List[MessageDict]:
        """Handle context events (always emitted immediately as system messages)."""
        content = evt.get("prompt", "")
        if not content:
            return []
        return [{"role": "system", "content": content}]

    # =========================================================================
    # HELPERS
    # =========================================================================

    @classmethod
    def _match_tool_call(cls, call_id: Optional[str], state: Dict) -> Optional[str]:
        """Match a tool response to a pending call by ID or FIFO."""
        if call_id:
            if call_id in state["look_ahead"] and state["look_ahead"][call_id]:
                return call_id
            return None

        return state["pending_batch_ids"][0] if state["pending_batch_ids"] else None

    @classmethod
    def _flush_buffered_events(
        cls, call_id: str, state: Dict, channel_map: ChannelMap,
    ) -> List[MessageDict]:
        """Flush events buffered between tool call and response."""
        if call_id not in state["buffered"]:
            return []
        return [
            cls._format_regular_message(evt, channel_map)
            for evt in state["buffered"].pop(call_id, [])
        ]

    @classmethod
    def _format_regular_message(
        cls, evt: EventDict, channel_map: ChannelMap,
    ) -> MessageDict:
        """Format a regular event into an OpenAI message dict."""
        event_type = evt.get("event_type", "")
        extra = evt.get("extra_params") or {}

        prefix, role = EVENT_PREFIXES.get(event_type, ("[Desconocido]", "system"))

        # Agent responses: no prefix so LLM sees clean examples
        if event_type == "response_to_whatsapp_message":
            return {"role": role, "content": evt.get("prompt", "")}

        # Add sender info for received messages
        if event_type in {"received_whatsapp_message", "received_email", "message_to_whatsapp_agent"}:
            sender_name = extra.get("sender_name", "")
            contact = extra.get("sender_email", "") or extra.get("sender_phone", "")
            if sender_name or contact:
                contact_info = f"{sender_name} <{contact}>" if contact else sender_name
                prefix = f"{prefix} {contact_info}"

        ch_id = evt.get("channel_id")
        if channel_map and ch_id in channel_map:
            idx, ch_type = channel_map[ch_id]
            prefix = f"{prefix} [{idx}: {ch_type}]"

        content = evt.get("prompt", "")

        # Include attachment/reaction info for WhatsApp messages
        if event_type == "received_whatsapp_message":
            media_type = extra.get("type", "text")
            media_url = extra.get("url")
            if media_type == "reaction":
                content = f"[Reacción: {content}]" if content else "[Reacción]"
            elif media_type != "text" and media_url:
                attachment_label = f"[Archivo adjunto: {media_type}]"
                content = f"{content}\n{attachment_label}" if content else attachment_label

        speaker_name = EVENT_SPEAKER_NAMES.get(event_type)

        msg: MessageDict = {"role": role, "content": f"{prefix} {content}\n---"}
        if speaker_name:
            msg["name"] = speaker_name
        return msg

    @classmethod
    def _extract_latest_email_content(cls, body: str) -> str:
        """Extract latest email content, removing quoted replies."""
        if not body:
            return ""
        for splitter in cls._REPLY_SPLITTERS:
            match = splitter.search(body)
            if match:
                return body[: match.start()].strip()

        lines = []
        for line in body.splitlines():
            if line.lstrip().startswith(">"):
                break
            lines.append(line)
        return "\n".join(lines).strip()

    @classmethod
    def _handle_unmatched_calls(
        cls, state: Dict, channel_map: ChannelMap,
    ) -> List[MessageDict]:
        """Handle tool calls that never received responses (tail processing)."""
        output: List[MessageDict] = []

        for call_id, call_list in state["look_ahead"].items():
            for call_data in call_list:
                call_evt = call_data.get("event")
                if call_evt is not None:
                    output.append(cls._format_regular_message(call_evt, channel_map))

            output.extend(
                cls._format_regular_message(evt, channel_map)
                for evt in state["buffered"].get(call_id, [])
            )

        if output:
            logger.warning(f"Processed {len(output)} unmatched tool calls/buffered events")
        return output
