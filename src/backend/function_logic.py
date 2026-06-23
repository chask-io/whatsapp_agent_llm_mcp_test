"""
WhatsappAgentLlmFn - Business Logic

WhatsApp agent using the generic AgentFunctionBackend with WhatsApp-specific
configuration, custom event formatting, and operator reminder injection.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from chask_foundation.backend.agent_wrapper import (
    AgentConfig,
    AgentFunctionBackend,
    AgentWrapper,
)
from chask_foundation.backend.models import OrchestrationEvent
from api.orchestrator_requests import orchestrator_api_manager
from api.internal_whatsapp_requests import internal_whatsapp_api_manager

from .whatsapp_prompt_builder import (
    apply_template_variables,
    build_auth_rematch_directive_text,
    build_whatsapp_system_prompt,
    get_whatsapp_prompt_data,
)
from .whatsapp_event_formatter import WhatsAppEventFormatter, WHATSAPP_DEFAULT_EVENTS

LAMBDA_NAME = os.getenv("AWS_LAMBDA_FUNCTION_NAME", "whatsapp_agent_llm")
logger = logging.getLogger(__name__)

# =============================================================================
# Operator reminder
# =============================================================================

OPERATOR_REMINDER_TEXT = (
    "IMPORTANTE: El requerimiento está esperando una respuesta con la información "
    "solicitada. Usa la herramienta EnviarMensajeAlRequerimientoFn para enviar la "
    "información recopilada del usuario. Si ya enviaste la información o no aplica, "
    "ignora este mensaje."
)

PIPELINE_COLLECTION_REMINDER = (
    "Recuerda: hay un flujo activo esperando datos. "
    "Cuando tengas la información requerida, llama a SendPipelineDataFn "
    "para iniciar la ejecución."
)

_PAUSE_BLOCK_FIELDS = (
    ("reason", "reason"),
    ("awaiting_for", "awaiting_for"),
    ("related_task", "related_task"),
    ("node", "nodo_pausado"),
)

_PAUSE_PROMPT_RE = re.compile(
    r"Pausando por:\s*(?P<reason>.*?),\s*"
    r"Esperando:\s*(?P<awaiting_for>.*?),\s*"
    r"Nodo:\s*(?P<related_task>.*)$"
)


def _should_inject_operator_reminder(events: List[Dict[str, Any]]) -> bool:
    """Return True when an operator reminder should be injected.

    Conditions:
    1. At least one operator message exists (message_to_whatsapp_agent).
    2. The last relevant message is from the user (received_whatsapp_message).
    """
    sorted_events = sorted(events, key=lambda x: x.get("created_at", ""))

    has_operator = False
    last_relevant = None

    for evt in sorted_events:
        event_type = evt.get("event_type", "")
        if event_type == "message_to_whatsapp_agent":
            has_operator = True
            last_relevant = "operator"
        elif event_type == "received_whatsapp_message":
            last_relevant = "user"

    return has_operator and last_relevant == "user"


def _format_pause_block_value(value: Any) -> str:
    """Render structured pause-block values without inventing fields."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _build_pause_block_system_message(
    pause_block: Optional[Dict[str, Any]],
) -> Optional[Dict[str, str]]:
    """Build the pause context message from non-empty pause_block values."""
    if not isinstance(pause_block, dict):
        return None

    fields = []
    for field_name, label in _PAUSE_BLOCK_FIELDS:
        value = pause_block.get(field_name)
        if value is None or value == "":
            continue
        fields.append(f"{label}={_format_pause_block_value(value)}")

    if not fields:
        return None

    content = (
        "[Bloque de pausa] Esta sesión estaba pausada. "
        f"{'; '.join(fields)}. "
        "Aplica el Protocolo de reanudación desde pausa: si el nuevo mensaje "
        "NO es awaiting_for, NO lo reenvíes con EnviarMensajeAlRequerimiento; "
        "primero enruta la nueva intención."
    )
    return {"role": "system", "content": content}


def _pause_block_from_tool_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize PauseOrchestrationFn args into the pause_block shape."""
    return {
        "reason": args.get("reason") or args.get("reason_to_pause"),
        "awaiting_for": args.get("awaiting_for"),
        "related_task": args.get("related_task"),
        "node": args.get("node") or args.get("related_task"),
    }


def _extract_pause_block_from_events(
    events: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Best-effort fallback using already-fetched same-session event history."""
    for evt in reversed(events):
        extra = evt.get("extra_params")
        if not isinstance(extra, dict):
            continue

        pause_block = extra.get("pause_block")
        if isinstance(pause_block, dict):
            return pause_block

        for tool_call in extra.get("tool_calls") or []:
            if tool_call.get("name") != "PauseOrchestrationFn":
                continue
            args = tool_call.get("args") or {}
            if isinstance(args, dict):
                return _pause_block_from_tool_args(args)

        if evt.get("event_type") == "pause_orchestration":
            prompt = evt.get("prompt", "")
            match = _PAUSE_PROMPT_RE.search(prompt)
            if match:
                pause_args = match.groupdict()
                pause_args["node"] = pause_args.get("related_task")
                return pause_args

    return None


# =============================================================================
# AgentConfig
# =============================================================================

WHATSAPP_CONFIG = AgentConfig(
    source_name="agent",
    request_event_type="received_whatsapp_message",
    response_event_type="response_to_whatsapp_message",
    enabled_event_types={
        "received_whatsapp_message",
        "response_to_whatsapp_message",
        "message_to_whatsapp_agent",
        "function_call",
        "function_call_response",
        "function_call_async_error",
        "analyst_request",
        "analyst_response",
        "context",
        "batch_tool_execution",
        "execute_plan",
    },
    prompt_builder=build_whatsapp_system_prompt,
    trigger_event_types=[
        "need_agent_whatsapp",
        "function_call_response",
        "function_call_async_error",
        "execute_plan",
        "user_authenticated",
        "notify_whatsapp",
        "message_to_whatsapp_agent",
    ],
    socket_name="whatsapp_agent",
    enable_dynamic_tools=True,
    dynamic_tool_slug="chask-dev",
    dynamic_tool_organization_id="faed8f52-3a25-41be-8a13-cc10b51e05a7",
    dynamic_tool_branch="test",
    dynamic_tool_top_k=5,
    forward_topic="orchestrator",
    default_prompt=(
        "Eres un modelo de lenguaje desarrollado por Chask. "
        "Asiste al usuario con sus requerimientos."
    ),
)


# =============================================================================
# Custom AgentWrapper
# =============================================================================

class _WhatsAppAgentWrapper(AgentWrapper):
    """AgentWrapper subclass with WhatsApp-specific message preparation.

    Overrides _prepare_messages to:
    - Use WhatsAppEventFormatter instead of the generic AgentEventFormatter
    - Inject operator reminder when applicable
    - Inject special event messages for user_authenticated / notify_whatsapp
    """

    def _build_system_prompt(self) -> str:
        """Build system prompt, applying template variables to socket context.

        When an admin assigns a socket context (via LLM context UI), the base
        AgentWrapper returns it raw — skipping the prompt_builder and leaving
        {bot_name}, {organizacion_name}, etc. unresolved.
        """
        oe = self.orchestration_event

        if self.config.socket_name:
            socket_prompt = self._fetch_socket_context()
            if socket_prompt:
                logger.info("Applying template variables to socket-assigned context")
                data = get_whatsapp_prompt_data(oe)
                return apply_template_variables(socket_prompt, data)

        return build_whatsapp_system_prompt(oe)

    def _call_llm(
        self, messages: List[Dict[str, Any]], force_tool_call: bool = True,
    ) -> Dict[str, Any]:
        """Call the LLM with WhatsApp-specific metadata and optional tool enforcement.

        Args:
            messages: Prepared message list to send to the LLM.
            force_tool_call: When True and tools are available, sets
                tool_choice="required" to ensure the model responds with a
                tool call. Set to False for notify_whatsapp direct responses.
        """
        temperature = 1.0 if self.model.startswith("gpt-5") else 0.7

        extra_kwargs: Dict[str, Any] = {}
        if force_tool_call and self.function_schemas:
            extra_kwargs["tool_choice"] = "required"

        response = self.llm_client.chat(
            messages=messages,
            tools=self.function_schemas if self.function_schemas else None,
            temperature=temperature,
            caller_function="whatsapp_agent.get_response",
            metadata={
                "event_type": self.orchestration_event.event_type,
                "lambda_name": LAMBDA_NAME,
            },
            **extra_kwargs,
        )

        if not response.get("success", True):
            error_msg = response.get("error", "Unknown LLM error")
            raise Exception(f"LLM call failed: {error_msg}")

        return response

    def _is_collecting_pipeline_data(self) -> bool:
        """Check if the session is in collecting_pipeline_data status."""
        try:
            response = orchestrator_api_manager.call(
                "get_active_requirement_for_os",
                orchestration_session_uuid=self.orchestration_event.orchestration_session_uuid,
                access_token=self.orchestration_event.access_token,
                organization_id=self.orchestration_event.organization.organization_id,
            )
            return response.get("session_status") == "collecting_pipeline_data"
        except Exception:
            return False

    def _get_conversation_history_for_tools(self) -> List[Dict[str, Any]]:
        """Use WhatsApp-formatted history for tenant MCP preflight discovery."""
        if self._conversation_history_cache is None:
            self._conversation_history_cache = self._build_whatsapp_conversation_history()
        return self._conversation_history_cache

    def _prepare_messages(self) -> List[Dict[str, Any]]:
        system_prompt = self._build_system_prompt()
        conversation_history = self._build_whatsapp_conversation_history()
        current_extra = self.orchestration_event.extra_params or {}
        pause_block = current_extra.get("pause_block")
        pause_source = "current_event"
        if not isinstance(pause_block, dict):
            pause_block = _extract_pause_block_from_events(getattr(self, "_raw_events", []))
            pause_source = "event_history"

        pause_block_message = _build_pause_block_system_message(pause_block)

        # Operator reminder
        if hasattr(self, "_raw_events") and _should_inject_operator_reminder(self._raw_events):
            conversation_history.append({"role": "system", "content": OPERATOR_REMINDER_TEXT})
            logger.info("Injected operator reminder")

        # Special trigger events
        event_type = self.orchestration_event.event_type
        if event_type in ("user_authenticated", "notify_whatsapp"):
            conversation_history.append(
                {"role": "system", "content": self.orchestration_event.prompt}
            )

        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)

        # Pipeline collection reminder (appended as last message)
        if self._is_collecting_pipeline_data():
            reminder = {"role": "system", "content": PIPELINE_COLLECTION_REMINDER}
            messages.append(reminder)
            logger.info("Injected pipeline collection reminder")

        auth_rematch_directive = build_auth_rematch_directive_text(self.orchestration_event)
        if auth_rematch_directive:
            messages.append({"role": "system", "content": auth_rematch_directive})
            logger.info("Injected auth_rematch directive as trailing system message")

        if pause_block_message:
            messages.append(pause_block_message)
            logger.info("Injected pause_block context from %s", pause_source)

        return messages

    def _build_whatsapp_conversation_history(self) -> List[Dict[str, Any]]:
        """Fetch events and format with WhatsAppEventFormatter."""
        try:
            response = orchestrator_api_manager.call(
                "get_orchestration_events",
                orchestration_session_id=self.orchestration_event.orchestration_session_uuid,
                access_token=self.orchestration_event.access_token,
                organization_id=self.orchestration_event.organization.organization_id,
            )

            orchestration_events = response.get("orchestration_events", [])
            logger.info(f"Retrieved {len(orchestration_events)} orchestration events")

            # Store raw events for operator reminder check
            self._raw_events = orchestration_events

            # Build channel map
            channel_map: Dict[str, Any] = {}
            if self.orchestration_event.channel_id:
                channel_map[self.orchestration_event.channel_id] = (0, "whatsapp")

            relevant = [
                evt for evt in orchestration_events
                if evt.get("event_type") in WHATSAPP_DEFAULT_EVENTS
            ]

            return WhatsAppEventFormatter.format_events(
                relevant,
                channel_map=channel_map,
                enabled_events=WHATSAPP_DEFAULT_EVENTS,
            )

        except Exception as e:
            logger.error(f"Failed to build WhatsApp conversation history: {e}")
            return []


# =============================================================================
# FunctionBackend
# =============================================================================

class FunctionBackend(AgentFunctionBackend):
    """WhatsApp agent backend.

    Preserves the handler.py contract:
        FunctionBackend(oe, key, model).process_request()
    """

    def __init__(
        self,
        orchestration_event: OrchestrationEvent,
        openai_api_key: str,
        model: str,
    ):
        model = model or "gpt-5.1-2025-11-13"
        super().__init__(
            config=WHATSAPP_CONFIG,
            orchestration_event=orchestration_event,
            openai_api_key=openai_api_key,
            model=model,
        )

    def _handle_agent_request(self) -> str:
        """Use _WhatsAppAgentWrapper for WhatsApp-specific message preparation.

        WhatsApp-specific flow:
        1. Get initial LLM response with tool_choice="required"
        2. If tool call -> invoke tool, return "requested_orchestrator_assistance"
        3. If no tool call (edge case) -> re-invoke via Kafka with explicit
           instruction to use a tool
        """
        agent = None
        try:
            agent = _WhatsAppAgentWrapper(
                config=self.config,
                orchestration_event=self.orchestration_event,
                openai_api_key=self.openai_api_key,
                model=self.model,
            )

            # Handle notify_whatsapp specially - direct response, no tools
            if self.orchestration_event.event_type == "notify_whatsapp":
                return self._handle_notify_whatsapp(agent)

            response_message = agent.get_response()

            if response_message == "requested_orchestrator_assistance":
                self.response_event_sent = True
                return response_message

            # Safety net: tool_choice="required" should prevent this, but if it
            # happens, re-invoke the whatsapp agent so it tries again with an
            # explicit instruction to use a tool.
            logger.warning(
                "LLM returned no tool calls despite tool_choice=required — re-invoking"
            )
            self._re_invoke_whatsapp_agent()
            return response_message
        finally:
            if agent:
                agent.shutdown()

    def _handle_notify_whatsapp(self, agent: _WhatsAppAgentWrapper) -> str:
        """Handle notify_whatsapp events with a direct LLM response (no tools)."""
        messages = agent._prepare_messages()

        response = agent._call_llm(messages, force_tool_call=False)
        content = response.get("content", "")

        if content:
            self._send_whatsapp_response(content)

        return content

    def _re_invoke_whatsapp_agent(self) -> None:
        """Re-invoke the whatsapp agent when the LLM fails to produce a tool call.

        Emits an execute_plan event with source=agent so the orchestrator
        re-targets the whatsapp agent. The prompt instructs the LLM to always
        respond with a tool call. Failures in the API calls are not raised —
        the current invocation already returned without a tool call, so this
        is a best-effort recovery step.
        """
        oe = self.orchestration_event
        extra_params = {"original_source": "agent"}
        re_invoke_prompt = (
            "DEBES responder con una llamada a herramienta. "
            "Analiza la conversación y ejecuta la herramienta apropiada. "
            "Si necesitas enviar un mensaje de WhatsApp, usa la herramienta correspondiente."
        )

        try:
            evolve_response = orchestrator_api_manager.call(
                "evolve_event",
                parent_event_uuid=str(oe.event_id),
                event_type="execute_plan",
                source="agent",
                target="orchestrator",
                prompt=re_invoke_prompt,
                extra_params=extra_params,
                access_token=oe.access_token,
                organization_id=oe.organization.organization_id,
            )

            re_invoke_event = oe.model_copy(deep=True)
            re_invoke_event.event_type = "execute_plan"
            re_invoke_event.source = "agent"
            re_invoke_event.target = "orchestrator"
            re_invoke_event.prompt = re_invoke_prompt
            re_invoke_event.event_id = evolve_response.get("uuid", oe.event_id)
            re_invoke_event.extra_params = evolve_response.get("extra_params", extra_params)

            orchestrator_api_manager.call(
                "forward_oe_to_kafka",
                orchestration_event=re_invoke_event.model_dump(),
                topic="orchestrator",
                access_token=oe.access_token,
                organization_id=oe.organization.organization_id,
            )
            logger.info(
                "Emitted execute_plan to re-invoke whatsapp agent with tool requirement"
            )
        except Exception as e:
            logger.error(f"Failed to re-invoke whatsapp agent: {e}")

    def _send_whatsapp_response(self, response_message: str) -> None:
        """Send response as response_to_whatsapp_message with phone numbers."""
        if self.response_event_sent:
            logger.warning("[DUPLICATE_GUARD] Response already sent, skipping")
            return

        oe = self.orchestration_event
        conversation_uuid = oe.channel_id
        if not conversation_uuid:
            raise ValueError("Missing channel_id (conversation_uuid)")

        extra_params = self._get_phone_numbers(oe, conversation_uuid)
        evolved_uuid = self._evolve_response_event(oe, response_message, extra_params)
        self._forward_to_kafka(oe, evolved_uuid, response_message, extra_params)

        self.response_event_sent = True
        logger.info(f"WhatsApp response sent [evolved from {oe.event_id} -> {evolved_uuid}]")

    def _get_phone_numbers(self, oe: OrchestrationEvent, conversation_uuid: str) -> Dict[str, Any]:
        """Get user and agent phone numbers from extra_params or API."""
        original_extra = oe.extra_params or {}
        user_phone = original_extra.get("user_phone_number")
        agent_phone = original_extra.get("agent_phone_number")

        if user_phone and agent_phone:
            return {
                "user_phone_number": user_phone,
                "agent_phone_number": agent_phone,
                "original_source": "agent",
            }

        phone_data = internal_whatsapp_api_manager.call(
            "get_phone_from_conversation",
            conversation_uuid=conversation_uuid,
            access_token=oe.access_token,
            organization_id=oe.organization.organization_id,
        )

        if not user_phone:
            user_phone = phone_data.get("phone_number")
            if not user_phone:
                raise ValueError("API returned no phone_number")

        if not agent_phone:
            agent_phone = phone_data.get("phone_number_id")
            if not agent_phone:
                raise ValueError("API returned no phone_number_id")

        return {
            "user_phone_number": user_phone,
            "agent_phone_number": agent_phone,
            "original_source": "agent",
        }

    def _evolve_response_event(
        self, oe: OrchestrationEvent, response_message: str, extra_params: Dict[str, Any]
    ) -> str:
        """Evolve the orchestration event and return the new UUID."""
        evolve_response = orchestrator_api_manager.call(
            "evolve_event",
            parent_event_uuid=str(oe.event_id),
            event_type="response_to_whatsapp_message",
            source="agent",
            target="whatsapp",
            prompt=response_message,
            extra_params=extra_params,
            access_token=oe.access_token,
            organization_id=oe.organization.organization_id,
        )

        status_code = evolve_response.get("status_code")
        if status_code and status_code not in (200, 201):
            raise Exception(f"Failed to evolve event: {evolve_response.get('error', 'Unknown')}")

        evolved_uuid = evolve_response.get("uuid")
        if not evolved_uuid:
            raise Exception("API response missing uuid for evolved event")

        return evolved_uuid

    def _forward_to_kafka(
        self, oe: OrchestrationEvent, evolved_uuid: str, response_message: str, extra_params: Dict[str, Any]
    ) -> None:
        """Forward the response event to Kafka."""
        response_event = oe.model_copy(deep=True)
        response_event.event_id = evolved_uuid
        response_event.event_type = "response_to_whatsapp_message"
        response_event.source = "agent"
        response_event.target = "whatsapp"
        response_event.prompt = response_message
        response_event.extra_params = extra_params
        response_event.extra_params["_already_persisted"] = True

        orchestrator_api_manager.call(
            "forward_oe_to_kafka",
            orchestration_event=response_event.model_dump(),
            topic="orchestrator",
            access_token=response_event.access_token,
            organization_id=response_event.organization.organization_id,
        )
