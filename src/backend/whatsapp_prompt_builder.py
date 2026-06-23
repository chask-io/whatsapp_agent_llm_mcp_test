"""
WhatsApp Agent - System Prompt Builder

Builds the system prompt for the WhatsApp agent by fetching organization context,
user validation data, and active requirements from the Chask APIs.

Extracted from the old WhatsappAgent._build_system_prompt() method.
Returns a plain string (no LangChain dependency).
"""

import os
import logging
from typing import Dict, Any, Optional

from chask_foundation.backend.models import OrchestrationEvent

logger = logging.getLogger(__name__)

PROMPT_FILE_PATH = "backend/prompts/whatsapp_agent_prompt.txt"

AUTH_REMATCH_SYSTEM_DIRECTIVE = (
    "Tu llamada anterior a IniciarRequerimientoFn coincidió con un flujo al que "
    "este usuario NO tiene acceso. DEBES volver a llamar ÚNICAMENTE a la "
    "herramienta IniciarRequerimientoFn ahora. NO llames a SendPipelineDataFn, "
    "NO llames a WhatsappAlUsuarioFn, NO le respondas al usuario — tu única "
    "acción válida es ejecutar IniciarRequerimientoFn. Reformula el "
    "requerimiento con palabras clave que coincidan DIRECTAMENTE con UNO de "
    "estos flujos a los que el usuario SÍ tiene acceso (no repitas la solicitud "
    "original que falló): {accessible_flow_names}."
)

AUTH_REMATCH_DIRECTIVE_BLOCK_TEMPLATE = (
    "\n\n[INICIO DIRECTIVA AUTH_REMATCH]\n"
    "{directive}\n"
    "[FIN DIRECTIVA AUTH_REMATCH]"
)


def _load_prompt_template() -> str:
    """Load WhatsApp agent prompt template from file."""
    prompt_path = os.path.join(os.getcwd(), PROMPT_FILE_PATH)

    if not os.path.exists(prompt_path):
        prompt_path = os.path.join(
            os.path.dirname(__file__), "prompts", "whatsapp_agent_prompt.txt"
        )

    with open(prompt_path, "r", encoding="utf-8") as fh:
        return fh.read()


def _get_api_credentials(oe: OrchestrationEvent) -> Dict[str, str]:
    """Return common API call kwargs."""
    return {
        "access_token": oe.access_token,
        "organization_id": str(oe.organization.organization_id),
    }


def _fetch_whatsapp_channel_id(oe: OrchestrationEvent) -> str:
    """Fetch the WhatsApp channel_id from channels API."""
    from api.channels_requests import channels_api_manager

    creds = _get_api_credentials(oe)
    channels_response = channels_api_manager.call("get-channels", **creds)

    if "channels" not in channels_response:
        error_detail = channels_response.get("detail", "Unknown error")
        error_status = channels_response.get("status_code", "Unknown status")
        raise ValueError(
            f"Channels API error: {error_detail} (status: {error_status})"
        )

    for channel in channels_response["channels"]:
        if channel["name"] == "whatsapp":
            return channel["uuid"]

    raise ValueError("WhatsApp channel not found")


def _fetch_organization_context(oe: OrchestrationEvent, channel_id: str) -> str:
    """Fetch organization context description for the channel."""
    from api.agent_requests import agent_api_manager

    creds = _get_api_credentials(oe)
    context = agent_api_manager.call(
        "get-context-by-channel", channel_id=channel_id, **creds,
    )
    return context.get("context", "")


def _fetch_user_data(oe: OrchestrationEvent) -> Dict[str, Any]:
    """Fetch user validation data from orchestrator."""
    from api.orchestrator_requests import orchestrator_api_manager

    creds = _get_api_credentials(oe)
    return orchestrator_api_manager.call(
        "get_orchestration_session_user_data",
        orchestration_session_uuid=oe.orchestration_session_uuid,
        internal_orchestration_session_uuid=oe.internal_orchestration_session_uuid,
        **creds,
    )


def _fetch_active_requirements(oe: OrchestrationEvent) -> str:
    """Fetch and format active requirements for the session."""
    from api.orchestrator_requests import orchestrator_api_manager

    creds = _get_api_credentials(oe)
    response = orchestrator_api_manager.call(
        "get_active_requirement_for_os",
        orchestration_session_uuid=oe.orchestration_session_uuid,
        **creds,
    )

    if not response or not isinstance(response, dict):
        return "None"

    active_pipeline = response.get("active_pipeline")
    if not active_pipeline:
        return "None"

    pipeline_id = active_pipeline.get("id", "N/A")
    title = active_pipeline.get("title", "N/A")
    description = active_pipeline.get("description", "N/A")
    status = active_pipeline.get("status", "N/A")

    return (
        f"- ID: {pipeline_id} | Estado: {status}\n"
        f"  Título: {title}\n"
        f"  Descripción: {description}"
    )


def apply_template_variables(template: str, data: Dict[str, Any]) -> str:
    """Replace known template variables using str.replace().

    Safer than str.format() because unknown {variable} placeholders
    (e.g. from admin-configured LLM contexts) are left untouched
    instead of raising KeyError.
    """
    for key, value in data.items():
        template = template.replace(f"{{{key}}}", str(value))
    return template


def _is_auth_rematch_directive_enabled(extra_params: Dict[str, Any]) -> bool:
    return (
        extra_params.get("agent_directive") == "auth_rematch"
        or bool(extra_params.get("auth_rematch_retry"))
    )


def _format_accessible_flow_names(extra_params: Dict[str, Any]) -> str:
    flow_names = extra_params.get("accessible_flow_names") or []

    if isinstance(flow_names, str):
        names = [flow_names]
    elif isinstance(flow_names, (list, tuple, set)):
        names = [str(name) for name in flow_names if str(name).strip()]
    else:
        names = []

    return ", ".join(names) if names else "ninguno informado"


def render_auth_rematch_system_directive(oe: OrchestrationEvent) -> str:
    extra_params = oe.extra_params or {}
    directive = AUTH_REMATCH_SYSTEM_DIRECTIVE.format(
        accessible_flow_names=_format_accessible_flow_names(extra_params)
    )
    return AUTH_REMATCH_DIRECTIVE_BLOCK_TEMPLATE.format(directive=directive)


def build_auth_rematch_directive_text(oe: OrchestrationEvent) -> Optional[str]:
    extra_params = oe.extra_params or {}
    if not _is_auth_rematch_directive_enabled(extra_params):
        return None

    return render_auth_rematch_system_directive(oe)


def get_whatsapp_prompt_data(oe: OrchestrationEvent) -> Dict[str, Any]:
    """Fetch all dynamic data for the WhatsApp prompt template.

    Returns a dict of template variable name -> value.
    """
    channel_id = _fetch_whatsapp_channel_id(oe)
    org_context = _fetch_organization_context(oe, channel_id)
    user_data = _fetch_user_data(oe)
    client_requirements = _fetch_active_requirements(oe)

    validate_user = (
        "Hay que validar al usuario"
        if user_data.get("user_data") == "No hay datos del usuario"
        else "USUARIO VALIDADO"
    )

    bot_name = (
        oe.target_agent.agent_alias if oe.target_agent else "Asistente WhatsApp"
    )

    return {
        "bot_name": bot_name,
        "organizacion_name": oe.organization.organization_name,
        "organizacion_description": org_context,
        "client_requirements": client_requirements,
        "validate_user": validate_user,
        "user_data": user_data.get("user_data", "No hay datos del usuario"),
        "factual_summary": "None",
    }


def build_whatsapp_system_prompt(oe: OrchestrationEvent) -> str:
    """Build the complete system prompt for the WhatsApp agent.

    This is the prompt_builder callable used by AgentConfig.

    Args:
        oe: The orchestration event with all request context.

    Returns:
        Formatted system prompt string.
    """
    template = _load_prompt_template()
    data = get_whatsapp_prompt_data(oe)
    return apply_template_variables(template, data)
