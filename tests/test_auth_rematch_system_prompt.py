import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


models_module = types.ModuleType("chask_foundation.backend.models")
models_module.OrchestrationEvent = object
agent_wrapper_module = types.ModuleType("chask_foundation.backend.agent_wrapper")


class _StubAgentConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _StubAgentFunctionBackend:
    pass


class _StubAgentWrapper:
    pass


agent_wrapper_module.AgentConfig = _StubAgentConfig
agent_wrapper_module.AgentFunctionBackend = _StubAgentFunctionBackend
agent_wrapper_module.AgentWrapper = _StubAgentWrapper
sys.modules.setdefault("chask_foundation", types.ModuleType("chask_foundation"))
sys.modules.setdefault("chask_foundation.backend", types.ModuleType("chask_foundation.backend"))
sys.modules.setdefault("chask_foundation.backend.agent_wrapper", agent_wrapper_module)
sys.modules.setdefault("chask_foundation.backend.models", models_module)

api_module = types.ModuleType("api")
orchestrator_requests_module = types.ModuleType("api.orchestrator_requests")
internal_whatsapp_requests_module = types.ModuleType("api.internal_whatsapp_requests")
orchestrator_requests_module.orchestrator_api_manager = SimpleNamespace(call=lambda *a, **k: {})
internal_whatsapp_requests_module.internal_whatsapp_api_manager = SimpleNamespace(
    call=lambda *a, **k: {}
)
sys.modules.setdefault("api", api_module)
sys.modules.setdefault("api.orchestrator_requests", orchestrator_requests_module)
sys.modules.setdefault("api.internal_whatsapp_requests", internal_whatsapp_requests_module)

module_path = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "backend"
    / "whatsapp_prompt_builder.py"
)
spec = importlib.util.spec_from_file_location("whatsapp_prompt_builder", module_path)
whatsapp_prompt_builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(whatsapp_prompt_builder)

build_auth_rematch_directive_text = (
    whatsapp_prompt_builder.build_auth_rematch_directive_text
)
render_auth_rematch_system_directive = (
    whatsapp_prompt_builder.render_auth_rematch_system_directive
)

backend_package = types.ModuleType("backend")
backend_package.__path__ = [str(Path(__file__).resolve().parents[1] / "src" / "backend")]
sys.modules.setdefault("backend", backend_package)
sys.modules.setdefault("backend.whatsapp_prompt_builder", whatsapp_prompt_builder)
formatter_module = types.ModuleType("backend.whatsapp_event_formatter")
formatter_module.WHATSAPP_DEFAULT_EVENTS = set()
formatter_module.WhatsAppEventFormatter = SimpleNamespace(
    format_events=lambda *a, **k: []
)
sys.modules.setdefault("backend.whatsapp_event_formatter", formatter_module)
function_logic_path = (
    Path(__file__).resolve().parents[1] / "src" / "backend" / "function_logic.py"
)
function_logic_spec = importlib.util.spec_from_file_location(
    "backend.function_logic", function_logic_path
)
function_logic = importlib.util.module_from_spec(function_logic_spec)
function_logic_spec.loader.exec_module(function_logic)


def test_auth_rematch_marker_builds_directive_with_accessible_flows():
    event = SimpleNamespace(
        extra_params={
            "agent_directive": "auth_rematch",
            "accessible_flow_names": [
                "Inicio de Jornada para conductores",
                "Crear orden de retiro",
            ],
            "denied_pipeline_name": "Flujo restringido",
        }
    )

    directive = build_auth_rematch_directive_text(event)

    assert directive is not None
    assert "IniciarRequerimientoFn" in directive
    assert "NO llames a SendPipelineDataFn" in directive
    assert "NO llames a WhatsappAlUsuarioFn" in directive
    assert "Inicio de Jornada para conductores" in directive
    assert "Crear orden de retiro" in directive
    assert "Flujo restringido" not in directive


def test_auth_rematch_retry_truthy_marker_appends_system_directive():
    event = SimpleNamespace(
        extra_params={
            "auth_rematch_retry": True,
            "accessible_flow_names": "Inicio de Jornada para conductores",
        }
    )

    directive = render_auth_rematch_system_directive(event)

    assert "IniciarRequerimientoFn" in directive
    assert "NO llames a SendPipelineDataFn" in directive
    assert "NO llames a WhatsappAlUsuarioFn" in directive
    assert "Inicio de Jornada para conductores" in directive


def test_without_auth_rematch_marker_directive_is_absent():
    event = SimpleNamespace(
        extra_params={
            "accessible_flow_names": ["Inicio de Jornada para conductores"],
        }
    )

    assert build_auth_rematch_directive_text(event) is None


def test_auth_rematch_directive_is_last_message_and_not_base_prompt():
    event = SimpleNamespace(
        extra_params={
            "agent_directive": "auth_rematch",
            "accessible_flow_names": ["Inicio de Jornada para conductores"],
        },
        event_type="need_agent_whatsapp",
    )
    wrapper = SimpleNamespace(
        orchestration_event=event,
        _build_system_prompt=lambda: "PROMPT BASE",
        _build_whatsapp_conversation_history=lambda: [
            {"role": "user", "content": "quiero iniciar jornada"},
            {"role": "assistant", "content": "[Operador]"},
        ],
        _is_collecting_pipeline_data=lambda: True,
    )

    messages = function_logic._WhatsAppAgentWrapper._prepare_messages(wrapper)

    assert "INICIO DIRECTIVA AUTH_REMATCH" not in messages[0]["content"]
    assert messages[-1]["role"] == "system"
    assert "INICIO DIRECTIVA AUTH_REMATCH" in messages[-1]["content"]
    assert messages[-2]["content"] == function_logic.PIPELINE_COLLECTION_REMINDER
