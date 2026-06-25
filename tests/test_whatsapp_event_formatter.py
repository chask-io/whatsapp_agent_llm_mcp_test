import importlib.util
from pathlib import Path


def _load_formatter_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "backend"
        / "whatsapp_event_formatter.py"
    )
    spec = importlib.util.spec_from_file_location("whatsapp_event_formatter_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tenant_mcp_tool_response_is_wrapped_for_history():
    module = _load_formatter_module()

    messages = module.WhatsAppEventFormatter.format_events([
        {
            "created_at": "2026-06-25T15:37:05Z",
            "event_type": "function_call",
            "source": "agent",
            "extra_params": {
                "tool_calls": [{
                    "id": "call_tenant_1",
                    "name": "tenant_mcp",
                    "args": {
                        "function_name": "list-clientes",
                        "action": "list",
                        "params": {"limit": 5},
                    },
                }],
            },
        },
        {
            "created_at": "2026-06-25T15:37:06Z",
            "event_type": "function_call_response",
            "source": "function",
            "prompt": (
                '{"items":[{"nombre":"AIA"}],"page":{"limit":5,'
                '"offset":0,"returned":1},"status_code":200}'
            ),
            "extra_params": {
                "tool_call_id": "call_tenant_1",
                "tool_name": "tenant_mcp",
            },
        },
    ])

    assert [message["role"] for message in messages] == ["assistant", "tool"]
    tool_message = messages[1]
    assert tool_message["tool_call_id"] == "call_tenant_1"
    assert "Resultado Tenant MCP completado." in tool_message["content"]
    assert "Función: list-clientes." in tool_message["content"]
    assert "Acción: list." in tool_message["content"]
    assert 'Parámetros usados: {"limit": 5}.' in tool_message["content"]
    assert "status_code=200" in tool_message["content"]
    assert "items=1" in tool_message["content"]
    assert "WhatsappAlUsuarioFn" in tool_message["content"]
    assert "No repitas la misma llamada tenant_mcp" in tool_message["content"]
    assert "AIA" in tool_message["content"]


def test_regular_tool_response_keeps_original_content():
    module = _load_formatter_module()

    messages = module.WhatsAppEventFormatter.format_events([
        {
            "created_at": "2026-06-25T15:38:05Z",
            "event_type": "function_call",
            "source": "agent",
            "extra_params": {
                "tool_calls": [{
                    "id": "call_whatsapp_1",
                    "name": "WhatsappAlUsuarioFn",
                    "args": {"message": "Hola"},
                }],
            },
        },
        {
            "created_at": "2026-06-25T15:38:06Z",
            "event_type": "function_call_response",
            "source": "function",
            "prompt": "Mensaje enviado correctamente",
            "extra_params": {
                "tool_call_id": "call_whatsapp_1",
                "tool_name": "WhatsappAlUsuarioFn",
            },
        },
    ])

    assert [message["role"] for message in messages] == ["assistant", "tool"]
    assert messages[1]["tool_call_id"] == "call_whatsapp_1"
    assert messages[1]["content"] == "Mensaje enviado correctamente"
    assert "Resultado Tenant MCP" not in messages[1]["content"]
