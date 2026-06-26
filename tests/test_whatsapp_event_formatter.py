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
    assert "Evita repetir la misma llamada tenant_mcp" in tool_message["content"]
    assert "AIA" in tool_message["content"]


def test_tenant_mcp_validation_error_is_wrapped_as_failure_with_missing_params():
    module = _load_formatter_module()

    messages = module.WhatsAppEventFormatter.format_events([
        {
            "created_at": "2026-06-25T15:39:05Z",
            "event_type": "function_call",
            "source": "agent",
            "extra_params": {
                "tool_calls": [{
                    "id": "call_tenant_error",
                    "name": "tenant_mcp",
                    "args": {
                        "function_name": "gammavet_registrar_retiro",
                        "action": "create",
                        "params": {},
                        "function_data": {
                            "mcp_actions": {
                                "create": {
                                    "request_schema": {
                                        "type": "object",
                                        "required": [
                                            "source_event_uuid",
                                            "requested_by_phone",
                                        ],
                                        "properties": {
                                            "source_event_uuid": {
                                                "type": "string",
                                                "description": "UUID del evento de WhatsApp que originó la solicitud",
                                            },
                                            "requested_by_phone": {
                                                "type": "string",
                                                "description": "Teléfono del solicitante",
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                }],
            },
        },
        {
            "created_at": "2026-06-25T15:39:06Z",
            "event_type": "function_call_response",
            "source": "function",
            "prompt": (
                '{"status_code":422,"detail":[{"type":"missing",'
                '"loc":["body","source_event_uuid"],"msg":"Field required"},'
                '{"type":"missing","loc":["body","requested_by_phone"],'
                '"msg":"Field required"}]}'
            ),
            "extra_params": {
                "tool_call_id": "call_tenant_error",
                "tool_name": "tenant_mcp",
            },
        },
    ])

    assert [message["role"] for message in messages] == ["assistant", "tool"]
    content = messages[1]["content"]
    assert "Resultado Tenant MCP fallido." in content
    assert "Resultado Tenant MCP completado." not in content
    assert "Función: gammavet_registrar_retiro." in content
    assert "Acción: create." in content
    assert "Parámetros usados: sin parámetros." in content
    assert "status_code=422" in content
    assert "source_event_uuid" in content
    assert "requested_by_phone" in content
    assert "UUID del evento de WhatsApp" in content
    assert "Teléfono del solicitante" in content
    assert "Podrías intentar nuevamente corrigiendo los parámetros" in content
    assert "Evita repetir la misma llamada con los mismos parámetros fallidos" in content
    assert "debes" not in content.lower()
    assert "no debes" not in content.lower()


def test_tenant_mcp_success_with_nested_domain_status_code_is_not_failure():
    module = _load_formatter_module()

    messages = module.WhatsAppEventFormatter.format_events([
        {
            "created_at": "2026-06-25T15:40:05Z",
            "event_type": "function_call",
            "source": "agent",
            "extra_params": {
                "tool_calls": [{
                    "id": "call_tenant_nested_status",
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
            "created_at": "2026-06-25T15:40:06Z",
            "event_type": "function_call_response",
            "source": "function",
            "prompt": (
                '{"items":[{"nombre":"Cliente con estado interno",'
                '"status_code":500}],"page":{"limit":5,"offset":0,'
                '"returned":1}}'
            ),
            "extra_params": {
                "tool_call_id": "call_tenant_nested_status",
                "tool_name": "tenant_mcp",
            },
        },
    ])

    content = messages[1]["content"]
    assert "Resultado Tenant MCP completado." in content
    assert "Resultado Tenant MCP fallido." not in content
    assert "items=1" in content
    assert "Cliente con estado interno" in content


def test_tenant_mcp_success_with_business_error_text_is_not_failure():
    module = _load_formatter_module()

    messages = module.WhatsAppEventFormatter.format_events([
        {
            "created_at": "2026-06-25T15:41:05Z",
            "event_type": "function_call",
            "source": "agent",
            "extra_params": {
                "tool_calls": [{
                    "id": "call_tenant_business_text",
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
            "created_at": "2026-06-25T15:41:06Z",
            "event_type": "function_call_response",
            "source": "function",
            "prompt": (
                '{"items":[{"nombre":"Cliente",'
                '"nota":"texto de negocio con validation error histórico"}],'
                '"page":{"limit":5,"offset":0,"returned":1}}'
            ),
            "extra_params": {
                "tool_call_id": "call_tenant_business_text",
                "tool_name": "tenant_mcp",
            },
        },
    ])

    content = messages[1]["content"]
    assert "Resultado Tenant MCP completado." in content
    assert "Resultado Tenant MCP fallido." not in content
    assert "validation error histórico" in content


def test_tenant_mcp_success_with_falsey_top_level_error_is_not_failure():
    module = _load_formatter_module()

    messages = module.WhatsAppEventFormatter.format_events([
        {
            "created_at": "2026-06-25T15:42:05Z",
            "event_type": "function_call",
            "source": "agent",
            "extra_params": {
                "tool_calls": [{
                    "id": "call_tenant_falsey_error",
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
            "created_at": "2026-06-25T15:42:06Z",
            "event_type": "function_call_response",
            "source": "function",
            "prompt": '{"ok":true,"error":false,"items":[{"nombre":"AIA"}]}',
            "extra_params": {
                "tool_call_id": "call_tenant_falsey_error",
                "tool_name": "tenant_mcp",
            },
        },
    ])

    content = messages[1]["content"]
    assert "Resultado Tenant MCP completado." in content
    assert "Resultado Tenant MCP fallido." not in content
    assert "AIA" in content


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
