"""Reference prompts rendered by Qwen3.5's own chat template (chat_template.jinja) through Hugging
Face `transformers`. Run with the reference environment:

    ~/Dev/LMmodels/.venv/bin/python testdata/gen_chat_template.py ~/Dev/LMmodels/Qwen/Qwen3.5-0.8B
"""
import json, os, sys
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1])
hours = {"type": "function", "function": {"name": "clinic_hours", "description": "Opening hours of the clinic for one day",
         "parameters": {"type": "object", "properties": {"day": {"type": "string", "description": "day of the week"}}, "required": ["day"]}}}
book = {"type": "function", "function": {"name": "book_appointment", "description": "Book an appointment",
        "parameters": {"type": "object", "properties": {"day": {"type": "string"}, "hour": {"type": "integer"}, "urgent": {"type": "boolean"}}, "required": ["day", "hour"]}}}
sys_msg = {"role": "system", "content": "Eres la recepcionista de Clínica San Miguel."}
cases = [
    {"name": "plain", "messages": [sys_msg, {"role": "user", "content": "Hola"}], "tools": None},
    {"name": "no_system", "messages": [{"role": "user", "content": "Hola"}], "tools": None},
    {"name": "with_tools", "messages": [sys_msg, {"role": "user", "content": "¿A qué hora abren los lunes?"}], "tools": [hours, book]},
    {"name": "tool_round_trip", "tools": [hours], "messages": [
        sys_msg, {"role": "user", "content": "¿A qué hora abren los lunes?"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "clinic_hours", "arguments": {"day": "lunes"}}}]},
        {"role": "tool", "content": "8:00-20:00"}]},
    {"name": "two_tool_results", "tools": [hours, book], "messages": [
        sys_msg, {"role": "user", "content": "Reserva el lunes a las 9 si abren"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "clinic_hours", "arguments": {"day": "lunes"}}},
            {"type": "function", "function": {"name": "book_appointment", "arguments": {"day": "lunes", "hour": 9, "urgent": False}}}]},
        {"role": "tool", "content": "8:00-20:00"}, {"role": "tool", "content": "reservado"}]},
    {"name": "multi_turn", "tools": None, "messages": [
        sys_msg, {"role": "user", "content": "Hola"}, {"role": "assistant", "content": "¡Hola! ¿En qué puedo ayudar?"},
        {"role": "user", "content": "Gracias"}]},
]
for c in cases:
    c["prompt"] = tok.apply_chat_template(c["messages"], tools=c["tools"], tokenize=False, add_generation_prompt=True)
json.dump(cases, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_template_cases.json"), "w"), ensure_ascii=False, indent=1)
print("cases", len(cases))
