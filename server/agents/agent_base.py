import json
from typing import Any, Dict, Optional
from openai import OpenAI

def call_gpt5_json(api_key: str, developer_text: str, user_text: str) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)
    resp = client.responses.create(
        model="gpt-5",
        input=[
            {"role": "developer", "content": [{"type": "input_text", "text": developer_text}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
        ],
        text={"format": {"type": "text"}, "verbosity": "medium"},
        reasoning={"effort": "high", "summary": "detailed"},
        tools=[{"type": "web_search_preview", "user_location": {"type": "approximate", "country": "US"}, "search_context_size": "medium"}],
        store=True,
    )
    output_text = getattr(resp, "output_text", None)
    try:
        if isinstance(output_text, str):
            return json.loads(output_text)
    except Exception:
        pass
    return {"raw": output_text}


