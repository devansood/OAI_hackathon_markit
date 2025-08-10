from .agent_base import call_gpt5_json

PROMPT = (
    "You are the Email Agent. Using web search only, produce 2 short cold emails and 1 nurture sequence: \n"
    "{\"cold\": [{\"subject\": \"\", \"body\": \"\"}], \"nurture\": [{\"subject\": \"\", \"body\": \"\"}]}"
)

def run(api_key: str, email: str) -> dict:
    return call_gpt5_json(api_key, PROMPT, email)


