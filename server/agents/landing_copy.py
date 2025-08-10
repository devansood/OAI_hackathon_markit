from .agent_base import call_gpt5_json

PROMPT = (
    "You are the Landing Copy Agent. Using web search only, generate concise hero+subhead and 3 bullet benefits: \n"
    "{\"hero\": \"\", \"subhead\": \"\", \"bullets\": [\"\", \"\", \"\"]}"
)

def run(api_key: str, email: str) -> dict:
    return call_gpt5_json(api_key, PROMPT, email)


