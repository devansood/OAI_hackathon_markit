from .agent_base import call_gpt5_json

PROMPT = (
    "You are the Paid Ads Agent. Using web search only, produce 3 ad variants for Meta/Google: \n"
    "{\"ads\": [{\"headline\": \"\", \"primary\": \"\", \"cta\": \"\"}], \"keywords\": [\"\"]}"
)

def run(api_key: str, email: str) -> dict:
    return call_gpt5_json(api_key, PROMPT, email)


