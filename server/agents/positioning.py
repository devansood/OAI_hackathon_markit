from .agent_base import call_gpt5_json

PROMPT = (
    "You are the Positioning Agent. Using web search only, produce tight positioning: \n"
    "{\"tagline\": \"\", \"category\": \"\", \"value_props\": [\"\"], \"proof_points\": [\"\"]}"
)

def run(api_key: str, email: str) -> dict:
    return call_gpt5_json(api_key, PROMPT, email)


