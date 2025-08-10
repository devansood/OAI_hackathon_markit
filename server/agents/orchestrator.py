import asyncio
from typing import Any, Dict
import httpx

from . import run_positioning, run_landing_copy, run_ads, run_emails

async def run_all_parallel(api_key: str, email: str) -> Dict[str, Any]:
    loop = asyncio.get_running_loop()
    # Run CPU-bound wrappers in threads since OpenAI client is sync
    positioning_fut = loop.run_in_executor(None, run_positioning, api_key, email)
    landing_fut = loop.run_in_executor(None, run_landing_copy, api_key, email)
    ads_fut = loop.run_in_executor(None, run_ads, api_key, email)
    emails_fut = loop.run_in_executor(None, run_emails, api_key, email)
    positioning, landing, ads, emails = await asyncio.gather(positioning_fut, landing_fut, ads_fut, emails_fut)
    return {
        "positioning": positioning,
        "landing_copy": landing,
        "ads": ads,
        "emails": emails,
    }

async def save_to_airtable(base_id: str, table: str, api_key: str, record_id: str, payload: Dict[str, Any]) -> None:
    # Store the consolidated orchestration result into one long-text field by id
    fields = {"fldNLJlEqVwvOg100": __import__("json").dumps(payload)}
    async with httpx.AsyncClient(timeout=30) as http:
        url = f"https://api.airtable.com/v0/{base_id}/{table}"
        await http.patch(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json={"records": [{"id": record_id, "fields": fields}]})


