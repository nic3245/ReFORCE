import time
import logging
import openai
import httpx

logger = logging.getLogger("custom-client")

def call_llm(payload):
    payload_model = payload["model"]
    payload_messages = payload["messages"]
    payload_max_tokens = payload["max_tokens"]
    payload_top_p = payload["top_p"]
    payload_temperature = payload["temperature"]

    https_client = httpx.Client(proxy=None)
    client = openai.OpenAI(
        api_key="INSERT API KEY HERE",
        base_url="INSERT URL HERE",
        http_client=https_client
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=payload_model,
                messages=payload_messages,
                temperature=payload_temperature,
                top_p=payload_top_p,
                max_tokens=payload_max_tokens
            )
            return True, response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_info = e.response.json()
                    code_value = error_info['error']['code']
                    logger.error(f"API error code: {code_value}")
                except Exception:
                    code_value = 'unknown_error'
            else:
                code_value = 'unknown_error'
            if attempt < max_retries - 1:
                sleep_time = 4 * (2 ** (attempt + 1))
                logger.info(f"Retrying after {sleep_time} seconds...")
                time.sleep(sleep_time)

    return False, code_value
