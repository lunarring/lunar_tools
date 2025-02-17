import os
import requests

def search(query: str) -> list:
    # Determine whether to use simulation mode via an environment variable.
    simulate = os.environ.get("SIMULATE_WEB_SEARCH", "1") == "1"
    if simulate:
        return [{'title': 'OpenAI', 'url': 'https://www.openai.com'}]
    else:
        try:
            # For demonstration, using DuckDuckGo API's JSON response.
            response = requests.get(
                'https://api.duckduckgo.com',
                params={'q': query, 'format': 'json'},
                timeout=5
            )
            data = response.json()
            results = []
            # Parse results: method dependent on API response structure.
            for item in data.get('RelatedTopics', []):
                if isinstance(item, dict) and 'Text' in item and 'FirstURL' in item:
                    results.append({'title': item['Text'], 'url': item['FirstURL']})
            if not results:
                results = [{'title': query, 'url': f'https://duckduckgo.com/?q={query}'}]
            return results
        except Exception:
            # Fallback to simulation mode if any error occurs.
            return [{'title': 'OpenAI', 'url': 'https://www.openai.com'}]
