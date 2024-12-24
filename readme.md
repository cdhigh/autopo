# Brief
Use AI to automatically translate gettext PO files.

# Requirements
1. python >= 3.8
2. polib

# Usage
```bash
python autopo.py --config config.json --dest fr path/to/messages.po
```

# config.json format
```json
{
  "provider": "openai/google/anthropic/xai/mistral/groq/perplexity/alibaba",
  "model": "",
  "api_key": "",
  "api_host": "",
  "chat_type": "multi_turn/single_turn"
}
```
