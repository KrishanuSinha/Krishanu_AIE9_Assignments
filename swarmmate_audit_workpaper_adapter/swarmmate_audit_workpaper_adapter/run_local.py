from dotenv import load_dotenv
from app.graphs.workpaper_adapter_with_guardrails import graph

load_dotenv()

result = graph.invoke(
    {
        "messages": [
            {
                "role": "human",
                "content": (
                    "Draft an audit planning memo for this client. "
                    "Highlight likely risk areas, open requests, and include citations to the source files."
                ),
            }
        ]
    }
)

messages = result.get("messages", [])
if messages:
    print(messages[-1].content)
else:
    print(result)