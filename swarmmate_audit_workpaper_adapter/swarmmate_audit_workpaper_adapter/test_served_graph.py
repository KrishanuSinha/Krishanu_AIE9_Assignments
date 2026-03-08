from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from app.graphs.workpaper_adapter_with_guardrails import graph


def main():
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)

    input_state = {
        "messages": [
            HumanMessage(
                content=(
                    "Draft an audit planning memo for this client. "
                    "Highlight likely risk areas, open requests, and include citations to the source files."
                )
            )
        ]
    }

    result = graph.invoke(input_state)

    print("\n" + "=" * 80)
    print("REPORT GENERATED")
    print("=" * 80)
    print(f"Title: {result.get('artifact_title', 'N/A')}")
    print(f"Saved to: {result.get('artifact_path', 'N/A')}")
    print("=" * 80)
    print(result.get("final_response", "No final report generated."))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
