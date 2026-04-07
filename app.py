import os
from flask import Flask, jsonify, request
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.route("/")
def hello_world():
    return jsonify({"message": "Hello, World!"})


@app.route("/health")
def health_check():
    return jsonify({"status": "ok"})


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    OpenAI Chat Completions API 호출
    Request body:
      - messages: list of {role, content}  (required)
      - model: string                       (optional, default: gpt-4o-mini)
      - temperature: float                  (optional, default: 1.0)
      - max_tokens: int                     (optional)
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON"}), 400

    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        return jsonify({"error": "'messages' field is required and must be a list"}), 400

    model = body.get("model", "gpt-4o-mini")
    temperature = body.get("temperature", 1.0)
    max_tokens = body.get("max_tokens")

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    response = client.chat.completions.create(**kwargs)

    choice = response.choices[0]
    return jsonify({
        "id": response.id,
        "model": response.model,
        "message": {
            "role": choice.message.role,
            "content": choice.message.content,
        },
        "finish_reason": choice.finish_reason,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    })


@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
