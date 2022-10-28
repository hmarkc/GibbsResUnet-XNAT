from flask import Flask, jsonify, request

from main import handle_request

app = Flask(__name__)


@app.route("/", methods=["POST"])
def fetch_args():
    json_data = request.get_json()
    if json_data is None:
        # handle_request(None, None)
        pass
    else:
        image_path: str = json_data.get("image_path", None)
        seg_path: str = json_data.get("seg_path", None)
        print(image_path, seg_path)
        # handle_request(image_path, seg_path)
    return jsonify({"ok": True})
