from task import processor
from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        print("フォームが送信されました")

        file = request.files.get("audio")
        language = request.form.get("lang_type")
        prompt_type = request.form.get("prompt_type", "").strip()
        print(f"プロンプトタイプ: {prompt_type}")

        if file and file.filename:
            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)
            print(f"ファイルを保存しました: {path}")

            print("processor関数を実行します")
            transcription_list, summary_list, summary = processor(path,language,prompt_type)
            print("processor関数が完了しました")

            return render_template("index.html",
                                   transcription_list=transcription_list,summary_list=summary_list,
                                   summary=summary)

        print("音声ファイルが未選択です")
        return "音声ファイルがアップロードされていません", 400

    print("GETリクエストでindexページにアクセスされました")
    return render_template("index.html")

if __name__ == "__main__":
    print("Flaskアプリを起動します")
    app.run(debug=True)
