from dotenv import load_dotenv
import os
from together import Together
import whisper
from transformers import AutoTokenizer
from pydub import AudioSegment
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import time

# === 事前準備 ===
def get_audio_format(file_path):
    ext = os.path.splitext(file_path)[1].lower().replace('.', '')
    return ext 
def get_language_name(code):
    return {
        "ja": "Japanese",
        "en": "English",
        "zh": "Chinese"
    }.get(code, "Japanese")

#仮想環境での開発なので、必ず使用する際には「.\.venv\Scripts\Activate」で実行し、停止させる際には「deactivate」
def processor(audio_file:str,language,prompt_type:str)->str:
    load_dotenv()
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("APIキーが読み込まれていません")
    llm = ChatOpenAI(
        openai_api_key = api_key,
        openai_api_base  = "https://api.together.xyz/v1",
        model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        temperature = 0.2,
        max_tokens=800
        )
    #音声の抽出(テストのみ)
    # === 最初の10分だけ切り出す（10分 = 600,000ミリ秒）===
    audio_format = get_audio_format(audio_file)
    audio = AudioSegment.from_file(audio_file, format=audio_format)
    duration_ms = len(audio)
    segment_ms = 600_000  # 10分
    whole_text = []
    # Whisperで音声をテキストに変換
    print("Whisper モデル読み込み中...")
    model = whisper.load_model("base")
    print("Whisper モデル読み込み完了")
    for i, start in enumerate(range(0, duration_ms, segment_ms)):
        end = min(start + segment_ms, duration_ms)
        chunk = audio[start:end]
        chunk_filename = f"segment_{i+1}.mp3"
        chunk.export(chunk_filename, format="mp3")
        print(f"{chunk_filename} にエクスポート完了")

        # Whisperで文字起こし(言語を日本語だと指定)
        result = model.transcribe(chunk_filename)
        whole_text.append(result["text"])
        os.remove(chunk_filename)

    #print("===== 文字起こし結果（先頭100字）=====")
    #print(transcribed_text[:100], "...")

    # === トークン制限対応：トークン数で分割して処理 ===
    #Tokenをgpt2に変更
    tokenizer = AutoTokenizer.from_pretrained("gpt2",model_max_length=99999)

    def split_by_token_limit(text, max_tokens=2000):
        words = text.split()
        chunks = []
        current_chunk = []
        token_count = 0

        for word in words:
            current_tokens = len(tokenizer.encode(word, add_special_tokens=False))
            if token_count + current_tokens > max_tokens:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                token_count = current_tokens
            else:
                current_chunk.append(word)
                token_count += current_tokens
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    #print("===== 文字起こし結果（先頭100字）=====")
    full_text = "\n".join(whole_text)
    chunks =split_by_token_limit(full_text)

    # === 各チャンクを要約 ===
    #LangchainのLLMChainを設定
    messages = """
        以下は会議の録音内容の一部です。この内容から、話し合われたトピック・参加者・会社名・議論内容・決定事項など、重要なポイントを箇条書きまたは短い段落でまとめてください。書式は不要です。
        また、言語の出力は{language}に合わせてください。
        {chunk}
        """
    prompt = PromptTemplate(
        input_variables=["chunk","language"],
        template=messages)
    partial_chain = LLMChain(llm=llm,prompt=prompt)
    #チャンクごとに出力
    partial_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"\n--- 要約中: チャンク {i+1}/{len(chunks)} ---")
        response = partial_chain.run({"chunk":chunk,"language":language})
        time.sleep(10)
        partial_summaries.append(response)

    # === 全体を統合して議事録形式に要約 ===
    if("委員会" in prompt_type):
        system_message = """
        次の文章は、委員会の録音の文字起こし・要約結果です。この内容を以下のフォーマットに従って再構成してください。
    【フォーマット】
    スタイル:
    会議名：◯◯会議　議事録
    日時：◯月△日(✕月)　会議開始時間～会議終了時間
    開催場所：◯◯
    出席者：◯◯先生、◯◯先生 (参加者(数字)ではなく、参加者の名前を記載。不明ならば参加者(数字)で記載) 議事録作成：AI
    欠席者：◯◯先生（出張のため）
    審査案件: 文書に記載されている通りの会社名を記載 (会社名が1つでなければ、1つ以上記載)
    <議題>：新規審査(新規ならば)or定期報告(定期的ならば)or変更審査(以前の審査で存在し、変更された審査) 
    機関、会社名
    管理者名
    この会社に関する議論内容
    事務局受領日:審査を受け付けた日付
    決議不参加: 参加者の中でこの審査に関わっていないか、 ー(無し)
    説明者:ー(無し)

    決定事項：
    ・記載されている決定事項
    ・無ければ(無)と記載

    次回の会議:
    会議名：◯◯会議　議事録
    日時：◯月△日(✕月□日)　会議開始時間～会議終了時間
    無ければ(未定)と記載
    """
    elif("勉強会" in prompt_type):
        system_message = """次の文章は、勉強会の録音の文字起こし・要約結果です。この内容を以下のフォーマットに従って再構成してください。
        【フォーマット】
        第(数字)回 勉強会 プログラム
        日時 ： 記載されている日付（曜日）何時開始~何時終了
        場所 ： 記載されている開催場所
        主催 ：  記載されている主催者名(役職名含む)
        開会　   (開会式を行った人物の名前と役職名)
        テーマ	     
        (討論会で討論した内容のテーマ(1~))
        発表者	 (発表を行った人物の名前と役職名)
        内容　 ①(発表内容の文字起こし(記載されて
        司会　  (司会を行った人物の名前と役職名)
        閉会　 (閉会式を行った人物の名前と役職名)
        ※下記のプロンプトもユーザーに必要かどうか聞かずに続行してください。
        <喋っている人物の名前>その人物が喋った内容の文字起こし(記載されている内容を要約・中略せずにそのまま掲載ただし、綺麗な文章にして掲載してください)
        ※すでにドキュメントに記載されていても、綺麗な文章にして全文(概要では無くを)出力し直してください。
        テーマの移り変わり(記載されているテーマ名 人物名)"""
    else:
        system_message = """{prompt_type}のフォーマットに沿って要約・成型してください。また出力言語は{language}に合わせてください。"""
    #トークン数を制限
    tokenizer = AutoTokenizer.from_pretrained("gpt2", model_max_length=99999)

    # 1. 全チャンク要約を連結
    final_input = "\n\n".join(partial_summaries)

    # 2. トークン数を計算して制限（8000未満）
    input_tokens = tokenizer.encode(final_input, add_special_tokens=False)
    max_input_tokens = 7300  # 安全
    if len(input_tokens) > max_input_tokens:
        print(f"入力トークン数が多すぎます: {len(input_tokens)} → 切り詰めます")
        final_input = tokenizer.decode(input_tokens[:max_input_tokens])
    else:
        print(f"入力トークン数 OK: {len(input_tokens)}")

    ###
    ##LangchainのLLMChainを設定
    final_messages = system_message + "\n\n以下は要約結果です。\n\n{final_input}"
    prompt = PromptTemplate(
        input_variables=["final_input","language"],
        template=final_messages)
    final_chain = LLMChain(llm=llm,prompt=prompt)
    final_response = final_chain.run({"final_input":final_input, "prompt_type": prompt_type,"language":language})
    #final_response = client.chat.completions.create(
   #    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
   #    messages=[
   #        {
   #            "role": "system",
   #            "content": system_message
   #        },
   #        {
   #            "role": "user",
   #            "content": final_input
   #        }
   #    ],
   #    max_tokens=800,
   #    temperature=0.2
   #)

    print("\n===== 最終議事録の要約結果 =====")
    summary = final_response
    return whole_text,partial_summaries,summary

