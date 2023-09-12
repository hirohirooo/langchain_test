import os
from dotenv import load_dotenv

# ↓もうサポートしていなかった
# from langchain.llms import OpenAI 
from langchain.chat_models import ChatOpenAI #代わりにこっちつかう

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# .env ファイルから環境変数をロード
load_dotenv()

# 環境変数を取得
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# ChatOpenAIモデルを初期化
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=500)

# テキストドキュメントを読み込む
# ここではwikiのチェーンソーマンのデンジの部分をコピペしたものを用意した
loader = TextLoader('./source/denji.txt')
data = loader.load()

# テキストを指定されたチャンクサイズに分割
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
texts = text_splitter.split_documents(data)

# OpenAIの埋め込み機能を使用
embeddings = OpenAIEmbeddings()
# テキストのベクトル表現を保存するデータベースを作成
db = Chroma.from_documents(texts, embeddings)

# ベクトルデータベースを検索するためのリトリーバーを作成
retriever = db.as_retriever()

# 質問応答チェーンを初期化
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# 質問を入力して、回答を取得
query = "チェンソーの悪魔とはなんですか？"
print(chain.run(query))