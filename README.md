# Mutual Funds Insight

投資信託の値動きを分析し、チャート表示・サマリー生成・AI解説まで提供する Streamlit アプリケーションです。Google Spreadsheet に保存した各銘柄データを取得し、テクニカル指標の計算や銘柄間相関分析、Slack 通知、OpenAI を使ったレポート生成を行います。

## 現在の進捗
- Streamlit ベースの UI を実装（詳細分析／銘柄一覧／相関分析タブ構成）
- Google Sheets 連携ユーティリティとテクニカル指標計算モジュールを整備
- OpenAI を利用した詳細分析生成・チャット機能を追加
- Slack 通知（様子見→買い推奨の変更検知）と相関分析の可視化を実装
- Poetry / requirements.txt 双方で依存関係を管理

## ディレクトリ構成
```
mutual-funds-app/
├── data/                    # ローカル検証用サンプルデータ
├── streamlit_app/
│   ├── app.py               # Streamlit アプリ本体
│   └── utils/               # Google Sheets / 指標計算 / 相関分析 ほか
├── src/app/                 # 今後の再利用向けライブラリコード（FastAPI スケルトン含む）
├── tests/                   # ユニットテスト
├── .env.example             # 環境変数テンプレート
├── requirements.txt
├── pyproject.toml
└── README.md
```

## セットアップ手順
1. **Python 3.11 以上**をインストールします。
2. 仮想環境を作成して有効化します。
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # macOS/Linux
   ```
3. 依存パッケージをインストールします。
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   # または Poetry を利用する場合は次の2行
   pip install poetry
   poetry install
   ```

## 環境変数の設定
`.env.example` を `.env` にコピーし、以下を設定してください。
- `GOOGLE_SERVICE_ACCOUNT_KEY` : Google Sheets API 用サービスアカウント JSON を文字列化したもの
- `GOOGLE_SPREADSHEET_ID` : 分析対象のスプレッドシート ID（既定値を変更する場合）
- `OPENAI_API_KEY` : OpenAI API キー
- `SLACK_WEBHOOK_URL` : 通知を行う場合に設定（任意）

Streamlit の `st.secrets` を使う場合は、`[gcp_service_account]` セクションに同じ JSON を格納すれば動作します。

## アプリの起動
```bash
streamlit run streamlit_app/app.py
```
起動後、ブラウザに表示されるメニューから銘柄を選択して分析を進めます。Slack 通知テストやシミュレーションは「銘柄一覧」タブから利用できます。

## テスト
```bash
pytest
```

## 今後の拡張アイデア
- Google Sheets から高値・安値などの列を取得して DMI 精度を向上
- FastAPI ベースの API を公開し、別フロントエンドからも利用できるよう整理
- OpenAI プロンプトや出力テンプレートのカスタマイズ
- データキャッシュやバックグラウンド更新の最適化