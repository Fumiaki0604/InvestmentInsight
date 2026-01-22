# Mutual Funds Insight

投資信託の基準価額データを分析し、チャート表示・サマリー生成・AI分析まで提供する Streamlit アプリケーションです。Google Spreadsheet に保存した各銘柄データを取得し、テクニカル指標の計算や銘柄間相関分析、Slack 通知、OpenAI を使った分析レポート生成を行います。

## 現在の進捗
- Streamlit ベースの UI を実装（詳細分析／銘柄一覧／相関分析タブ表示）
- Google Sheets ユーティリティとテクニカル指標計算モジュールを整備
- OpenAI を利用した詳細分析生成・チャット機能を追加
- Slack 通知（投資判断ステータスの変更検知）と相関分析の可視化を実装
- Poetry / requirements.txt 双方で依存関係を管理

## ディレクトリ構成
```
mutual-funds-app/
├── data/                    # ローカル検証用サンプルデータ
├── streamlit_app/
│   ├── app.py               # Streamlit アプリ本体
│   └── utils/               # Google Sheets / 指標計算 / 相関分析 ほか
├── src/app/                 # 将来の再利用向けライブラリコード（FastAPI スケルトン含む）
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

Streamlit で `st.secrets` を使う場合は、`[gcp_service_account]` セクションに同等 JSON を格納すれば動作します。

## アプリの起動
```bash
streamlit run streamlit_app/app.py
```
起動後、ブラウザに表示されるメニューから銘柄を選択して分析を進めます。Slack 通知テストやシミュレーションは「銘柄一覧」タブから利用できます。

## テスト
```bash
pytest
```

## 将来の拡張アイデア
- Google Sheets から高値・安値などの列を取得して DMI 精度を向上
- FastAPI ベースの API を公開し、別フロントエンドからも利用できるようにする
- OpenAI プロンプトや出力テンプレートのカスタマイズ
- データキャッシュやバックグラウンド更新の最適化

---

## Claude Code ガイド (旧 claude.md)

### プロジェクト概要
投資信託の基準価額データを分析し、テクニカル指標の計算、チャート表示、AI分析レポートを提供するStreamlitアプリケーション。Google Sheetsからデータを取得し、OpenAI APIを使った投資分析とSlack通知機能を備える。

### コマンド

#### 開発・実行
```bash
# 仮想環境のアクティベート（Windows）
.venv\Scripts\activate

# 仮想環境のアクティベート（Linux/macOS）
source .venv/bin/activate

# 依存パッケージのインストール
pip install -r requirements.txt

# Streamlitアプリの起動
streamlit run streamlit_app/app.py

# FastAPI サーバーの起動（将来の拡張用）
uvicorn src.app.main:app --reload
```

#### テスト・品質管理
```bash
# テストの実行
pytest

# コードフォーマット
black .
isort .

# リンター
ruff check .
```

### アーキテクチャ

#### ディレクトリ構成
- **streamlit_app/**: Streamlit UIのメインコード
  - **app.py**: エントリーポイント。3つのタブ（詳細分析・銘柄一覧・相関分析）を管理
  - **utils/**: ヘルパーモジュール群
    - **gsheet_helper.py**: Google Sheets APIからデータ取得
    - **chart_helper.py**: テクニカル指標計算（RSI, MACD, DMI, ボリンジャーバンド）とPlotlyチャート生成
    - **gpt_analysis.py**: OpenAI APIを使った投資分析レポート生成
    - **correlation_helper.py**: 銘柄間の相関分析
    - **slack_notifier.py**: 銘柄ステータス変更のSlack通知

- **src/app/**: FastAPIベースの将来の拡張用コード
  - **main.py**: FastAPI エンドポイント（/health, /tickers）
  - **indicators/technical.py**: テクニカル指標計算ロジック（SMA, MACD, ボリンジャーバンド）
  - **services/analytics_service.py**: データ取得と分析ロジック
  - **data_sources/loader.py**: Google Sheets / ローカルCSVからのデータロード

- **data/**: ローカルテスト用サンプルデータ

#### データフロー
1. `gsheet_helper.py`がGoogle Sheetsから各銘柄のシート（基準価額の時系列データ）を取得
2. `chart_helper.py`でテクニカル指標（RSI, MACD, 移動平均線など）を計算
3. Plotlyで価格チャートとテクニカル指標を可視化
4. `gpt_analysis.py`が計算されたテクニカルデータを基にOpenAI APIで投資分析を生成
5. チャット機能では、ユーザーの質問に対してAIが追加の分析を提供

#### 認証とシークレット管理
- **Google Sheets API**: サービスアカウントJSONキーを`GOOGLE_SERVICE_ACCOUNT_KEY`環境変数に設定（JSON文字列またはファイルパス）
  - Streamlit Cloud: `.streamlit/secrets.toml`に`[gcp_service_account]`セクションで設定
  - ローカル: `.env`ファイルまたは環境変数で設定
- **OpenAI API**: `OPENAI_API_KEY`環境変数を設定
- **Google Spreadsheet ID**: `GOOGLE_SPREADSHEET_ID`環境変数（デフォルト値: `1O3nYKIHCrDbjz1yBGrrAnq883Lgotfvvq035tC9wMVM`）
- **Slack通知**: `SLACK_WEBHOOK_URL`環境変数（オプション）
- **ローカルデータパス**: `DEFAULT_LOCAL_DATA_PATH`環境変数（デフォルト: `data/sample_nav.csv`）

#### キャッシング戦略
- `@st.cache_data(ttl=1800)`: Google Sheetsデータの取得結果を30分キャッシュ（`load_available_sheets`, `load_sheet_data`）
- `@lru_cache(maxsize=1)`: Google認証情報をメモリキャッシュ（`get_credentials`）
- セッションステート: チャット履歴を`st.session_state.chat_history_per_fund`に保存

### 重要な技術的ポイント

#### テクニカル指標の計算
- **RSI**: 14日間のRSI（買われすぎ/売られすぎ判定）
- **MACD**: 12/26/9日のEMAベース（トレンド転換シグナル）
- **移動平均線**: 25日/200日SMA（ゴールデンクロス/デッドクロス判定）
- **DMI**: ADX, +DI, -DIでトレンド強度を測定
- **ボリンジャーバンド**: 20日移動平均±2標準偏差

計算ロジックは`streamlit_app/utils/chart_helper.py`と`src/app/indicators/technical.py`に分散。将来的にはsrc配下に統一予定。

#### Slack通知の仕組み
`previous_fund_status.json`に前回の銘柄ステータスを保存し、投資判断（強気買い・買い・中立・売り・強気売り）が変更された場合にSlack通知を送信。

#### AIチャット機能
各銘柄ごとにチャット履歴を保持し、テクニカル指標データをコンテキストとしてOpenAI APIに送信。会話履歴は`st.session_state.chat_history_per_fund[sheet_name]`に保存。

#### OpenAI APIモデル
現在GPT-5 (`gpt-5-2024-10-14`) を使用。詳細分析生成とチャット機能の両方で利用。

### デプロイメント

#### Streamlit Cloud
`.streamlit/secrets.toml`に以下を設定:
```toml
OPENAI_API_KEY = "sk-..."
GOOGLE_SPREADSHEET_ID = "1O3nYKIHCrDbjz1yBGrrAnq883Lgotfvvq035tC9wMVM"
SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/..."

[gcp_service_account]
type = "service_account"
project_id = "..."
# ... (Google Cloud サービスアカウントのJSON内容)
```

#### ローカル開発
`.env.example`を`.env`にコピーして必要な値を設定。
