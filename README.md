# Environment State Server

Dify 判断用の state snapshot と、表示用 indicator snapshot を返すローカル集約サーバーです。

## Responsibility

- Home Assistant bridge の action state を保持する。
- Camera Hub topic と追加 vision topic を購読し、gesture、camera、room-light の snapshot を保持する。
- Dify 向けの認証付き API と、表示向けの loopback API を分ける。
- module health を軽く集約する。

このサーバーは物理カメラを開きません。gesture 推論も映像配信も担当しません。

## 初期セットアップ

```powershell
uv sync
uv run python -m unittest discover -s tests
```

この organ は Camera Hub、Vision Snapshot Processor、Home Assistant bridge などの状態を集約します。
単体で物理カメラや家電を初期化するものではありません。

## dotenv / local config

この repo 自身には標準の `.env.example` はありません。通常は Home Assistant bridge の
`.env` を `uv run --env-file ..\home-assistant-server\.env ...` で読みます。

必要な token:

- `ENVIRONMENT_API_TOKEN`: Environment API 専用 token。未設定なら Home Assistant bridge の
  `HOME_CONTROL_API_TOKEN` を共有できます。
- `HOME_CONTROL_API_TOKEN`: Dify / Thought Core / AITuberKit などが local bridge に送る token。
- `HOME_ASSISTANT_TOKEN`: Home Assistant bridge 側で使う長期 access token。

実 token、feedback JSONL、`.cache/`、`.venv/` はコミットしません。

## 通常起動

```powershell
cd <workspace>\environment-state-server
uv run --env-file ..\home-assistant-server\.env python -m environment_state_server.main `
  --host 127.0.0.1 `
  --port 8790 `
  --state-query-feedback-path .cache\environment_state_server\state_query_feedback.jsonl `
  --camera-hub-url ws://127.0.0.1:8765 `
  --vision-topic-url ws://127.0.0.1:8776 `
  --home-assistant-health-url http://127.0.0.1:8787/health `
  --aituber-url http://127.0.0.1:3000 `
  --dify-url http://127.0.0.1:8080 `
  --voicevox-health-url http://127.0.0.1:50021/version
```

`--disable-camera-hub` はテストや縮退運転だけに使います。代わりに別プロセスでカメラを開かないでください。

## API

| Endpoint | Consumer | Auth | Purpose |
|---|---|---|---|
| `GET /environment/current` | Dify | Bearer token | 家電、gesture、camera、module 状態、Dify 向け state query |
| `GET /environment/current?wait_for=room_light&after=<iso>&timeout_ms=1500` | Dify | Bearer token | 操作後の room-light snapshot を短時間待つ |
| `GET /environment/relations` | Dify | Bearer token | Dify issue ID と action ID の関連情報 |
| `POST /feedback/state-query` | Dify | Bearer token | 状態照会へのユーザー訂正ラベルを append-only 保存 |
| `GET /feedback/state-query/recent` | Dify / debug | Bearer token | 最近の状態照会 feedback を確認 |
| `GET /feedback/state-query/summary` | Dify / debug | Bearer token | label/status 件数、学習レベル、最新 feedback を確認 |
| `GET /indicators/current` | HUD / display-runtime | loopback only | 表示用に安全化した状態 |
| `GET /health` | launcher / check | none on loopback | process diagnostics |
| `GET /ready` | launcher / check | none on loopback | Dify 判断に使える鮮度か |

Auth token は `ENVIRONMENT_API_TOKEN` または `HOME_CONTROL_API_TOKEN` を使います。

## State Model

- Camera Hub topic と追加 vision topic は TTL を持つ snapshot として扱います。
- stale な camera / gesture / module は reason を付けます。
- `state_queries.room_light` は Dify が状態照会に使う projection です。画像判定の authority は `vision_snapshot_processor`、projection の担当は `environment_state_server` です。
- `state_queries.room_light.learning` には `/feedback/state-query/summary` と同じ学習レベルの要約が入ります。これは user_feedback の蓄積状況であり、画像判定そのものを上書きする authority ではありません。
- `state_queries.room_light` は `observed_at`、`updated_at`、`source_snapshot_id`、`stale_reason` を含みます。操作後確認では古い snapshot を post-action evidence として扱わないでください。
- `wait_for=room_light` の `wait_result.matched=true` は、`state_queries.room_light.observed_at` が `after` より新しく、`source_snapshot_id` が存在する場合だけです。HTTP 200 のまま `wait_result.matched=false` / `reason=timeout` を返すことがあります。
- Dify は「電気ついてる？」のような照会では `state_queries.room_light.available/stale/state/answer_hint/evidence` を第一参照し、probability 閾値や画像推論を持ちません。
- `POST /feedback/state-query` は `authority=user_feedback` の学習用ラベルだけを保存します。現在の `vision_snapshot_processor` 判定は即時に上書きしません。
- feedback は `schema_version`、`workflow_version`、`received_at`、`received_snapshot_id`、`idempotency_key`、`status`、`warnings` を保持します。古い pending は `accepted_with_warning` として保存します。
- `/feedback/state-query/summary` は `label_counts`、`status_counts`、`reason_counts`、`source_context_counts`、`action_counts`、`expected_state_counts`、`learning` を返します。`status_counts` は `accepted`、`accepted_with_warning`、`duplicate`、`rejected` の固定キーです。`duplicate` と `rejected` は実行中プロセスの診断カウントです。
- `learning.level` は `none`、`collecting`、`seeded`、`usable`、`reinforced` の5段階です。`learning.problems[]` には `code`、`severity`、`message` が入り、feedbackが少ない、on/offの片側がない、操作後確認がない、rejectがある、といった「学習できていない理由」を機械判定できます。
- `/environment/relations` は関連メタデータであり、Dify ID や Home Assistant ID の authority ではありません。
- display 用 endpoint は Dify relation や raw Home Assistant event history を省きます。

## Action Registry

Action registry は Home Assistant bridge 由来の action を Dify が選びやすい形に整えます。action ID と実行結果の authority は Home Assistant bridge 側にあります。

## Security

- Dify 用 API は Bearer token を要求します。
- 表示用 API は loopback 限定です。
- CORS はローカルの projection visual origins だけに限定します。
- API key、token、ローカルパスをログに出さないでください。
