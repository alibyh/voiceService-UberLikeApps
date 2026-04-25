# Hosting on `mauri-alpha.com`

The plan: put the API on a **subdomain** like `api.mauri-alpha.com` so it lives separately from the existing Squarespace website. The Squarespace-hosted site at `mauri-alpha.com` is untouched — we only add one DNS record.

---

## Recommended: Railway

Railway is the easiest path. It deploys straight from a GitHub repo, gives you a public HTTPS URL, supports custom domains, and provides persistent volumes for the FAISS index. Pricing is ~$5/month on the Hobby plan, plus a small amount for resource usage.

### Step 1 — Push the project to GitHub

```bash
cd "/Users/alibyh/Desktop/Projects/voiceApp 2/voiceService-UberLikeApps"
git init
git add .
git commit -m "initial commit"
# create a private repo on github.com first, then:
git remote add origin git@github.com:<you>/voiceService-UberLikeApps.git
git push -u origin main
```

⚠️ Before pushing: make sure your real `.env` is **not** staged. The `.gitignore` shipped with this repo excludes `.env` already. `git status` should not show it. If it does, stop and fix `.gitignore` first.

### Step 2 — Add a Dockerfile

Railway will auto-detect Python and try to install requirements, but the BGE-M3 model + FAISS install needs more memory than Railway's default builder gives. A small Dockerfile fixes that and makes builds reproducible:

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build indexes at container build time so cold starts are fast.
# Comment this out if you'd rather rebuild on first boot via the Railway shell.
RUN python -m matcher.index_build

EXPOSE 8000
CMD ["uvicorn", "matcher.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

And a `.dockerignore`:

```
.venv
.env
__pycache__
.pytest_cache
logs
matcher/data/index
```

### Step 3 — Create the Railway project

1. Go to **railway.app** → sign up / log in.
2. **New Project** → **Deploy from GitHub repo** → pick your repo.
3. Railway detects the Dockerfile and starts building. First build is slow (~5–10 min) because BGE-M3 downloads.

### Step 4 — Set environment variables

In the Railway service → **Variables** tab → add:

```
ANTHROPIC_API_KEY=<your real key>
OPENAI_API_KEY=<your real key>
RERANKER_MODEL=claude-haiku-4-5
EMBEDDING_MODEL=BAAI/bge-m3
ASR_MODEL=whisper-1
PLACES_PATH=matcher/data/places.json
INDEX_DIR=matcher/data/index
CONFIDENCE_THRESHOLD=0.85
CONFIDENCE_MARGIN=0.1
```

Railway redeploys automatically once you save.

### Step 5 — Smoke test

In the Railway service → **Settings** → **Networking** → **Generate Domain**. You'll get something like `voiceservice-production.up.railway.app`. Test it:

```bash
curl https://voiceservice-production.up.railway.app/health
curl -X POST https://voiceservice-production.up.railway.app/resolve \
  -H "Content-Type: application/json" \
  -d '{"query":"بتروديس الشارة","top_k":3}'
```

### Step 6 — Connect `api.mauri-alpha.com`

In Railway → **Settings** → **Networking** → **Custom Domain** → enter `api.mauri-alpha.com`. Railway shows you a CNAME target like `xyz.up.railway.app`.

Then in Squarespace:

1. Squarespace Dashboard → your domain → **DNS Settings** → **Add Record**.
2. Type: `CNAME`
3. Host: `api`
4. Data: the CNAME target Railway gave you (`xyz.up.railway.app`)
5. Save.

DNS propagates in ~5–30 minutes. Railway issues a Let's Encrypt certificate automatically once the CNAME resolves. Your website on `mauri-alpha.com` keeps running normally — only the `api` subdomain points elsewhere.

Then point your React Native app's `API_BASE` at `https://api.mauri-alpha.com` and you're live.

---

## Resource sizing

The matcher needs ~3 GB RAM resident (BGE-M3 model + PyTorch overhead). Railway's Hobby plan gives 8 GB RAM and 8 vCPU — fine. Pro tier gives 32 GB.

If you want a leaner deploy, build the indexes with `--skip-semantic` (in the Dockerfile change `RUN python -m matcher.index_build` to `RUN python -m matcher.index_build --skip-semantic`). The pipeline still works — recall is lower but the LLM reranker compensates well in practice.

---

## Alternatives

If Railway doesn't fit, all of these are similar in spirit (managed PaaS, GitHub-based deploys, custom domains, automatic HTTPS):

| Service | Notes |
|---|---|
| **Render** | Closest Railway equivalent. Free tier exists but spins down after 15 min idle (cold starts hurt). Paid Starter is $7/mo. |
| **Fly.io** | More powerful, more knobs. Closer to a VPS in spirit. Free tier is generous. Good if you want regional control (e.g., deploy in Frankfurt for low Mauritania latency). |
| **Hugging Face Spaces** | Built for ML services, free tier with limits. No custom domain on free tier. |
| **Northflank, Koyeb** | Less mainstream but similar UX. |

For all of them: the DNS step is the same — add a CNAME record `api.mauri-alpha.com → <provider's hostname>` in Squarespace.

---

## Things to know

- **Cold-start latency.** First request after a deploy or sleep is slow (~10–20 s) because BGE-M3 has to load into RAM. Railway keeps containers warm on Hobby+; on free tiers (Render free) you'll feel this every ~15 min. A simple cron that pings `/health` every 10 minutes keeps the container warm if needed.
- **API keys.** Live in Railway's **Variables** tab — never commit them. `.env` is for local dev only.
- **Logs.** Railway has built-in log viewing; the JSONL resolution log (`logs/resolutions.jsonl`) lives in the container's ephemeral filesystem. If you want to keep that data, mount a volume on `/app/logs` (Railway → service → **Volumes**).
- **CORS.** The mobile app doesn't need CORS. If you ever build a web client at `mauri-alpha.com`, add `fastapi.middleware.cors` to `matcher/api.py` and allow the website's origin.
- **Cost ballpark.** Railway Hobby ~$5/mo + usage (~$5–10/mo for a small instance running 24/7) + ~$0.0002 per Haiku call + ~$0.006 per minute of Whisper audio. Realistically $10–20/month all-in for a test deployment.
