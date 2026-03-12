# Dobbi CS Assistant 🧺

AI-powered customer support tool that helps Dobbi's CS team draft responses faster.

## What it does

1. CS employee pastes a customer message (email/WhatsApp)
2. Tool classifies the question (pricing, order status, complaint, etc.)
3. Tool finds relevant info from FAQ, price list, and terms & conditions
4. Tool generates a draft response in the same language (Dutch or English)
5. CS employee reviews, edits if needed, and sends to customer

## Live Demo

The tool is deployed and accessible at:
**[https://dobbi-cs-tool-production.up.railway.app](https://dobbi-cs-tool-production.up.railway.app)**

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| LLM | Claude API (Anthropic) |
| Embeddings | HuggingFace Inference API |
| Vector Database | ChromaDB |
| Hosting | Railway |
| Language | Python 3.12 |

## Architecture

```
Customer Message
       │
       ▼
   [Claude API] ──→ Category (pricing/complaint/etc)
       │
       ▼
   [HuggingFace API] ──→ Embedding vector
       │
       ▼
   [ChromaDB] ──→ 15 relevant FAQ/price items
       │
       ▼
   [Claude API] ──→ Draft response in Dutch/English
       │
       ▼
   Display to CS employee
```

## Knowledge Base

| Source | Items | Description |
|--------|-------|-------------|
| faq_en.json | 28 | English FAQ |
| faq_nl.json | 28 | Dutch FAQ |
| terms_en.json | 25 | Terms & Conditions |
| prices.csv | 70 | Price list |
| **Total** | **151** | |

## Local Development

### Prerequisites

- Python 3.10 or higher
- Claude API key from Anthropic
- HuggingFace API token

### Setup

```bash
# Clone the repo
git clone https://github.com/johben31/dobbi-cs-tool.git
cd dobbi-cs-tool

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
```

Then open `.env` and add your API keys:
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
HF_API_TOKEN=hf_your-token-here
```

### Index the knowledge base

```bash
python src/indexer.py
```

You should see:
```
Initializing indexer with HuggingFace API...
Indexer ready!
Embedding 28 FAQ items...
Indexed 28 FAQ items from knowledge_base/faq_en.json
...
Total items in index: 151
```

### Run the app

```bash
streamlit run src/app.py
```

Opens in your browser at `http://localhost:8501`

### Stop the app

Press `Ctrl + C` in the terminal.

## Project Structure

```
dobbi-cs-tool/
├── knowledge_base/
│   ├── faq_en.json       # English FAQ (28 items)
│   ├── faq_nl.json       # Dutch FAQ (28 items)
│   ├── terms_en.json     # Terms & Conditions (25 items)
│   └── prices.csv        # Price list (70 items)
├── src/
│   ├── indexer.py        # Embeds knowledge via HuggingFace API
│   ├── retriever.py      # Semantic search via HuggingFace API
│   ├── classifier.py     # Categorizes questions (Claude API)
│   ├── generator.py      # Generates responses (Claude API)
│   ├── pipeline.py       # Connects everything
│   └── app.py            # Streamlit UI
├── chroma_db/            # Vector database (created after indexing)
├── requirements.txt
├── Procfile              # Railway deployment config
├── .python-version       # Python version for Railway
├── .env.example
└── README.md
```

## Updating the Knowledge Base

When you add new FAQ items or update prices:

1. Edit the relevant file in `knowledge_base/`
2. Delete the old index: `rm -rf chroma_db`
3. Re-run: `python src/indexer.py`
4. Push to GitHub (Railway will auto-redeploy)

## Deployment

The app is deployed on Railway. Any push to the `main` branch triggers auto-deployment.

### Environment Variables (Railway)

| Variable | Description |
|----------|-------------|
| ANTHROPIC_API_KEY | Claude API key |
| HF_API_TOKEN | HuggingFace API token |

## API Costs

| API | Cost | Free Tier |
|-----|------|-----------|
| Claude (Anthropic) | ~$0.01 per message | - |
| HuggingFace | Free | 30,000 requests/month |
| Railway | ~$5-10/month | 500 hours/month |

## Future Improvements (V2)

- [ ] GDPR-compliant anonymization before sending to Claude
- [ ] Usage statistics dashboard
- [ ] Feedback buttons (👍👎) for quality tracking
- [ ] Integration with Dobbi backoffice API

## Team

Built by Digital Impact Lab team (UvA 2026) for Dobbi.

## License

Private repository - Dobbi internal use only.
