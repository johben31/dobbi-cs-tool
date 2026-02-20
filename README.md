# Dobbi CS Assistant 🧺

AI-powered customer support tool that helps Dobbi's CS team draft responses faster.

## What it does

1. CS employee pastes a customer message (email/WhatsApp)
2. Tool classifies the question (pricing, order status, complaint, etc.)
3. Tool finds relevant info from FAQ and price list
4. Tool generates a draft response
5. CS employee reviews, edits if needed, and sends to customer

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Claude API key from Anthropic

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

Then open `.env` and add your Claude API key:
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Index the knowledge base
```bash
python src/indexer.py
```

You should see:
```
Loading embedding model...
Indexer ready!
Indexed 3 FAQ items...
Indexed 4 price items...
Total items in index: 7
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
│   ├── faq_en.json      # English FAQ (real data to be added)
│   ├── faq_nl.json      # Dutch FAQ (real data to be added)
│   └── prices.csv       # Price list (real data to be added)
├── src/
│   ├── indexer.py       # Embeds knowledge into vector DB
│   ├── retriever.py     # Searches for relevant info
│   ├── classifier.py    # Categorizes questions (Claude API)
│   ├── generator.py     # Generates responses (Claude API)
│   ├── pipeline.py      # Connects everything
│   └── app.py           # Streamlit UI
├── chroma_db/           # Vector database (created after indexing)
├── requirements.txt
├── .env.example
└── README.md
```

## Updating the Knowledge Base

When you add new FAQ items or update prices:

1. Edit `knowledge_base/faq_en.json` or `knowledge_base/prices.csv`
2. Delete the old index: `rm -rf chroma_db`
3. Re-run: `python src/indexer.py`
