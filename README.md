# Memoir Prototype

Memoir is a webapp prototype for **AR-glasses–style memory support**.  
It helps early-stage dementia patients (and power users like doctors, teachers, delegates) **recognize faces and recall past conversations**.

---

## 🌟 Core Features (First Sprint)

- Face recognition handled by **face-api.js** (in browser).
- Speech-to-text handled by **Web Speech API** (in browser).
- Backend: **Flask + SQLite**, with conversation storage and summarization via Gemini.
- Memory Bank site: contact list, unknown profiles, and conversation history with summaries + full transcripts.

---

## 🛠 Requirements

- **Python 3.9.x**  
  (Tested on **3.9.13** for Windows and macOS. Any 3.9.x patch release is safe.)
- Git
- (Optional) Google Gemini API key for summarization (`.env` file)

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/mukhokironmoy/Memoir_2.0.git
cd memoir
```

---

### 2. Create Virtual Environment

#### Windows (PowerShell)

```powershell
# Create venv with Python 3.9
py -3.9 -m venv .venv

# Activate
.\.venv\Scripts\Activate.ps1

# Check version
python --version
```

#### macOS

```bash
# Install Python 3.9 if not already (via Homebrew)
brew install python@3.9

# Create and activate venv
python3.9 -m venv .venv
source .venv/bin/activate

python --version
```

#### Linux

```bash
# Ensure Python 3.9 is installed (example: Ubuntu)
sudo apt-get update
sudo apt-get install python3.9 python3.9-venv

# Create and activate venv
python3.9 -m venv .venv
source .venv/bin/activate

python --version
```

---

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

### 4. Setup Environment Variables

Create a `.env` file in the project root:

```ini
SECRET_KEY=dev-secret

# Optional: add Gemini API key for summarization
# GEMINI_API_KEY=your_api_key_here
```

---

### 5. Initialize Database with Seed Data

This will create `instance/memoir.db` and populate it with sample people, conversations, and turns.

```bash
python -m scripts.seed_db
```

Example output:

```
Seed complete!
People: 4 | Conversations: 4 | Turns: 12 | Embeddings: 2
```

---

### 6. Run the App

```bash
python run.py
```

Open in your browser:

- [http://127.0.0.1:5000/ping](http://127.0.0.1:5000/ping) → check server status
- [http://127.0.0.1:5000/glasses/home](http://127.0.0.1:5000/glasses/home) → blueprint test
- [http://127.0.0.1:5000/memory_bank/home](http://127.0.0.1:5000/memory-bank/home) → blueprint test

---

## 📂 Project Structure

```
memoir/
├─ app/
│  ├─ __init__.py          # Flask app factory
│  ├─ models.py            # Database models
│  ├─ blueprints/          # Feature routes
│  │  ├─ glasses.py
│  │  └─ memory_bank.py
│  ├─ services/            # AI clients (summarizer, face API)
│  ├─ templates/           # HTML templates
│  └─ static/              # Static assets (CSS/JS/img)
├─ scripts/
│  └─ seed_db.py           # Seeds DB with sample data
├─ instance/               # SQLite DB lives here (gitignored)
├─ run.py                  # App entrypoint
├─ requirements.txt        # Python deps
├─ .gitignore
└─ README.md
```

---

## 📌 Notes

- The backend is **API-agnostic**: you can swap face/STT providers later without schema changes.
- For production or deployment, replace SQLite with PostgreSQL/MySQL and configure via `SQLALCHEMY_DATABASE_URI`.

---
