echo "🚀 Starting Streamlit UI for Yelp Algorithm Prototype..."

# Ensure virtual environment exists
if [ ! -d ".venv" ]; then
  echo "🔧 Creating virtual environment (.venv)..."
  python3.10 -m venv .venv
fi

# Activate environment
source .venv/bin/activate

# Install dependencies if needed
echo "📦 Installing dependencies..."
pip install -r requirements.txt --quiet

# Run Streamlit
echo "🌐 Launching app at http://localhost:8501 ..."
streamlit run app_streamlit.py
