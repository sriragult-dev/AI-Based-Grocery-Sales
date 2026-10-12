# AI Based Grocery Sales Prediction & Forecasting

Random Forest model + Streamlit web app.

## Files
- `app.py` - the web app (trains the Random Forest on first load, then predicts)
- `requirements.txt` - libraries to install
- `K_class_sales_clean.csv` - dataset (must be in the same folder as app.py)
- `grocery_sales_rf.py` - optional: full training/EDA/tuning script for your report

## Run on your laptop
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501

## Host it FREE on Streamlit Community Cloud
1. Create a GitHub account and a new **public** repository (example: `grocery-sales-predictor`).
2. Upload these 3 files to the repo root: `app.py`, `requirements.txt`, `K_class_sales_clean.csv`
   (GitHub website -> Add file -> Upload files -> Commit).
3. Go to https://share.streamlit.io and sign in with GitHub.
4. Click **Create app** -> choose your repo -> branch `main` -> main file path `app.py` -> **Deploy**.
5. Wait 2-3 minutes. You get a public link like `https://your-app-name.streamlit.app`.

The first load trains the model (a few seconds); after that it is cached.

## Other free options
- Hugging Face Spaces (choose the Streamlit SDK, upload the same 3 files)
- Render.com web service with start command:
  `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## Common errors
- `FileNotFoundError: K_class_sales_clean.csv` -> the CSV is not in the repo root.
- App sleeps after some inactivity on the free tier; just open the link and it wakes up.
