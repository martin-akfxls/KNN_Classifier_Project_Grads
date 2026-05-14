# ⚡ Pokémon Legendary KNN Classifier
### K-Nearest Neighbor Mini-System Prototype
**Course:** Data Mining and Business Intelligence | Special Laboratory Project

---

## 📋 Project Description
A complete KNN classification mini-system that predicts whether a Pokémon is
**Legendary or Non-Legendary** using its base battle statistics (HP, Attack, Defense,
Sp. Attack, Sp. Defense, Speed).

---

## 📁 Folder Structure
```
KNN_Classifier_Project/
│
├── data/
│   └── knn_dataset.csv           ← Place your Pokémon CSV file here
│
├── src/
│   ├── app.py                    ← Main Streamlit application
│   └── knn_functions.py          ← Standalone KNN helper functions
│
├── notebook/
│   └── KNN_Classifier_Prototype.ipynb  ← Jupyter Notebook version
│
├── outputs/
│   └── screenshots/              ← Save your screenshots here
│
├── report/
│   └── KNN_Project_Report.docx   ← Your written report
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Install and Run

### Step 1 — Install Python
Make sure you have **Python 3.9 or higher** installed.
Download from: https://www.python.org/downloads/

### Step 2 — Install dependencies
Open a terminal/command prompt in this project folder and run:
```bash
pip install -r requirements.txt
```

### Step 3 — Place your CSV file
Copy your Pokémon CSV file into the `data/` folder.
Expected columns: `Name`, `HP`, `Attack`, `Defense`, `Sp_Atk`, `Sp_Def`, `Speed`, `Class`

### Step 4 — Launch the Streamlit app
```bash
streamlit run src/app.py
```
The app will open automatically in your browser at `http://localhost:8501`

---

## 🔄 How to Use the App

| Step | Page | What to Do |
|------|------|------------|
| 1 | 🏠 Home | Read the project overview |
| 2 | 📂 Dataset Preview | Upload your CSV file and inspect the data |
| 3 | ⚙️ Model Setup & Train | Select features, K value, split ratio, then click **Train** |
| 4 | 📊 Evaluation Results | View confusion matrix and metrics |
| 5 | 🔮 New Record Prediction | Enter a Pokémon's stats and classify it |
| 6 | 📈 Visualizations | Explore charts and data distributions |

---

## 📊 Expected CSV Format

```
Name,HP,Attack,Defense,Sp_Atk,Sp_Def,Speed,Class
Bulbasaur,45,49,49,65,65,45,Non-Legendary
Mewtwo,106,110,90,154,90,130,Legendary
...
```

---

## 🔬 KNN Algorithm Used

1. Load and clean the dataset
2. Normalize features using Min-Max scaling
3. Split into 80% training / 20% testing
4. For each test record:
   - Compute Euclidean distance to every training record
   - Select the K nearest neighbors
   - Predict class by majority vote
5. Evaluate with confusion matrix, accuracy, precision, recall, F1

---

## 📦 Dependencies
- `streamlit` — web interface
- `pandas` — data handling
- `numpy` — numerical computation
- `matplotlib` + `seaborn` — visualizations
- `scikit-learn` — KNN classifier and evaluation metrics
