Tennis Match Outcome Prediction (ATP \& WTA)

This repository contains a complete set of machine learning models for predicting tennis match outcomes across four markets: Match Winner, Total Games, Games Handicap, and First Set Winner. Models are trained separately for ATP (men) and WTA (women) using advanced feature engineering and the CatBoost algorithm.



The project demonstrates:



Feature engineering from historical match data



Model training with temporal validation and isotonic calibration



Comprehensive evaluation reports and diagnostic plots



A clean, modular codebase for reproducibility



⚠️ Educational Purpose Only: This project is intended for learning and demonstration. The models show strong validation metrics, but backtesting on 10,000 matches revealed negative ROI for the winner model. Real‑world betting involves significant financial risk; use at your own discretion.



📁 Repository Structure

text

.

├── .env.example                 # Example environment variables (not used in current code)

├── metrics\_models.txt           # Summary of model performance

├── ATP/                         # Saved ATP models and evaluation reports

│   ├── WINNER/model/winner\_atp

│   ├── TOTAL/model/total\_games\_atp

│   ├── HANDICAP/model/games\_diff\_atp

│   └── FIRSTSET/model/first\_set\_winner\_atp

├── WTA/                         # Saved WTA models and reports

│   └── ...

├── scr2/                        # Source code

│   ├── feature\_selection.py     # Feature importance analysis \& selection

│   ├── retrain\_final\_models.py  # Retrain all final models using selected features

│   ├── ATP/                     # Training scripts for ATP

│   │   ├── train\_winner\_atp\_clean.py

│   │   ├── train\_total\_games\_atp\_clean.py

│   │   └── ...

│   └── WTA/                     # Training scripts for WTA

│       └── ...

└── data/                        # Data directory (not included)

&nbsp;   └── dbexample.db              # Empty database placeholder

Note: The actual data files (.csv, .db) are not included in this repository. You must provide your own historical tennis data (see Data Preparation).



🔧 Requirements

Python 3.9+



Required packages: catboost, pandas, numpy, scikit-learn, matplotlib, seaborn, joblib



Install dependencies:



bash

pip install -r requirements.txt

(Create a requirements.txt with the packages listed above if you don't have one.)



📊 Data Preparation

The training scripts expect CSV files with the following columns (exact names may vary per script; see individual scripts for details):



date – match date (used for temporal split)



gender – 'ATP' or 'WTA'



player1\_id, player2\_id – unique player identifiers



surface – court surface (e.g., 'Clay', 'Hard', 'Grass')



winner\_1 – binary target: 1 if player1 won, else 0



total\_games – total games played in the match



games\_diff – games won by player1 minus games won by player2



first\_set\_bin – binary target: 1 if player1 won the first set, else 0



plus numerous feature columns (e.g., player form, head‑to‑head, surface statistics)



Place your prepared data files in the data/ folder. The main training scripts expect files like:



clean\_multimarket\_features.csv



winner\_features.csv



Refer to the code comments in each training script for the exact required column set.



🚀 How to Reproduce the Models

1\. Feature Selection

Run feature\_selection.py to identify the most important features for each target and gender. This script handles missing values, removes constant and highly correlated features, and computes feature importances using CatBoost.



bash

python scr2/feature\_selection.py

Outputs (saved in data/selected\_features/):



selected\_features\_<target>\_<gender>.txt – list of selected feature names



feature\_importance\_<target>\_<gender>.csv – feature importance values



2\. Train the Final Models

Run retrain\_final\_models.py to train all eight models (4 markets × 2 genders) using the top 11 features from the previous step. The script automatically saves models, calibrators, evaluation reports, and diagnostic plots.



bash

python scr2/retrain\_final\_models.py

Alternatively, you can train individual models using the scripts in scr2/ATP/ and scr2/WTA/, for example:



bash

python scr2/ATP/train\_winner\_atp\_clean.py

3\. Model Outputs

Each trained model is stored in a dedicated folder, e.g. ATP/WINNER/model/winner\_atp/, containing:



model.cbm – trained CatBoost model



isotonic\_calibrator.pkl – calibrator (for classification models)



evaluation\_report.json – metrics on validation set



feature\_importance.png – top‑20 feature importance plot



learning\_curve.png – training/validation loss



roc\_curve.png / residuals.png – diagnostic plots



📈 Model Performance

All models were validated on the most recent 20% of matches (temporal split). Probabilities are calibrated using isotonic regression where indicated.



ATP Models

Model	Metric	Value	Notes

Winner	ROC-AUC	0.7513	LogLoss 0.5869, Brier 0.2019

ECE (cal)	0.0125	Isotonic calibration

Total Games	MAE	2.006	R² 0.808, RMSE 3.750

ECE (cal)	0.0125	Isotonic calibration

Games Diff	MAE	3.217	R² 0.223, RMSE 4.903

ECE (cal)	0.0117	Isotonic calibration

First Set	ROC-AUC	0.6988	LogLoss 0.6261, Brier 0.2189

ECE (raw)	0.0063	No calibration needed

WTA Models

Model	Metric	Value	Notes

Winner	ROC-AUC	0.7130	LogLoss 0.6107, Brier 0.2130

ECE (cal)	0.0144	Isotonic calibration

Total Games	MAE	1.566	R² 0.707, RMSE 3.131

ECE (cal)	0.0144	Isotonic calibration

Games Diff	MAE	3.450	R² 0.183, RMSE 5.291

ECE (cal)	0.0206	Isotonic calibration

First Set	ROC-AUC	0.7130	LogLoss 0.6107, Brier 0.2130

ECE (raw)	0.0065\*	Expected <1% (similar to ATP)

ECE = Expected Calibration Error; cal = after isotonic calibration; raw = before calibration.



Key Observations:



ATP Winner and ATP Total models perform at a high level compared to public benchmarks.



Calibration is excellent (ECE <2% for all), crucial for probability estimation.



Games Diff (handicap) is the weakest group (R² ~0.2), reflecting the high variance of game differentials.



📝 License

This project is open‑source under the MIT License.



👤 Author

Kirill Chernyshev / https://github.com/Gotodataru



🤝 Contributing

Issues and pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.



Disclaimer: This project is for educational and research purposes only. Betting involves financial risk. Past performance does not guarantee future results. Use these models at your own discretion.

