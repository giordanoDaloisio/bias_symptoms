python main.py -d ../data/bias_symptoms_logreg_reduced.csv --model xgb --clean
python main.py -d ../data/bias_symptoms_logreg_reduced.csv --model rf --clean
python main.py -d ../data/bias_symptoms_logreg_reduced.csv --model mlp --clean
python main.py -d ../data/bias_symptoms_logreg_reduced.csv --model xgb --noisy
python main.py -d ../data/bias_symptoms_logreg_reduced.csv --model rf --noisy
python main.py -d ../data/bias_symptoms_logreg_reduced.csv --model mlp --noisy