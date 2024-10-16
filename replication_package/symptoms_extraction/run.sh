for d in baselines/*; do
echo $d
python main.py -d $d -m logreg
done