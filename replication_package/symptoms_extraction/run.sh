for d in data/*; do
echo $d
python main.py -d $d -m logreg
done