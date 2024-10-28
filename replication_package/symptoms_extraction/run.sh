for d in data/*; do
echo $d
python main.py -d $d -m mlp
python main.py -d $d -m rf
done