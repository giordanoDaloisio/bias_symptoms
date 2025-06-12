for d in data/*; do
echo $d
python noise.py --data $d
done