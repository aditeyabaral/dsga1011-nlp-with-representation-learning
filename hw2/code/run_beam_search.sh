for i in {1..5}
do
    echo "Running beam_size = $i"
    python main.py --beam_search --beam_size $i > out_beam_$i.txt
done