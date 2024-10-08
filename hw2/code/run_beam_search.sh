for i in {1..5}
do
    echo "Running beam_size = $i at $(date)"
    start=$(date +%s)
    python main.py --beam_search --beam_size $i > out_beam_$i.txt
    end=$(date +%s)
    elapsed=$((end-start))
    echo "Completed beam_size = $i in $((elapsed/60)) minutes"
done

