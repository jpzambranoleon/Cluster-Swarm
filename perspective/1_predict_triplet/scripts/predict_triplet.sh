
for dataset in banking77
do
    link_path=sampled_triplet_results/${dataset}_embed=e5_s=small_m=1024_d=77.0_sf_choice_seed=100.json
    # link_path=sampled_triplet_results/${dataset}_embed=instructor_s=large_m=1024_d=67.0_sf_choice_seed=100.json
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python predict_triplet.py \
        --dataset $dataset \
        --data_path $link_path \
        --model_name gpt-4o-mini \
        --temperature 0
done