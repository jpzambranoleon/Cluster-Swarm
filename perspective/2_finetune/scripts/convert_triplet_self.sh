scale=small
dataset=banking77
python convert_triplet_self.py \
    --dataset $dataset \
    --pred_path ../1_predict_triplet/predicted_triplet_results/${dataset}_embed=e5_s=${scale}_m=1024_d=77.0_sf_choice_seed=100-gpt-4o-mini-pred.json \
    --output_path converted_triplet_results \
    --feat_path ../../datasets/${dataset}/${scale}_embeds_e5.hdf5 \
    --data_path ../../datasets/${dataset}/${scale}.jsonl