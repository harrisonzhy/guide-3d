python extract_segment_everything_masks.py --image_root "/home/zhanghy/orcd/scratch/zhanghy/guide-3d/mats/data/images1/kitchen" --sam_checkpoint_path "third_party/segment-anything-model/sam_vit_h_4b8939.pth" --downsample 4
python get_scale.py --image_root "../mats/data/images1/kitchen" --source "../mats/data/images1/kitchen" --model_path "../gaussian-splatting/output/ee635da2-0"
popd
sh clip.sh
pushd SegAnyGAussians
python get_clip_features.py --image_root "../mats/data/images1/kitchen"
python train_contrastive_feature.py -m "../gaussian-splatting/output/ee635da2-0" --iteration 30000 --num_sampled_rays 1000
popd
