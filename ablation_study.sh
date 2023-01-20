python src/experiments/stacking/optimise_stacking.py --experiment_name all_classifier --stacking_type all
python src/experiments/stacking/optimise_stacking.py --experiment_name intermediate_classifier --stacking_type less_stacking
python src/experiments/stacking/optimise_stacking.py --experiment_name less_classifier --stacking_type only_single
python src/experiments/stacking/optimise_stacking.py --experiment_name deactivate_triplet_loss --stacking_type less_stacking --deactivate_triplet_loss
