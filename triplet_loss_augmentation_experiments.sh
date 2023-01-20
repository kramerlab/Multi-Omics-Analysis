python src/experiments/early_integration/optimise_early_integration.py --experiment_name early_integration_with_triplet_loss
python src/experiments/moli/optimise_moli.py --experiment_name moli_without_triplet_loss --deactivate_triplet_loss
python src/experiments/moma/optimise_moma.py --experiment_name moma_with_triplet_loss --add_triplet_loss
python src/experiments/omiEmbed/optimise_omiEmbed.py --experiment_name omiEmbed_with_triplet_loss --add_triplet_loss
python src/experiments/super.felt/optimise_super_felt.py --experiment_name super_felt_without_triplet_loss --deactivate_triplet_loss