.PHONY: clean-ej1 run-ej1

EJ1_DIR = experiments/ej1

clean-ej1:
	rm -f $(EJ1_DIR)/*.json $(EJ1_DIR)/*.npz
	rm -f $(EJ1_DIR)/plots/*.png
	@echo "Outputs de Ej1 eliminados."

run-ej1:
	uv run python -m src.ej1.explore
	uv run python -m src.ej1.train_linear
	uv run python -m src.ej1.train_nonlinear
	uv run python -m src.ej1.compare_models
	uv run python -m src.ej1.train_generalization
	uv run python -m src.ej1.threshold_analysis
