.PHONY: clean-outputs clean-ej1 inspect-digit run-ej1 run-ej2 run-ej3 run-xor test

EJ1_DIR = outputs/ej1_fraud
EJ2_CONFIG ?= configs/ej2_digits/baseline.json
EJ3_CONFIG ?= configs/ej3_more_digits/softmax.json
DIGIT_DATASET ?= test
DIGIT_INDEX ?= 0

clean-ej1:
	rm -rf $(EJ1_DIR)
	@echo "Outputs de Ej1 eliminados."

clean-outputs:
	rm -rf outputs
	@echo "Outputs eliminados."

run-ej1:
	uv run python -m exercises.ej1_fraud.explore
	uv run python -m exercises.ej1_fraud.train_linear
	uv run python -m exercises.ej1_fraud.train_sigmoid
	uv run python -m exercises.ej1_fraud.compare_models
	uv run python -m exercises.ej1_fraud.train_generalization
	uv run python -m exercises.ej1_fraud.threshold_analysis

run-xor:
	uv run python -m exercises.xor.run configs/xor/default.json

run-ej2:
	uv run python -m exercises.ej2_digits.run $(EJ2_CONFIG)

run-ej3:
	uv run python -m exercises.ej3_more_digits.run $(EJ3_CONFIG)

inspect-digit:
	uv run python -m exercises.ej2_digits.inspect_sample --dataset $(DIGIT_DATASET) --index $(DIGIT_INDEX)

test:
	uv run pytest
