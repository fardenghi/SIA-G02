PYTHON   := uv run python
PYTEST   := uv run pytest

# --- Kohonen ---
KOHONEN_CONFIG ?= configs/kohonen_europe.json

# --- PCA (sklearn) ---
PCA_N ?= 1

.PHONY: all kohonen pca test clean clean-kohonen clean-pca help

all: kohonen pca

kohonen:
	$(PYTHON) kohonen_europe.py --config $(KOHONEN_CONFIG)

pca:
	$(PYTHON) pca_europe.py --n-components $(PCA_N)

test:
	$(PYTEST) -v

clean-kohonen:
	rm -rf output/kohonen

clean-pca:
	rm -rf output/[0-9]*

clean: clean-kohonen clean-pca

help:
	@echo "Targets disponibles:"
	@echo "  make kohonen                          Entrena SOM con config default"
	@echo "  make kohonen KOHONEN_CONFIG=<path>    Entrena SOM con config alternativo"
	@echo "  make pca                              PCA con 1 componente"
	@echo "  make pca PCA_N=3                      PCA con N componentes"
	@echo "  make all                              Corre kohonen + pca"
	@echo "  make test                             Corre la suite de tests"
	@echo "  make clean-kohonen                    Borra output/kohonen/"
	@echo "  make clean-pca                        Borra output de PCA (output/1/, output/2/, ...)"
	@echo "  make clean                            Borra todo el output"
