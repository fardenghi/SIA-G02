PYTHON   := uv run python
PYTEST   := uv run pytest

# --- Kohonen ---
KOHONEN_CONFIG ?= configs/kohonen_europe.json

# --- PCA (sklearn) ---
PCA_N ?= 1

# --- Hopfield ---
HOPFIELD_CONFIG ?= configs/hopfield.json

.PHONY: all kohonen pca hopfield test clean clean-kohonen clean-pca clean-hopfield help

all: kohonen pca hopfield

kohonen:
	$(PYTHON) kohonen_europe.py --config $(KOHONEN_CONFIG)

pca:
	$(PYTHON) pca_europe.py --n-components $(PCA_N)

hopfield:
	$(PYTHON) hopfield_letters.py --config $(HOPFIELD_CONFIG)

test:
	$(PYTEST) -v

clean-kohonen:
	rm -rf output/kohonen

clean-pca:
	rm -rf output/[0-9]*

clean-hopfield:
	rm -rf output/hopfield

clean: clean-kohonen clean-pca clean-hopfield

help:
	@echo "Targets disponibles:"
	@echo "  make kohonen                            Entrena SOM con config default"
	@echo "  make kohonen KOHONEN_CONFIG=<path>      Entrena SOM con config alternativo"
	@echo "  make pca                                PCA con 1 componente"
	@echo "  make pca PCA_N=3                        PCA con N componentes"
	@echo "  make hopfield                           Ejecuta red de Hopfield con config default"
	@echo "  make hopfield HOPFIELD_CONFIG=<path>    Ejecuta Hopfield con config alternativo"
	@echo "  make all                                Corre kohonen + pca + hopfield"
	@echo "  make test                               Corre la suite de tests"
	@echo "  make clean-kohonen                      Borra output/kohonen/"
	@echo "  make clean-pca                          Borra output de PCA"
	@echo "  make clean-hopfield                     Borra output/hopfield/"
	@echo "  make clean                              Borra todo el output"
