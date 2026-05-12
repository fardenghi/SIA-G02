PYTHON   := uv run python
PYTEST   := uv run pytest

# --- Kohonen ---
KOHONEN_CONFIG ?= configs/kohonen_europe.json

# --- Oja ---
OJA_CONFIG ?= configs/oja_europe.json

.PHONY: all kohonen pca pca-plots pca-boxplots oja test clean clean-kohonen clean-pca clean-oja help

all: kohonen pca oja

kohonen:
	$(PYTHON) kohonen_europe.py --config $(KOHONEN_CONFIG)

pca: pca-plots pca-boxplots
	cd pca_test && $(PYTHON) pca.py

pca-plots:
	cd pca_test && $(PYTHON) plots.py

pca-boxplots:
	cd pca_test && $(PYTHON) boxplots.py

oja:
	cd oja && $(PYTHON) oja_europe.py --config ../$(OJA_CONFIG)

test:
	$(PYTEST) -v

clean-kohonen:
	rm -rf output/kohonen

clean-pca:
	rm -rf pca_test/plots pca_test/pca_scores.csv pca_test/europe_standardized.csv

clean-oja:
	rm -rf oja/output/oja

clean: clean-kohonen clean-pca clean-oja

help:
	@echo "Targets disponibles:"
	@echo "  make kohonen                          Entrena SOM con config default"
	@echo "  make kohonen KOHONEN_CONFIG=<path>    Entrena SOM con config alternativo"
	@echo "  make pca                              PCA completo (analisis + plots + boxplots)"
	@echo "  make pca-plots                        Solo variance, biplot y ranking"
	@echo "  make pca-boxplots                     Solo boxplots pre/post estandarizacion"
	@echo "  make oja                              Entrena Oja con config default"
	@echo "  make oja OJA_CONFIG=<path>            Entrena Oja con config alternativo"
	@echo "  make all                              Corre kohonen + pca + oja"
	@echo "  make test                             Corre la suite de tests"
	@echo "  make clean-kohonen                    Borra output/kohonen/"
	@echo "  make clean-pca                        Borra pca_test/plots/ y CSVs generados"
	@echo "  make clean-oja                        Borra oja/output/oja/"
	@echo "  make clean                            Borra todo el output"
