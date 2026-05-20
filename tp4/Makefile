PYTHON   := uv run python
PYTEST   := uv run pytest

# --- Kohonen ---
KOHONEN_CONFIG ?= configs/kohonen_europe.json

# --- Oja ---
OJA_CONFIG ?= configs/oja_europe.json

# --- Hopfield ---
HOPFIELD_CONFIG ?= configs/hopfield.json
HOPFIELD_K      ?= 4
LET_START       ?= A
LET_END         ?= Z

.PHONY: all kohonen pca pca-plots pca-boxplots oja hopfield hopfield-alphabet \
	hopfield-orthogonality hopfield-capacity test clean clean-kohonen \
	clean-pca clean-oja clean-hopfield help

all: kohonen pca oja hopfield

kohonen:
	$(PYTHON) -m kohonen.kohonen_europe --config $(KOHONEN_CONFIG)

pca: pca-plots pca-boxplots
	$(PYTHON) -m pca_test.pca

pca-plots:
	$(PYTHON) -m pca_test.plots

pca-boxplots:
	$(PYTHON) -m pca_test.boxplots

oja:
	$(PYTHON) -m oja.oja_europe --config $(OJA_CONFIG)

hopfield:
	$(PYTHON) -m hopfield.hopfield_runner --config $(HOPFIELD_CONFIG)

hopfield-alphabet:
	$(PYTHON) -m hopfield.plot_letters --start $(LET_START) --end $(LET_END)

hopfield-orthogonality:
	$(PYTHON) -m hopfield.orthogonality --k $(HOPFIELD_K)

hopfield-capacity:
	$(PYTHON) -m hopfield.capacity

test:
	$(PYTEST) -v

clean-kohonen:
	rm -rf output/kohonen

clean-pca:
	rm -rf pca_test/plots pca_test/pca_scores.csv pca_test/europe_standardized.csv

clean-oja:
	rm -rf oja/output/oja

clean-hopfield:
	rm -rf output/hopfield

clean: clean-kohonen clean-pca clean-oja clean-hopfield

help:
	@echo "Targets disponibles:"
	@echo "  make kohonen                          Entrena SOM con config default"
	@echo "  make kohonen KOHONEN_CONFIG=<path>    Entrena SOM con config alternativo"
	@echo "  make pca                              PCA completo (analisis + plots + boxplots)"
	@echo "  make pca-plots                        Solo variance, biplot y ranking"
	@echo "  make pca-boxplots                     Solo boxplots pre/post estandarizacion"
	@echo "  make oja                              Entrena Oja con config default"
	@echo "  make oja OJA_CONFIG=<path>            Entrena Oja con config alternativo"
	@echo "  make hopfield                         Corre Hopfield (parte a + b)"
	@echo "  make hopfield-alphabet LET_START=C LET_END=H   Grafica letras en grilla"
	@echo "  make hopfield-orthogonality HOPFIELD_K=4       Analisis de ortogonalidad"
	@echo "  make hopfield-capacity                Metrica vs cantidad de patrones"
	@echo "  make all                              Corre kohonen + pca + oja + hopfield"
	@echo "  make test                             Corre la suite de tests"
	@echo "  make clean-kohonen                    Borra output/kohonen/"
	@echo "  make clean-pca                        Borra pca_test/plots/ y CSVs generados"
	@echo "  make clean-oja                        Borra oja/output/oja/"
	@echo "  make clean-hopfield                   Borra output/hopfield/"
	@echo "  make clean                            Borra todo el output"
