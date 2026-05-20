PYTHON   := uv run python
PYTEST   := uv run pytest

# --- Kohonen ---
KOHONEN_CONFIG ?= configs/kohonen_europe.json

# --- Oja ---
OJA_CONFIG ?= configs/oja_europe.json

# --- Hopfield ---
HOPFIELD_CONFIG ?= configs/hopfield.json

# --- Analysis ---
MIDDLE_THRESHOLD ?= 0.5

.PHONY: all kohonen pca pca-plots pca-boxplots oja hopfield hopfield-alphabet \
        compare standardization analysis \
        test clean clean-kohonen clean-pca clean-oja clean-hopfield clean-analysis help

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
	$(PYTHON) hopfield_letters.py --config $(HOPFIELD_CONFIG)

hopfield-alphabet:
	$(PYTHON) hopfield_letters.py --alphabet

compare:
	$(PYTHON) -m analysis.compare \
	  --kohonen-config $(KOHONEN_CONFIG) \
	  --middle-threshold $(MIDDLE_THRESHOLD)

standardization:
	$(PYTHON) -m analysis.standardization_experiment \
	  --kohonen-config $(KOHONEN_CONFIG)

analysis: compare standardization

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

clean-analysis:
	rm -rf output/compare output/standardization

clean: clean-kohonen clean-pca clean-oja clean-hopfield clean-analysis

help:
	@echo "Targets disponibles:"
	@echo "  make kohonen                            Entrena SOM con config default"
	@echo "  make kohonen KOHONEN_CONFIG=<path>      Entrena SOM con config alternativo"
	@echo "  make pca                                PCA completo (analisis + plots + boxplots)"
	@echo "  make pca-plots                          Solo variance, biplot y ranking"
	@echo "  make pca-boxplots                       Solo boxplots pre/post estandarizacion"
	@echo "  make oja                                Entrena Oja con config default"
	@echo "  make oja OJA_CONFIG=<path>              Entrena Oja con config alternativo"
	@echo "  make hopfield                           Ejecuta red de Hopfield (4 letras, config default)"
	@echo "  make hopfield HOPFIELD_CONFIG=<path>    Ejecuta Hopfield con config alternativo"
	@echo "  make hopfield-alphabet                  Experimento con alfabeto completo (A-Z, grilla adaptativa)"
	@echo "  make compare                            Validación cruzada PCA vs Kohonen (Secciones 1.1, 1.2, 2.1)"
	@echo "  make compare MIDDLE_THRESHOLD=0.8       Umbral personalizado para países 'promedio'"
	@echo "  make standardization                    Experimento 2x2 estandarización vs datos crudos"
	@echo "  make analysis                           Corre compare + standardization"
	@echo "  make all                                Corre kohonen + pca + oja + hopfield"
	@echo "  make test                               Corre la suite de tests"
	@echo "  make clean-kohonen                      Borra output/kohonen/"
	@echo "  make clean-pca                          Borra pca_test/plots/ y CSVs generados"
	@echo "  make clean-oja                          Borra oja/output/oja/"
	@echo "  make clean-hopfield                     Borra output/hopfield/"
	@echo "  make clean-analysis                     Borra output/compare/ y output/standardization/"
	@echo "  make clean                              Borra todo el output"
