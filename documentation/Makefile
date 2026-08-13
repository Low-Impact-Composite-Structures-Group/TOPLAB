TEXMFHOME := /Users/dmrraso/dv/aces/texmf
export TEXMFHOME

# Use MacPorts TeX Live to avoid conflicts with conda-installed texlive-core
PATH := /opt/local/bin:$(PATH)
export PATH

MAIN := main
LATEXMK := latexmk
LATEXMK_FLAGS := -pdf -f -interaction=nonstopmode

.PHONY: all clean

all:
	$(LATEXMK) $(LATEXMK_FLAGS) $(MAIN).tex

clean:
	$(LATEXMK) -C $(MAIN).tex
	rm -f $(MAIN).bbl $(MAIN).run.xml
