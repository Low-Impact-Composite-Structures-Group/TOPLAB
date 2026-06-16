TEXMFHOME := /Users/dmrraso/dv/aces/texmf
export TEXMFHOME

MAIN := main
LATEXMK := latexmk
LATEXMK_FLAGS := -pdf -f -interaction=nonstopmode

.PHONY: all clean

all:
	$(LATEXMK) $(LATEXMK_FLAGS) $(MAIN).tex

clean:
	$(LATEXMK) -C $(MAIN).tex
	rm -f $(MAIN).bbl $(MAIN).run.xml
