install:
    uv sync --all-groups
    uv pip install -e ~/projects/stats-packages/did-imputation/
    uv pip install -e /home/jsr-p/gh-repos/documentation-stuff/sphinx-autodoc2

prep_test:
    mkdir -p data
    stata scripts/stata/harmon_sim_data.do

docs:
    mkdir -p _output
    quarto render scripts/example_docs.py --output-dir ../_output
    pandoc --to=gfm ~/tex-notes/stats/did/pkgdocs/harmon.tex > docs/harmon.md
    quarto render README.qmd

examples:
    python scripts/harmon_ex.py > output/harmon_ex.txt
    python scripts/harmon_simexperiment.py > output/harmon_simexperiment.txt

sphinx: 
    cp README.md docs/source
    # mkdir -p docs/source/figs
    # copy example images to docs
    # cp figs/*.png docs/source/figs/
    # cp figs/*.svg docs/source/figs/
    just sphinx-clean 
    just sphinx-build
    
sphinx-build:
    sphinx-build docs/source docs/build

sphinx-clean:
    sphinx-build -M clean docs/source docs/build
    rm docs/source/did_imp.rst || true
    rm docs/source/modules.rst || true
    rm -rf docs/source/apidocs/ || true
