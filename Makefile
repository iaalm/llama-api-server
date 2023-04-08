.PHONY : help format tags
PYTHON=env/bin/python 
UT=${PYTHON} -m unittest -b

help:
	@LC_ALL=C $(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/(^|\n)# Files(\n|$$)/,/(^|\n)# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

format:
	$(PYTHON) -m black llama_api_server

tags:
	ctags -R pretzel

pack:
	$(PYTHON) -m build
