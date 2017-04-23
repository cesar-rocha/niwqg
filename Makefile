install:
	python setup.py develop
requirements:
	conda install --yes --file requirements.txt
test:
	python -m pytest niwqg
