all:
	@echo 'Install'
	python setup.py develop

nose:
	@# nosetests -vs --logging-level=DEBUG --with-coverage --cover-package=gargantua tests
	nosetests -vs --logging-level=DEBUG tests

changelog: CHANGELOG.md
	sh ./.scripts/generate_changelog.sh
