test:
	@tox --recreate
	@tox

changelog: CHANGELOG.md
	pdm run sh ./.scripts/generate_changelog.sh
