.PHONY: modified_only_fixup quality style  test

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = marie

check_dirs := marie

modified_only_fixup:
	$(eval modified_py_files := $(shell python utils/get_modified_files.py $(check_dirs)))
	@if test -n "$(modified_py_files)"; then \
		echo "Checking/fixing $(modified_py_files)"; \
		#black $(modified_py_files); \
		isort $(modified_py_files); \
		flake8 $(modified_py_files); \
	else \
		echo "No library .py files were modified"; \
	fi

# this target runs checks on all files

quality:
	#black --check $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)

# this target runs checks on all files and potentially modifies some of them
# We will only apply `black` on idividual basis or on modified files only

style:
	#black $(check_dirs)
	isort $(check_dirs)

# Run tests for the library

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/
