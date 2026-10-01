PORT ?= 8000
LT_CACHE ?= $(HOME)/.cache/overhear-subs

# Optional local overrides (copy .env.example). Plain KEY=value lines.
-include .env

# Keep this list in sync with the app's env vars.
APP_VARS = PORT LT_CACHE LT_LOOKAHEAD LT_CHUNK LT_REMUX LT_TRANSLATE_MODEL \
           LT_TTS LT_TTS_MODEL LT_TTS_VOICE LT_TTS_LANG \
           LT_OLLAMA_URL LT_CHAPTER_MODEL LT_CHAPTER_AIM LT_CHAPTER_MIN LT_CHAPTER_MAX \
           PHONEMIZER_ESPEAK_LIBRARY PHONEMIZER_ESPEAK_DATA_PATH \
           HF_TOKEN HUGGING_FACE_HUB_TOKEN
# Export only names that are actually set. `export FOO` on an undefined name
# injects FOO="" into the child, and the float-parsed LT_* vars crash on that.
export_if_set = $(if $(filter undefined,$(origin $(1))),,$(eval export $(1)))
$(foreach v,$(APP_VARS),$(call export_if_set,$(v)))

.DEFAULT_GOAL := help
.PHONY: help setup run dev test check clean cache-clean

help:
	@echo "overhear-subs"
	@echo
	@echo "  make setup       install deps"
	@echo "  make run         start the server on :$(PORT)"
	@echo "  make dev         start with auto-reload"
	@echo "  make test        run the test suite"
	@echo "  make check       byte-compile the python modules"
	@echo "  make clean       remove .venv and caches"
	@echo "  make cache-clean remove derived PCM/remuxed media ($(LT_CACHE))"
	@echo
	@echo "Override the port with: make run PORT=9000"
	@echo "Pass a token with   : HF_TOKEN=hf_xxx make run"

setup:
	uv sync

run:
	uv run uvicorn app:app --port $(PORT)

dev:
	uv run uvicorn app:app --port $(PORT) --reload

test:
	uv run pytest

check:
	uv run python -m py_compile app.py pipeline.py backends.py

clean:
	rm -rf .venv __pycache__ .pytest_cache tests/__pycache__

cache-clean:
	rm -rf "$(LT_CACHE)"
