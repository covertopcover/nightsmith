package main

import (
	"os"
	"path/filepath"
)

// Everything nightsmith owns lives under $HOME. That is not tidiness — it is
// what makes "never asks for your password" true, and what makes
// `nightsmith remove` a plain delete with nothing left behind.

func homeDir() string {
	h, err := os.UserHomeDir()
	if err != nil {
		return "."
	}
	return h
}

func stateDir() string   { return filepath.Join(homeDir(), ".nightsmith") }
func modelsDir() string  { return filepath.Join(stateDir(), "models") }
func runtimeDir() string { return filepath.Join(stateDir(), "runtime") } // the venv
func uvDir() string      { return filepath.Join(stateDir(), "uv") }
func pythonDir() string  { return filepath.Join(stateDir(), "python") } // uv's own Python
func uvCacheDir() string { return filepath.Join(stateDir(), "uv-cache") }
func configPath() string { return filepath.Join(stateDir(), "config.toml") }
func pidPath() string    { return filepath.Join(stateDir(), "server.pid") }
func binPath() string    { return filepath.Join(homeDir(), ".local", "bin", "nightsmith") }

// hfCacheDir is shared with Ollama and LM Studio. It is a gift when a model is
// already there — "Found it already on your Mac. Nothing to download." — and it
// is never ours to delete.
func hfCacheDir() string { return filepath.Join(homeDir(), ".cache", "huggingface") }
