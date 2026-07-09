package config

import "testing"

func TestLoadDefaults(t *testing.T) {
	t.Setenv("MARIE_PLUGIN_DAEMON_ADDR", "")
	t.Setenv("MARIE_PLUGIN_STORAGE_ROOT", "")
	t.Setenv("MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID", "")
	t.Setenv("MARIE_PLUGIN_DAEMON_SIGNING_SECRET", "")
	t.Setenv("MARIE_PLUGIN_DAEMON_DEV_INSECURE", "")

	c, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Addr != "127.0.0.1:8099" {
		t.Errorf("Addr default = %q", c.Addr)
	}
	if c.StorageRoot != "./storage" {
		t.Errorf("StorageRoot default = %q", c.StorageRoot)
	}
	if c.DevInsecure {
		t.Errorf("DevInsecure should default false")
	}
	if c.SigningConfigured() {
		t.Errorf("SigningConfigured should be false with no key")
	}
}

func TestLoadFromEnv(t *testing.T) {
	t.Setenv("MARIE_PLUGIN_DAEMON_ADDR", "0.0.0.0:9000")
	t.Setenv("MARIE_PLUGIN_STORAGE_ROOT", "/tmp/storage")
	t.Setenv("MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID", "marie-local-key")
	t.Setenv("MARIE_PLUGIN_DAEMON_SIGNING_SECRET", "secret")
	t.Setenv("MARIE_PLUGIN_DAEMON_DEV_INSECURE", "1")

	c, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Addr != "0.0.0.0:9000" {
		t.Errorf("Addr = %q", c.Addr)
	}
	if c.StorageRoot != "/tmp/storage" {
		t.Errorf("StorageRoot = %q", c.StorageRoot)
	}
	if c.SigningKeyID != "marie-local-key" || c.SigningSecret != "secret" {
		t.Errorf("signing = %q/%q", c.SigningKeyID, c.SigningSecret)
	}
	if !c.DevInsecure {
		t.Errorf("DevInsecure should be true for '1'")
	}
	if !c.SigningConfigured() {
		t.Errorf("SigningConfigured should be true")
	}
}
