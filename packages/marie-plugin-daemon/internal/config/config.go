// Package config centralizes the marie-plugin-daemon's environment configuration.
//
// Instead of hand-rolling os.Getenv lookups, settings are declared once as a
// struct and populated with kelseyhightower/envconfig (the same pattern the
// upstream dify-plugin-daemon uses). Explicit `envconfig` tags pin the exact
// env var names so existing deployments keep working.
package config

import (
	"strings"

	"github.com/kelseyhightower/envconfig"
)

// Config holds all environment-driven settings for the daemon.
type Config struct {
	// Addr is the HTTP listen address. The -addr flag overrides this.
	Addr string `envconfig:"MARIE_PLUGIN_DAEMON_ADDR"`

	// StorageRoot is the plugin storage root. The -storage-root flag overrides this.
	StorageRoot string `envconfig:"MARIE_PLUGIN_STORAGE_ROOT"`

	// SigningKeyID / SigningSecret are the HMAC key the daemon verifies envelopes
	// against. When unset (and DevInsecure is false), signed routes reject all
	// requests.
	SigningKeyID  string `envconfig:"MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID"`
	SigningSecret string `envconfig:"MARIE_PLUGIN_DAEMON_SIGNING_SECRET"`

	// DevInsecure, when true, accepts envelopes WITHOUT verifying their signature.
	// DEV ONLY — never enable in a shared environment. Forgiving parsing: unset or
	// empty -> false; "1"/"true"/"yes"/"on" -> true.
	DevInsecure ForgivingBool `envconfig:"MARIE_PLUGIN_DAEMON_DEV_INSECURE"`

	// Logging.
	LogLevel string `envconfig:"MARIE_PLUGIN_DAEMON_LOG_LEVEL"` // DEBUG|INFO|WARN|ERROR (default INFO)
	// LogFormat selects "json" for structured JSON output; anything else is text.
	LogFormat string `envconfig:"MARIE_PLUGIN_DAEMON_LOG_FORMAT"`
	// LogFile, when set, tees logs to this file in addition to stdout.
	LogFile string `envconfig:"MARIE_PLUGIN_DAEMON_LOG_FILE"`
}

// LogJSON reports whether JSON log output is requested.
func (c Config) LogJSON() bool {
	return strings.EqualFold(strings.TrimSpace(c.LogFormat), "json")
}

// ForgivingBool is a bool that tolerates unset/empty values (treated as false)
// and accepts the usual truthy strings. Implements envconfig.Decoder.
type ForgivingBool bool

// Decode implements envconfig.Decoder.
func (b *ForgivingBool) Decode(value string) error {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "yes", "on":
		*b = true
	default:
		*b = false
	}
	return nil
}

// Load populates a Config from the environment and applies defaults.
func Load() (Config, error) {
	var c Config
	if err := envconfig.Process("", &c); err != nil {
		return Config{}, err
	}
	c.SetDefaults()
	return c, nil
}

// SetDefaults fills empty/unset values with their defaults (matches the upstream
// dify-plugin-daemon's SetDefault pattern; tolerant of vars explicitly set empty).
func (c *Config) SetDefaults() {
	if strings.TrimSpace(c.Addr) == "" {
		c.Addr = "127.0.0.1:8099"
	}
	if strings.TrimSpace(c.StorageRoot) == "" {
		c.StorageRoot = "./storage"
	}
}

// SigningConfigured reports whether a real HMAC signing key is available.
func (c Config) SigningConfigured() bool {
	return c.SigningKeyID != "" && c.SigningSecret != ""
}
