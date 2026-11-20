use std::{
    fmt,
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Config {
    #[serde(default = "default_fabric")]
    pub default_fabric: String,
    #[serde(default)]
    pub gateways: Vec<GatewayConfig>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            default_fabric: default_fabric(),
            gateways: Vec::new(),
        }
    }
}

impl Config {
    fn normalize(&mut self) -> Result<()> {
        self.default_fabric = self.default_fabric.trim().to_owned();
        if self.default_fabric.is_empty() {
            bail!("default_fabric cannot be empty");
        }
        for gateway in &mut self.gateways {
            if gateway.fabric.trim().is_empty() {
                gateway.fabric.clone_from(&self.default_fabric);
            } else {
                gateway.fabric = gateway.fabric.trim().to_owned();
            }
            if gateway.token.is_none()
                && gateway.token_env.as_deref().is_some_and(is_marie_api_token)
            {
                gateway.token = gateway.token_env.take();
            }
        }
        Ok(())
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub struct GatewayConfig {
    pub name: String,
    pub url: String,
    #[serde(default)]
    pub fabric: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_env: Option<String>,
}

impl fmt::Debug for GatewayConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GatewayConfig")
            .field("name", &self.name)
            .field("url", &self.url)
            .field("fabric", &self.fabric)
            .field("token", &self.token.as_ref().map(|_| "<redacted>"))
            .field("token_env", &self.token_env)
            .finish()
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct SelectionState {
    pub fabric: Option<String>,
    pub gateway: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Paths {
    pub config: PathBuf,
    pub state: PathBuf,
}

fn default_fabric() -> String {
    "default".to_owned()
}

pub fn is_token_env_name(value: &str) -> bool {
    let mut characters = value.chars();
    matches!(characters.next(), Some('_' | 'A'..='Z' | 'a'..='z'))
        && characters.all(|character| character == '_' || character.is_ascii_alphanumeric())
}

fn is_marie_api_token(value: &str) -> bool {
    value.len() == 58 && (value.starts_with("mas_") || value.starts_with("mau_"))
}

impl Paths {
    pub fn discover(config_override: Option<PathBuf>) -> Result<Self> {
        let dirs = ProjectDirs::from("ai", "marie", "m3top")
            .context("could not determine the user configuration directory")?;
        Ok(Self {
            config: config_override.unwrap_or_else(|| dirs.config_dir().join("config.toml")),
            state: dirs
                .state_dir()
                .unwrap_or_else(|| dirs.data_local_dir())
                .join("state.toml"),
        })
    }
}

pub fn load(paths: &Paths) -> Result<(Config, SelectionState)> {
    let mut config = match fs::read_to_string(&paths.config) {
        Ok(config_text) => {
            // TOML errors include source lines, which may contain tokens.
            toml::from_str::<Config>(&config_text).map_err(|_| {
                anyhow::anyhow!("invalid configuration in {}", paths.config.display())
            })?
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => Config::default(),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("could not read {}", paths.config.display()));
        }
    };

    config
        .normalize()
        .with_context(|| format!("invalid configuration in {}", paths.config.display()))?;

    let state = fs::read_to_string(&paths.state)
        .ok()
        .and_then(|text| toml::from_str(&text).ok())
        .unwrap_or_default();
    Ok((config, state))
}

pub fn save_config(path: &Path, config: &Config) -> Result<()> {
    let contents = toml::to_string_pretty(config).context("could not encode configuration")?;
    atomic_write(path, contents.as_bytes())
}

pub fn save_selection(path: &Path, selection: &SelectionState) -> Result<()> {
    let contents = toml::to_string(selection).context("could not encode selection state")?;
    atomic_write(path, contents.as_bytes())
}

fn atomic_write(path: &Path, contents: &[u8]) -> Result<()> {
    let parent = path.parent().context("configuration path has no parent")?;
    fs::create_dir_all(parent).with_context(|| format!("could not create {}", parent.display()))?;
    let temporary = path.with_extension(format!("tmp.{}", std::process::id()));
    let mut options = OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(&temporary)
        .with_context(|| format!("could not write {}", temporary.display()))?;
    file.write_all(contents)
        .with_context(|| format!("could not write {}", temporary.display()))?;
    file.sync_all()
        .with_context(|| format!("could not sync {}", temporary.display()))?;
    fs::rename(&temporary, path)
        .with_context(|| format!("could not replace {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Config, is_token_env_name};

    #[test]
    fn gateway_without_fabric_can_use_default() {
        let mut config: Config = toml::from_str(
            r#"
default_fabric = "local"

[[gateways]]
name = "gateway-1"
url = "http://localhost:51000"
"#,
        )
        .unwrap();
        config.normalize().unwrap();
        assert_eq!(config.default_fabric, "local");
        assert_eq!(config.gateways[0].fabric, "local");
    }

    #[test]
    fn legacy_raw_token_env_is_migrated_in_memory() {
        let token = format!("mau_{}", "x".repeat(54));
        let mut config: Config = toml::from_str(&format!(
            r#"
[[gateways]]
name = "gateway-1"
url = "http://localhost:51000"
token_env = "{token}"
"#,
        ))
        .unwrap();

        config.normalize().unwrap();

        assert_eq!(config.gateways[0].token.as_deref(), Some(token.as_str()));
        assert!(config.gateways[0].token_env.is_none());
        assert!(!format!("{:?}", config.gateways[0]).contains(&token));
    }

    #[test]
    fn token_environment_name_uses_shell_identifier_syntax() {
        assert!(is_token_env_name("M3TOP_GATEWAY_TOKEN"));
        assert!(is_token_env_name("_M3TOP_TOKEN_2"));
        assert!(!is_token_env_name(""));
        assert!(!is_token_env_name("2_M3TOP_TOKEN"));
        assert!(!is_token_env_name("raw-token-value"));
    }
}
