use std::collections::BTreeMap;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use reqwest::Url;
use tokio::time::Instant;

use crate::{
    client::GatewayTarget,
    config::{Config, GatewayConfig, SelectionState, save_config, save_selection},
    model::{AttentionItem, Executor, GatewaySummary, Overview, Target},
};

use super::{ApiCommand, App, Focus, POLL_INTERVAL, Screen, UserAction};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OnboardingField {
    Name,
    Url,
    Fabric,
    Token,
}

impl OnboardingField {
    pub(super) fn next(self) -> Self {
        match self {
            Self::Name => Self::Url,
            Self::Url => Self::Fabric,
            Self::Fabric => Self::Token,
            Self::Token => Self::Name,
        }
    }

    pub(super) fn previous(self) -> Self {
        match self {
            Self::Name => Self::Token,
            Self::Url => Self::Name,
            Self::Fabric => Self::Url,
            Self::Token => Self::Fabric,
        }
    }
}

#[derive(Clone)]
pub struct OnboardingForm {
    pub field: OnboardingField,
    pub name: String,
    pub url: String,
    pub fabric: String,
    pub token: String,
    pub error: Option<String>,
}

impl OnboardingForm {
    pub(super) fn new(default_fabric: &str) -> Self {
        Self {
            field: OnboardingField::Name,
            name: String::new(),
            url: "http://localhost:51000".to_owned(),
            fabric: default_fabric.to_owned(),
            token: String::new(),
            error: None,
        }
    }

    pub(super) fn active_value(&mut self) -> &mut String {
        match self.field {
            OnboardingField::Name => &mut self.name,
            OnboardingField::Url => &mut self.url,
            OnboardingField::Fabric => &mut self.fabric,
            OnboardingField::Token => &mut self.token,
        }
    }
}

pub(super) fn fabric_targets(config: &Config) -> Vec<Target> {
    let mut gateways_by_fabric: BTreeMap<&str, Vec<&GatewayConfig>> = BTreeMap::new();
    for gateway in &config.gateways {
        gateways_by_fabric
            .entry(gateway.fabric.as_str())
            .or_default()
            .push(gateway);
    }
    gateways_by_fabric
        .into_iter()
        .map(|(fabric, gateways)| Target {
            group_id: fabric.to_owned(),
            name: fabric.to_owned(),
            environment: "direct".to_owned(),
            region: None,
            status: "unknown".to_owned(),
            is_default: fabric == config.default_fabric,
            gateway_count: gateways.len() as u64,
            gateway: gateways.first().map(|gateway| GatewaySummary {
                id: gateway.name.clone(),
                name: gateway.name.clone(),
                url: gateway.url.clone(),
                service_type: None,
            }),
        })
        .collect()
}

impl App {
    #[must_use]
    pub fn selected_fabric(&self) -> Option<&Target> {
        self.fabrics.get(self.fabric_index)
    }

    #[must_use]
    pub fn gateways_for_selected_fabric(&self) -> Vec<&GatewayConfig> {
        let Some(fabric) = self.selected_fabric() else {
            return Vec::new();
        };
        self.config
            .gateways
            .iter()
            .filter(|gateway| gateway.fabric == fabric.group_id)
            .collect()
    }

    #[must_use]
    pub fn selected_gateway(&self) -> Option<&GatewayConfig> {
        self.gateways_for_selected_fabric()
            .get(self.gateway_index)
            .copied()
    }

    #[must_use]
    pub fn selected_executor(&self) -> Option<&Executor> {
        self.overview
            .as_ref()
            .and_then(|overview| overview.executors.get(self.executor_index))
    }

    #[must_use]
    pub fn selected_attention(&self) -> Option<&AttentionItem> {
        self.overview
            .as_ref()
            .and_then(|overview| overview.attention.get(self.attention_index))
    }

    pub(super) fn handle_splash_key(&mut self, key: KeyEvent) -> UserAction {
        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => UserAction::Quit,
            KeyCode::Char('q') => UserAction::Quit,
            KeyCode::Char(' ') => {
                if self.config.gateways.is_empty() {
                    self.screen = Screen::Onboarding;
                    self.status = "Add the first Marie gateway".to_owned();
                } else {
                    self.screen = Screen::Targets;
                    self.status = "Select a gateway and press Enter".to_owned();
                }
                UserAction::None
            }
            _ => UserAction::None,
        }
    }

    pub(super) fn handle_onboarding_key(&mut self, key: KeyEvent) -> UserAction {
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            return UserAction::Quit;
        }
        match key.code {
            KeyCode::Esc => {
                self.screen = if self.config.gateways.is_empty() {
                    Screen::Splash
                } else {
                    Screen::Targets
                };
            }
            KeyCode::Tab | KeyCode::Down => self.onboarding.field = self.onboarding.field.next(),
            KeyCode::BackTab | KeyCode::Up => {
                self.onboarding.field = self.onboarding.field.previous();
            }
            KeyCode::Backspace => {
                self.onboarding.active_value().pop();
                self.onboarding.error = None;
            }
            KeyCode::Enter if self.onboarding.field != OnboardingField::Token => {
                self.onboarding.field = self.onboarding.field.next();
            }
            KeyCode::Enter => self.finish_onboarding(),
            KeyCode::Char(character)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                self.onboarding.active_value().push(character);
                self.onboarding.error = None;
            }
            _ => {}
        }
        UserAction::None
    }

    pub(super) fn finish_onboarding(&mut self) {
        let name = self.onboarding.name.trim();
        if name.is_empty() {
            self.onboarding.field = OnboardingField::Name;
            self.onboarding.error = Some("Gateway name is required".to_owned());
            return;
        }
        if self
            .config
            .gateways
            .iter()
            .any(|gateway| gateway.name == name)
        {
            self.onboarding.field = OnboardingField::Name;
            self.onboarding.error = Some(format!("Gateway already exists: {name}"));
            return;
        }
        let gateway_url = self.onboarding.url.trim().trim_end_matches('/');
        match Url::parse(gateway_url) {
            Ok(url) if matches!(url.scheme(), "http" | "https") && url.host().is_some() => {}
            _ => {
                self.onboarding.field = OnboardingField::Url;
                self.onboarding.error =
                    Some("Enter a complete http:// or https:// gateway URL".to_owned());
                return;
            }
        }
        let fabric = match self.onboarding.fabric.trim() {
            "" => self.config.default_fabric.clone(),
            value => value.to_owned(),
        };
        let token = self.onboarding.token.trim().to_owned();
        let token = (!token.is_empty()).then_some(token);
        let gateway = GatewayConfig {
            name: name.to_owned(),
            url: gateway_url.to_owned(),
            fabric: fabric.clone(),
            token,
            token_env: None,
        };
        self.config.gateways.push(gateway);
        if let Err(error) = save_config(&self.config_path, &self.config) {
            self.config.gateways.pop();
            self.onboarding.error = Some(error.to_string());
            return;
        }
        self.fabrics = fabric_targets(&self.config);
        self.fabric_index = self
            .fabrics
            .iter()
            .position(|target| target.group_id == fabric)
            .unwrap_or(0);
        let gateways = self.gateways_for_selected_fabric();
        self.gateway_index = gateways
            .iter()
            .position(|item| item.name == name)
            .unwrap_or(0);
        self.screen = Screen::Targets;
        self.focus = Focus::Gateways;
        self.connection = "idle".to_owned();
        self.status = format!("Added {name} to fabric {fabric}; press Enter to load");
        self.onboarding = OnboardingForm::new(&self.config.default_fabric);
        self.invalidate_target();
    }

    pub(super) fn open_onboarding(&mut self) {
        let fabric = self
            .selected_fabric()
            .map_or(self.config.default_fabric.as_str(), |target| {
                target.group_id.as_str()
            });
        self.onboarding = OnboardingForm::new(fabric);
        self.navigate(Screen::Onboarding, self.focus);
        self.status = "Add a Marie gateway".to_owned();
    }

    pub(super) fn start_overview_request(&mut self, initial: bool) -> Option<ApiCommand> {
        if self.in_flight {
            self.status = "Refresh already in progress".to_owned();
            return None;
        }
        let gateway = self.selected_gateway()?.clone();
        if initial {
            self.generation = self.generation.wrapping_add(1);
            self.overview = None;
            self.last_success = None;
        }
        self.in_flight = true;
        self.connection = if initial { "connecting" } else { "refreshing" }.to_owned();
        self.status = format!("Loading overview from {}", gateway.name);
        Some(ApiCommand::Overview {
            generation: self.generation,
            target: GatewayTarget {
                name: gateway.name,
                url: gateway.url,
                fabric: gateway.fabric,
                token: gateway.token,
                token_env: gateway.token_env,
            },
        })
    }

    pub(super) fn record_overview(&mut self, overview: Overview) {
        let previous_executor = self.selected_executor().map(|executor| executor.id.clone());
        self.executor_index = previous_executor
            .and_then(|id| {
                overview
                    .executors
                    .iter()
                    .position(|executor| executor.id == id)
            })
            .unwrap_or(0)
            .min(overview.executors.len().saturating_sub(1));
        self.attention_index = self
            .attention_index
            .min(overview.attention.len().saturating_sub(1));
        if let Some(fabric) = self.fabrics.get_mut(self.fabric_index) {
            fabric.status.clone_from(&overview.health.state);
        }
        let selection = SelectionState {
            fabric: Some(overview.target.group_id.clone()),
            gateway: Some(overview.target.gateway.name.clone()),
        };
        self.overview = Some(overview);
        if self.screen == Screen::Targets {
            self.screen = Screen::Overview;
            self.focus = Focus::Executors;
        }
        self.connection = "connected".to_owned();
        self.status = "Overview refreshed".to_owned();
        self.last_success = Some(Instant::now());
        self.failures = 0;
        self.next_refresh = Instant::now() + POLL_INTERVAL;
        if let Err(error) = save_selection(&self.state_path, &selection) {
            self.status = format!("Overview refreshed; could not save selection: {error}");
        }
    }
}
