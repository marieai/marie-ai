use super::App;

impl App {
    pub(super) fn invalidate_target(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.in_flight = false;
        self.last_success = None;
        self.connection = "idle".to_owned();
        self.overview = None;
        self.executor_fleet = None;
        self.executor_detail = None;
        self.detail_executor = None;
        self.scheduler = None;
        self.jobs = None;
        self.job_detail = None;
        self.execution_history = None;
        self.dags = None;
        self.dag_detail = None;
        self.throughput = None;
        self.events = None;
        self.flow = None;
        self.attempts = None;
        self.dependency_health = None;
        self.dispatch = None;
    }
}
