from logging import LogRecord, Handler


class NeptuneLogHandler(Handler):
    def __init__(self, config):
        super(NeptuneLogHandler, self).__init__()
        if config["neptune_enabled"]:
            import neptune
            neptune.init(project_qualified_name=config["neptune_project_name"],
                         api_token=config["neptune_api_token"])
            self.neptune = neptune.create_experiment(name="L2RPN", params=config)
        self.config = config

    def emit(self, record: LogRecord) -> None:
        if self.config["neptune_enabled"]:
            raw_msg = record.msg
            arr = raw_msg.split("|||")
            self.neptune.log_metric(arr[0], float(arr[1]))
