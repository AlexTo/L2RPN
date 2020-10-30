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
        raw_msg = record.msg
        arr = raw_msg.split("|||")
        if self.config["neptune_enabled"]:
            self.neptune.log_metric(arr[0], float(arr[1]))
        else:
            print(f'{float(arr[1])}: {raw_msg}')
