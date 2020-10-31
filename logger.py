from logging import LogRecord, Handler


class NeptuneLogHandler(Handler):
    def __init__(self, config):
        super(NeptuneLogHandler, self).__init__()
        if config["neptune_enabled"]:
            import neptune
            neptune.init(project_qualified_name=config["neptune_project_name"],
                         api_token=config["neptune_api_token"])
            self.neptune = neptune.create_experiment(name="L2RPN", params=config, upload_source_files=[
                                      'a3c.py', 'train.py', 'eval.py', 'expert_rules.py', 'utils.py'])
        self.config = config

    def emit(self, record: LogRecord) -> None:
        raw_msg = record.msg
        arr = raw_msg.split("|||")
        value = float(arr[1]) if arr[1].replace(".", "").isdigit() else arr[1]
        if self.config["neptune_enabled"]:
            if isinstance(value, float):
                self.neptune.log_metric(arr[0], value)
            else:
                self.neptune.log_text(arr[0], value)
        else:
            print(f'{value}: {raw_msg}')
