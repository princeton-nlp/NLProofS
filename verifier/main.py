from common import *
from pytorch_lightning.utilities.cli import LightningCLI
from verifier.datamodule import EntailmentDataModule
from verifier.model import EntailmentClassifier


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: Any) -> None:
        parser.link_arguments("model.model_name", "data.model_name")
        parser.link_arguments("data.max_input_len", "model.max_input_len")


def main() -> None:
    cli = CLI(EntailmentClassifier, EntailmentDataModule, save_config_overwrite=True)
    print("Configuration: \n", cli.config)


if __name__ == "__main__":
    main()
