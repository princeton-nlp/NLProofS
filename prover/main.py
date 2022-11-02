from common import *
from pytorch_lightning.utilities.cli import LightningCLI
from prover.datamodule import ProofDataModule
from prover.model import EntailmentWriter


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: Any) -> None:
        parser.link_arguments("model.model_name", "data.model_name")
        parser.link_arguments("model.stepwise", "data.stepwise")
        parser.link_arguments("data.dataset", "model.dataset")
        parser.link_arguments("data.max_input_len", "model.max_input_len")


def main() -> None:
    cli = CLI(EntailmentWriter, ProofDataModule, save_config_overwrite=True)
    print("Configuration: \n", cli.config)


if __name__ == "__main__":
    main()
