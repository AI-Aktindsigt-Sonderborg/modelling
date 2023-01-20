import argparse
from datetime import datetime


class Modelling:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.total_steps = None

        self.output_name = f'{self.args.model_name.replace("/", "_")}-' \
                           f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        self.args.output_name = self.output_name

        self.train_data = None
        self.eval_data = None
        self.model = None
        self.scheduler = None

        if not self.args.freeze_layers:
            self.args.freeze_layers_n_steps = 0
            self.args.lr_freezed_warmup_steps = None
            self.args.lr_freezed = None



if __name__ == '__main__':
    from shared.modelling_utils.input_args import ModellingArgParser
    model_parser = ModellingArgParser()
    args = model_parser.parser.parse_args()

    modelling = Modelling(args=args)
