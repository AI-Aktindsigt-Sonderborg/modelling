import argparse


class MLMArgParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_data_params()
        self.add_model_params()
        self.add_training_params()
        self.add_eval_params()
        self.add_dp_params()

    def add_data_params(self):
        data_params = self.parser.add_argument_group('data')
        data_params.add_argument("--train_data", type=str, default='train.json',
                                 help="training data file name")
        data_params.add_argument("--eval_data", type=str, default='validation.json',
                                 help="validation data file name")

    def add_model_params(self):
        model_params = self.parser.add_argument_group('modelling')
        model_params.add_argument("--model_name", type=str, default='NbAiLab/nb-bert-base',
                                  help="foundation model from huggingface")
        model_params.add_argument("--save_config", type=bool, default=True,
                                  help="Whether to save input args to file")
        model_params.add_argument("--save_model_at_end", type=bool, default=True,
                                  help="Whether to save model after training.")


    def add_training_params(self):
        training_params = self.parser.add_argument_group('training')
        training_params.add_argument("--max_length", type=int, default=128,
                                     help="Max length for a text input")
        training_params.add_argument("--epochs", type=int, default=20,
                                     help="Number of epochs to train model")
        training_params.add_argument("--lr", type=float, default=0.00002,
                                     help="Learning rate")
        training_params.add_argument("--weight_decay", type=float, default=0.01,
                                     help="Weight decay")
        training_params.add_argument("--use_fp16", type=bool, default=False,
                                     help="Set to True, if your GPU supports FP16 operations")
        training_params.add_argument("--lot_size", type=int, default=8,
                                     help="Lot size specifies the sample size of which noise is "
                                          "injected into. Must be larger and multiple of batch size")
        training_params.add_argument("--train_batch_size", type=int, default=2,
                                     help="Batch size specifies the sample size of which the gradients are "
                                          "computed. Depends on memory available")
        training_params.add_argument("--whole_word_mask", type=bool, default=False,
                                     help="If set to true, whole words are masked")
        training_params.add_argument("--mlm_prob", type=float, default=0.15,
                                     help="Probability that a word is replaced by a [MASK] token")
        training_params.add_argument("--device", type=str, default='cuda',
                                     help="device to train on, can be either 'cuda' or 'cpu'")

    def add_eval_params(self):
        eval_params = self.parser.add_argument_group('evaluation')
        # ToDo: evaluate steps must be smaller than number of steps in each epoch
        eval_params.add_argument("--evaluate_steps", type=int, default=20,
                                 help="evaluate model accuracy after number of steps")
        eval_params.add_argument("--logging_steps", type=int, default=100,
                                 help="Log model accuracy after number of steps")
        eval_params.add_argument("--save_steps", type=int, default=None,
                                 help="save checkpoint after number of steps")
        eval_params.add_argument("--eval_batch_size", type=int, default=8,
                                 help="Batch size for evaluation")
        eval_params.add_argument("--evaluate_during_training", type=bool, default=True,
                                 help="Whether to evaluate model during training")

    def add_dp_params(self):
        dp_params = self.parser.add_argument_group('dp')
        dp_params.add_argument("--epsilon", type=float, default=1, help="privacy parameter epsilon")
        dp_params.add_argument("--delta", type=float, default=0.00002,
                               help="privacy parameter delta")
        dp_params.add_argument("--max_grad_norm", type=float, default=1.2,
                               help="maximum norm to clip gradient")
