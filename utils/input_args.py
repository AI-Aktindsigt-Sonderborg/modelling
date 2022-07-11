import argparse
from distutils.util import strtobool
class MLMArgParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
        model_params.add_argument("--save_config", type=lambda x: bool(strtobool(x)), default=True,
                                  help="Whether to save input args to file")
        model_params.add_argument("--replace_head", type=lambda x: bool(strtobool(x)), default=True,
                                  help="Whether to replace bert head")
        model_params.add_argument("--save_model_at_end", type=lambda x: bool(strtobool(x)), default=True,
                                  help="Whether to save final model after training.")



    def add_training_params(self):
        training_params = self.parser.add_argument_group('training')
        training_params.add_argument("--local_testing", type=lambda x: bool(strtobool(x)), default=False,
                                     help="Whether to test on local machine with small subset")
        training_params.add_argument("--max_length", type=int, default=128,
                                     help="Max length for a text input")
        training_params.add_argument("--epochs", type=int, default=20,
                                     help="Number of epochs to train model")
        training_params.add_argument("--weight_decay", type=float, default=0.01,
                                     help="Weight decay")
        training_params.add_argument("--use_fp16", type=lambda x: bool(strtobool(x)), default=False,
                                     help="Set to True, if your GPU supports FP16 operations")
        training_params.add_argument("--train_batch_size", type=int, default=8,
                                     help="Batch size specifies the sample size of which the gradients are "
                                          "computed. Depends on memory available")
        training_params.add_argument("--whole_word_mask", type=lambda x: bool(strtobool(x)), default=False,
                                     help="If set to true, whole words are masked")
        training_params.add_argument("--mlm_prob", type=float, default=0.15,
                                     help="Probability that a word is replaced by a [MASK] token")
        training_params.add_argument("--device", type=str, default='cuda',
                                     help="device to train on, can be either 'cuda' or 'cpu'")
        training_params.add_argument("--freeze_layers", type=lambda x: bool(strtobool(x)), default=True,
                                     help="whether to freeze all bert layers until freeze_layers_n_steps is reached")
        training_params.add_argument("--freeze_layers_n_steps", type=int, default=10000,
                                     help="number of steps to train head only")
        training_params.add_argument("--lr_freezed", type=float, default=0.0005,
                                     help="number of steps to train head only")
        training_params.add_argument("--lr_freezed_warmup_steps", type=int, default=1000,
                                     help="number of steps to train head only")
        training_params.add_argument("--lr", type=float, default=0.00005,
                                     help="Learning rate")
        training_params.add_argument("--lr_warmup_steps", type=int, default=4000,
                                     help="warmup learning rate - set to 1 if no warmup")
        training_params.add_argument("--lr_start_decay", type=int, default=40000,
                                     help="after which step to start decaying learning rate ")
        # training_params.add_argument("--grad_accum_steps", type=int, default=1,
        #                              help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass.")
        # training_params.add_argument("--start_lr", type=float, default=0.00001,
        #                              help="Learning rate")
        # training_params.add_argument("--end_lr", type=float, default=0.00001,
        #                              help="Learning rate")


    def add_eval_params(self):
        eval_params = self.parser.add_argument_group('evaluation')
        # ToDo: evaluate steps must be smaller than number of steps in each epoch
        eval_params.add_argument("--evaluate_steps", type=int, default=1200,
                                 help="evaluate model accuracy after number of steps")
        eval_params.add_argument("--logging_steps", type=int, default=1200,
                                 help="Log model accuracy after number of steps")
        eval_params.add_argument("--save_steps", type=int, default=None,
                                 help="save checkpoint after number of steps")
        eval_params.add_argument("--eval_batch_size", type=int, default=8,
                                 help="Batch size for evaluation")
        eval_params.add_argument("--evaluate_during_training", type=bool, default=True,
                                 help="Whether to evaluate model during training")
        eval_params.add_argument("--make_plots", type=lambda x: bool(strtobool(x)), default=True,
                                 help="Whether to plot running learning rate, loss and accuracies")


    def add_dp_params(self):
        dp_params = self.parser.add_argument_group('dp')
        dp_params.add_argument("--epsilon", type=float, default=1, help="privacy parameter epsilon")
        dp_params.add_argument("--delta", type=float, default=4.6e-05,
                               help="privacy parameter delta. Usually a good delta is 1/len(train)")
        dp_params.add_argument("--max_grad_norm", type=float, default=1.2,
                               help="maximum norm to clip gradient")
        dp_params.add_argument("--lot_size", type=int, default=64,
                                     help="Lot size specifies the sample size of which noise is "
                                        "injected into. Must be larger and multiple of batch size")

