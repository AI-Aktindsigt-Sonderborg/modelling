import argparse
from distutils.util import strtobool


class MLMArgParser:
    """
    Class to handle input args for unsupervised Masked Language Modelling
    Methods
    -------
    add_data_params()
        add data parameters
    add_model_params()
        Add model parameters
    add_training_params()
        Add parameters relevant for training - including learning rate scheduling
    add_eval_params()
        Add parameters relevant for evaluation
    add_dp_params()
        Add parameters relevant for differential privacy
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.add_helper_params()
        self.add_data_params()
        self.add_model_params()
        self.add_training_params()
        self.add_eval_params()
        self.add_dp_params()

    def add_helper_params(self):
        helper_params = self.parser.add_argument_group('helpers')
        helper_params.add_argument("-p", type=lambda x: bool(strtobool(x)),
                                  default=False, help="whether to only print args and exit")

    def add_data_params(self):
        """
        Add data parameters
        """
        data_params = self.parser.add_argument_group('data')
        data_params.add_argument("--train_data", type=str, default='train.json',
                                 help="training data file name")
        data_params.add_argument("--eval_data", type=str, default='validation.json',
                                 help="validation data file name")

    def add_model_params(self):
        """
        Add model parameters
        """
        model_params = self.parser.add_argument_group('modelling')
        model_params.add_argument("--load_alvenir_pretrained", type=lambda x: bool(strtobool(x)),
                                  default=True, help="Whether to load local alvenir model")
        model_params.add_argument("--model_name", type=str, default='NbAiLab_nb-bert-base-2022-08-11_14-28-23',
                                  help="foundation model from huggingface")
        model_params.add_argument("--save_config", type=lambda x: bool(strtobool(x)), default=True,
                                  help="Whether to save input args to file")
        model_params.add_argument("--replace_head", type=lambda x: bool(strtobool(x)), default=True,
                                  help="Whether to replace bert head")
        model_params.add_argument("--freeze_embeddings", type=lambda x: bool(strtobool(x)), default=True,
                                  help="Whether to freeze embeddings layer")
        model_params.add_argument("--save_model_at_end", type=lambda x: bool(strtobool(x)),
                                  default=True,
                                  help="Whether to save final model after training.")

    def add_training_params(self):
        """
        Add parameters relevant for training - including learning rate scheduling
        """
        training_params = self.parser.add_argument_group('training')
        training_params.add_argument("--local_testing", type=lambda x: bool(strtobool(x)),
                                     default=False,
                                     help="Whether to test on local machine with small subset")
        training_params.add_argument("--dp", type=lambda x: bool(strtobool(x)), default=True,
                                     help="Whether to train model with differential privacy")
        training_params.add_argument("--max_length", type=int, default=64,
                                     help="Max length for a text input")
        training_params.add_argument("--epochs", type=int, default=20,
                                     help="Number of epochs to train model")
        training_params.add_argument("--weight_decay", type=float, default=0.01,
                                     help="Weight decay")
        training_params.add_argument("--use_fp16", type=lambda x: bool(strtobool(x)), default=False,
                                     help="Set to True, if your GPU supports FP16 operations")
        training_params.add_argument("--train_batch_size", type=int, default=8,
                                     help="Batch size specifies the sample size of which the "
                                          "gradients are computed. Depends on memory available")
        training_params.add_argument("--whole_word_mask", type=lambda x: bool(strtobool(x)),
                                     default=False,
                                     help="If set to true, whole words are masked")
        training_params.add_argument("--mlm_prob", type=float, default=0.15,
                                     help="Probability that a word is replaced by a [MASK] token")
        training_params.add_argument("--device", type=str, default='cuda',
                                     help="device to train on, can be either 'cuda' or 'cpu'")
        training_params.add_argument("--freeze_layers", type=lambda x: bool(strtobool(x)),
                                     default=False,
                                     help="whether to freeze all bert layers until "
                                          "freeze_layers_n_steps is reached")
        training_params.add_argument("--freeze_layers_n_steps", type=int, default=20000,
                                     help="number of steps to train head only")

        lr_params = self.parser.add_argument_group('learning rate')
        lr_params.add_argument("--auto_lr_scheduling", type=lambda x: bool(strtobool(x)),
                                     default=True,
                                     help="Whether to compute lr_warmup and decay automatically\n"
                                          "freeze_layers_n_steps = 10%% of total_steps\n"
                                          "lr_freezed_warmup_steps = 10%% of freeze_layers_n_steps\n"
                                          "lr_warmup_steps = 10%% of steps training full model\n"
                                          "lr_start_decay = 50%% of training full model")
        lr_params.add_argument("--lr_freezed", type=float, default=0.0005,
                                     help="number of steps to train head only")
        lr_params.add_argument("--lr_freezed_warmup_steps", type=int, default=1000,
                                     help="number of steps to train head only")
        lr_params.add_argument("--lr", type=float, default=0.00005,
                                     help="Learning rate")
        lr_params.add_argument("--lr_warmup_steps", type=int, default=6000,
                                     help="warmup learning rate - set to 1 if no warmup")
        lr_params.add_argument("--lr_start_decay", type=int, default=46000,
                                     help="after which step to start decaying learning rate")


    def add_eval_params(self):
        """
        Add parameters relevant for evaluation
        """
        eval_params = self.parser.add_argument_group('evaluation')
        # ToDo: evaluate steps must be smaller than number of steps in each epoch
        eval_params.add_argument("--evaluate_steps", type=int, default=50000,
                                 help="evaluate model accuracy after number of steps")
        eval_params.add_argument("--logging_steps", type=int, default=5000,
                                 help="Log model accuracy after number of steps")
        eval_params.add_argument("--save_steps", type=int, default=50000,
                                 help="save checkpoint after number of steps")
        eval_params.add_argument("--save_only_best_model", type=lambda x: bool(strtobool(x)),
                                 default=True,
                             help="Whether to only save best model - overwrites save_steps if True")
        eval_params.add_argument("--eval_batch_size", type=int, default=6,
                                 help="Batch size for evaluation")

        eval_params.add_argument("--evaluate_during_training", type=bool, default=True,
                                 help="Whether to evaluate model during training")
        eval_params.add_argument("--make_plots", type=lambda x: bool(strtobool(x)), default=True,
                                 help="Whether to plot running learning rate, loss and accuracies")

    def add_dp_params(self):
        """
        Add parameters relevant for differential privacy
        """
        dp_params = self.parser.add_argument_group('dp')
        dp_params.add_argument("--epsilon", type=float, default=1, help="privacy parameter epsilon")
        dp_params.add_argument("--delta", type=float, default=4.6e-05,
                               help="privacy parameter delta. Usually a good delta is 1/len(train)")
        dp_params.add_argument("--compute_delta", type=lambda x: bool(strtobool(x)), default=True,
                                 help="Whether to compute delta such that delta=1/len(train)")
        dp_params.add_argument("--max_grad_norm", type=float, default=1.2,
                               help="maximum norm to clip gradient")
        dp_params.add_argument("--lot_size", type=int, default=64,
                               help="Lot size specifies the sample size of which noise is "
                                    "injected into. Must be larger and multiple of batch size")
        # dp_params.add_argument("--simulate_batches", type=lambda x: bool(strtobool(x)), default=False,
        #                        help="Simulate larger batch size due to GPU limitations")
        # dp_params.add_argument("--batch_multiplier", type=int, default=2,
        #                        help="simulate batches the size of train_batch_size*batch_multiplier")

