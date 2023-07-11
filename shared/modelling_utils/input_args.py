import argparse
from distutils.util import strtobool


class ModellingArgParser:
    """
    General class to handle input args for model-training. Below arguments are
    grouped for a better overview.

    *Model*

    :param str --custom_model_dir: Whether to load model from specific directory
        (default: None)
    :param str --custom_model_name: Whether to give model specific output name
        (default: None)
    :param bool --save_config: Whether to save input args to file
        (default: True)
    :param bool --freeze_embeddings: Whether to freeze embeddings layer.
        Must be freezed for DP at the moment (default: True)
    :param bool --save_model_at_end: Whether to save final model after training.
        (default: True)
    :param bool --freeze_layers: Whether to freeze all bert layers until
        freeze_layers_n_steps is reached. True is mandatory for DP at the moment
        (default: False)
    :param int --freeze_layers_n_steps: Number of steps to train head only -
        if freeze_layers is true freeze_layers_n_steps must be > 0 (default: 0)
    :param bool --sc_demo: Whether to use model for demo purposes -
        this is not for training (default: False)

    *Training*

    :param bool --local_testing: Whether to test on local machine with small
        subset (default:False)
    :param bool --differential_privacy: Whether to train model with
        differential privacy (default: True)
    :param int --max_length: Max length for a text input (default: 64)
    :param int --epochs: Number of epochs to train model (default: 20)
    :param float --weight_decay: Weight decay (default: 0.01)
    :param bool --use_fp16: Set to True, if your GPU supports FP16 operations
        (default: False)
    :param int --lot_size: Lot size specifies the sample size of which noise
        is injected into. Must be larger and multiple of batch size -- in
        None-DP models lot size is set to train_batch_size, which is then used as
        training batch size (default: 64)
    :param int --train_batch_size: Batch size specifies the sample size of
        which the gradients are computed (default: 32)
    :param str --device: Device to train on, can be either 'cuda' or 'cpu' (
        default: cuda)
    :param bool --auto_lr_scheduling: Whether to compute lr_warmup and decay
        automatically (lr_warmup_steps = 10% of steps training full
        model lr_start_decay = 50% of training full model) (default: True)
    :param float --lr_freezed: Number of steps to train head only
        (default: 0.0005)
    :param int --lr_freezed_warmup_steps: Number of steps to train head only
        (default: 1000)
    :param float --learning_rate: Learning rate (default: 5e-05)
    :param int --lr_warmup_steps: Warmup learning rate - set to 1 if no
        warmup (default: 6000)
    :param int --lr_start_decay: After which step to start decaying learning
        rate (default: 46000)

    *Evaluation*

    :param int --evaluate_steps: Evaluate model accuracy after number of
        steps (default: 50000)
    :param int --logging_steps: Log model accuracy after number of steps
        (default: 5000)
    :param int --save_steps: Save checkpoint after number of steps (default:
        50000)
    :param bool --save_only_best_model: Whether to only save best model -
        overwrites save_steps if True (default: True)
    :param int --eval_batch_size: Batch size for evaluation (default: 32)
    :param bool --evaluate_during_training: Whether to evaluate model during
        training (default: True)
    :param bool --make_plots: Whether to plot running learning rate,
        loss and accuracies (default: True)
    :param List[str] --eval_metrics: Define eval metrics to evaluate best
        model (default: ['loss', 'accuracy'])

    *Differential privacy*

    :param float --epsilon: Privacy parameter epsilon (default: 1)
    :param bool --compute_delta: Whether to compute delta such that
        delta=1/len(train) (default:True)
    :param float --delta: Privacy parameter delta. Usually a good delta is
        1/len(train) (default: 4.6e-05)
    :param float --max_grad_norm: Maximum norm to clip gradient (default: 1.2)

    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument("-p", '--print_only_args', action='store_true',
                                 help="whether to only print args and exit")
        self.add_data_params()
        self.add_model_params()
        self.add_training_params()
        self.add_eval_params()
        self.add_dp_params()

    def add_data_params(self):
        """
        Add data parameters
        """
        pass

    def add_model_params(self):
        """
        Add model parameters
        """
        model_params = self.parser.add_argument_group('modelling')
        model_params.add_argument(
            "--custom_model_dir",
            type=str,
            default=None,
            metavar='<str>',
            help="Whether to load model from specific directory")
        model_params.add_argument(
            "--custom_model_name", "-cmn",
            type=str,
            default=None,
            metavar='<str>',
            help="Whether to give model specific output name")
        model_params.add_argument(
            "--save_config",
            type=lambda x: bool(strtobool(x)),
            default=True,
            help="Whether to save input args to file",
            metavar='<bool>')
        model_params.add_argument(
            '-fe', "--freeze_embeddings",
            type=lambda x: bool(strtobool(x)),
            default=True,
            help="Whether to freeze embeddings layer. Must be freezed for DP"
                 " at the moment",
            metavar='<bool>')
        model_params.add_argument(
            "--save_model_at_end",
            type=lambda x: bool(strtobool(x)),
            default=True,
            metavar='<bool>',
            help="Whether to save final model after training.")
        model_params.add_argument(
            "-fl", "--freeze_layers",
            type=lambda x: bool(strtobool(x)),
            default=False,
            metavar='<bool>',
            help="whether to freeze all bert layers until "
                 "freeze_layers_n_steps is reached."
                 "True is mandatory for DP at the moment")
        model_params.add_argument(
            "-flns", "--freeze_layers_n_steps",
            type=int,
            default=0,
            help="number of steps to train head only - if freeze_layers is true"
                 " freeze_layers_n_steps must be > 0",
            metavar='<int>')
        model_params.add_argument(
            "--sc_demo",
            type=lambda x: bool(strtobool(x)),
            default=False,
            metavar='<bool>',
            help="whether to use model for demo purposes - this is not for"
                 "training")

    def add_training_params(self):
        """
        Add parameters relevant for training - including learning rate
        scheduling
        """
        training_params = self.parser.add_argument_group('training')
        training_params.add_argument(
            "--local_testing",
            type=lambda x: bool(strtobool(x)),
            default=False,
            metavar='<bool>',
            help="Whether to test on local machine with small subset")
        training_params.add_argument(
            "-dp", "--differential_privacy",
            type=lambda x: bool(strtobool(x)),
            default=True,
            metavar='<bool>',
            help="Whether to train model with differential privacy")
        training_params.add_argument(
            "--max_length",
            type=int,
            default=64,
            metavar='<int>',
            help="Max length for a text input")
        training_params.add_argument(
            "--epochs",
            type=int,
            default=20,
            metavar='<int>',
            help="Number of epochs to train model")
        training_params.add_argument(
            "--weight_decay",
            type=float,
            default=0.01,
            help="Weight decay",
            metavar='<float>')
        training_params.add_argument(
            "--use_fp16",
            type=lambda x: bool(strtobool(x)),
            default=False,
            metavar='<bool>',
            help="Set to True, if your GPU supports FP16 operations")
        training_params.add_argument(
            "--lot_size", "-ls",
            type=int,
            default=64,
            metavar='<int>',
            help="Lot size specifies the sample size of which noise is "
                 "injected into. Must be larger and multiple of batch size - "
                 "- in None-DP models lot size is set to train_batch_size, "
                 "which is then used as training batch size")
        training_params.add_argument(
            "-tbs", "--train_batch_size",
            type=int,
            default=32,
            metavar='<int>',
            help="Batch size specifies the sample size of which the "
                 "gradients are computed")
        training_params.add_argument(
            "--device",
            type=str,
            default='cuda',
            metavar='<str>',
            help="device to train on, can be either 'cuda' or 'cpu'")
        training_params.add_argument(
            "-wc", "--weight_classes",
            type=lambda x: bool(strtobool(x)),
            metavar='<bool>',
            default=False,
            help="Whether to add class weights for loss - only implemented for NER for now")
        training_params.add_argument(
            "-mcw", "--manual_class_weighting",
            type=lambda x: bool(strtobool(x)),
            metavar='<bool>',
            default=False,
            help="Whether to weight classes manually.")
        training_params.add_argument(
            "--n_trials",
            type=int,
            default=10,
            metavar='<int>',
            help="Number of trials to run when hp optimizing")

        lr_params = self.parser.add_argument_group('learning rate')
        lr_params.add_argument(
            "-alrs", "--auto_lr_scheduling",
            type=lambda x: bool(strtobool(x)),
            default=True,
            metavar='<bool>',
            help="Whether to compute lr_warmup and decay automatically\n"
                 "lr_warmup_steps = 10%% of steps training full model\n"
                 "lr_start_decay = 50%% of training full model")
        lr_params.add_argument(
            "-lrf", "--lr_freezed",
            type=float,
            default=0.0005,
            metavar='<float>',
            help="number of steps to train head only - must be higher than lr")
        lr_params.add_argument(
            "-lrfws", "--lr_freezed_warmup_steps",
            type=int,
            default=None,
            help="number of steps to train head only",
            metavar='<int>')
        lr_params.add_argument(
            "-lr", "--learning_rate",
            type=float,
            default=0.00005,
            help="Learning rate", metavar='<float>')
        lr_params.add_argument(
            "-lrws", "--lr_warmup_steps",
            type=int, default=6000,
            metavar='<int>',
            help="warmup learning rate - set to 1 if no warmup")
        lr_params.add_argument(
            "-lrsd", "--lr_start_decay", type=int, default=46000,
            metavar='<int>',
            help="after which step to start decaying learning rate")

    def add_eval_params(self):
        """
        Add parameters relevant for evaluation
        """
        eval_params = self.parser.add_argument_group('evaluation')
        eval_params.add_argument(
            "-esteps", "--evaluate_steps",
            type=int,
            default=50000,
            metavar='<int>',
            help="evaluate model accuracy after number of steps")
        eval_params.add_argument(
            "-lsteps", "--logging_steps",
            type=int,
            default=5000,
            metavar='<int>',
            help="Log model accuracy after number of steps")
        eval_params.add_argument(
            "-ssteps", "--save_steps",
            type=int,
            default=50000,
            metavar='<int>',
            help="save checkpoint after number of steps")
        eval_params.add_argument(
            "--save_only_best_model",
            type=lambda x: bool(strtobool(x)),
            metavar='<bool>',
            default=True,
            help="Whether to only save best model - overwrites save_steps if "
                 "True")
        eval_params.add_argument(
            "-ebs", "--eval_batch_size",
            type=int,
            default=32,
            metavar='<int>',
            help="Batch size for evaluation")
        eval_params.add_argument(
            "--evaluate_during_training",
            type=lambda x: bool(strtobool(x)),
            default=True,
            metavar='<bool>',
            help="Whether to evaluate model during training")
        eval_params.add_argument(
            "--make_plots",
            type=lambda x: bool(strtobool(x)),
            default=True,
            metavar='<bool>',
            help="Whether to plot running learning rate, loss and accuracies")
        eval_params.add_argument(
            "--eval_metrics",
            type=str,
            nargs='*',
            default=['loss', 'accuracy'],
            metavar='<str>',
            help="define eval metrics to evaluate best model")

    def add_dp_params(self):
        """
        Add parameters relevant for differential privacy
        """
        dp_params = self.parser.add_argument_group('dp')
        dp_params.add_argument(
            "--epsilon",
            type=float,
            default=1,
            metavar='<float>',
            help="privacy parameter epsilon")
        dp_params.add_argument(
            "--delta",
            type=float,
            default=4.6e-05,
            metavar='<float>',
            help="privacy parameter delta. Usually a "
                 "good delta is 1/len(train)")
        dp_params.add_argument(
            "--compute_delta",
            type=lambda x: bool(strtobool(x)),
            default=True,
            metavar='<bool>',
            help="Whether to compute delta such that delta=1/len(train)")
        dp_params.add_argument(
            "--max_grad_norm",
            type=float,
            default=1.2,
            metavar='<float>',
            help="maximum norm to clip gradient")
