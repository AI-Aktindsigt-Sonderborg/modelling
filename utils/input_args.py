import argparse
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=1, help="privacy parameter epsilon")
    parser.add_argument("--lot_size", type=int, default=8,
                        help="Lot size specifies the sample size of which noise is injected into. "
                             "Must be larger and multiple of batch size")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size specifies the sample size of which the gradients are "
                             "computed. Depends on memory available")
    parser.add_argument("--delta", type=float, default=0.00002,
                        help="privacy parameter delta")
    parser.add_argument("--max_grad_norm", type=float, default=1.2,
                        help="maximum norm to clip gradient")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs to train model")
    parser.add_argument("--lr", type=float, default=0.00002,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    # ToDo: evaluate steps must be smaller than number of steps in each epoch
    parser.add_argument("--evaluate_steps", type=int, default=10,
                        help="evaluate model accuracy after number of steps")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log model accuracy after number of steps")
    parser.add_argument("--model_name", type=str, default='jonfd/electra-small-nordic',
                        help="foundation model from huggingface")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="save checkpoint after number of steps")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Max length for a text input")
    parser.add_argument("--use_fp16", type=bool, default=False,
                        help="Set to True, if your GPU supports FP16 operations")
    parser.add_argument("--whole_word_mask", type=bool, default=False,
                        help="If set to true, whole words are masked")
    parser.add_argument("--mlm_prob", type=float, default=0.15,
                        help="Probability that a word is replaced by a [MASK] token")
    parser.add_argument("--data_file", type=str, default='da_DK_subset.json',
                        help="Probability that a word is replaced by a [MASK] token")
    return parser

