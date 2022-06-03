
class MLMUnsupervisedModelling:
    def __init__(self, model_tag: str = 'NbAiLab/nb-bert-base',
                 train_data: str = 'data/train.json',
                 eval_data: str = None):
        self.model_tag = model_tag
        self.train_data = train_data
        self.eval_data = eval_data

