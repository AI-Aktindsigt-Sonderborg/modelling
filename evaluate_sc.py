from data_utils.custom_dataclasses import LoadModelType
from local_constants import DATA_DIR
from modelling_utils.supervised_text_modelling import SupervisedTextModelling

# MODEL_NAME = 'last_model_all_data'
MODEL_NAME = 'last_model-2022-12-16_21-55-28'
# MODEL_NAME = 'NbAiLab/nb-sbert-base'

label_dict = {'Beskæftigelse og integration': 0, 'Børn og unge': 1, 'Erhverv og turisme': 2,
              'Klima, teknik og miljø': 3, 'Kultur og fritid': 4, 'Socialområdet': 5,
              'Sundhed og ældre': 6, 'Økonomi og administration': 7}

LABELS = list(label_dict)

supervised_text_modelling = SupervisedTextModelling(labels=LABELS,
                                        data_dir=DATA_DIR,
                                        model_name=MODEL_NAME, load_model_type=LoadModelType.EVAL,
                                        alvenir_pretrained=True)

y_true, y_pred, eval_result = supervised_text_modelling.eval_model()

f_score, f1 = supervised_text_modelling.calc_f1_score(y_list=y_true, prediction_list=y_pred,
                                                labels=LABELS,
                                                conf_plot=True, normalize='true')

print()
