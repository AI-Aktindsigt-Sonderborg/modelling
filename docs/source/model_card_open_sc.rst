.. _model-card-open-sc:

Tværkommunal Sprogmodel Del 2 (ts2)
===================================
Beskrivelse
-----------
Den semantiske søgemodel er en finetunet version af vores :doc:`model_card_open_mlm` til at klassificere et givet KL område baseret på en input sætning.
Modellen er trænet på 2672566 sætninger genereret på baggrund af data skrabet fra hovedparten af alle danske kommuner.

Brug
----
Modellen kan benyttes til at søge efter similære sætninger og klassificere hvilken KL-kategori en givet sætning stammer fra.

Datasæt
-------
Modellen er trænet på **N** unikke sætninger og valideret på **M** sætninger.
Se vores `GitHub repository <https://github.com/AI-Aktindsigt-Sonderborg/modelling>`_ for
for beskrivelse af koden benyttet til præprocessering af det skrabede data.
Se :ref:`data-home` for en beskrivelse af datasættet.

Eksempel
--------

.. code-block:: python
   :linenos:

	"""
	Inferens på semantisk søgemodel
	===============================

	Example script to predict a "KL category" from an input sentence.
	Script should be run as module from project root folder "modelling" e.g.
	- python -m example_scripts.sc_inference
	"""
	from sc.local_constants import MODEL_DIR
	from sc.modelling_utils.input_args import SequenceModellingArgParser
	from sc.modelling_utils.sequence_classification import SequenceClassification

	sc_parser = SequenceModellingArgParser()

	label_dict = {'Beskæftigelse og integration': 0, 'Børn og unge': 1,
	              'Erhverv og turisme': 2, 'Klima, teknik og miljø': 3,
	              'Kultur og fritid': 4, 'Socialområdet': 5,
	              'Sundhed og ældre': 6, 'Økonomi og administration': 7}

	LABELS = list(label_dict)

	args = sc_parser.parser.parse_args()
	args.labels = LABELS

	args.load_alvenir_pretrained = True

	args.model_name = 'ts1'
	args.custom_model_dir = MODEL_DIR

	modelling = SequenceClassification(args)

	model = modelling.get_model()

	prediction = modelling.predict(model=model,
	                                sentence="Dette er en sætning om en børnehave",
	                                label="Børn og unge")

	print(f"Sætning: {prediction.sentence}")
	print(f"Sand KL kategori: {prediction.label}")
	print(f"Modellens forudsagte KL kategori: {prediction.prediction}")


Træningsprocedure
-----------------

Hyperparametre
^^^^^^^^^^^^^^
.. list-table::
   :header-rows: 1

   * - Model
     - learning_rate
     - train_batch_size
     - eval_batch_size
     - optimizer
     - num_epochs
   * - ts2
     - :math:`5e^{-5}`
     - 32
     - 32
     - AdamW med betas=(0.9,0.999) og epsilon=1e-08
     - 10

Framework versioner
^^^^^^^^^^^^^^^^^^^
- transformers 4.19.2
- opacus 1.2.0
- datasets 2.2.2
- pandas
- seaborn
- numpy==1.22.3
- pytorch 1.13.0+cu11