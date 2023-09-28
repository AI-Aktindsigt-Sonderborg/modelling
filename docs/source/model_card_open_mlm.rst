.. _model-card-open-mlm:

Tværkommunal Sprogmodel Del 1 (ts1)
===================================
Beskrivelse
-----------
Den første del af den tværkommunale sprogmodel modellen er en generel bert-base videretrænet fra modellen `NB-BERT-base <https://huggingface.co/NbAiLab/nb-bert-base>`_.
Modellen er trænet på 2672566 sætninger genereret på baggrund af data skrapet fra hovedparten af alle danske kommuner.

Brug
----
Modellen er trænet som en videreførsel af den prætrænede model NB-BERT-base model og derfor
bør den finetunes til specifikke opgaver fx. Named Entity Recognition (NER) eller Semantisk Søgning.

Datasæt
-------
Modellen er trænet på 2672566 unikke sætninger og valideret på 54543 sætninger.
Den rå html tekst skrapet fra de 94 kommuner og 8 KL områder er blevet filtreret,
opdelt i unikke sætninger og derefter inddelt i trænings- og valideringssæt - se vores `GitHub repository <https://github.com/AI-Aktindsigt-Sonderborg/modelling>`_ for
for beskrivelse af koden benyttet til præprocessering af det skrabede data.
Se :ref:`data-home` for en beskrivelse af datasættet.

Mere information
----------------
I denne `artikel <https://arxiv.org/pdf/2004.10964.pdf>`_ fra 2020 beskrives hvordan
en `Masked Language Model (MLM) <https://www.sbert.net/examples/unsupervised_learning/MLM/README.html>`_ kan
benyttes til at videretræne prætrænede modeller på ikke-annoteret domænespecifikt
data kan forhøje kvaliteten af domænerelevante vector embeddings signifikant.
Modellen optimeres ved at maskere enkelte ord i sætninger for derefter at forudsige hvilket ord, der er maskeret.

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
   * - ts1
     - :math:`0.0001`
     - 64
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