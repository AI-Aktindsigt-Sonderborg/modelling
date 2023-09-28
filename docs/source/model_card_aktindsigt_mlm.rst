.. _model-card-aktindsigt-mlm:

Sønderborg Aktindsigt Sprogmodel (sas)
======================================
Beskrivelse
-----------
SAS-modellen er en generel bert-base model videretrænet fra modellen `NB-BERT-base <https://huggingface.co/NbAiLab/nb-bert-base>`_.
Modellen er trænet på 1125808 sætninger genereret på baggrund af data baseret på aktindsigter udtrukket hos Sønderborg Kommune.
Modellen er trænet uden differential privacy. Der er også trænet en model (sas-dp) med :math:`(\varepsilon, \delta)`-differential privacy med parametre :math:`\varepsilon = 8`, :math:`\delta = 8.9e^{-7}`. DP-modellen anbefales dog ikke at benyttes, ydeevnen er meget lav.

Brug
----
Modellen er trænet som en videreførsel af den prætrænede model NB-BERT-base model og derfor
bør den finetunes til specifikke opgaver fx. Named Entity Recognition (NER) eller Semantisk Søgning.

Datasæt
-------
Modellen er trænet på 1125808 unikke sætninger og valideret på 5658 sætninger.
Den rå data er genereret af **Aktio** og overleveret til Alvenir som **8** jsonlines filer. Data er blevet filtreret,
opdelt i unikke sætninger og derefter inddelt i trænings- og valideringssæt af Alvenir.
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
     - lot_size
     - epsilon
     - delta
     - num_epochs
   * - sas
     - :math:`2.4e^{-5}`
     - 32
     - 32
     - AdamW med betas=(0.9,0.999) og epsilon=1e-08
     - NA
     - NA
     - NA
     - 10
   * - sas-dp-8
     - 0.00015
     - 8
     - 8
     - AdamW med betas=(0.9,0.999) og epsilon=1e-08
     - 64
     - 8
     - :math:`8.89e^{-7}`
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