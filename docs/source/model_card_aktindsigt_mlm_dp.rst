.. _model-card-aktindsigt-mlm-dp:

Sønderborg Aktindsigt Sprogmodel (SAS)
======================================
Beskrivelse
-----------
SAS-modellen er en generel bert-base model videretrænet fra modellen `NB-BERT-base <https://huggingface.co/NbAiLab/nb-bert-base>`_.
Modellen er trænet på 1125808 sætninger genereret på baggrund af data baseret på aktindsigter udtrukket hos Sønderborg Kommune.
Modellen er trænet med :math:`(\epsilon, \delta)`-differential privacy med parametre :math:`\epsilon = 8`, :math:`\delta = 0.002`.


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

Eksempel
--------

.. code-block:: python
	:linenos:

	# Python code here
	def test(a: bool = False):
   		print("hello")
