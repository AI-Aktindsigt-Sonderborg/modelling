.. _model-card-aktindsigt-ner-dp:

Assisteret Anonymisering af Personhenførbar Data på Aktindsigter (AAPDA)
========================================================================
Beskrivelse
-----------
MLM modellen er en generel bert-base videretrænet fra modellen `NB-BERT-base <https://huggingface.co/NbAiLab/nb-bert-base>`_.
Modellen er trænet på 2672566 sætninger genereret på baggrund af data skrapet fra hovedparten af alle danske kommuner.

Brug
----
Modellen er trænet som en videreførsel af den prætrænede model NB-BERT-base model og derfor
bør den finetunes til specifikke opgaver fx. Named Entity Recognition (NER) eller Semantisk Søgning.

Datasæt
-------
Modellen er trænet på 2672566 unikke sætninger og valideret på 54543 sætninger.
Den rå html tekst skrapet fra de **N antal** kommuner og **M antal** KL områder (**Mangler databeskrivelse her fra Aktio**) er blevet filtreret,
opdelt i unikke sætninger og derefter inddelt i trænings- og valideringssæt - se :class:`.RawScrapePreprocessing` for
for beskrivelse af koden benyttet til præprocessering af det skrabede data.
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
