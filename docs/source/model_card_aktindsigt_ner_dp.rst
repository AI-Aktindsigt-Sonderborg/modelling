.. _model-card-aktindsigt-ner-dp:

Assisteret Anonymisering af Personhenførbar Data på Aktindsigter (AAPDA)
========================================================================
Beskrivelse
-----------
AAPDA-modellen er en model finetunet til Named Entity Recognition med det formål at semi-automatisk anonymisere aktindsigtsdokumenter fra Sønderborg Kommune.
Modellen er finetunet på :ref:`model-card-aktindsigt-mlm-dp`.
Modellen er trænet på 49191 sætninger genereret på baggrund af data annoteret baseret på aktindsigter fra Sønderborg Kommune.

Brug
----
Modellen er finetunet til Named Entity Recognition (NER) og er trænet til at forudsige følgende kategorier:

.. list-table::
   :header-rows: 1

   * - Tag
     - Navn
     - Beskrivelse
   * - PERSON
     - Person
     - Navn på en person (fx *Kasper Hansen* eller *Birgitte*)
   * - ORGANISATION
     - Organisation
     - Navn på en organisation (fx *Alvenir Aps* eller *Aktio*)
   * - LOKATION
     - Lokation
     - Navn på en lokation (fx *Danmark* eller *Kongens Have*)
   * - HELBRED
     - Helbred
     - Ord relaterede til helbred (fx *hovedpine* eller *OCD*)
   * - KOMMUNE
     - Kommune
     - Navn på en kommune (fx *Sønderborg Kommune*)
   * - TELEFONNUMMER
     - Telefonnummer
     - Telefonnummer (fx *11 22 33 69*, *11223344* eller *1122 3344*)


Datasæt
-------
Modellen er trænet på 49191 unikke sætninger og valideret på 2359 sætninger under træningen. Herefter modellen evalueret på et udeholdt test-sæt bestående af 25 entiteter **beskriv i data** fra hver kategori.
Den rå data er genereret af **Aktio** og overleveret til Alvenir som en jsonlines fil. Data er blevet filtreret,
opdelt i unikke sætninger og derefter inddelt i trænings- og valideringssæt af Alvenir.
Se :ref:`data-home` for en beskrivelse af datasættet.

Mere information
----------------

Eksempel
--------
Du kan benytte modellen til at forudsige entiteter sådan her:



.. code-block:: python
	:linenos:

	from transformers import pipeline
	import pandas as pd

	ner = pipeline(task='ner',
       		       model='../ner/models/21-dp-base-BIO/best_model',
            	   aggregation_strategy='first')

	sentence = 'Kasper Schjødt-Hansen er medarbejder i virksomheden Alvenir Aps og har ofte ekstrem hovedpine.' \
           ' Han bor på Blegdamsvej 85, 2100 København Ø som ligger i Københavns Kommune.' \
           ' Hans tlf nummer er 12345560 og han er fra Danmark. Blegamsvej er tæt på Fælledparken.'


	result = ner(sentence)
	print(pd.DataFrame.from_records(result))

Resultater
----------
Da NER-modellerne er finetunet på andre kategorier end de eksisterende open-source NER modeller er disse svære at sammenligne direkte. Nedenstående tabel viser de forskellige modellers Macro-F1 NER score på det førnævnte test-sæt. 

.. list-table::
   :header-rows: 1

   * - Model
     - F1-score
     - Beskrivelse
   * - last-model-ner-akt
     - 0
     - Sønderborg aktindsigt sprogmodel finetunet på NER-annoterede aktindsigter
   * - base-ner-akt
     - 0
     - NBailab-base finetunet på NER-annoterede aktindsigter
   * - akt-mlm-ner-akt
     - 0
     - Sønderborg aktindsigt sprogmodel finetunet på NER-annoterede aktindsigter
   * - akt-mlm-ner-akt-dp-8
     - 0
     - Sønderborg aktindsigt sprogmodel finetunet på NER-annoterede aktindsigter med DP - :math:`\varepsilon = 8`
   * - akt-mlm-ner-akt-dp-1
     - 0
     - Sønderborg aktindsigt sprogmodel finetunet på NER-annoterede aktindsigter med DP - :math:`\varepsilon = 1`


Træningsprocedure
-----------------

Hyperparametre
^^^^^^^^^^^^^^

Træningsresultater
^^^^^^^^^^^^^^^^^^

Framework versioner
^^^^^^^^^^^^^^^^^^^

 - transformers 4.19.2
 - opacus 1.2.0
 - datasets 2.2.2
 - pandas
 - seaborn
 - numpy==1.22.3
 - wandb