.. _model-card-aktindsigt-ner-dp:

Assisteret Anonymisering af Personhenførbar Data på Aktindsigter (AAPDA)
========================================================================
Beskrivelse
-----------
AAPDA-modellerne er modeller finetunet til Named Entity Recognition med det formål at semi-automatisk anonymisere aktindsigtsdokumenter fra Sønderborg Kommune.
Modellerne er finetunet på :ref:`model-card-aktindsigt-mlm-dp`.
Modellerne er trænet på 49191 sætninger genereret på baggrund af data annoteret baseret på aktindsigter fra Sønderborg Kommune.


Delmodeller
^^^^^^^^^^^
NER-modellerne er trænet på tre forskellige måder: En ikke-privat baseline model og to modeller trænet med differential privacy med hhv. :math:`\varepsilon = 8` og :math:`\varepsilon = 1` **(kilde?)**. Den anden DP parameter :math:`\delta` er sat til 1 over længden af træningsdatasættet **(kilde?)**. Derudover er modellerne trænet på to

.. list-table::
   :header-rows: 1

   * - Model
     - Beskrivelse
   * - sas-ner
     - Sønderborg Aktindsigt Sprogmodel finetunet på NER-annoterede aktindsigter excl. Forbrydelse og CPR
   * - sas-ner-dp-8
     - Sønderborg Aktindsigt Sprogmodel finetunet på NER-annoterede aktindsigter med DP - :math:`\varepsilon = 8` excl. Forbrydelse og CPR
   * - sas-ner-dp-1
     - Sønderborg Aktindsigt Sprogmodel finetunet på NER-annoterede aktindsigter med DP - :math:`\varepsilon = 1` excl. Forbrydelse og CPR
   * - sas-ner-fc
     - Sønderborg Aktindsigt Sprogmodel finetunet på NER-annoterede aktindsigter
   * - sas-ner-fc-dp-8
     - Sønderborg Aktindsigt Sprogmodel finetunet på NER-annoterede aktindsigter med DP - :math:`\varepsilon = 8`
   * - sas-ner-fc-dp-1
     - Sønderborg Aktindsigt Sprogmodel finetunet på NER-annoterede aktindsigter med DP - :math:`\varepsilon = 1`


Brug
----
Modellerne er finetunet til Named Entity Recognition (NER) og er trænet til at forudsige følgende kategorier:

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
   * - CPR
     - CPR nummer
     - CPR - **høj usikkerhed** (fx *01011990 1234*, *01011990-1234* eller *010119901234*)
   * - FORBRYDELSE
     - Forbrydelse
     - Forbrydelse - **høj usikkerhed** (fx *tyveri*, *vold* eller *psykisk vold*)


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
     - Macro F1-score
   * - sas-ner
     - 0
   * - sas-ner-dp-8
     - 0
   * - sas-ner-dp-8
     - 0
   * - sas-ner-fc
     - 0
   * - sas-ner-fc-dp-8
     - 0
   * - sas-ner-fc-dp-1
     - 0


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