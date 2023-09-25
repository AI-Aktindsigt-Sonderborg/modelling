.. _model-card-aktindsigt-ner:

Assisteret Anonymisering af Personhenførbar Data på Aktindsigter (AAPDA)
========================================================================
Beskrivelse
-----------
AAPDA-modellerne er modeller finetunet til Named Entity Recognition med det formål at semi-automatisk anonymisere aktindsigtsdokumenter fra Sønderborg Kommune.
Modellerne er finetunet på :ref:`model-card-aktindsigt-mlm`.
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
   * - PER
     - Person
     - Navn på en person (fx *Kasper Hansen* eller *Birgitte*)
   * - LOK
     - Lokation
     - Navn på en lokation (fx *Danmark* eller *Kongens Have*)
   * - ADR
     - Adresse
     - Navn på en adresse (fx *Rådhuspladsen, 2400  København*)
   * - HEL
     - Helbred
     - Ord relaterede til helbred (fx *hovedpine* eller *OCD*)
   * - ORG
     - Organisation
     - Navn på en organisation (fx *Alvenir Aps* eller *Aktio*)
   * - KOM
     - Kommune
     - Navn på en kommune (fx *Sønderborg Kommune*)
   * - TEL
     - Telefonnummer
     - Telefonnummer (fx *11 22 33 69*, *11223344* eller *1122 3344*)
   * - CPR
     - CPR nummer
     - CPR - **høj usikkerhed** (fx *01011990 1234*, *01011990-1234* eller *010119901234*)
   * - FOR
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
Da NER-modellerne er finetunet på andre kategorier end de eksisterende open-source NER modeller er disse svære at sammenligne direkte. Nedenstående tabel viser de forskellige modellers Macro-F1 NER et `tilfældigt genereret og ikke-manuelt valideret test-sæt`.

.. list-table::
   :header-rows: 1

   * - Model
     - Macro F1-score
     - PER F1
     - LOK F1
     - ADR F1
     - HEL F1
     - ORG F1
     - KOM F1
     - TEL F1
     - CPR F1
     - FOR F1
   * - sas-ner
     - 0.86
     - 0.95
     - 0.77
     - 0.90
     - 0.63
     - 0.77
     - 0.96
     - 0.92
     - NA
     - NA
   * - sas-ner-dp-8
     - 0.84
     - 0.97
     - 0.64
     - 0.85
     - 0.65
     - 0.73
     - 0.94
     - 0.95
     - NA
     - NA 
   * - sas-ner-dp-1
     - 0.75
     - 0.94
     - 0.51
     - 0.82
     - 0.59
     - 0.62
     - 0.86
     - 0.75
     - NA
     - NA 
   * - sas-ner-fc
     - 0.80
     - 0.96
     - 0.70
     - 0.88
     - 0.43
     - 0.70
     - 0.97
     - 0.90
     - 0.55
     - 0.95    
   * - sas-ner-fc-dp-8
     - 0.82
     - 0.96
     - 0.66
     - 0.87
     - 0.62
     - 0.69
     - 0.94
     - 0.92
     - 0.67
     - 0.93
   * - sas-ner-fc-dp-1
     - 0.69
     - 0.95
     - 0.56
     - 0.84
     - 0.40
     - 0.66
     - 0.88
     - 0.78
     - 0.06
     - 0.87


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
   * - sas-ner
     - 0.86
     - 0.95
     - 0.77
     - 0.90
     - 0.63
     - 0.77
     - 0.96
     - 0.92
   * - sas-ner-dp-8
     - 0.84
     - 0.97
     - 0.64
     - 0.85
     - 0.65
     - 0.73
     - 0.94
     - 0.95
   * - sas-ner-dp-1
     - 0.75
     - 0.94
     - 0.51
     - 0.82
     - 0.59
     - 0.62
     - 0.86
     - 0.75
   * - sas-ner-fc
     - 0.80
     - 0.96
     - 0.70
     - 0.88
     - 0.43
     - 0.70
     - 0.97
     - 0.90
   * - sas-ner-fc-dp-8
     - 0.82
     - 0.96
     - 0.66
     - 0.87
     - 0.62
     - 0.69
     - 0.94
     - 0.92
   * - sas-ner-fc-dp-1
     - 0.69
     - 0.95
     - 0.56
     - 0.84
     - 0.40
     - 0.66
     - 0.88
     - 0.78

Framework versioner
^^^^^^^^^^^^^^^^^^^

 - transformers 4.19.2
 - opacus 1.2.0
 - datasets 2.2.2
 - pandas
 - seaborn
 - numpy==1.22.3
 - pytorch 1.13.0+cu11