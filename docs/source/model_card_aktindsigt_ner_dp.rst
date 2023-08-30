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

.. table::

   +--------------+--------------+------------------------------------------------------------------------------------------------------+
   | Label        | Navn         | Beskrivelse                                                                                          |
   +==============+==============+======================================================================================================+
   | PERSON       | Person       | Navn på en person (fx *Kasper Hansen* eller *Birgitte*)                                              |
   +--------------+--------------+------------------------------------------------------------------------------------------------------+
   | ORGANISATION | Organisation | Navn på en organisation (fx *Alvenir Aps* eller *Aktio*)                                             | 
   +--------------+--------------+------------------------------------------------------------------------------------------------------+
   | LOKATION     | Lokation     | Navn på en lokation (fx *Danmark* eller *Fælledparken*)                                              | 
   +--------------+--------------+------------------------------------------------------------------------------------------------------+
   | HELBRED      | Helbred      | Ord relaterede til helbred (fx *hovedpine* eller *OCD*)                                              | 
   +--------------+--------------+------------------------------------------------------------------------------------------------------+
   | KOMMUNE      | Kommune      | Navn på en kommune (fx *Sønderborg Kommune*)                                                         | 
   +--------------+--------------+------------------------------------------------------------------------------------------------------+
   | TELEFONNUMMER| Telefonnummer| Telefonnummer (fx *11 22 33 69*, *11223344* eller *1122 3344*)                                       | 
   +--------------+--------------+------------------------------------------------------------------------------------------------------+


Datasæt
-------
Modellen er trænet på 49191 unikke sætninger og valideret på 2359 sætninger under træningen. Herefter modellen evalueret på et udeholdt test-sæt bestående af 25 entiteter fra hver kategori.
Den rå data er genereret af **Aktio** og overleveret til Alvenir som en jsonlines fil. Data er blevet filtreret,
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

Resultater
----------

