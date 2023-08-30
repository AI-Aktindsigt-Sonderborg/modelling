.. _model-card-open-sc:

Tværkommunal sprogmodel (Del 2)
===============================
Beskrivelse
-----------
Den semantiske søgemodel er en finetunet version af vores :doc:`model_card_open_mlm` til at klassificere et givet KL område baseret på en input sætning.

**Modellen er trænet på 2672566 sætninger genereret på baggrund af data skrapet fra hovedparten af alle danske kommuner.**

Brug
----
Semantisk Søgning.

Datasæt
-------
Modellen er trænet på **N** unikke sætninger og valideret på **M** sætninger.
- se :class:`.ClassifiedScrapePreprocessing` for
for beskrivelse af koden benyttet til præprocessering af det skrabede data.
Se :ref:`data-home` for en beskrivelse af datasættet.

Mere information
----------------

Eksempel
--------



.. code-block:: python
   :linenos:

	# Python code here
	def test(a: bool = False):
   		print("hello")

