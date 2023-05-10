.. _code-data-prep:

Data præprocessering
====================
Indsæt Eksempel her
-------------------

.. code-block::

	This is some code


Følgende klasser metoder er benyttet til præprocessering data af forskellige typer.

Klasser
-------

.. autoclass:: mlm.data_utils.data_prep_input_args::DataPrepArgParser

.. autoclass:: mlm.data_utils.prep_scrape::RawScrapePreprocessing
	:members: run, extract_danish_and_save_from_raw, create_unique_sentences, split_train_val, create_dump_data, save_datasets, is_correct_danish,
		fix_utf8_encodings

.. autoclass:: sc.data_utils.prep_scrape::ClassifiedScrapePreprocessing
	:members: read_xls_save_json

Hjælpefunktioner
----------------
.. automethod:: mlm.data_utils.helpers::split_sentences
