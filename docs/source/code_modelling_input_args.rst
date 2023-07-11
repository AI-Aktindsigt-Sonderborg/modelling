.. _code-modelling-args:

Input parametre til modellering
===============================
Her ses input parametre til de forskellige typer af modeller. Generelle parametre relevante for alle modeller er indeholdt i :class:`.ModellingArgParser`, mens de tre afledte klasser
:class:`.MLMArgParser`, :class:`.SequenceModellingArgParser`, :class:`.NERArgParser` indeholder parametre specifikt relevante for de tre modeltyper MLM, SS og NER. 

.. autoclass:: shared.modelling_utils.input_args.ModellingArgParser

.. autoclass:: mlm.modelling_utils.input_args.MLMArgParser

.. autoclass:: ner.modelling_utils.input_args.NERArgParser

.. autoclass:: sc.modelling_utils.input_args.SequenceModellingArgParser
