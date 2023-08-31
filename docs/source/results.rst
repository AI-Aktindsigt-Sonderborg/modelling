Resultater
==========

Semantisk Søgning resultater
----------------------------
De semantiske søgemodeller er blevet benchmarket på dansk op mod andre open-source modeller vha.  
`scandeval <https://github.com/saattrupdan/ScandEval>`_.

.. list-table::
   :header-rows: 1

   * - Model
     - DA
     - Beskrivelse
   * - chcaa/dfm-encoder-large-v1
     - 0
     - text
   * - last-model
     - 0
     - 0
   * - akt-mlm
     - 0
     - text
   * - akt-mlm-dp-8
     - 0
     - text


NER resultater
--------------
Da NER-modellerne er finetunet på andre kategorier end de eksisterende open-source NER modeller er disse svære at sammenligne direkte.  

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
