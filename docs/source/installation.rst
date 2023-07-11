Kom i gang
==========

Installation
------------

.. code-block:: console

	(.venv) $ conda install pytorch torchvision torchaudio cudatoolkit=<cuda_version> -c pytorch
	(.venv) $ pip install -r requirements.txt
	(.venv) $ python setup.py

Simpelt eksempel
----------------
.. code-block::

	import bla bla from bla bla
	argparser = ModellingArgParser()
	args = argparser.parser.parse_args()
	args.train_data = "filename in data_dir"
	args.test_data = "filename in data_dir"
	args.model_name = "huggingface model name"
	modelling = Modelling(args)
	modelling.load_data(train=true, test=true)

	modelling.train_model()
	modelling.evaluate(modelling.data.test)
	modelling.predict("teksteksempel")
