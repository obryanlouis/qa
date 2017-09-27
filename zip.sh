rm -f files.tar.gz
tar -czvf files.tar.gz model/*.py preprocessing/*.py train/*.py setup.py \
    perform_training.py flags.py
