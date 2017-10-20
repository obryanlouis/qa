rm -f files.tar.gz
tar -czvf files.tar.gz model/*.py preprocessing/*.py train/*.py datasets/*.py \
    test/*.py *.py requirements.txt
