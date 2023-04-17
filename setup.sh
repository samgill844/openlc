echo "------- Installing  -------"
python setup.py -q install
echo "------- Cleaning up -------"
rm -rf openlc/__pycache__
rm -rf build dist openlc.egg-info
echo "------- Finished    -------"

