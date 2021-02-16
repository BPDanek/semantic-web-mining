import csv
import pickle
from collections import defaultdict
import time
import os.path


if __name__ == '__main__':
    LENGTH_OF_GRAPHDB = 1048576
    PICKLE_FILE_NAME = "knowledgegraph.pkl"
    start = time.time()
    entities = defaultdict()
    # Try to open the .pkl file, and also validate it by checking if it's the right length
    if os.path.isfile(PICKLE_FILE_NAME):
        with open(PICKLE_FILE_NAME, 'rb') as f:
            entities = pickle.load(f)
        if len(entities.keys()) != LENGTH_OF_GRAPHDB:
            print(f"Incorrect length: {len(entities.keys())} rows of {LENGTH_OF_GRAPHDB} present in .pkl file")
        # The .pkl file loads in 2 seconds! Nice!
        print(f"Finished in {time.time() - start} seconds")
    else:
        with open("knowledgegraphdata.csv", encoding="utf-8") as f:
            csv_reader = csv.reader(f, delimiter=",")
            for i, row in enumerate(csv_reader):
                # Split along the tab to get each entry in the CSV separated. Also, drop the last term since those URLS
                # are incomprehensible as is and many of them don't even work anymore.
                record = row[0].split("\t")[:-1]
                # This separation also leaves a string of length 0 as an entry each time, so try to strip it out here
                # There's two instances of this, so do it twice
                try:
                    record.remove('')
                except ValueError:
                    pass
                try:
                    record.remove('')
                except ValueError:
                    pass
                entities[i] = record
            # Loads entire knowledge graph in 54 seconds...how much faster is pkl?
        print(f"Finished in {time.time() - start} seconds")
        with open(PICKLE_FILE_NAME, "wb") as f:
            pickle.dump(entities, f)
    for value in entities.values():
        print(value)
        print("\n\n")


