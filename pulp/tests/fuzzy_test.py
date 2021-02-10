from fuzzysearch import find_near_matches

from definitions import from_root_dir

with open(from_root_dir('data/fuzzy_test'),'r') as f:
    text = f.read()
    result = find_near_matches('augmentation) might help eliminate these problems. In the mean time, spatial aggregation in multiple 2D views (as proposed here) might be a very efficient (and computationally les',text,max_l_dist=32)
result