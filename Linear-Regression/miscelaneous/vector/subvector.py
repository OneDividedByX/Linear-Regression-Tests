from miscelaneous.miscelaneous import GetVectorFromIndexes
from miscelaneous.miscelaneous import GetVectorComplement

def with_indexes(v,index_vector):
  return GetVectorFromIndexes(v,index_vector)

def complement(v,index_vector):
  return GetVectorComplement(v,index_vector)