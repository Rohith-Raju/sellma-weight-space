import sys; sys.path.insert(0,"editor_runs/gemma_b25/scripts")
from lib import *
e=scale_edit([13,14,15,16,17,18],512,1.0)
print({k:v for k,v in wb.edit_size(e).items() if k!="tensors"})
items=evalset()
ans=wb.answer(items,edit=e)
report(items,ans,"scale L13-18 k512 g1.0")
