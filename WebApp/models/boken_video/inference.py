import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from ensemble import Ensemble
submit = Ensemble()
prediction = submit.inference("./temp/delete.mp4")
#print(prediction)