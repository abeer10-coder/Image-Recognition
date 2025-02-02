# First "clone" git repo in our system
# Second we move our files in .git folder
# Third We check the "status", if not not added to repo then,
# Fourth We use "git add <file> or git add ." to add files in load ing process one by one or all at once respectively.
# Fifth We us "git commit -m 'random msg'" to commit the changes in the file with a message (optional)
# Sixth We use "git push" to upload in the repo final step

from imageai.Classification import ImageClassification
import os

execution_path = os.getcwd()

prediction = ImageClassification()
prediction.setModelTypeAsDenseNet121()
prediction.setModelPath(os.path.join(execution_path, "densenet121-a639ec97.pth"))
prediction.loadModel()

predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "giraffe.jpg"), result_count = 7 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)