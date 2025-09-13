import utils
import os
import json

question = "is diabetes hereditary?"

answer = utils.get_answer_for_question(question)
with open("answer.json", "w") as f:
    json.dump(answer, f)

print(answer)
